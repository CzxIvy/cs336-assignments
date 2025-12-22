import os
import json
import random
import torch
import typer
import wandb
import numpy as np
from typing import Optional, List
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
from unittest.mock import patch
import pathlib

# Import helper functions from your local package (assumed structure based on assignment)
# You must have implemented these in the previous problems.
from tests.adapters import (
    run_tokenize_prompt_and_output,
    run_get_response_log_probs,
    run_sft_microbatch_train_step,
    run_masked_mean,
    run_compute_entropy
)
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from data_utils import *

app = typer.Typer()

CUR_DIR = pathlib.Path(__file__).resolve()
ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data" / "gsm8k"
MODEL_DIR = ROOT_DIR / "model"

# --- vLLM Helper Functions (Source: PDF Page 13-14) ---

def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.6):
    """
    Start the inference process, here we use vLLM to hold a model on
    [cite_start]a GPU separate from the policy. [cite: 366-391]
    """
    from vllm.model_executor import set_random_seed as vllm_set_random_seed
    vllm_set_random_seed(seed)
    
    # Monkeypatch from TRL to handle distributed settings/profiling
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None
    )
    
    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
            device=device,
            dtype="bfloat16",
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )

def load_policy_into_vllm_instance(policy: torch.nn.Module, llm: LLM):
    """
    Copied from TRL grpo_trainer.py. [cite_start]Loads policy weights into vLLM instance. [cite: 392-399]
    """
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())

# --- Data Loading Utilities ---

def load_data(data_path: str, dataset_size: Optional[int] = None, filter_correct: bool = False):
    """
    [cite_start]Loads and optionally filters/subsamples the dataset. [cite: 408-410, 413]
    """
    with open(data_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line.strip()) for line in f]
    
    """
    # Filter for correctness if requested (Part 2 of experiment)
    if filter_correct:
        print(f"Filtering dataset for correctness... Initial size: {len(data)}")
        filtered_data = []
        for example in tqdm(data, desc="Filtering"):
            # Check correctness using the reward function logic
            # Assuming example has 'prompt', 'response' and 'solution' (ground truth)
            # If 'solution' is missing, we might need to parse it from the prompt metadata
            # For this assignment, we assume standard format or provided ground truth.
            
            # Note: The prompt description implies we use the reward function to check.
            # We treat the 'response' as the model output.
            ground_truth = example.get('solution') or example.get('answer') # Adjust based on actual jsonl key
            if ground_truth:
                # We reuse r1_zero_reward_fn logic. 
                # Note: r1_zero_reward_fn expects the full response text.
                reward_dict = r1_zero_reward_fn(example['response'], ground_truth)
                if reward_dict['reward'] == 1.0:
                    filtered_data.append(example)
        
        data = filtered_data
        print(f"Filtered size: {len(data)}")
    """

    # Shuffle and subsample (Part 1 of experiment)
    if dataset_size is not None and dataset_size < len(data):
        random.shuffle(data)
        data = data[:dataset_size]
        print(f"Subsampled dataset to {len(data)} examples.")
    
    return data

# --- Evaluation Loop ---

def evaluate_vllm(llm: LLM, validation_path: str, num_eval_examples: int = 500):
    """
    [cite_start]Evaluates the model using vLLM on a subset of validation data. [cite: 146-157, 363]
    """
    # Load validation data
    with open(validation_path, 'r') as f:
        val_data = [json.loads(line) for line in f]
    
    # Subsample for speed
    if len(val_data) > num_eval_examples:
        val_data = val_data[:num_eval_examples]
    
    prompts = [format_r1_zero_prompt(ex['question']) for ex in val_data] # Ensure prompts are formatted for r1_zero
    ground_truths = [ex.get('solution').split("####", 1)[-1] or ex.get('answer').split("####", 1)[-1] for ex in val_data]
    
    # [cite_start]Sampling params: temp 1.0, max_tokens 1024, stop at </answer> [cite: 141-144]
    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=1024,
        stop=["</answer>"],
        include_stop_str_in_output=True
    )
    
    print("Running vLLM generation for evaluation...")
    outputs = llm.generate(prompts, sampling_params)
    
    correct_count = 0
    total_count = 0
    
    for output, ground_truth in zip(outputs, ground_truths):
        generated_text = output.outputs[0].text
        reward_dict = r1_zero_reward_fn(generated_text, ground_truth)
        if reward_dict['reward'] == 1.0:
            correct_count += 1
        total_count += 1
        
    accuracy = correct_count / total_count
    return accuracy

def log_generations(llm: LLM, validation_data: List[dict], step: int, num_examples: int = 10):
    """
    Generate responses for a few examples and log them to WandB. [cite: 348-356]
    """
    print(f"Logging {num_examples} generations...")
    
    # Randomly sample examples
    examples = random.sample(validation_data, min(num_examples, len(validation_data)))
    prompts = [format_r1_zero_prompt(ex['question']) for ex in examples] # Ensure prompts are formatted for r1_zero
    ground_truths = [ex.get('solution').split("####", 1)[-1] or ex.get('answer').split("####", 1)[-1] for ex in examples]
    
    # Sampling parameters for generation
    # Note: vLLM usually needs logprobs=1 to calculate entropy, 
    # but calculating exact entropy from top-1 logprob is an approximation.
    sampling_params = SamplingParams(
        temperature=1.0,
        max_tokens=1024,
        stop=["</answer>"],
        include_stop_str_in_output=True
    )
    
    outputs = llm.generate(prompts, sampling_params)
    
    # Create a WandB Table
    table = wandb.Table(columns=[
        "Prompt", "Response", "Ground Truth", 
        "Reward", "Format Reward", "Answer Reward", 
        "Length"
    ])
    
    total_len = 0
    correct_len = 0
    incorrect_len = 0
    correct_count = 0
    
    for output, ground_truth, prompt in zip(outputs, ground_truths, prompts):
        response_text = output.outputs[0].text
        
        # Calculate rewards using the provided grader
        reward_dict = r1_zero_reward_fn(response_text, ground_truth)
        
        # Update statistics
        resp_len = len(output.outputs[0].token_ids)
        total_len += resp_len
        
        if reward_dict['reward'] == 1.0:
            correct_len += resp_len
            correct_count += 1
        else:
            incorrect_len += resp_len
            
        # Add row to table
        table.add_data(
            prompt[:100] + "...",  # Truncate prompt for display
            response_text,
            ground_truth,
            reward_dict['reward'],
            reward_dict['format_reward'],
            reward_dict['answer_reward'],
            resp_len
        )
        
    # Log the table
    wandb.log({f"eval/generations_step_{step}": table}, step=step)
    
    # Log length statistics [cite: 356]
    avg_len = total_len / len(outputs) if outputs else 0
    avg_len_correct = correct_len / correct_count if correct_count > 0 else 0
    avg_len_incorrect = incorrect_len / (len(outputs) - correct_count) if (len(outputs) - correct_count) > 0 else 0
    
    wandb.log({
        "eval/avg_response_length": avg_len,
        "eval/avg_len_correct": avg_len_correct,
        "eval/avg_len_incorrect": avg_len_incorrect,
        "eval_step": step
    })

# --- Main Training Function ---

@app.command()
def train(
    dataset_path: str = DATA_DIR / "train_convert.jsonl",
    validation_path: str = DATA_DIR / "test.jsonl",
    model_path: str = MODEL_DIR / "Qwen2.5-Math-1.5B",
    output_dir: str = CUR_DIR / "checkpoints/sft_run",
    dataset_size: int = typer.Option(None, help="Number of examples to use (128, 256, etc.)"),
    filter_correct: bool = typer.Option(False, help="Filter dataset for only correct reasoning traces"),
    learning_rate: float = 1e-5,
    batch_size: int = 4, # Global batch size
    micro_batch_size: int = 1, # Per-device micro batch size
    epochs: int = 2,
    eval_every_steps: int = 50,
    seed: int = 42,
):
    # Set seeds
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # [cite_start]1. Setup Devices [cite: 364]
    # Policy on GPU 0, vLLM on GPU 1
    policy_device = "cuda:0"
    vllm_device = "cuda:1"
    
    # [cite_start]2. Initialize WandB [cite: 402-405]
    run_name = f"sft_ds{dataset_size}_filter{filter_correct}_lr{learning_rate}"
    wandb.init(project="cs336_alignment_sft", name=run_name, config={
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "dataset_size": dataset_size,
        "filter_correct": filter_correct
    })
    wandb.define_metric("train_step")
    wandb.define_metric("eval_step")
    wandb.define_metric("train/*", step_metric="train_step")
    wandb.define_metric("eval/*", step_metric="eval_step")

    # 3. Load Data
    train_data = load_data(dataset_path, dataset_size, filter_correct)
    val_data = load_data(validation_path, dataset_size, filter_correct)
    
    # [cite_start]4. Load Policy Model [cite: 190-196]
    print(f"Loading policy model on {policy_device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to(policy_device)
    model.train()

    # [cite_start]5. Initialize vLLM on separate GPU [cite: 367-391]
    print(f"Initializing vLLM on {vllm_device}...")
    # Note: we point vLLM to the base model path initially.
    # We will swap weights using load_policy_into_vllm_instance later.
    llm = init_vllm(model_path, device=vllm_device, seed=seed)

    # 6. Optimizer & Gradient Accumulation Setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    gradient_accumulation_steps = batch_size // micro_batch_size
    assert batch_size % micro_batch_size == 0, "Batch size must be divisible by micro batch size"

    # [cite_start]7. Training Loop (Algorithm 1) [cite: 167-176]
    global_step = 0
    optimizer.zero_grad()
    
    total_steps = (len(train_data) // batch_size) * epochs
    progress_bar = tqdm(total=total_steps, desc="Training")

    for epoch in range(epochs):
        random.shuffle(train_data)
        
        # Iterate through microbatches
        for i in range(0, len(train_data), micro_batch_size):
            batch_data = train_data[i : i + micro_batch_size]
            if len(batch_data) == 0: continue

            # [cite_start]Tokenize [cite: 247-261]
            prompt_strs = [ex['prompt'] for ex in batch_data]
            output_strs = [ex['response'] for ex in batch_data]
            
            # This calls your helper function
            tokenized = run_tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer)
            
            input_ids = tokenized['input_ids'].to(policy_device)
            labels = tokenized['labels'].to(policy_device)
            response_mask = tokenized['response_mask'].to(policy_device)

            # Forward pass
            outputs = model(input_ids)
            
            # [cite_start]Get Log Probs [cite: 290-305]
            log_probs_dict = run_get_response_log_probs(
                model=model, # Passed just for signature, likely not used inside if logits passed
                input_ids=input_ids,
                labels=labels, # Not strictly needed for just log_probs extraction if implementation varies
                return_token_entropy=True
            )
            # Depending on your implementation of get_response_log_probs, 
            # it might take logits or run the model itself. 
            # If it takes model, it does the forward pass. 
            # Assuming here we passed the model and it returned the log probs derived from it.
            # *Correction based on standard patterns*: Usually we compute log_probs from logits.
            # However, the helper `get_response_log_probs` in the assignment takes the model.
            # So we might not need the explicit `outputs = model(input_ids)` line above if the helper does it.
            # Let's assume the helper handles the forward pass or extracting from logits.
            
            policy_log_probs = log_probs_dict["log_probs"]
            token_entropy = log_probs_dict["token_entropy"]

            # [cite_start]Compute Loss & Backward (Microbatch Step) [cite: 329-347]
            # This helper handles loss.backward() internally
            loss, metadata = run_sft_microbatch_train_step(
                policy_log_probs=policy_log_probs,
                response_mask=response_mask,
                gradient_accumulation_steps=gradient_accumulation_steps
            )

            # Logging microbatch metrics
            if i % (micro_batch_size * 10) == 0:
                avg_entropy = run_masked_mean(token_entropy, response_mask).item()
                wandb.log({
                    "train/loss_micro": loss.item() * gradient_accumulation_steps,
                    "train/entropy": avg_entropy,
                    "train_step": global_step
                })

            # [cite_start]Optimizer Step (Gradient Accumulation) [cite: 211-238]
            # Check if we completed a full global batch
            current_micro_step = i // micro_batch_size
            if (current_micro_step + 1) % gradient_accumulation_steps == 0:
                # [cite_start]Gradient Clipping [cite: 406]
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                progress_bar.update(1)

                # [cite_start]Periodic Evaluation [cite: 146, 363]
                if global_step % eval_every_steps == 0:
                    print(f"\nEvaluating at step {global_step}...")
                    
                    # [cite_start]Sync weights to vLLM [cite: 392-399]
                    load_policy_into_vllm_instance(model, llm)
                    
                    val_acc = evaluate_vllm(llm, validation_path)
                    
                    log_generations(llm, val_data, global_step, num_examples=10)

                    print(f"Validation Accuracy: {val_acc:.4f}")
                    wandb.log({
                        "eval/accuracy": val_acc,
                        "eval_step": global_step
                    })
                    
                    # [cite_start]Save checkpoint [cite: 204-208]
                    ckpt_path = os.path.join(output_dir, f"step_{global_step}")
                    model.save_pretrained(ckpt_path)
                    tokenizer.save_pretrained(ckpt_path)

    # Final Save
    model.save_pretrained(os.path.join(output_dir, "final"))
    tokenizer.save_pretrained(os.path.join(output_dir, "final"))
    print("Training Complete.")

if __name__ == "__main__":
    app()
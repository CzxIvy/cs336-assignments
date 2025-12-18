import json
import os
from typing import List, Dict, Callable, Any
from datetime import datetime
import pathlib

import torch
from vllm import LLM, SamplingParams
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

def load_math_validation_data(data_path: str) -> List[Dict[str, Any]]:
    """Load MATH validation examples from JSONL file."""
    examples = []
    with open(data_path, 'r') as f:
        for line in f:
            example = json.loads(line.strip())
            examples.append(example)
    return examples

import pyarrow.parquet as pq
from typing import List, Dict, Any

def format_prompt_with_r1_zero(question: str) -> str:
    """Format math question using r1_zero prompt template."""
    # Based on the r1_zero_reward_fn function requirements, the prompt should
    # ask the model to output in the format: </think> <answer>...</answer>
    prompt = f"""A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.
    User: {question}
    Assistant: <think>"""
    return prompt

def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], Dict[str, float]],
    prompts: List[str],
    eval_sampling_params: SamplingParams,
    ground_truths: List[Any],
    examples: List[Dict[str, Any]]
) -> None:
    """
    Evaluate a language model on a list of prompts, compute evaluation metrics,
    and serialize results to disk.
    """
    # Generate outputs for all prompts
    print(f"Generating outputs for {len(prompts)} prompts...")
    outputs = vllm_model.generate(prompts, eval_sampling_params)
    
    # Calculate evaluation metrics
    results = []
    total_correct = 0
    total_format_correct = 0
    
    for i, (example, output, gt) in enumerate(zip(examples, outputs, ground_truths)):
        generated_text = output.outputs[0].text
        reward_dict = reward_fn(generated_text, gt)
        
        result = {
            "example_id": i,
            "question": example["question"],
            "ground_truth": gt,
            "generated_output": generated_text,
            "reward_dict": reward_dict,
            "is_correct": reward_dict["answer_reward"] == 1.0,
            "is_format_correct": reward_dict["format_reward"] == 1.0
        }
        
        results.append(result)
        total_correct += 1 if result["is_correct"] else 0
        total_format_correct += 1 if result["is_format_correct"] else 0
    
    # Compute metrics
    accuracy = total_correct / len(results)
    format_accuracy = total_format_correct / len(results)
    
    metrics = {
        "total_examples": len(results),
        "correct_answers": total_correct,
        "accuracy": accuracy,
        "format_correct": total_format_correct,
        "format_accuracy": format_accuracy,
        "evaluation_time": datetime.now().isoformat()
    }
    
    # Serialize results to disk
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save detailed results
    results_path = os.path.join(output_dir, "math_evaluation_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Save metrics summary
    metrics_path = os.path.join(output_dir, "math_evaluation_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Print evaluation summary
    print("\n=== Evaluation Summary ===")
    print(f"Total examples: {metrics['total_examples']}")
    print(f"Correct answers: {metrics['correct_answers']}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Format correct: {metrics['format_correct']}")
    print(f"Format accuracy: {metrics['format_accuracy']:.4f}")
    print(f"\nResults saved to: {results_path}")
    print(f"Metrics saved to: {metrics_path}")

def main():
    ROOT_DIR = pathlib.Path(__file__).parent.parent
    DATA_DIR = ROOT_DIR / "data" / "gsm8k"
    # Configuration
    MODEL_NAME = "Qwen2.5-Math-1.5B-Instruct"
    MODEL_PATH = str(ROOT_DIR / "model" / MODEL_NAME)
    DATA_PATH = str(DATA_DIR / "test.jsonl")
    
    # Load validation data
    print(f"Loading MATH validation data from {DATA_PATH}...")
    examples = load_math_validation_data(DATA_PATH)
    print(f"Loaded {len(examples)} examples.")
    
    # Prepare prompts and ground truths
    prompts = []
    ground_truths = []
    
    for example in examples:
        # Format question with r1_zero prompt
        prompt = format_prompt_with_r1_zero(example["question"])
        prompts.append(prompt)
        
        # Extract ground truth answer from MATH dataset
        # MATH dataset typically has:
        # - "answer": Final answer (sometimes in \boxed{} format)
        # - "solution": Full solution with steps
        if "answer" in example:
            gt = example["answer"]
        elif "solution" in example:
            # Extract final answer from solution
            solution = example["solution"]
            # Look for \boxed{} in solution as final answer
            if "\\boxed" in solution:
                gt = solution
            else:
                # Fallback to solution as ground truth
                gt = solution
        else:
            # Fallback to problem itself if no answer found
            gt = example["question"]
        
        ground_truths.append(gt)
    
    # Initialize vllm model
    print(f"Initializing vLLM with model: {MODEL_NAME}...")
    # llm = LLM(
    #     model=MODEL_PATH,
    #     dtype=torch.float16,
    #     gpu_memory_utilization=0.8,
    #     max_model_len=4096
    # )
    llm = LLM(model=MODEL_PATH)
    
    # Configure sampling parameters for evaluation
    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=1024,
        stop=["</answer>"]
    )
    
    # Run evaluation
    print("Starting evaluation...")
    evaluate_vllm(
        vllm_model=llm,
        reward_fn=r1_zero_reward_fn,
        prompts=prompts,
        eval_sampling_params=sampling_params,
        ground_truths=ground_truths,
        examples=examples
    )

if __name__ == "__main__":
    main()

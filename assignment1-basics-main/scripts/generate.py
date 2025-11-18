import os
import sys
import json
import torch
import pickle
import pathlib
import argparse
from model import BasicsTransformerLM
project_root = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
from tests.adapters import *

tokenizer_dir = pathlib.Path(__file__).resolve().parent / 'tokenizer'
vocab_path = os.path.join(tokenizer_dir, 'tinystories_bpe_vocab.pkl')
merges_path = os.path.join(tokenizer_dir, 'tinystories_bpe_merges.pkl')
special_tokens = ['<|endoftext|>']

device = 'cuda'

with open(vocab_path, 'r') as f:
    vocab = pickle.load(vocab_path)
with open(merges_path, 'r') as f:
    merges = pickle.load(merges_path)
    
tokenizer = Tokenizer(
    vocab=vocab,
    merges=merges,
    special_tokens=special_tokens,
)

ckpt_path = pathlib.Path(__file__).resolve().parent / 'checkpoints/ckpt_iter5000.pt'
config_path = pathlib.Path(__file__).resolve().parent / 'config.json'

def main():
    parser = argparse.ArgumentParser(description='Text Generation Inference')
    parser.add_argument('--prompt', type=str, default='Once upon a time, there was a pretty girl', help='Input prompt for generation')
    parser.add_argument('--max_new_tokens', type=int, default=256, help='Maximum new tokens to generate')
    parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature')
    parser.add_argument('--top_p', type=int, default=30, help='Top-p sampling')
    args = parser.parse_args()
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    model = BasicsTransformerLM(**config['model'])
    model.to(device)
    
    with open(ckpt_path, 'r') as f:
        checkpoint = torch.load(f, weights_only=False)
    
    model.load_state_dict(checkpoint['model_checkpoint'])
    
    input_ids = tokenizer.encode(args.prompt)
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    
    output_tokens = model.generate(
        x=input_tensor,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        eos_token_id=256
    )
    output_ids = output_tokens[0].cpu().numpy().tolist()
    
    full_ids = input_ids + output_ids
    text = tokenizer.decode(full_ids)
    print("输入：", args.prompt)
    print("输出：", text)
    

if __name__ == '__main__':
    main()
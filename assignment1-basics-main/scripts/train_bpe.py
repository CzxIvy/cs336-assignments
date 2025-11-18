import os
import sys
import pickle
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
from tests.adapters import run_train_bpe

current_dir = os.path.dirname(os.path.abspath(__file__))
train_data_path = os.path.join(current_dir, 'data/TinyStoriesV2-GPT4-train.txt')

# tokenizer保存路径
tokenizer_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tokenizer')
vocab_path = os.path.join(tokenizer_dir, 'tinystories_bpe_vocab.pkl')
merges_path = os.path.join(tokenizer_dir, 'tinystories_bpe_merges.pkl')

vocab_size = 10000
special_tokens = ['<|endoftext|>']

if __name__ == '__main__':
    vocab, merges = run_train_bpe(
        input_path=train_data_path,
        vocab_size=vocab_size,
        special_tokens=special_tokens
    )

    os.makedirs(tokenizer_dir, exist_ok=True)
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    with open(merges_path, 'wb') as f:
        pickle.dump(merges, f)
        
    longest_token = max(vocab.values(), key=len)
    print('最长token：', longest_token, '长度：', len(longest_token))
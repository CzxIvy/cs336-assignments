import pickle
import os
import pathlib
import sys
import numpy as np
import regex
from tqdm import tqdm
import multiprocessing
project_root = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
from tests.adapters import Tokenizer
from cs336_basics.pretokenization_example import find_chunk_boundaries

tokenizer_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tokenizer')
vocab_path = os.path.join(tokenizer_dir, 'tinystories_bpe_vocab.pkl')
merges_path = os.path.join(tokenizer_dir, 'tinystories_bpe_merges.pkl')

data_dir = pathlib.Path(__file__).resolve().parent / 'data'
train_txt_data_path = os.path.join(data_dir, 'TinyStoriesV2-GPT4-train.txt')
valid_txt_data_path = os.path.join(data_dir, 'TinyStoriesV2-GPT4-valid.txt')
train_data_path = os.path.join(data_dir, 'train.dat')
valid_data_path = os.path.join(data_dir, 'valid.dat')
temp_dir = os.path.join(data_dir, 'temp')
os.makedirs(temp_dir, exist_ok=True)

special_tokens = ['<|endoftext|>']

with open(vocab_path, 'rb') as f:
    vocab = pickle.load(f)
with open(merges_path, 'rb') as f:
    merges = pickle.load(f)
    
tokenizer = Tokenizer(
    vocab=vocab,
    merges=merges,
    special_tokens=special_tokens
)

def encode_chunk(chunk_param):
    chunk, special_tokens, chunk_index, temp_dir = chunk_param
    # 为每个chunk创建单独的文件
    temp_file = os.path.join(temp_dir, f"chunk_{chunk_index}.txt")
    
    sorted_special_tokens = sorted(special_tokens, key=len, reverse=True)
    special_tokens_pattern = '|'.join(map(regex.escape, sorted_special_tokens))
    pattern_with_capture = f'({special_tokens_pattern})'
    
    all_tokens = []
    out_f = open(temp_file, 'w')
    for sub_chunk in regex.split(pattern_with_capture, chunk):
        all_tokens.extend(tokenizer.encode(sub_chunk))
        out_f.write(' '.join(map(str, tokenizer.encode(sub_chunk))) + '\n')
    out_f.close()
    
    return len(all_tokens), temp_file

def parallel_encode_file(filename, temp_dir, save_path, special_tokens):
    # 创建临时目录
    os.makedirs(temp_dir, exist_ok=True)
    
    with open(filename, "rb") as f:
        num_processes = 12
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
        
        with multiprocessing.Pool(processes=num_processes) as pool:
            chunk_params = []
            for i, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
                f.seek(start)
                chunk = f.read(end - start).decode("utf-8", errors="ignore")
                chunk_params.append((chunk, special_tokens, i, temp_dir))

            results = pool.map(encode_chunk, chunk_params)
            
        pool.close()
        pool.join()
        
        # 收集所有token并合并
        total_tokens = 0
        all_temp_files = []
        for tokens_len, temp_file in results:
            total_tokens += tokens_len
            all_temp_files.append(temp_file)
        
        print(f"total tokens: {total_tokens}")
        
        # 按顺序合并所有临时文件
        dtype = np.int32
        tokens_mm = np.memmap(save_path, mode='w+', dtype=dtype, shape=(total_tokens,))
        
        pos = 0
        for temp_file in sorted(all_temp_files):  # 按chunk顺序排序
            with open(temp_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        token_ids = list(map(int, line.split()))
                        tokens_mm[pos:pos+len(token_ids)] = token_ids
                        pos += len(token_ids)
        
        tokens_mm.flush()
    
    # 删除临时目录
    os.rmdir(temp_dir)
    print(f"Encoded data saved to {save_path}")

if __name__ == '__main__':
    print("=== 测试 Tokenizer ===")
    test_texts = [
        "Once upon a time, there was a little robot.",
        "Hello world! <|endoftext|> Some more text.",
        "<|endoftext|>",
        "你好，世界！"
    ]

    for text in test_texts:
        print(f"\n原文: {text}")
        encoded = tokenizer.encode(text)
        print("编码:", encoded)

        byte_tokens = [tokenizer.vocab[token_id] for token_id in encoded]
        str_tokens = [b.decode("utf-8", errors="replace") for b in byte_tokens]
        print("分词（可读）:", str_tokens)

        decoded = tokenizer.decode(encoded)
        print("解码:", decoded)
        print("是否完全还原:", decoded == text)
        
    parallel_encode_file(train_txt_data_path, temp_dir, train_data_path, special_tokens)
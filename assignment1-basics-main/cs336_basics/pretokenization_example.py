import os
from typing import BinaryIO, List, Tuple, Dict

import multiprocessing
from collections import Counter
import re
import regex
import time
import pathlib

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


## Usage(serial version) ##
def serial_process_file(filename: str):
    with open(filename, "rb") as f:
        num_processes = 4
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            # Run pre-tokenization on your chunk and store the counts for each pre-token

def to_bytes_tuple(word: str):
    return tuple(bytes([b]) for b in word.encode("utf-8"))

## Usage(parallel version) ##
def process_chunk(chunk_param) -> Dict[Tuple[bytes], int]:
    chunk, special_tokens = chunk_param
    # handle special tokens
    sorted_special_tokens = sorted(special_tokens, key=len, reverse=True)
    special_tokens_pattern = '|'.join(map(regex.escape, sorted_special_tokens))
    counter = Counter()
    for sub_chunk in regex.split(special_tokens_pattern, chunk):
        for token in regex.finditer(PAT, sub_chunk):
            counter[to_bytes_tuple(token.group(0))] += 1
    return dict(counter)
        
    
def parallel_process_file(filename: str, special_tokens: list[str]) -> Dict[Tuple[bytes], int]:
    with open(filename, "rb") as f:
        num_processes = 8
        # num_processes = multiprocessing.cpu_count()
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        # Create a pool of workers
        with multiprocessing.Pool(processes=num_processes) as pool:
            chunk_params = []
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                f.seek(start)
                chunk = f.read(end - start).decode("utf-8", errors="ignore")
                chunk_params.append(((chunk, special_tokens)))

            # Map the pre-tokenization function to the chunks
            results = pool.map(process_chunk, chunk_params)
        
        # 关闭进程池
        pool.close()
        pool.join()

        # TODO: Combine results from all processes
        total_counter = Counter()
        for counter_dict in results:
            total_counter.update(counter_dict)

    return dict(total_counter)

if __name__ == "__main__":
    root_dir = pathlib.Path(__file__).parent.parent
    data_dir = root_dir / "tests" / "fixtures"
    txt_data_path = data_dir / "tinystories_sample.txt"
    
    special_tokens = ['<|endoftext|>']
    
    parallel_process_file(str(txt_data_path), special_tokens)
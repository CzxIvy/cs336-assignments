import json
from common import gpt2_bytes_to_unicode
from pathlib import Path

vocab_path = Path(__file__).parent / "fixtures/gpt2_vocab.json"
g2b = gpt2_bytes_to_unicode()
gpt2_byte_decoder = {v: k for k, v in g2b.items()}

print("Checking vocabulary file:", vocab_path)
print("Number of characters in gpt2_byte_decoder:", len(gpt2_byte_decoder))
print("Sample of valid characters:", list(gpt2_byte_decoder.keys())[:10])

try:
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
        print("\nFirst 10 vocab entries:")
        for i, (token, idx) in enumerate(list(vocab.items())[:10]):
            print(f"{idx}: {repr(token)}")
except Exception as e:
    print("Error reading vocab file:", e)

bad = []
for token_str, idx in vocab.items():
    for ch in token_str:
        if ch not in gpt2_byte_decoder:
            bad.append((ch, repr(ch), token_str, idx))
            break

if not bad:
    print("\nNo problematic tokens found.")
else:
    print(f"\nFound {len(bad)} problematic tokens (char, repr(char), token_string, index):")
    for entry in bad[:50]:
        print(entry)
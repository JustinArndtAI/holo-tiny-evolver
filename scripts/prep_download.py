"""
HKM Pipeline - Dataset and Model Download Script
Phase: Prep - Step 5
"""

from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer
import os

print("Starting downloads...")

# 1. Download WikiText dataset (small test subset)
print("\n1. Downloading WikiText dataset...")
try:
    wikitext = load_dataset('Salesforce/wikitext', 'wikitext-103-raw-v1', split='train[:1%]')
    print(f"   [OK] WikiText subset loaded: {len(wikitext)} samples")
except Exception as e:
    print(f"   [ERROR] WikiText error: {e}")

# 2. Download FB15k-237 dataset
print("\n2. Downloading FB15k-237 dataset...")
try:
    fb15k = load_dataset('KGraph/FB15k-237')
    print(f"   [OK] FB15k-237 loaded: {list(fb15k.keys())} splits")
except Exception as e:
    print(f"   [ERROR] FB15k-237 error: {e}")

# 3. Download DistilBERT model
print("\n3. Downloading DistilBERT model...")
try:
    model = AutoModel.from_pretrained('distilbert/distilbert-base-uncased')
    tokenizer = AutoTokenizer.from_pretrained('distilbert/distilbert-base-uncased')
    print("   [OK] DistilBERT model and tokenizer downloaded")
except Exception as e:
    print(f"   [ERROR] DistilBERT error: {e}")

# 4. Download Microsoft Phi-1.5 model
print("\n4. Downloading Microsoft Phi-1.5 model...")
try:
    phi_model = AutoModel.from_pretrained('microsoft/phi-1_5', trust_remote_code=True)
    print("   [OK] Microsoft Phi-1.5 model downloaded")
except Exception as e:
    print(f"   [ERROR] Phi-1.5 error: {e}")

# Check cache location
cache_dir = os.path.expanduser("~/.cache/huggingface")
if os.path.exists(cache_dir):
    import shutil
    total_size = sum(os.path.getsize(os.path.join(dirpath, filename))
                    for dirpath, dirnames, filenames in os.walk(cache_dir)
                    for filename in filenames)
    print(f"\nTotal cache size: {total_size / (1024**3):.2f} GB")
    print(f"Cache location: {cache_dir}")

print("\nDownload script completed!")
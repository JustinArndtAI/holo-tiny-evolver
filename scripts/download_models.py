"""
Download DistilBERT and Phi models - Fixed version
"""

from transformers import AutoModel, AutoTokenizer
import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# 1. Download DistilBERT
print("\n1. Downloading DistilBERT...")
try:
    model = AutoModel.from_pretrained('distilbert-base-uncased')
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    print("   [OK] DistilBERT model and tokenizer downloaded")
    
    # Test the model
    inputs = tokenizer("Hello world", return_tensors="pt")
    outputs = model(**inputs)
    print(f"   [OK] Model test successful, output shape: {outputs.last_hidden_state.shape}")
except Exception as e:
    print(f"   [ERROR] DistilBERT: {e}")

# 2. Download Microsoft Phi-1.5
print("\n2. Downloading Microsoft Phi-1.5...")
try:
    from transformers import AutoModelForCausalLM
    phi_model = AutoModelForCausalLM.from_pretrained(
        "microsoft/phi-1_5", 
        trust_remote_code=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    phi_tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1_5", trust_remote_code=True)
    print("   [OK] Microsoft Phi-1.5 model downloaded")
    
    # Test the model
    inputs = phi_tokenizer("Once upon a time", return_tensors="pt")
    with torch.no_grad():
        outputs = phi_model.generate(**inputs, max_length=20)
    result = phi_tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"   [OK] Model test: '{result[:50]}...'")
except Exception as e:
    print(f"   [ERROR] Phi-1.5: {e}")

print("\nModel downloads completed!")
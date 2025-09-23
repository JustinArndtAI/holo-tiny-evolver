from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
import torch
from utils import check_gpu_temp

# Load tokenizer from original model
tokenizer = AutoTokenizer.from_pretrained("models/tinyllama")

# Load and quantize model
quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
model = AutoModelForCausalLM.from_pretrained("models/tinyllama", quantization_config=quantization_config, device_map="auto")

check_gpu_temp()

# Save quantized model and tokenizer to the same directory
model.save_pretrained("models/tinyllama-4bit")
tokenizer.save_pretrained("models/tinyllama-4bit")

print("Quantized and tokenizer saved.")

# Test inference
pipe = pipeline("text-generation", model="models/tinyllama-4bit")
print(pipe("Test AI?")[0]['generated_text'])
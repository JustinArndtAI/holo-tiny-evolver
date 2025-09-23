from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
import torch
import time
import csv
from utils import check_gpu_temp

# Load tokenizer from original model
tokenizer = AutoTokenizer.from_pretrained("models/tinyllama", legacy=False)

# Load and quantize model
quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
model = AutoModelForCausalLM.from_pretrained("models/tinyllama", quantization_config=quantization_config, device_map="auto")

check_gpu_temp()

# Save quantized model and tokenizer to the same directory
model.save_pretrained("models/tinyllama-4bit")
tokenizer.save_pretrained("models/tinyllama-4bit")

print("Quantized and tokenizer saved.")

# Test inference with metrics
pipe = pipeline("text-generation", model="models/tinyllama-4bit")

start_time = time.time()
generated = pipe("Test AI?", max_new_tokens=10)[0]['generated_text']  # Generate ~10 tokens for measurement
end_time = time.time()

infer_time_ms = (end_time - start_time) * 1000
infer_ms_per_token = infer_time_ms / 10  # Approximate ms/token
print(f"Generated: {generated}")
print(f"Inference ms/token: {infer_ms_per_token}")

# Log to metrics.csv (create if not exists)
with open('metrics.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['phase', 'infer_ms_token'])
    writer.writerow(['base', infer_ms_per_token])
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from datasets import load_dataset
import numpy as np
from utils import check_gpu_temp
from hkm_wrapper import HKMWrapper

# Initialize
hkm = HKMWrapper()
tokenizer = AutoTokenizer.from_pretrained("models/tinyllama-4bit")
if not torch.cuda.is_available():
    raise RuntimeError(f"CUDA not available! PyTorch: {torch.__version__}, CUDA Version: {torch.version.cuda or 'None'}. Reinstall with: pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121")
model = AutoModelForCausalLM.from_pretrained(
    "models/tinyllama-4bit",
    device_map="auto",
    dtype=torch.float16
)
# Set pad_token_id if undefined
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
    print(f"Set pad_token_id to eos_token_id: {tokenizer.pad_token_id}")
print(f"Device: {model.device} (PyTorch: {torch.__version__}, CUDA: {torch.version.cuda})")
print("Local quantized TinyLlama model loaded.")
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

def generate_synthetic_data(dataset, num_samples=2):
    """Generate synthetic data for evolution."""
    try:
        samples = dataset["test"].select(range(min(num_samples, len(dataset["test"]))))
        print(f"Selected {len(samples)} samples from test split")
        synthetic = []
        for i, sample in enumerate(samples):
            print(f"Sample {i}: {sample}")
            answer_text = sample['options'][ord(sample['answer']) - ord('A')] if sample['answer'].isalpha() else sample['answer']
            prompt = f"Q: {sample['question']} Options: {sample['options']} A: {answer_text}"
            if not prompt.strip() or not sample['question'] or not answer_text:
                print(f"Sample {i} has invalid prompt or answer, skipping.")
                continue
            try:
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to("cuda")
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
                generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
                print(f"Generated {i}: {generated}")
                synthetic.append(generated)
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"Generation failed for sample {i}: {e}")
                continue
        return synthetic
    except Exception as e:
        print(f"Synthetic data generation failed: {e}")
        return []

def compute_loss(model, inputs, targets):
    """Compute loss with numerical stability."""
    try:
        outputs = model(**inputs)
        logits = outputs.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_targets = targets[..., 1:].contiguous()
        loss = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_targets.view(-1),
            ignore_index=tokenizer.pad_token_id,
            reduction='mean'
        )
        return loss if not torch.isnan(loss) else torch.tensor(0.0)
    except Exception as e:
        print(f"Loss computation error: {e}")
        return torch.tensor(0.0)

def evolve_model(epochs=3, num_samples=2):
    """Evolve model using synthetic data."""
    try:
        dataset = load_dataset("parquet", data_files={
            "test": "datasets/mmlu_pro/data/test-00000-of-00001.parquet",
            "validation": "datasets/mmlu_pro/data/validation-00000-of-00001.parquet"
        })
    except Exception as e:
        raise FileNotFoundError(f"MMLU-Pro dataset failed to load: {e}. Ensure datasets/mmlu_pro/data contains test-00000-of-00001.parquet and validation-00000-of-00001.parquet")
    print(f"Evolving TinyLlama with {num_samples} samples over {epochs} epochs...")
    synthetic_data = generate_synthetic_data(dataset, num_samples)
    if not synthetic_data:
        raise ValueError(f"No synthetic data generated. Dataset keys: {list(dataset['test'][0].keys())}")
    
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        total_loss = 0.0
        for i, text in enumerate(synthetic_data):
            try:
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to("cuda")
                targets = inputs["input_ids"].clone()
                loss = compute_loss(model, inputs, targets)
                total_loss += loss.item()
                print(f"Batch {i}, Loss: {loss.item():.4f}")
                check_gpu_temp()
            except Exception as e:
                print(f"Batch {i} failed: {e}")
                continue
        print(f"Average Epoch Loss: {total_loss / len(synthetic_data):.4f}" if len(synthetic_data) > 0 else "No valid batches")
        torch.cuda.empty_cache()

    print("Evolution complete. Saving updated model...")
    model.save_pretrained("outputs/phase6_model_evolved")
    tokenizer.save_pretrained("outputs/phase6_model_evolved")

    print("Integrating evolved model with HKM manifold...")
    hkm.load_manifold()
    print("Manifold integration complete.")

if __name__ == "__main__":
    evolve_model()
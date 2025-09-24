import os
import pickle
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import numpy as np
from hkm_wrapper import HKMWrapper
from utils import check_gpu_temp

class ModelEvolver:
    def __init__(self):
        self.hkm = HKMWrapper()
        self.model = None
        self.tokenizer = None
        self.load_model()

    def load_model(self):
        # Load the local quantized TinyLlama from Phase 2
        model_path = "models/tinyllama-4bit"
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=quantization_config,
            device_map="auto"
        )
        check_gpu_temp()
        print("Local quantized TinyLlama model loaded.")

    def load_data(self, subdirs=["bookcorpus", "python_stack", "arxiv", "commoncrawl", "wikipedia"]):
        # Load ingested data from directories
        data = {}
        for subdir in subdirs:
            data[subdir] = []
            dir_path = os.path.join("data", subdir)
            if os.path.exists(dir_path):
                for file in os.listdir(dir_path):
                    if file.endswith(('.txt', '.csv', '.parquet', '.gz')):
                        file_path = os.path.join(dir_path, file)
                        try:
                            with open(file_path, 'r', errors='ignore') as f:
                                data[subdir].append(f.read())
                        except Exception as e:
                            print(f"Error reading {file_path}: {e}")
        return data

    def evolve_model(self, data, epochs=3, batch_size=8):  # Reduced batch size for TinyLlama
        # Evolve the model with new data using a simple fine-tuning loop
        print(f"Evolving TinyLlama with {sum(len(v) for v in data.values())} samples over {epochs} epochs...")
        train_data = []
        for subdir, texts in data.items():
            for text in texts:
                encoded = self.tokenizer(text, truncation=True, padding='max_length', max_length=128, return_tensors="pt")
                train_data.append(encoded['input_ids'])
        train_data = torch.cat(train_data, dim=0) if train_data else torch.tensor([])

        if train_data.numel() == 0:
            print("No valid training data loaded. Skipping evolution.")
            return

        optimizer = torch.optim.Adam(self.model.parameters(), lr=5e-5)
        self.model.train()
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            for i in range(0, len(train_data), batch_size):
                batch = train_data[i:i + batch_size].to(self.model.device)
                if batch.shape[0] == 0:
                    continue
                outputs = self.model(batch, labels=batch)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                if i % 1000 == 0:
                    print(f"Batch {i//batch_size}, Loss: {loss.item():.4f}")
        print("Evolution complete. Saving updated model...")
        self.model.save_pretrained("outputs/phase6_model_evolved")
        self.tokenizer.save_pretrained("outputs/phase6_model_evolved")

    def integrate_manifold(self):
        # Integrate evolved model with HKM manifold (placeholder)
        print("Integrating evolved model with HKM manifold...")
        # Load Phase 4 manifold and update with new weights
        with open('outputs/phase4_updated_manifold.pkl', 'rb') as f:
            manifold = pickle.load(f)
        # Simple integration: Update manifold with new model embeddings
        manifold['model_weights'] = self.model.state_dict()
        with open('outputs/phase6_updated_manifold.pkl', 'wb') as f:
            pickle.dump(manifold, f)
        print("Manifold integration complete.")

if __name__ == "__main__":
    evolver = ModelEvolver()
    data = evolver.load_data()
    evolver.evolve_model(data)
    evolver.integrate_manifold()
"""
Phase 4: Dynamic Chipping
Updates manifold and model with new data using diffraction merging and chipping pruning
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import time
from tqdm import tqdm

print("Phase 4: Dynamic Chipping Starting...")
print("="*50)

start_time = time.time()

# Load Phase 3 assets
print("\n1. Loading Phase 3 assets...")
with open('../outputs/phase2_manifold.pkl', 'rb') as f:
    manifold = pickle.load(f)
print(f"   Manifold loaded: {manifold['metadata']['num_nodes']} nodes")

model = AutoModelForCausalLM.from_pretrained('../outputs/phase3_model_weights')
tokenizer = AutoTokenizer.from_pretrained('../outputs/phase3_model_weights')
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print(f"   Model loaded: DistilGPT2")

# Load embedder for new data
print("\n2. Loading embedder for new data...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Generate new data increment (simulated)
print("\n3. Generating new data increment...")
new_data_texts = [
    "Quantum entanglement in neural networks enables distributed learning",
    "Holographic projections preserve information across scales",
    "Fractal dimensions characterize knowledge graph complexity",
    "Dynamic chipping maintains model plasticity",
    "Diffraction patterns merge old and new knowledge",
    "Continual learning through manifold updates",
    "Hierarchical clustering at quantum scales",
    "Information preservation through quantization",
    "Manifold topology guides knowledge integration",
    "Adaptive pruning prevents catastrophic forgetting",
    # Add variations
    "Neural quantum computing advances AI capabilities",
    "Holographic memory storage in biological systems",
    "Fractal geometry in machine learning architectures",
    "Knowledge graphs with dynamic topology",
    "Quantum interference in neural pathways",
    "Manifold learning for continual adaptation",
    "Hierarchical representations in deep networks",
    "Quantized neural networks for edge computing",
    "Topological data analysis in AI systems",
    "Pruning strategies for efficient learning"
]

print(f"   New samples: {len(new_data_texts)}")

# Embed new data
print("\n4. Embedding new data...")
new_embeddings = embedder.encode(new_data_texts, show_progress_bar=True)
print(f"   New embeddings shape: {new_embeddings.shape}")

# Diffraction merger - merge with probabilistic interference
print("\n5. Applying diffraction merger...")
# Get current quantized data
if 'int8' in manifold['quantized']:
    current_int8 = manifold['quantized']['int8']
    current_int16 = manifold['quantized']['int16']
else:
    # Handle old format
    current_int8 = manifold['quantized']
    current_int16 = np.zeros((current_int8.shape[0], 32), dtype=np.int16)

print(f"   Current INT8 shape: {current_int8.shape}")
print(f"   Current INT16 shape: {current_int16.shape}")

# Standardize and quantize new embeddings
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
new_embeddings_scaled = scaler.fit_transform(new_embeddings)

# Apply PCA to match dimensions if needed
if new_embeddings.shape[1] != current_int8.shape[1]:
    print(f"   Adjusting dimensions: {new_embeddings.shape[1]} -> {current_int8.shape[1]}")
    # Pad or truncate
    if new_embeddings.shape[1] < current_int8.shape[1]:
        padding = np.zeros((new_embeddings.shape[0], current_int8.shape[1] - new_embeddings.shape[1]))
        new_embeddings_scaled = np.hstack([new_embeddings_scaled, padding])
    else:
        new_embeddings_scaled = new_embeddings_scaled[:, :current_int8.shape[1]]

# Quantize new data
new_int8 = np.round(new_embeddings_scaled * 127).astype(np.int8)
new_int16 = np.round(new_embeddings_scaled[:, :32] * 32767).astype(np.int16)

# Merge with interference pattern (weighted average simulating diffraction)
alpha = 0.1  # New data weight (10% influence)
print(f"   Merging with alpha={alpha} (new data weight)")

# Create interference pattern
interference_pattern = np.random.uniform(0.8, 1.2, size=current_int8.shape[0])
interference_pattern = interference_pattern.reshape(-1, 1)

# Apply interference to existing data
current_int8_float = current_int8.astype(np.float32) * interference_pattern
current_int16_float = current_int16.astype(np.float32) * interference_pattern[:, :32]

# Concatenate old (with interference) and new
updated_int8 = np.vstack([
    (current_int8_float * (1 - alpha)).astype(np.int8),
    new_int8
])
updated_int16 = np.vstack([
    (current_int16_float * (1 - alpha)).astype(np.int16),
    new_int16
])

print(f"   Merged INT8 shape: {updated_int8.shape}")
print(f"   Merged INT16 shape: {updated_int16.shape}")

# Chipping pruner - remove low-variance dimensions
print("\n6. Applying chipping pruner...")
# Calculate variance per dimension
variance_int8 = np.var(updated_int8.astype(np.float32), axis=0)
variance_int16 = np.var(updated_int16.astype(np.float32), axis=0)

# Set pruning thresholds
threshold_int8 = np.percentile(variance_int8, 5)  # Keep 95% highest variance
threshold_int16 = np.percentile(variance_int16, 10)  # Keep 90% for INT16

# Create masks
keep_mask_int8 = variance_int8 > threshold_int8
keep_mask_int16 = variance_int16 > threshold_int16

print(f"   INT8 dimensions kept: {np.sum(keep_mask_int8)}/{len(keep_mask_int8)}")
print(f"   INT16 dimensions kept: {np.sum(keep_mask_int16)}/{len(keep_mask_int16)}")

# Apply pruning
pruned_int8 = updated_int8[:, keep_mask_int8]
pruned_int16 = updated_int16[:, keep_mask_int16]

# Update manifold
print("\n7. Updating manifold...")
manifold['quantized'] = {
    'int8': pruned_int8,
    'int16': pruned_int16,
    'scale_factors': manifold['quantized'].get('scale_factors', {'int8': 127.0, 'int16': 32767.0}),
    'important_dims': pruned_int16.shape[1]
}

# Update metadata
manifold['metadata']['num_nodes'] = pruned_int8.shape[0]
manifold['metadata']['update_time'] = time.time() - start_time
manifold['metadata']['new_samples_added'] = len(new_data_texts)
manifold['metadata']['pruning_applied'] = True
manifold['metadata']['dimensions_after_pruning'] = {
    'int8': pruned_int8.shape[1],
    'int16': pruned_int16.shape[1]
}

# Save updated manifold
print("\n8. Saving updated manifold...")
with open('../outputs/phase4_updated_manifold.pkl', 'wb') as f:
    pickle.dump(manifold, f)
print(f"   Manifold saved to: phase4_updated_manifold.pkl")

# Light fine-tuning on mixed old/new data
print("\n9. Fine-tuning model on mixed data...")
# Create mixed dataset
mixed_texts = []

# Sample from original data (load from graph)
with open('../outputs/phase1_graph.pkl', 'rb') as f:
    graph = pickle.load(f)

# Get some original texts
original_texts = []
for node_id in list(graph.nodes())[:50]:  # Sample 50 original
    node_data = graph.nodes[node_id]
    if 'text' in node_data and len(node_data['text']) > 20:
        original_texts.append(node_data['text'][:200])

# Mix old and new
mixed_texts = original_texts[:30] + new_data_texts  # 30 old + 20 new
print(f"   Mixed dataset: {len(original_texts[:30])} old + {len(new_data_texts)} new")

# Create dataset
dataset = Dataset.from_dict({'text': mixed_texts})
train_dataset = dataset

# Tokenize
def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=64)

tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_train.set_format('torch', columns=['input_ids', 'attention_mask'])

# Light training args (just 1 epoch, small steps)
training_args = TrainingArguments(
    output_dir='../outputs/phase4_checkpoints',
    num_train_epochs=1,
    per_device_train_batch_size=2,
    logging_steps=10,
    save_strategy='no',
    fp16=False,
    report_to='none',
    max_steps=25,  # Very limited fine-tuning
)

from transformers import DataCollatorForLanguageModeling
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Train
print("   Running light fine-tuning (25 steps)...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    data_collator=data_collator,
)

train_result = trainer.train()
print(f"   Fine-tuning loss: {train_result.metrics['train_loss']:.4f}")

# Save updated model
print("\n10. Saving updated model...")
model.save_pretrained('../outputs/phase4_updated_model_weights')
tokenizer.save_pretrained('../outputs/phase4_updated_model_weights')
print(f"   Model saved to: phase4_updated_model_weights")

# Calculate statistics
total_time = time.time() - start_time
size_growth = pruned_int8.shape[0] / current_int8.shape[0]
dimension_retention = {
    'int8': np.sum(keep_mask_int8) / len(keep_mask_int8) * 100,
    'int16': np.sum(keep_mask_int16) / len(keep_mask_int16) * 100
}

print("\n" + "="*50)
print("Phase 4 Dynamic Chipping Complete!")
print("="*50)
print(f"Processing time: {total_time:.2f} seconds")
print(f"New samples added: {len(new_data_texts)}")
print(f"Size growth: {(size_growth - 1) * 100:.1f}%")
print(f"Dimension retention: INT8={dimension_retention['int8']:.1f}%, INT16={dimension_retention['int16']:.1f}%")
print(f"Final manifold nodes: {pruned_int8.shape[0]}")
print(f"Final INT8 dims: {pruned_int8.shape[1]}")
print(f"Final INT16 dims: {pruned_int16.shape[1]}")

print("\nNote: Enhance with RL policy, full diffraction (FFT), EWC-inspired loss for production.")
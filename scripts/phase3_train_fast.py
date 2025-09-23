"""
Phase 3: Training Integration (Fast PoC Version)
Minimal training for proof of concept
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset
import pickle
import numpy as np
import random
import time

print("Phase 3: Fast Training Integration Starting...")
print("="*50)

# Check GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

start_time = time.time()

# Load manifold
with open('../outputs/phase2_manifold.pkl', 'rb') as f:
    manifold = pickle.load(f)

# Load graph
with open('../outputs/phase1_graph.pkl', 'rb') as f:
    graph = pickle.load(f)

# Generate minimal dataset (100 samples for fast PoC)
print("\nGenerating minimal dataset...")
texts = []
for node_id in list(graph.nodes())[:100]:
    node_data = graph.nodes[node_id]
    if 'text' in node_data and len(node_data['text']) > 20:
        texts.append(node_data['text'][:200])  # Truncate for speed

# Ensure we have enough samples
while len(texts) < 100:
    texts.append(f"Holographic sample {len(texts)}: Manifold projection")

dataset = Dataset.from_dict({'text': texts[:100]})
split = dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = split['train']
val_dataset = split['test']

print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

# Load small model
print("\nLoading model...")
model_name = 'distilgpt2'
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Tokenize
def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=64)

tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_val = val_dataset.map(tokenize_function, batched=True)
tokenized_train.set_format('torch', columns=['input_ids', 'attention_mask'])
tokenized_val.set_format('torch', columns=['input_ids', 'attention_mask'])

# Minimal training args
training_args = TrainingArguments(
    output_dir='../outputs/phase3_checkpoints',
    num_train_epochs=1,  # Just 1 epoch for PoC
    per_device_train_batch_size=2,  # Very small batch
    per_device_eval_batch_size=2,
    logging_steps=10,
    eval_strategy='steps',
    eval_steps=20,
    save_strategy='no',  # Don't save checkpoints
    fp16=False,  # Disable for CPU
    report_to='none',
    max_steps=50,  # Limit to 50 steps
)

from transformers import DataCollatorForLanguageModeling
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Train
print("\nTraining (50 steps max)...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    data_collator=data_collator,
)

train_result = trainer.train()

# Save
print("\nSaving model...")
model.save_pretrained('../outputs/phase3_model_weights')
tokenizer.save_pretrained('../outputs/phase3_model_weights')

total_time = time.time() - start_time
print(f"\nComplete! Time: {total_time:.1f}s")
print(f"Final loss: {train_result.metrics['train_loss']:.4f}")
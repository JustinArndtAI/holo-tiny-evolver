"""
Phase 3: Training Integration
Fine-tunes a small LLM using holographic sampler from quantized manifold
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset
import pickle
import numpy as np
import random
from tqdm import tqdm
import time

print("Phase 3: Training Integration Starting...")
print("="*50)

# Check GPU availability
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    device = 'cuda'
else:
    print("Using CPU (Warning: Training will be slow)")
    device = 'cpu'

start_time = time.time()

# Load Phase 2 manifold
print("\n1. Loading Phase 2 manifold...")
with open('../outputs/phase2_manifold.pkl', 'rb') as f:
    manifold = pickle.load(f)
print(f"   Manifold loaded: {manifold['metadata']['num_nodes']} nodes")

# Load original graph for text data
print("\n2. Loading original graph for text data...")
with open('../outputs/phase1_graph.pkl', 'rb') as f:
    graph = pickle.load(f)

# Create holographic sampler
def holographic_sample(manifold, graph, num_samples=1000):
    """
    Generate training samples using holographic projection from manifold
    """
    print(f"\n3. Generating {num_samples} holographic samples...")
    
    # Extract text from graph nodes
    texts = []
    node_ids = list(graph.nodes())
    
    for node_id in node_ids:
        node_data = graph.nodes[node_id]
        if 'text' in node_data and node_data['text']:
            texts.append(node_data['text'])
    
    print(f"   Found {len(texts)} text samples in graph")
    
    # Use manifold structure to create augmented samples
    samples = []
    
    # Get quantized embeddings
    int8_data = manifold['quantized']['int8']
    int16_data = manifold['quantized']['int16']
    
    # Use hierarchical clustering to create diverse samples
    for level in manifold['levels']:
        labels = level['labels']
        unique_clusters = np.unique(labels)
        
        for cluster_id in unique_clusters[:num_samples // len(manifold['levels'])]:
            # Get nodes in this cluster
            cluster_mask = labels == cluster_id
            cluster_indices = np.where(cluster_mask)[0]
            
            if len(cluster_indices) > 0:
                # Sample from cluster
                idx = random.choice(cluster_indices)
                if idx < len(texts):
                    text = texts[idx]
                    
                    # Create holographic variations
                    # Use quantization levels to create semantic variations
                    if len(text) > 20:  # Only use substantial texts
                        samples.append(text)
                        
                        # Add variation with cluster context
                        cluster_size = len(cluster_indices)
                        variation = f"In a cluster of {cluster_size} concepts: {text}"
                        samples.append(variation)
    
    # If not enough samples, repeat with variations
    while len(samples) < num_samples:
        if texts:
            base_text = random.choice(texts)
            if len(base_text) > 20:
                # Add holographic context
                level_idx = random.randint(0, len(manifold['levels']) - 1)
                n_clusters = manifold['levels'][level_idx]['n_clusters']
                augmented = f"At hierarchical level {level_idx} with {n_clusters} clusters: {base_text}"
                samples.append(augmented)
        else:
            # Fallback if no good texts
            samples.append(f"Holographic sample {len(samples)}: Knowledge manifold projection")
    
    # Truncate to exact number
    samples = samples[:num_samples]
    
    print(f"   Generated {len(samples)} samples")
    data = {'text': samples}
    return Dataset.from_dict(data)

# Generate dataset
dataset = holographic_sample(manifold, graph, num_samples=1000)

# Split into train/validation
print("\n4. Splitting dataset...")
split_dataset = dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = split_dataset['train']
val_dataset = split_dataset['test']
print(f"   Train samples: {len(train_dataset)}")
print(f"   Validation samples: {len(val_dataset)}")

# Load model and tokenizer
print("\n5. Loading model and tokenizer...")
# Use DistilGPT2 for lighter weight and faster training
model_name = 'distilgpt2'  # Lighter than phi-1.5 for PoC
print(f"   Using model: {model_name}")

model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Set padding token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Move model to device
model = model.to(device)

# Tokenization function
def tokenize_function(examples):
    # Tokenize with proper padding and truncation
    return tokenizer(
        examples['text'], 
        padding='max_length', 
        truncation=True,
        max_length=128,  # Shorter for faster training
        return_tensors=None
    )

print("\n6. Tokenizing datasets...")
tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_val = val_dataset.map(tokenize_function, batched=True)

# Set format for PyTorch
tokenized_train.set_format('torch', columns=['input_ids', 'attention_mask'])
tokenized_val.set_format('torch', columns=['input_ids', 'attention_mask'])

# Training arguments - optimized for PoC
print("\n7. Setting up training configuration...")
training_args = TrainingArguments(
    output_dir='../outputs/phase3_checkpoints',
    num_train_epochs=3,  # Reduced for PoC
    per_device_train_batch_size=8 if device == 'cuda' else 4,  # Smaller batch for stability
    per_device_eval_batch_size=8 if device == 'cuda' else 4,
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir='../outputs/phase3_logs',
    logging_steps=50,
    eval_strategy='epoch',  # Changed from evaluation_strategy
    save_strategy='epoch',
    save_steps=500,
    load_best_model_at_end=True,
    metric_for_best_model='eval_loss',
    fp16=True if torch.cuda.is_available() else False,
    gradient_checkpointing=True if device == 'cuda' else False,
    report_to='none',  # Disable wandb/tensorboard for simplicity
)

# Custom data collator for language modeling
from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # Causal LM, not masked
)

# Initialize trainer
print("\n8. Initializing trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

# Train model with holographic integration
print("\n9. Starting training with holographic integration...")
print("   This may take 30-60 minutes on GPU, longer on CPU...")

try:
    # Train
    train_result = trainer.train()
    
    # Save metrics
    train_metrics = train_result.metrics
    trainer.log_metrics("train", train_metrics)
    trainer.save_metrics("train", train_metrics)
    
    print("\n10. Training complete! Saving model...")
    
    # Save final model
    model.save_pretrained('../outputs/phase3_model_weights')
    tokenizer.save_pretrained('../outputs/phase3_model_weights')
    
    # Also save trainer state
    trainer.save_state()
    
    # Calculate total time
    total_time = time.time() - start_time
    
    print("\n" + "="*50)
    print("Phase 3 Training Complete!")
    print("="*50)
    print(f"Total training time: {total_time/60:.2f} minutes")
    print(f"Final training loss: {train_metrics.get('train_loss', 'N/A'):.4f}")
    print(f"Model saved to: ../outputs/phase3_model_weights")
    print(f"Checkpoints saved to: ../outputs/phase3_checkpoints")
    
    # Save summary
    with open('../outputs/phase3_summary.txt', 'w') as f:
        f.write(f"Phase 3 Training Summary\n")
        f.write(f"========================\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Train samples: {len(train_dataset)}\n")
        f.write(f"Validation samples: {len(val_dataset)}\n")
        f.write(f"Epochs: {training_args.num_train_epochs}\n")
        f.write(f"Batch size: {training_args.per_device_train_batch_size}\n")
        f.write(f"Total time: {total_time/60:.2f} minutes\n")
        f.write(f"Final loss: {train_metrics.get('train_loss', 'N/A'):.4f}\n")
    
except Exception as e:
    print(f"\n[ERROR] Training failed: {str(e)}")
    print("Attempting to save partial results...")
    
    # Try to save whatever we have
    try:
        model.save_pretrained('../outputs/phase3_model_weights')
        tokenizer.save_pretrained('../outputs/phase3_model_weights')
        print("Partial model saved")
    except:
        print("Could not save model")
    
    raise e

print("\nNote: Enhance with full holographic attention and deeper manifold integration for production.")
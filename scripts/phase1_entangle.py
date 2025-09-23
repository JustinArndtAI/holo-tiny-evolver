"""
Phase 1: Entanglement Phase of HKM Pipeline
Creates a swarm of small models to process datasets and form a proto-manifold graph
"""

import torch
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset
import networkx as nx
from sentence_transformers import SentenceTransformer
import numpy as np
from scipy.stats import entropy
import pickle
import time
from tqdm import tqdm
import os

print("Phase 1: Entanglement Process Starting...")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    device = 'cuda'
else:
    print("Using CPU (this will be slower)")
    device = 'cpu'

start_time = time.time()

# Load datasets
print("\n1. Loading datasets...")
print("   Loading WikiText...")
wikitext = load_dataset('Salesforce/wikitext', 'wikitext-103-raw-v1', split='train[:1%]')  # Using 1% for PoC
print(f"   WikiText samples loaded: {len(wikitext)}")

print("   Loading FB15k-237...")
fb15k = load_dataset('KGraph/FB15k-237')
print(f"   FB15k splits: {list(fb15k.keys())}")

# Initialize entanglers (swarm of small models)
print("\n2. Initializing entangler swarm...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')  # Faster, smaller model
embedder.to(device)
print(f"   Embedder loaded on {device}")

# Process WikiText data
print("\n3. Processing WikiText for entanglement...")
# Use a reasonable subset for PoC
sample_size = min(500, len(wikitext))  # Process up to 500 samples
texts = []
for i in tqdm(range(sample_size), desc="   Extracting texts"):
    text = wikitext[i]['text'].strip()
    if len(text) > 10:  # Skip very short texts
        # Take first 200 chars for efficiency
        texts.append(text[:200])

print(f"   Valid texts extracted: {len(texts)}")

# Generate embeddings
print("\n4. Generating embeddings...")
batch_size = 32
embeddings = []
for i in tqdm(range(0, len(texts), batch_size), desc="   Embedding batches"):
    batch = texts[i:i+batch_size]
    batch_embeddings = embedder.encode(batch, convert_to_tensor=True, device=device)
    embeddings.append(batch_embeddings.cpu().numpy())

embeddings = np.vstack(embeddings)
print(f"   Embeddings shape: {embeddings.shape}")

# Create entangled graph
print("\n5. Creating entangled graph...")
G = nx.Graph()

# Add nodes with embeddings and text
for i in range(len(texts)):
    G.add_node(i, 
               text=texts[i][:50] + "...",  # Store truncated text for reference
               embedding=embeddings[i].tolist(),  # Convert to list for serialization
               source='wikitext')

# Create edges based on similarity (diffusion-like process)
print("\n6. Establishing entanglements (edges)...")
similarity_threshold = 0.6  # Adjusted threshold for better connectivity
edge_count = 0

for i in tqdm(range(len(texts)), desc="   Processing nodes"):
    for j in range(i+1, len(texts)):
        # Calculate cosine similarity
        sim = np.dot(embeddings[i], embeddings[j]) / (np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]))
        
        # Probabilistic entanglement with diffusion
        # Apply a probabilistic threshold based on similarity
        prob_threshold = np.random.random() * 0.3 + 0.5  # Random threshold between 0.5 and 0.8
        
        if sim > prob_threshold:
            G.add_edge(i, j, weight=float(sim))
            edge_count += 1

print(f"   Edges created: {edge_count}")

# Add FB15k knowledge graph entities (sample integration)
print("\n7. Integrating FB15k knowledge...")
fb15k_train = fb15k['train']
fb15k_sample_size = min(100, len(fb15k_train))  # Add 100 FB15k relationships

node_offset = len(texts)
fb15k_nodes = {}

for i in tqdm(range(fb15k_sample_size), desc="   Adding FB15k entities"):
    sample = fb15k_train[i]['text']
    # Parse the tab-separated format
    parts = sample.split('\t')
    if len(parts) >= 3:
        head = parts[0]
        relation = parts[1]
        tail = parts[2]
    else:
        continue
    
    # Create nodes for entities if not exists
    if head not in fb15k_nodes:
        fb15k_nodes[head] = node_offset
        G.add_node(node_offset, text=head, source='fb15k', type='entity')
        node_offset += 1
    
    if tail not in fb15k_nodes:
        fb15k_nodes[tail] = node_offset
        G.add_node(node_offset, text=tail, source='fb15k', type='entity')
        node_offset += 1
    
    # Add edge with relation
    G.add_edge(fb15k_nodes[head], fb15k_nodes[tail], 
               weight=1.0, relation=relation, source='fb15k')

print(f"   FB15k entities added: {len(fb15k_nodes)}")

# Calculate graph statistics
print("\n8. Graph Statistics:")
print(f"   Total nodes: {G.number_of_nodes()}")
print(f"   Total edges: {G.number_of_edges()}")
print(f"   Average degree: {sum(dict(G.degree()).values()) / G.number_of_nodes():.2f}")

# Check connectivity
components = list(nx.connected_components(G))
largest_component = max(components, key=len)
print(f"   Connected components: {len(components)}")
print(f"   Largest component size: {len(largest_component)} ({len(largest_component)/G.number_of_nodes()*100:.1f}%)")

# Save graph
print("\n9. Saving entangled graph...")
output_path = '../outputs/phase1_graph.pkl'
os.makedirs('../outputs', exist_ok=True)
with open(output_path, 'wb') as f:
    pickle.dump(G, f)
print(f"   Graph saved to: {output_path}")

# Save metadata
metadata = {
    'num_nodes': G.number_of_nodes(),
    'num_edges': G.number_of_edges(),
    'wikitext_samples': sample_size,
    'fb15k_samples': fb15k_sample_size,
    'similarity_threshold': similarity_threshold,
    'embedder_model': 'all-MiniLM-L6-v2',
    'device': device,
    'processing_time': time.time() - start_time
}

with open('../outputs/phase1_metadata.pkl', 'wb') as f:
    pickle.dump(metadata, f)

print(f"\nPhase 1 Complete!")
print(f"Total processing time: {metadata['processing_time']:.2f} seconds")
print(f"Graph saved with {metadata['num_nodes']} nodes and {metadata['num_edges']} edges")
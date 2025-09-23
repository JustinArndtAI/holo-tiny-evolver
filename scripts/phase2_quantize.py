"""
Phase 2: Quantization to Manifold
Applies mixed-precision quantization with fractal lattice structure
"""

import torch
import networkx as nx
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pickle
import time
from tqdm import tqdm

print("Phase 2: Quantization to Manifold Starting...")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    device = 'cuda'
else:
    print("Using CPU")
    device = 'cpu'

start_time = time.time()

# Load Phase 1 graph
print("\n1. Loading Phase 1 graph...")
with open('../outputs/phase1_graph.pkl', 'rb') as f:
    G = pickle.load(f)
print(f"   Loaded graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

# Extract embeddings
print("\n2. Extracting and preprocessing embeddings...")
node_ids = list(G.nodes())
embeddings = []
node_metadata = []

for node_id in node_ids:
    node_data = G.nodes[node_id]
    if 'embedding' in node_data:
        embeddings.append(node_data['embedding'])
        node_metadata.append({
            'id': node_id,
            'source': node_data.get('source', 'unknown'),
            'text': node_data.get('text', '')[:50]
        })
    else:
        # Create zero embedding for nodes without embeddings
        embeddings.append([0.0] * 384)  # Assuming 384 dimensions
        node_metadata.append({
            'id': node_id,
            'source': node_data.get('source', 'unknown'),
            'text': node_data.get('text', '')[:50]
        })

embeddings = np.array(embeddings)
print(f"   Embeddings shape: {embeddings.shape}")

# Standardize embeddings
scaler = StandardScaler()
embeddings_scaled = scaler.fit_transform(embeddings)

# Apply PCA for dimensionality reduction (helps with fractal structure)
print("\n3. Applying PCA for fractal base...")
pca = PCA(n_components=min(128, embeddings.shape[0], embeddings.shape[1]))
embeddings_pca = pca.fit_transform(embeddings_scaled)
print(f"   PCA components: {embeddings_pca.shape[1]}")
print(f"   Explained variance ratio: {pca.explained_variance_ratio_.sum():.3f}")

# Hierarchical clustering for fractal levels
print("\n4. Creating fractal hierarchy...")
manifold = {
    'levels': [],
    'quantized': {},
    'metadata': {
        'num_nodes': len(node_ids),
        'original_dim': embeddings.shape[1],
        'pca_dim': embeddings_pca.shape[1],
        'node_metadata': node_metadata
    }
}

# Create multiple clustering levels (fractal structure)
n_clusters_per_level = [5, 10, 20, 50, 100]  # Increasing granularity
prev_labels = None

for level_idx, n_clusters in enumerate(tqdm(n_clusters_per_level, desc="   Building levels")):
    if n_clusters > len(embeddings_pca):
        n_clusters = max(2, len(embeddings_pca) // 10)
    
    clustering = AgglomerativeClustering(
        n_clusters=n_clusters, 
        linkage='ward'
    )
    labels = clustering.fit_predict(embeddings_pca)
    
    # Store level information
    level_data = {
        'level': level_idx,
        'n_clusters': n_clusters,
        'labels': labels,
        'cluster_centers': []
    }
    
    # Calculate cluster centers
    for cluster_id in range(n_clusters):
        mask = labels == cluster_id
        if np.any(mask):
            center = embeddings_pca[mask].mean(axis=0)
            level_data['cluster_centers'].append(center)
    
    # Store inter-level connections (fractal property)
    if prev_labels is not None:
        # Map fine to coarse clusters
        mapping = {}
        for i, (fine, coarse) in enumerate(zip(labels, prev_labels)):
            if coarse not in mapping:
                mapping[coarse] = []
            mapping[coarse].append(fine)
        level_data['parent_mapping'] = mapping
    
    manifold['levels'].append(level_data)
    prev_labels = labels

# Mixed-precision quantization
print("\n5. Applying mixed-precision quantization...")

# INT8 for low-importance dimensions
quantized_int8 = np.round(embeddings_scaled * 127).astype(np.int8)

# INT16 for high-importance dimensions (top PCA components)
n_important = min(32, embeddings_pca.shape[1])
important_dims = embeddings_pca[:, :n_important]
quantized_int16 = np.round(important_dims * 32767).astype(np.int16)

# Store both quantization levels
manifold['quantized'] = {
    'int8': quantized_int8,
    'int16': quantized_int16,
    'scale_factors': {
        'int8': 127.0,
        'int16': 32767.0
    },
    'important_dims': n_important
}

# Calculate fractal dimension (box-counting approximation)
print("\n6. Computing fractal properties...")
box_sizes = [2, 4, 8, 16, 32]
counts = []

for box_size in box_sizes:
    # Discretize space into boxes
    bins = np.linspace(embeddings_pca.min(), embeddings_pca.max(), box_size)
    hist, _ = np.histogramdd(embeddings_pca[:, :3], bins=[bins, bins, bins])
    count = np.sum(hist > 0)
    counts.append(count)

# Estimate fractal dimension
if len(counts) > 1 and all(c > 0 for c in counts):
    log_sizes = np.log(box_sizes)
    log_counts = np.log(counts)
    fractal_dim = -np.polyfit(log_sizes, log_counts, 1)[0]
    manifold['metadata']['fractal_dimension'] = fractal_dim
    print(f"   Estimated fractal dimension: {fractal_dim:.3f}")

# Add graph structure to manifold
print("\n7. Integrating graph structure...")
edge_list = list(G.edges(data=True))
manifold['graph_structure'] = {
    'edges': [(e[0], e[1], e[2].get('weight', 1.0)) for e in edge_list],
    'num_edges': len(edge_list)
}

# Calculate compression metrics
original_size = embeddings.nbytes
quantized_size = quantized_int8.nbytes + quantized_int16.nbytes
compression_ratio = original_size / quantized_size

manifold['metadata']['compression'] = {
    'original_size_bytes': original_size,
    'quantized_size_bytes': quantized_size,
    'compression_ratio': compression_ratio,
    'processing_time': time.time() - start_time
}

# Save manifold
print("\n8. Saving quantized manifold...")
output_path = '../outputs/phase2_manifold.pkl'
with open(output_path, 'wb') as f:
    pickle.dump(manifold, f)
print(f"   Manifold saved to: {output_path}")

# Print summary
print("\n" + "="*50)
print("Phase 2 Complete!")
print("="*50)
print(f"Processing time: {manifold['metadata']['compression']['processing_time']:.2f} seconds")
print(f"Compression ratio: {compression_ratio:.2f}x")
print(f"Fractal levels: {len(manifold['levels'])}")
print(f"Original size: {original_size / 1024:.2f} KB")
print(f"Quantized size: {quantized_size / 1024:.2f} KB")
print(f"Space saved: {(1 - 1/compression_ratio) * 100:.1f}%")
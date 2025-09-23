"""
Evaluate Phase 2: Manifold Quality Metrics
Assesses the quality of the quantized manifold
"""

import pickle
import numpy as np
from scipy.stats import entropy
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings('ignore')

print("Phase 2 Quality Evaluation Starting...")
print("="*50)

# Load manifold
print("\n1. Loading manifold for evaluation...")
with open('../outputs/phase2_manifold.pkl', 'rb') as f:
    manifold = pickle.load(f)

# Load original graph for comparison
with open('../outputs/phase1_graph.pkl', 'rb') as f:
    original_graph = pickle.load(f)

print(f"   Manifold loaded: {manifold['metadata']['num_nodes']} nodes")
print(f"   Original graph: {original_graph.number_of_nodes()} nodes")

# Metric 1: Information Preservation
print("\n2. Evaluating information preservation...")
int8_data = manifold['quantized']['int8']
int16_data = manifold['quantized']['int16']

# Calculate entropy for quantized data
int8_entropy = []
for row in int8_data:
    if len(np.unique(row)) > 1:
        hist, _ = np.histogram(row, bins=20)
        hist = hist[hist > 0]
        if len(hist) > 0:
            int8_entropy.append(entropy(hist))

avg_int8_entropy = np.mean(int8_entropy) if int8_entropy else 0
print(f"   INT8 average entropy: {avg_int8_entropy:.3f}")

int16_entropy = []
for row in int16_data:
    if len(np.unique(row)) > 1:
        hist, _ = np.histogram(row, bins=20)
        hist = hist[hist > 0]
        if len(hist) > 0:
            int16_entropy.append(entropy(hist))

avg_int16_entropy = np.mean(int16_entropy) if int16_entropy else 0
print(f"   INT16 average entropy: {avg_int16_entropy:.3f}")

# Information retention score
info_score = (avg_int8_entropy + avg_int16_entropy) / 2
print(f"   Information retention score: {info_score:.3f}")

# Metric 2: Hierarchical Clustering Quality
print("\n3. Assessing hierarchical clustering quality...")
silhouette_scores = []

for i, level in enumerate(manifold['levels']):
    labels = level['labels']
    n_unique = len(np.unique(labels))
    
    if n_unique > 1 and n_unique < len(labels):
        # Use INT16 data for silhouette calculation (higher precision)
        try:
            score = silhouette_score(int16_data[:, :min(10, int16_data.shape[1])], labels)
            silhouette_scores.append(score)
            print(f"   Level {i} silhouette score: {score:.3f}")
        except:
            print(f"   Level {i} silhouette score: N/A (computation error)")
    else:
        print(f"   Level {i} silhouette score: N/A (single cluster)")

avg_silhouette = np.mean(silhouette_scores) if silhouette_scores else 0
print(f"   Average silhouette score: {avg_silhouette:.3f}")

# Metric 3: Locality Preservation
print("\n4. Testing locality preservation...")
# Compare nearest neighbors in original vs quantized space
sample_size = min(100, len(int16_data))
sample_indices = np.random.choice(len(int16_data), sample_size, replace=False)

# Original embeddings (from node data)
original_embeddings = []
node_ids = list(original_graph.nodes())
for idx in sample_indices:
    if idx < len(node_ids):
        node_id = node_ids[idx]
        if 'embedding' in original_graph.nodes[node_id]:
            original_embeddings.append(original_graph.nodes[node_id]['embedding'])

if len(original_embeddings) > 10:
    original_embeddings = np.array(original_embeddings)
    
    # Find k-nearest neighbors in original space
    k = 5
    nbrs_original = NearestNeighbors(n_neighbors=k+1)
    nbrs_original.fit(original_embeddings)
    distances_orig, indices_orig = nbrs_original.kneighbors(original_embeddings)
    
    # Find k-nearest neighbors in quantized space (using INT16)
    quantized_sample = int16_data[sample_indices[:len(original_embeddings)]]
    nbrs_quantized = NearestNeighbors(n_neighbors=k+1)
    nbrs_quantized.fit(quantized_sample)
    distances_quant, indices_quant = nbrs_quantized.kneighbors(quantized_sample)
    
    # Calculate preservation ratio
    preservation_ratios = []
    for i in range(len(indices_orig)):
        orig_neighbors = set(indices_orig[i][1:])  # Exclude self
        quant_neighbors = set(indices_quant[i][1:])
        if len(orig_neighbors) > 0:
            preservation = len(orig_neighbors.intersection(quant_neighbors)) / len(orig_neighbors)
            preservation_ratios.append(preservation)
    
    locality_preservation = np.mean(preservation_ratios) if preservation_ratios else 0
    print(f"   Locality preservation ratio: {locality_preservation:.3f}")
else:
    locality_preservation = 0
    print(f"   Locality preservation: N/A (insufficient data)")

# Metric 4: Compression Efficiency
print("\n5. Analyzing compression efficiency...")
compression = manifold['metadata']['compression']
compression_ratio = compression['compression_ratio']
processing_time = compression['processing_time']

# Calculate bits per dimension
original_bits = 32 * manifold['metadata']['original_dim']  # float32
quantized_bits = 8 * manifold['metadata']['original_dim'] + 16 * manifold['quantized']['important_dims']
bit_reduction = (original_bits - quantized_bits) / original_bits * 100

print(f"   Compression ratio: {compression_ratio:.2f}x")
print(f"   Bit reduction: {bit_reduction:.1f}%")
print(f"   Processing time: {processing_time:.2f} seconds")
print(f"   Compression speed: {manifold['metadata']['num_nodes'] / processing_time:.0f} nodes/sec")

# Metric 5: Fractal Structure Quality
print("\n6. Evaluating fractal structure...")
fractal_dim = manifold['metadata'].get('fractal_dimension', 0)
if fractal_dim < 0:
    print(f"   Fractal dimension: {fractal_dim:.3f} (negative - sparse data)")
    fractal_quality = 0.3  # Low score for negative dimension
else:
    print(f"   Fractal dimension: {fractal_dim:.3f}")
    # Ideal fractal dimension for knowledge graphs is between 1.5-2.5
    if 1.5 <= fractal_dim <= 2.5:
        fractal_quality = 1.0
    else:
        fractal_quality = max(0, 1 - abs(fractal_dim - 2.0) / 2.0)

print(f"   Fractal quality score: {fractal_quality:.3f}")

# Metric 6: Hierarchical Coherence
print("\n7. Checking hierarchical coherence...")
coherence_scores = []
for i in range(len(manifold['levels']) - 1):
    coarse_labels = manifold['levels'][i]['labels']
    fine_labels = manifold['levels'][i + 1]['labels']
    
    # Check if fine clusters are subsets of coarse clusters
    coherence = 0
    for coarse_id in np.unique(coarse_labels):
        coarse_mask = coarse_labels == coarse_id
        fine_in_coarse = fine_labels[coarse_mask]
        # Measure how concentrated fine clusters are within coarse clusters
        if len(fine_in_coarse) > 0:
            unique_fine = len(np.unique(fine_in_coarse))
            expected_fine = len(fine_in_coarse) / len(fine_labels) * len(np.unique(fine_labels))
            coherence += min(1.0, expected_fine / unique_fine) if unique_fine > 0 else 0
    
    coherence_score = coherence / len(np.unique(coarse_labels))
    coherence_scores.append(coherence_score)
    print(f"   Level {i}->{i+1} coherence: {coherence_score:.3f}")

avg_coherence = np.mean(coherence_scores) if coherence_scores else 0
print(f"   Average hierarchical coherence: {avg_coherence:.3f}")

# Overall Quality Assessment
print("\n" + "="*50)
print("MANIFOLD QUALITY ASSESSMENT")
print("="*50)

# Score thresholds
quality_metrics = {
    'Information Retention': (info_score, 0.5, info_score >= 0.5),
    'Clustering Quality': (avg_silhouette, 0.2, avg_silhouette >= 0.2),
    'Locality Preservation': (locality_preservation, 0.3, locality_preservation >= 0.3),
    'Compression Ratio': (compression_ratio, 3.0, compression_ratio >= 3.0),
    'Fractal Structure': (fractal_quality, 0.5, fractal_quality >= 0.5),
    'Hierarchical Coherence': (avg_coherence, 0.6, avg_coherence >= 0.6)
}

passed = 0
total = len(quality_metrics)

for metric, (value, threshold, passes) in quality_metrics.items():
    status = "[PASS]" if passes else "[FAIL]"
    print(f"{status} {metric}: {value:.3f} (threshold: {threshold})")
    if passes:
        passed += 1

quality_score = passed / total * 100
print(f"\nOverall Quality Score: {passed}/{total} ({quality_score:.0f}%)")

# Recommendations
print("\nRecommendations:")
if quality_score >= 60:
    print("[OK] Manifold quality is sufficient to proceed to Phase 3")
    decision = "PROCEED"
else:
    print("[WARNING] Manifold quality is below threshold")
    decision = "REVIEW"
    
if fractal_dim < 0:
    print("- Consider increasing graph density before Phase 3")
if avg_silhouette < 0.3:
    print("- Clustering quality could be improved with better parameters")
if locality_preservation < 0.5:
    print("- Locality preservation is suboptimal, may affect recall")
if avg_coherence < 0.7:
    print("- Hierarchical structure could be more coherent")

print(f"\nDecision: {decision}")

# Save evaluation results
eval_results = {
    'metrics': quality_metrics,
    'quality_score': quality_score,
    'decision': decision,
    'compression_ratio': compression_ratio,
    'processing_time': processing_time
}

with open('../outputs/phase2_evaluation.pkl', 'wb') as f:
    pickle.dump(eval_results, f)
print(f"\nEvaluation results saved to: phase2_evaluation.pkl")
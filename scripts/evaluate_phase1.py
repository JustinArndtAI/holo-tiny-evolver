"""
Phase 1 Quality Metrics Evaluation
"""
import networkx as nx
import pickle
import numpy as np
from scipy.stats import entropy
import random

# Load the graph
print("Loading graph and metadata...")
with open('../outputs/phase1_graph.pkl', 'rb') as f:
    G = pickle.load(f)

with open('../outputs/phase1_metadata.pkl', 'rb') as f:
    metadata = pickle.load(f)

print("\n=== ENTANGLEMENT QUALITY METRICS ===")

# 1. Entropy Analysis
print("\n1. Entropy Analysis:")
entropies = []
nodes_with_edges = [n for n in G.nodes() if G.degree(n) > 0]

for node in nodes_with_edges[:50]:  # Sample 50 nodes
    neighbors = list(G.neighbors(node))
    if len(neighbors) > 1:
        weights = []
        for neighbor in neighbors:
            if G.has_edge(node, neighbor):
                edge_data = G[node][neighbor]
                weights.append(edge_data.get('weight', 1.0))
        
        if weights:
            # Normalize weights to probabilities
            probs = np.array(weights) / sum(weights)
            ent = entropy(probs)
            entropies.append(ent)

if entropies:
    print(f"  Average entropy: {np.mean(entropies):.3f}")
    print(f"  Std entropy: {np.std(entropies):.3f}")
    print(f"  Min entropy: {np.min(entropies):.3f}")
    print(f"  Max entropy: {np.max(entropies):.3f}")
else:
    print("  No valid entropy values computed")

# 2. Semantic Coherence (sample check)
print("\n2. Semantic Coherence Check:")
wikitext_nodes = [(n, d) for n, d in G.nodes(data=True) if d.get('source') == 'wikitext']
sample_size = min(10, len(wikitext_nodes))
sampled = random.sample(wikitext_nodes, sample_size)

coherent_connections = 0
total_connections = 0

for node_id, node_data in sampled:
    if 'text' in node_data:
        print(f"\n  Node {node_id}: '{node_data['text'][:30]}...'")
        neighbors = list(G.neighbors(node_id))
        print(f"  Connected to {len(neighbors)} nodes")
        
        for neighbor in neighbors[:3]:  # Check first 3 connections
            neighbor_data = G.nodes[neighbor]
            if 'text' in neighbor_data:
                weight = G[node_id][neighbor].get('weight', 0)
                print(f"    -> Node {neighbor} (weight={weight:.3f}): '{neighbor_data.get('text', 'N/A')[:30]}...'")
                total_connections += 1
                if weight > 0.7:  # High similarity = likely coherent
                    coherent_connections += 1

# 3. Path Analysis
print("\n3. Path Analysis:")
# Find paths between random pairs
components = list(nx.connected_components(G))
largest_comp = max(components, key=len)

if len(largest_comp) > 10:
    comp_nodes = list(largest_comp)
    sample_pairs = 5
    path_lengths = []
    
    for _ in range(sample_pairs):
        if len(comp_nodes) >= 2:
            source, target = random.sample(comp_nodes, 2)
            try:
                path_len = nx.shortest_path_length(G, source, target)
                path_lengths.append(path_len)
            except nx.NetworkXNoPath:
                pass
    
    if path_lengths:
        print(f"  Average path length in largest component: {np.mean(path_lengths):.2f}")
        print(f"  Max path length: {max(path_lengths)}")
        print(f"  Min path length: {min(path_lengths)}")

# 4. Clustering Coefficient
print("\n4. Clustering Analysis:")
clustering = nx.clustering(G)
non_zero_clustering = [c for c in clustering.values() if c > 0]
if non_zero_clustering:
    print(f"  Average clustering coefficient: {np.mean(non_zero_clustering):.3f}")
    print(f"  Nodes with clustering > 0: {len(non_zero_clustering)}")

# 5. Manual Recall Estimation
print("\n5. Recall Estimation (conceptual links):")
# Check if expected relationships exist
test_queries = [
    ("WikiText sample connections", len([n for n in wikitext_nodes if G.degree(n[0]) > 2])),
    ("FB15k entity connections", len([n for n, d in G.nodes(data=True) 
                                     if d.get('source') == 'fb15k' and G.degree(n) > 0])),
    ("Cross-dataset links", len([e for e in G.edges(data=True) 
                                 if G.nodes[e[0]].get('source') != G.nodes[e[1]].get('source')]))
]

for query, count in test_queries:
    print(f"  {query}: {count}")

# Overall Quality Score
print("\n=== QUALITY ASSESSMENT ===")
quality_score = 0
max_score = 5

# Entropy score
if entropies and np.mean(entropies) > 0.2:
    quality_score += 1
    print("[PASS] Entropy > 0.2")
else:
    print("[FAIL] Entropy too low")

# Connectivity score
connectivity = (len(largest_comp) / len(G.nodes)) * 100
if connectivity > 5:
    quality_score += 1
    print(f"[PASS] Largest component covers {connectivity:.1f}% of nodes")
else:
    print(f"[FAIL] Largest component only {connectivity:.1f}%")

# Average degree score
avg_degree = sum(dict(G.degree()).values()) / len(G.nodes)
if avg_degree > 1:
    quality_score += 1
    print(f"[PASS] Average degree {avg_degree:.2f} > 1")
else:
    print(f"[FAIL] Average degree {avg_degree:.2f} < 1")

# Edge weight quality
edge_weights = [d['weight'] for _, _, d in G.edges(data=True) if 'weight' in d]
if edge_weights and np.mean(edge_weights) > 0.7:
    quality_score += 1
    print(f"[PASS] Mean edge weight {np.mean(edge_weights):.3f} > 0.7")
else:
    print(f"[FAIL] Mean edge weight below threshold")

# Node diversity
if len(wikitext_nodes) > 0 and len([n for n, d in G.nodes(data=True) if d.get('source') == 'fb15k']) > 0:
    quality_score += 1
    print("[PASS] Both WikiText and FB15k nodes present")
else:
    print("[FAIL] Missing dataset diversity")

print(f"\nOverall Quality Score: {quality_score}/{max_score}")
print(f"Quality Percentage: {(quality_score/max_score)*100:.0f}%")

if quality_score >= 3:
    print("\nRECOMMENDATION: Proceed with current graph (meets minimum criteria)")
else:
    print("\nRECOMMENDATION: Consider re-running with adjusted parameters")
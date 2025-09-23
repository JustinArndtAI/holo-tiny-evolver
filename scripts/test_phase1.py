"""
Phase 1 Graph Integrity Checks
"""
import networkx as nx
import matplotlib.pyplot as plt
import pickle
import numpy as np

# Load the graph
print("Loading graph...")
with open('../outputs/phase1_graph.pkl', 'rb') as f:
    G = pickle.load(f)

print("\n=== GRAPH STATISTICS ===")
print(f'Total Nodes: {len(G.nodes)}')
print(f'Total Edges: {len(G.edges)}')
print(f'Average degree: {sum(dict(G.degree()).values()) / len(G.nodes):.2f}')

# Connectivity analysis
components = list(nx.connected_components(G))
largest_component = max(components, key=len)
isolated_nodes = [comp for comp in components if len(comp) == 1]

print(f'\nConnected components: {len(components)}')
print(f'Largest component size: {len(largest_component)} ({len(largest_component)/len(G.nodes)*100:.1f}%)')
print(f'Isolated nodes: {len(isolated_nodes)} ({len(isolated_nodes)/len(G.nodes)*100:.1f}%)')

# Node source distribution
wikitext_nodes = [n for n, d in G.nodes(data=True) if d.get('source') == 'wikitext']
fb15k_nodes = [n for n, d in G.nodes(data=True) if d.get('source') == 'fb15k']
print(f'\nWikiText nodes: {len(wikitext_nodes)}')
print(f'FB15k nodes: {len(fb15k_nodes)}')

# Edge weight distribution
edge_weights = [d['weight'] for _, _, d in G.edges(data=True) if 'weight' in d]
if edge_weights:
    print(f'\nEdge weight statistics:')
    print(f'  Mean: {np.mean(edge_weights):.3f}')
    print(f'  Std: {np.std(edge_weights):.3f}')
    print(f'  Min: {np.min(edge_weights):.3f}')
    print(f'  Max: {np.max(edge_weights):.3f}')

# Degree distribution
degrees = dict(G.degree())
degree_values = list(degrees.values())
print(f'\nDegree distribution:')
print(f'  Max degree: {max(degree_values)}')
print(f'  Min degree: {min(degree_values)}')
print(f'  Nodes with degree > 5: {sum(1 for d in degree_values if d > 5)}')

# Create visualization
print("\nGenerating visualizations...")
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. Subgraph visualization
ax1 = axes[0, 0]
subgraph = G.subgraph(list(largest_component)[:50] if len(largest_component) > 50 else largest_component)
pos = nx.spring_layout(subgraph, k=2, iterations=50)
nx.draw(subgraph, pos, ax=ax1, node_size=30, with_labels=False, 
        node_color='lightblue', edge_color='gray', alpha=0.7)
ax1.set_title(f'Largest Component Subgraph ({len(subgraph.nodes)} nodes)')

# 2. Degree distribution histogram
ax2 = axes[0, 1]
ax2.hist(degree_values, bins=20, edgecolor='black', alpha=0.7)
ax2.set_xlabel('Degree')
ax2.set_ylabel('Number of Nodes')
ax2.set_title('Degree Distribution')
ax2.grid(True, alpha=0.3)

# 3. Edge weight distribution
ax3 = axes[1, 0]
if edge_weights:
    ax3.hist(edge_weights, bins=30, edgecolor='black', alpha=0.7, color='green')
    ax3.set_xlabel('Edge Weight (Similarity)')
    ax3.set_ylabel('Number of Edges')
    ax3.set_title('Edge Weight Distribution')
    ax3.grid(True, alpha=0.3)

# 4. Component size distribution
ax4 = axes[1, 1]
component_sizes = [len(comp) for comp in components]
unique_sizes, counts = np.unique(component_sizes, return_counts=True)
ax4.bar(range(len(unique_sizes[:20])), counts[:20], color='orange', alpha=0.7)
ax4.set_xlabel('Component Size')
ax4.set_ylabel('Number of Components')
ax4.set_title('Component Size Distribution (first 20 sizes)')
ax4.set_xticks(range(len(unique_sizes[:20])))
ax4.set_xticklabels(unique_sizes[:20], rotation=45)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../outputs/phase1_graph.png', dpi=150, bbox_inches='tight')
print(f"Visualization saved to ../outputs/phase1_graph.png")

# Connectivity verdict
connectivity_score = (len(largest_component) / len(G.nodes)) * 100
avg_degree = sum(dict(G.degree()).values()) / len(G.nodes)

print("\n=== INTEGRITY CHECK RESULTS ===")
if connectivity_score > 80:
    print(f"[OK] EXCELLENT: Connectivity is {connectivity_score:.1f}% (>80%)")
elif connectivity_score > 50:
    print(f"[WARN] MODERATE: Connectivity is {connectivity_score:.1f}% (50-80%)")
else:
    print(f"[FAIL] LOW: Connectivity is {connectivity_score:.1f}% (<50%)")

if avg_degree > 3:
    print(f"[OK] GOOD: Average degree is {avg_degree:.2f} (>3)")
elif avg_degree > 2:
    print(f"[WARN] FAIR: Average degree is {avg_degree:.2f} (2-3)")
else:
    print(f"[FAIL] LOW: Average degree is {avg_degree:.2f} (<2)")

# Overall assessment
if connectivity_score < 50 or avg_degree < 2:
    print("\nRecommendation: Consider adjusting similarity threshold or using more samples")
else:
    print("\nGraph structure is acceptable for Phase 2")
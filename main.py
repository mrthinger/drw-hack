import networkx as nx
from leidenalg import find_partition
import igraph as ig
import numpy as np

# Simulated data: 10 time periods with their embeddings (simplified to 3D for example)
periods = {
    "2022_Q1": np.array([0.8, 0.2, 0.1]),  # High inflation period
    "2022_Q2": np.array([0.9, 0.1, 0.2]),  # High inflation period
    "2022_Q3": np.array([0.7, 0.3, 0.1]),  # High inflation period
    "2020_Q2": np.array([0.1, 0.9, 0.1]),  # Covid shock period
    "2020_Q3": np.array([0.2, 0.8, 0.2]),  # Covid shock period
    "2019_Q1": np.array([0.4, 0.4, 0.4]),  # Stable period
    "2019_Q2": np.array([0.3, 0.5, 0.4]),  # Stable period
    "2018_Q4": np.array([0.5, 0.3, 0.4]),  # Stable period
    "2008_Q3": np.array([0.2, 0.2, 0.9]),  # Financial crisis
    "2008_Q4": np.array([0.1, 0.3, 0.8]),  # Financial crisis
}

# Create graph
G = nx.Graph()

# Add nodes
for period in periods.keys():
    G.add_node(period)

# Add edges with cosine similarity weights
for p1 in periods:
    for p2 in periods:
        if p1 < p2:  # Avoid duplicate edges
            sim = np.dot(periods[p1], periods[p2]) / (np.linalg.norm(periods[p1]) * np.linalg.norm(periods[p2]))
            if sim > 0.7:  # Only connect similar periods
                G.add_edge(p1, p2, weight=sim)

# Convert to igraph for Leiden
g_ig = ig.Graph.from_networkx(G)

# Apply Leiden clustering
partition = find_partition(g_ig, partition_type=ig.RBConfigurationVertexPartition)

# Print clusters
clusters = {}
for node, cluster in enumerate(partition.membership):
    period = list(periods.keys())[node]
    if cluster not in clusters:
        clusters[cluster] = []
    clusters[cluster].append(period)

for cluster_id, members in clusters.items():
    print(f"\nCluster {cluster_id}:")
    print(members)
    
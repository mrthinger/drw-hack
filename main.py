import networkx as nx
import leidenalg as la
import igraph as ig
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer

DEVICE = "mps"
model = SentenceTransformer(
    "dunzhang/stella_en_400M_v5",
    trust_remote_code=True,
    device=DEVICE,
    config_kwargs={
        "use_memory_efficient_attention": False,
        "unpad_inputs": False,
    },
)

# embeddings = model.encode(
#     list of str,
#     device=DEVICE,
# )


# similarities = model.similarity(
#     e1,
#     e2,
# )

# Example statements with metadata
statements = [
    # 2024 Q1 Statements
    {
        "id": "stmt_1",
        "text": "Inflation remains elevated at 4%",
        "period": "2024_Q1",
        "type": "inflation",
        "source": "FOMC",
    },
    {
        "id": "stmt_2",
        "text": "Labor markets remain tight",
        "period": "2024_Q1",
        "type": "employment",
        "source": "FOMC",
    },
    {
        "id": "stmt_3",
        "text": "Economic activity has been expanding at a solid pace",
        "period": "2024_Q1",
        "type": "growth",
        "source": "FOMC",
    },
    
    # 2023 Q4 Statements
    {
        "id": "stmt_4",
        "text": "Inflation showing persistent upward pressure at 3.8%",
        "period": "2023_Q4",
        "type": "inflation",
        "source": "FOMC",
    },
    {
        "id": "stmt_5",
        "text": "Job gains have moderated but remain strong",
        "period": "2023_Q4",
        "type": "employment",
        "source": "FOMC",
    },
    {
        "id": "stmt_6",
        "text": "GDP growth has slowed from its strong pace in the third quarter",
        "period": "2023_Q4",
        "type": "growth",
        "source": "FOMC",
    },

    # 2023 Q3 Statements
    {
        "id": "stmt_7",
        "text": "Inflation has eased from its peaks but remains above 3%",
        "period": "2023_Q3",
        "type": "inflation",
        "source": "FOMC",
    },
    {
        "id": "stmt_8",
        "text": "The unemployment rate has remained below 4 percent",
        "period": "2023_Q3",
        "type": "employment",
        "source": "FOMC",
    },
    {
        "id": "stmt_9",
        "text": "Recent indicators suggest robust economic growth",
        "period": "2023_Q3",
        "type": "growth",
        "source": "FOMC",
    },

    # 2008 Q4 Crisis Statements
    {
        "id": "stmt_10",
        "text": "Inflation pressures have diminished appreciably",
        "period": "2008_Q4",
        "type": "inflation",
        "source": "FOMC",
    },
    {
        "id": "stmt_11",
        "text": "Labor market conditions have deteriorated significantly",
        "period": "2008_Q4",
        "type": "employment",
        "source": "FOMC",
    },
    {
        "id": "stmt_12",
        "text": "Economic activity has dropped sharply in recent months",
        "period": "2008_Q4",
        "type": "growth",
        "source": "FOMC",
    },

    # 2020 Q2 Covid Statements
    {
        "id": "stmt_13",
        "text": "Inflation has fallen substantially due to weaker demand",
        "period": "2020_Q2",
        "type": "inflation",
        "source": "FOMC",
    },
    {
        "id": "stmt_14",
        "text": "Unemployment has risen to historic levels",
        "period": "2020_Q2",
        "type": "employment",
        "source": "FOMC",
    },
    {
        "id": "stmt_15",
        "text": "Economic activity has contracted at an unprecedented rate",
        "period": "2020_Q2",
        "type": "growth",
        "source": "FOMC",
    },
]

# Create graph
G = nx.Graph()

# First get embeddings for all statements
texts = [stmt["text"] for stmt in statements]
embeddings = model.encode(
    texts,
    device=DEVICE,
)

# Create embedding lookup for efficiency
stmt_embeddings = {
    stmt["id"]: embedding 
    for stmt, embedding in zip(statements, embeddings)
}

# Add statement nodes with all metadata
for stmt in statements:
    G.add_node(
        stmt["id"],
        text=stmt["text"],
        period=stmt["period"],
        type=stmt["type"],
        source=stmt["source"],
    )

# Add edges between similar statements using real embeddings
for i, stmt1 in enumerate(statements):
    for stmt2 in statements[i + 1:]:
        # Get embeddings for both statements
        emb1 = stmt_embeddings[stmt1["id"]]
        emb2 = stmt_embeddings[stmt2["id"]]
        
        # Calculate similarity
        sim = model.similarity(
            emb1,
            emb2
        )[0]
        
        # Add edge if similarity is high enough
        # if sim > 0.7:
        G.add_edge(stmt1["id"], stmt2["id"], weight=float(sim))

# Convert to igraph for Leiden
g_ig = ig.Graph.from_networkx(G)

# Apply Leiden clustering
partition = la.find_partition(g_ig, la.ModularityVertexPartition, weights="weight")

# Create a color map for nodes based on cluster
node_colors: list[str] = []  # Type annotation
for node, cluster in enumerate(partition.membership):
    node_colors.append(f"C{cluster}")  # Uses matplotlib default colors

# Create spring layout
pos = nx.spring_layout(G)

# Draw the graph with cluster colors
plt.figure(figsize=(12, 8))
nx.draw_networkx_nodes(G, pos, node_size=500, node_color=node_colors)  # type: ignore
nx.draw_networkx_edges(G, pos, width=[G[u][v]["weight"] * 2 for u, v in G.edges()])  # type: ignore
nx.draw_networkx_labels(G, pos)

# Add edge labels (optional)
edge_labels = nx.get_edge_attributes(G, "weight")
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

plt.title("Economic Period Similarity Network (Colored by Cluster)")
plt.axis("off")
plt.show()

# Print clusters
clusters = {}
for node, cluster in enumerate(partition.membership):
    stmt_id = list(G.nodes())[node]
    stmt = next(stmt for stmt in statements if stmt["id"] == stmt_id)
    if cluster not in clusters:
        clusters[cluster] = []
    clusters[cluster].append(stmt)

for cluster_id, members in clusters.items():
    print(f"\nCluster {cluster_id} (Color C{cluster_id}):")
    for stmt in members:
        print(
            f"Statement ID: {stmt['id']}, Text: {stmt['text']}, Period: {stmt['period']}, Type: {stmt['type']}, Source: {stmt['source']}"
        )

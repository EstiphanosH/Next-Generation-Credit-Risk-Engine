"""Build transaction graphs using NetworkX/PyTorch Geometric."""
import networkx as nx

def build_graph(transactions):
    """Creates user-merchant graph for fraud detection."""
    G = nx.Graph()
    for _, row in transactions.iterrows():
        G.add_edge(row['user_id'], row['merchant_id'], weight=row['amount'])
    return G

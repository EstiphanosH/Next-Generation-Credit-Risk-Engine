"""Test feature engineering logic."""
from scripts.feature_graphs import build_graph

def test_graph_construction():
    """Verify edge creation in transaction graph."""
    test_data = [{'user_id': 'u1', 'merchant_id': 'm1', 'amount': 100}]
    G = build_graph(test_data)
    assert G.number_of_edges() == 1

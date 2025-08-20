import pandas as pd, networkx as nx
def build_user_merchant_graph(df, user_col, merchant_col):
    B=nx.Graph(); users=df[user_col].astype(str).unique(); merchants=df[merchant_col].astype(str).unique()
    B.add_nodes_from([("u_"+u) for u in users]); B.add_nodes_from([("m_"+m) for m in merchants])
    edges=[("u_"+str(u),"m_"+str(m)) for u,m in zip(df[user_col].astype(str), df[merchant_col].astype(str))]; B.add_edges_from(edges); return B
def user_graph_metrics(df, user_col, merchant_col):
    G=build_user_merchant_graph(df,user_col,merchant_col); users=[n for n in G.nodes() if str(n).startswith("u_")]
    deg=dict(G.degree(users)); pr=nx.pagerank(G,alpha=0.85); user_pr={u:pr.get(u,0.0) for u in users}
    import pandas as pd
    return pd.DataFrame({user_col:[u.replace("u_","") for u in users],"graph_degree":[deg.get(u,0) for u in users],"graph_pagerank":[user_pr.get(u,0.0) for u in users]})

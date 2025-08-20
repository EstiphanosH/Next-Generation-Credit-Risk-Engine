def run_aequitas_audit(df, score_col: str, label_col: str, group_col: str):
    if group_col is None or group_col not in df.columns: return {"parity_delta": None}
    try:
        from aequitas.group import Group; from aequitas.bias import Bias
        aq=df[[group_col,score_col,label_col]].rename(columns={group_col:"attribute",score_col:"score",label_col:"label_value"})
        g=Group(); xtab,_=g.get_crosstabs(aq); b=Bias()
        _=b.get_disparity_predefined_groups(xtab, original_df=aq, ref_groups_dict={'attribute': 'all'}, alpha=0.05, check_significance=False)
        deltas=aq.groupby("attribute")["score"].mean().diff().abs().dropna()
        return {"parity_delta": float(deltas.max()) if not deltas.empty else None}
    except Exception as e:
        return {"parity_delta": None, "error": str(e)}

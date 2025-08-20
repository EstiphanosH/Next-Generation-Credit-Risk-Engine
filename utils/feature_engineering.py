import pandas as pd, numpy as np
from scipy.stats import entropy
STANDARD_COL_CANDIDATES = {
  "user_id":["user_id","customer_id","account_id","msisdn","user","uid"],
  "amount":["amount","transaction_amount","value","price"],
  "timestamp":["timestamp","date","transaction_date","datetime","trans_date"],
  "merchant":["merchant","provider","merchant_id","vendor"],
  "category":["category","product_category","merchant_category","product"],
  "region":["region","user_region","location","country"],
  "payment_method":["payment_method","channel","payment_type"],
  "device_os":["device_os","os"],
  "device_browser":["device_browser","browser"],
  "label_default":["default","defaulted","fraud_label","label"]}
def map_columns(df, overrides=None):
    overrides = overrides or {}; mapping = {}; cols_lower = {c.lower():c for c in df.columns}
    for std, cands in STANDARD_COL_CANDIDATES.items():
        if overrides.get(std): mapping[std]=overrides[std]; continue
        for cand in cands:
            if cand.lower() in cols_lower: mapping[std]=cols_lower[cand.lower()]; break
    return mapping
def compute_rfm_features(df, user_col, amount_col, timestamp_col):
    df=df.copy(); df[timestamp_col]=pd.to_datetime(df[timestamp_col]); ref=df[timestamp_col].max()
    r=(ref-df.groupby(user_col)[timestamp_col].max()).dt.days.rename("recency_days")
    f=df.groupby(user_col)[timestamp_col].count().rename("frequency")
    m=df.groupby(user_col)[amount_col].sum().rename("monetary")
    out=pd.concat([r,f,m],axis=1).reset_index(); out["tw_recency"]=np.log1p(1.0/(out["recency_days"].clip(lower=1))); return out
def compute_entropy_diversity(df, user_col, category_col):
    g=df.groupby([user_col,category_col]).size().rename("cnt").reset_index(); total=g.groupby(user_col)["cnt"].sum().rename("total")
    g=g.merge(total,on=user_col,how="left"); g["p"]=g["cnt"]/g["total"].replace(0,np.nan)
    ent=g.groupby(user_col)["p"].apply(lambda p: entropy(p.fillna(0.0))).rename("purchase_entropy")
    div=g.groupby(user_col)[category_col].nunique().rename("category_diversity"); return pd.concat([ent,div],axis=1).reset_index()
def compute_payment_diversity(df, user_col, method_col):
    if method_col is None or method_col not in df.columns: return pd.DataFrame({user_col:df[user_col].unique(),"payment_method_diversity":0})
    return df.groupby(user_col)[method_col].nunique().rename("payment_method_diversity").reset_index()
def compute_device_stats(df, user_col, os_col, br_col):
    out=pd.DataFrame({user_col:df[user_col].unique()})
    if os_col and os_col in df.columns: out=out.merge(df.groupby(user_col)[os_col].nunique().rename("device_os_diversity").reset_index(),on=user_col,how="left")
    if br_col and br_col in df.columns: out=out.merge(df.groupby(user_col)[br_col].nunique().rename("device_browser_diversity").reset_index(),on=user_col,how="left")
    return out.fillna(0)
def compute_user_volatility(df, user_col, amount_col):
    v=df.groupby(user_col)[amount_col].agg(["std","mean"]).rename(columns={"std":"amt_std","mean":"amt_mean"}).reset_index()
    v["amt_cv"]=(v["amt_std"]/v["amt_mean"].replace(0,np.nan)).fillna(0); return v

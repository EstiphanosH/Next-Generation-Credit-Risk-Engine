import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
def psi(expected, actual, buckets: int=10)->float:
    expected=pd.Series(expected).dropna(); actual=pd.Series(actual).dropna()
    if expected.empty or actual.empty: return 0.0
    bins=np.quantile(expected, np.linspace(0,1,buckets+1)); bins=np.unique(bins)
    if len(bins)<2: return 0.0
    e_counts,_=np.histogram(expected,bins=bins); a_counts,_=np.histogram(actual,bins=bins)
    e_perc=e_counts/(len(expected)+1e-8); a_perc=a_counts/(len(actual)+1e-8)
    a_perc=np.where(a_perc==0,1e-6,a_perc); e_perc=np.where(e_perc==0,1e-6,e_perc)
    return float(np.sum((e_perc-a_perc)*np.log(e_perc/a_perc)))
def auc_multiclass(y_true, y_proba)->float:
    try: return float(roc_auc_score(y_true, y_proba, multi_class="ovr"))
    except Exception: return float(roc_auc_score(y_true, y_proba))

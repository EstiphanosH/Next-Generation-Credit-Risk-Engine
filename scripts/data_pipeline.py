
#!/usr/bin/env python
import argparse, os, glob, json, pandas as pd, numpy as np
def main(i,o):
    csvs=sorted(glob.glob(os.path.join(i,'*.csv')))
    if not csvs: raise FileNotFoundError('No CSV')
    df=pd.concat([pd.read_csv(p) for p in csvs], ignore_index=True)
    os.makedirs(os.path.dirname(o), exist_ok=True)
    df.to_parquet(o, index=False)
    with open(os.path.join(os.path.dirname(o),'metadata.json'),'w') as f: json.dump({'mapping':{}},f)
if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--input', default='data/raw'); ap.add_argument('--output', default='data/processed/users.parquet'); a=ap.parse_args(); main(a.input,a.output)

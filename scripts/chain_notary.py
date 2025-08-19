# scripts/chain_notary.py
"""
Small helper to notarize a file hash on Ethereum-compatible chains.
This is a safe placeholder: it computes the SHA256 of a file and prints a
transaction payload that a real signing deployment would send.

In production, replace the payload construction with signed transactions using web3.py.
"""
import argparse
import hashlib
from pathlib import Path
import json
from datetime import datetime

def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def main(args):
    path = Path(args.file)
    if not path.exists():
        raise FileNotFoundError(path)
    h = file_hash(path)
    payload = {
        "hash": h,
        "file": str(path),
        "timestamp": datetime.utcnow().isoformat(),
        "note": args.note or ""
    }
    # Print the payload; in production this would be sent as transaction data
    print(json.dumps(payload, indent=2))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True)
    parser.add_argument("--note", required=False)
    args = parser.parse_args()
    main(args)

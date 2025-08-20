from pathlib import Path
import yaml
class Config:
    def __init__(self, path: str):
        self.path = Path(path)
        with open(self.path, "r", encoding="utf-8") as f: self.cfg = yaml.safe_load(f)
    def get(self, *keys, default=None):
        node = self.cfg
        for k in keys:
            node = node.get(k) if node is not None else None
        return default if node is None else node
    def root(self) -> Path: return self.path.parent.parent
    def path_of(self, *keys) -> Path:
        p = self.get(*keys); return None if p is None else self.root() / p

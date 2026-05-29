import json, os
from collections import Counter
from typing import List, Dict, Tuple, Iterable
from rdkit import Chem
from rdkit.Chem import BRICS

class MotifVocab:
    def __init__(self, vocab_path: str = None):
        self.token2id: Dict[str, int] = {}
        self.id2token: List[str] = []
        self.idf: List[float] = []
        if vocab_path and os.path.exists(vocab_path):
            self.load(vocab_path)

    def build(self, smiles_list: Iterable[str], fg_smarts: List[str] = None,
              min_freq: int = 3) -> None:
        doc_freq = Counter()
        n_docs = 0
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is None: 
                continue
            n_docs += 1
            tokens = set(BRICS.BRICSDecompose(mol, returnMols=False))
            if fg_smarts:
                for i, s in enumerate(fg_smarts):
                    patt = Chem.MolFromSmarts(s)
                    if patt and mol.HasSubstructMatch(patt):
                        tokens.add(f"FG_{i}")
            for t in tokens:
                doc_freq[t] += 1
        kept = [t for t, f in doc_freq.items() if f >= min_freq]
        kept.sort()
        self.id2token = kept
        self.token2id = {t: i for i, t in enumerate(kept)}
        import math
        self.idf = [math.log((1 + n_docs) / (1 + doc_freq[t])) + 1.0 for t in self.id2token]

    def save(self, path: str) -> None:
        obj = {"id2token": self.id2token, "idf": self.idf}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False)

    def load(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        self.id2token = obj["id2token"]
        self.token2id = {t: i for i, t in enumerate(self.id2token)}
        self.idf = obj["idf"]

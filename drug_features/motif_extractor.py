from typing import List, Dict
import numpy as np
from rdkit import Chem
from rdkit.Chem import BRICS
from .motif_vocab import MotifVocab

def smiles_to_tokens(smiles: str, fg_smarts: List[str] = None) -> List[str]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    tokens = list(BRICS.BRICSDecompose(mol, returnMols=False))
    if fg_smarts:
        for i, s in enumerate(fg_smarts):
            patt = Chem.MolFromSmarts(s)
            if patt and mol.HasSubstructMatch(patt):
                tokens.append(f"FG_{i}")
    return tokens

def tfidf_vector(smiles: str, vocab: MotifVocab, fg_smarts: List[str]) -> np.ndarray:
    tokens = smiles_to_tokens(smiles, fg_smarts)
    if len(tokens) == 0 or len(vocab.id2token) == 0:
        return np.zeros(len(vocab.id2token), dtype=np.float32)
    counts = {}
    for t in tokens:
        if t in vocab.token2id:
            counts[t] = counts.get(t, 0) + 1
    if not counts:
        return np.zeros(len(vocab.id2token), dtype=np.float32)
    vec = np.zeros(len(vocab.id2token), dtype=np.float32)
    for t, c in counts.items():
        j = vocab.token2id[t]
        vec[j] = float(c) * float(vocab.idf[j])
    norm = np.linalg.norm(vec) + 1e-8
    return (vec / norm).astype(np.float32)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
from collections import Counter
import pandas as pd

from rdkit import Chem
from rdkit.Chem import BRICS

BASIC_SMARTS = [
    "c1ccccc1", "c1ncccc1", "c1occc1", "n1cccc1", "c1cnccc1",
    "C=O", "C(=O)N", "C(=O)O", "OC(=O)O", "OC(=O)N",
    "S(=O)(=O)N", "S(=O)(=O)O", "S(=O)(=O)c1ccccc1",
    "Cl", "Br", "F", "I",
    "N", "[NH2]", "[NH3+]",
    "N1CCNCC1", "O1CCOCC1"
]

def _pick_smiles_column(df: pd.DataFrame) -> str:
    for c in ["isosmiles", "canonicalsmiles", "smiles", "SMILES"]:
        if c in df.columns:
            return c
    raise ValueError("drug_info.csv must contain one of: "
                     "isosmiles / canonicalsmiles / smiles / SMILES")

def _norm_smiles_strict(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.strip()
    if not s:
        return ""
    mol = Chem.MolFromSmiles(s)
    if mol is None:
        return ""
    return Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)

def _brics_frag_smiles(smi: str):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return []
    return list(BRICS.BRICSDecompose(mol, returnMols=False))

def _count_pat_in_mol(smi: str, pat_str: str) -> int:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return 0
    pat = Chem.MolFromSmarts(pat_str)
    if pat is None:
        pat = Chem.MolFromSmiles(pat_str)
    if pat is None:
        return 0
    return len(mol.GetSubstructMatches(pat))

def build_vocab(drug_info_path: str, out_path: str, top_k: int = 200, min_freq: int = 3):
    df = pd.read_csv(drug_info_path)
    smi_col = _pick_smiles_column(df)

    smiles_list = [ _norm_smiles_strict(s) for s in df[smi_col].astype(str).tolist() ]
    smiles_list = [s for s in smiles_list if s]
    smiles_list = list(dict.fromkeys(smiles_list))

    brics_ct = Counter()
    for smi in smiles_list:
        try:
            frags = _brics_frag_smiles(smi)
            for f in set(frags):
                brics_ct[f] += 1
        except Exception:
            pass

    smarts_ct = Counter()
    for smi in smiles_list:
        for pat in BASIC_SMARTS:
            try:
                if _count_pat_in_mol(smi, pat) > 0:
                    smarts_ct[pat] += 1
            except Exception:
                pass

    combined = Counter(); combined.update(brics_ct); combined.update(smarts_ct)
    items = [(k, v) for k, v in combined.items() if v >= min_freq]
    items.sort(key=lambda x: (-x[1], x[0]))
    items = items[:top_k]

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for k, _ in items:
            f.write(f"{k}\n")

    print(f"[OK] Extracted {len(items)} frequent fragments/functional groups from {drug_info_path} into {out_path}")
    if items:
        print("Top-10 preview:")
        for i,(k,v) in enumerate(items[:10], 1):
            print(f"{i:2d}. {k:40s} freq={v}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drug_info", type=str, default="./data/drug_info/drug_info.csv",
                    help="CSV file containing an isosmiles/canonicalsmiles/smiles column")
    ap.add_argument("--out", type=str, default="./data/drug_info/new_fg_groups.txt",
                    help="Output vocabulary, one SMARTS/SMILES pattern per line")
    ap.add_argument("--top_k", type=int, default=200, help="Maximum number of patterns to write")
    ap.add_argument("--min_freq", type=int, default=3, help="Minimum number of molecules containing a pattern")
    args = ap.parse_args()

    build_vocab(args.drug_info, args.out, args.top_k, args.min_freq)

if __name__ == "__main__":
    main()

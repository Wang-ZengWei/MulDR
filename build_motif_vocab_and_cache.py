
import argparse
import hashlib
import json
from pathlib import Path
from typing import List, Set

import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

from config import (
    FILE_DRUG_INFO, FG_SMARTS_PATH, RDKit_RANDOM_SEED, MOTIF_MIN_FREQ,
    normalize_split_mode
)
from drug_features.motif_vocab import MotifVocab
from split_utils import ensure_protocol_split_files


def pick_smiles_column(df: pd.DataFrame) -> str:
    for c in ['isosmiles', 'canonicalsmiles', 'smiles', 'SMILES']:
        if c in df.columns:
            return c
    raise ValueError('drug_info.csv must contain one of: isosmiles, canonicalsmiles, smiles, SMILES')


def norm_canonical_smiles(s: str) -> str:
    if not isinstance(s, str):
        return ''
    s = s.strip()
    if not s:
        return ''
    m = Chem.MolFromSmiles(s)
    if m is None:
        return ''
    return Chem.MolToSmiles(m, isomericSmiles=True, canonical=True)


def load_fg_smarts(path: Path):
    return [l.strip() for l in open(path, 'r', encoding='utf-8') if l.strip()]


def build_train_drug_smiles(drug_csv: Path, train_drug_ids: Set[str]):
    df = pd.read_csv(drug_csv)
    col = pick_smiles_column(df)
    id_col = None
    for c in ['pubchem_id', 'PubChem_ID', 'DRUG_ID', 'drug_id', 'drug_name', 'compound', 'id']:
        if c in df.columns:
            id_col = c
            break
    if id_col is None:
        id_col = df.columns[0]
    df[id_col] = df[id_col].astype(str)
    df = df[df[id_col].isin(train_drug_ids)].copy()
    smiles = [norm_canonical_smiles(x) for x in df[col].astype(str).tolist()]
    smiles = [s for s in smiles if s]
    return list(dict.fromkeys(smiles))


def save_vocab(vocab: MotifVocab, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vocab.save(str(out_path))


def safe_embed(mol, seed: int = 42):
    m = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed)
    params.maxAttempts = 50
    try:
        cid = AllChem.EmbedMolecule(m, params)
    except Exception:
        cid = -1
    if cid < 0:
        params2 = AllChem.ETKDGv3()
        params2.randomSeed = int(seed)
        params2.useRandomCoords = True
        params2.maxAttempts = 200
        try:
            cid = AllChem.EmbedMolecule(m, params2)
        except Exception:
            cid = -1
    conf_ids = [cid] if cid >= 0 else []
    if conf_ids:
        try:
            if AllChem.MMFFHasAllMoleculeParams(m):
                for cid in conf_ids:
                    try:
                        AllChem.MMFFOptimizeMolecule(m, confId=cid, maxIters=200)
                    except Exception:
                        pass
            else:
                for cid in conf_ids:
                    try:
                        AllChem.UFFOptimizeMolecule(m, confId=cid, maxIters=200)
                    except Exception:
                        pass
        except Exception:
            pass
    return m, conf_ids


def build_atom_feature(mol):
    feats = []
    for a in mol.GetAtoms():
        z = a.GetAtomicNum()
        onehot_z = [0] * 100
        if 0 < z <= 100:
            onehot_z[z - 1] = 1
        f = [a.GetFormalCharge(), int(a.GetIsAromatic()), a.GetTotalNumHs(), a.GetDegree()]
        v = onehot_z + f
        if len(v) < 128:
            v += [0] * (128 - len(v))
        feats.append(v[:128])
    return torch.tensor(feats, dtype=torch.float32)


def build_one_3d_graph(smi: str, seed: int = 42):
    mol0 = Chem.MolFromSmiles(smi)
    if mol0 is None:
        return {'node_feat': torch.zeros(1, 128), 'pos': torch.zeros(1, 3), 'edge_index': torch.zeros(2, 0, dtype=torch.long), 'N': 1}
    try:
        Chem.SanitizeMol(mol0)
    except Exception:
        pass
    m, conf_ids = safe_embed(mol0, seed=seed)
    if not conf_ids or m.GetNumConformers() == 0:
        return {'node_feat': torch.zeros(1, 128), 'pos': torch.zeros(1, 3), 'edge_index': torch.zeros(2, 0, dtype=torch.long), 'N': 1}
    conf = m.GetConformer(conf_ids[0])
    N = m.GetNumAtoms()
    nf = build_atom_feature(m)
    pos = torch.tensor([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z] for i in range(N)], dtype=torch.float32)
    src, dst = [], []
    for i in range(N):
        for j in range(N):
            if i != j:
                src.append(i); dst.append(j)
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    return {'node_feat': nf, 'pos': pos, 'edge_index': edge_index, 'N': N}


def maybe_build_cache(smiles_list: List[str], cache_dir: Path):
    cache_dir.mkdir(parents=True, exist_ok=True)
    ok, skipped = 0, 0
    for s in smiles_list:
        if not s:
            skipped += 1
            continue
        key = hashlib.md5(s.encode('utf-8')).hexdigest()[:16]
        outp = cache_dir / f'{key}.pt'
        if outp.exists():
            ok += 1
            continue
        g = build_one_3d_graph(s, seed=RDKit_RANDOM_SEED)
        if g is None or g.get('node_feat', None) is None:
            skipped += 1
            continue
        torch.save(g, outp)
        ok += 1
    print(f'[Cache] ready={ok} train drug graphs, skipped={skipped}, cache_dir={cache_dir}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split_mode', type=str, default='pair_level', choices=['pair_level', 'drug_coldstart', 'cell_coldstart'])
    parser.add_argument('--experiment_name', type=str, default=None)
    parser.add_argument('--min_freq', type=int, default=MOTIF_MIN_FREQ)
    parser.add_argument('--overwrite_splits', action='store_true')
    args = parser.parse_args()

    split_mode = normalize_split_mode(args.split_mode)
    train_df, val_df, test_df, split_meta, ctx = ensure_protocol_split_files(split_mode, args.experiment_name, overwrite=args.overwrite_splits)
    train_drug_ids = set(train_df['drug'].astype(str).unique().tolist())
    smiles_list = build_train_drug_smiles(FILE_DRUG_INFO, train_drug_ids)
    fg_smarts = load_fg_smarts(FG_SMARTS_PATH)

    vocab = MotifVocab()
    vocab.build(smiles_list, fg_smarts=fg_smarts, min_freq=args.min_freq)
    save_vocab(vocab, ctx['motif_vocab_path'])

    meta = {
        'build_scope': 'train_only',
        'protocol': split_meta['protocol'],
        'experiment_name': ctx['experiment_name'],
        'split_meta_path': str(ctx['split_meta_path']),
        'train_pairs_path': str(ctx['train_pairs_path']),
        'val_pairs_path': str(ctx['val_pairs_path']),
        'test_pairs_path': str(ctx['test_pairs_path']),
        'n_train_pairs': int(len(train_df)),
        'n_val_pairs': int(len(val_df)),
        'n_test_pairs': int(len(test_df)),
        'n_train_drugs': int(len(train_drug_ids)),
        'train_drug_ids': sorted(train_drug_ids),
        'n_vocab_tokens': int(len(vocab.id2token)),
        'motif_min_freq': int(args.min_freq),
        'vocab_path': str(ctx['motif_vocab_path']),
        'cache_dir': str(ctx['graph_cache_dir']),
    }
    with open(ctx['motif_vocab_meta_path'], 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[Vocab] protocol={split_meta['protocol']} | n_train_drugs={len(train_drug_ids)} | vocab_size={len(vocab.id2token)}")
    print(f"[Meta] saved to {ctx['motif_vocab_meta_path']}")
    maybe_build_cache(smiles_list, ctx['graph_cache_dir'])


if __name__ == '__main__':
    main()

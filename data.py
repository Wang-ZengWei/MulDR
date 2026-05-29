
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import hashlib

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from torch.utils.data import DataLoader

from config import (
    FILE_DRUG_INFO, FILE_PATH, FILE_EXP, FILE_MUT, FILE_METH,
    USE_DRUG_GRAPH_CACHE, RDKit_RANDOM_SEED,
    PATH_TOPK, KNN_K, KNN_TAU, KNN_MUTUAL, CELL_GRAPH_MODE,
    get_protocol_profile, normalize_split_mode,
)
from split_utils import ensure_protocol_split_files

EPS = 1e-6


def build_atom_feature(mol):
    feats = []
    for a in mol.GetAtoms():
        z = a.GetAtomicNum()
        onehot_z = [0] * 100
        if 0 < z <= 100:
            onehot_z[z - 1] = 1
        f = [a.GetFormalCharge(), int(a.GetIsAromatic()), a.GetTotalNumHs(), a.GetDegree()]
        vec = onehot_z + f
        if len(vec) < 128:
            vec += [0] * (128 - len(vec))
        feats.append(vec[:128])
    return torch.tensor(feats, dtype=torch.float32)

def _pick_smiles_column(df: pd.DataFrame) -> str:
    for c in ['isosmiles', 'canonicalsmiles', 'smiles', 'SMILES']:
        if c in df.columns:
            return c
    raise ValueError('drug_info.csv must contain one of: isosmiles, canonicalsmiles, smiles, SMILES')

def _norm_canonical_smiles(s: str) -> str:
    if not isinstance(s, str):
        return ''
    s = s.strip()
    if not s:
        return ''
    m = Chem.MolFromSmiles(s)
    if m is None:
        return ''
    return Chem.MolToSmiles(m, isomericSmiles=True, canonical=True)

def load_drug_smiles_map(drug_info_csv: Path) -> dict:
    df = pd.read_csv(drug_info_csv)
    smi_col = _pick_smiles_column(df)
    id_col = None
    for c in ['pubchem_id', 'PubChem_ID', 'DRUG_ID', 'drug_id', 'drug_name', 'compound', 'id']:
        if c in df.columns:
            id_col = c
            break
    if id_col is None:
        id_col = df.columns[0]
    ids = df[id_col].astype(str).tolist()
    smis = df[smi_col].astype(str).apply(_norm_canonical_smiles).tolist()
    out = {}
    for k, v in zip(ids, smis):
        if v:
            out[str(k)] = v
    return out

def _cache_paths_for_smiles_list(smiles_list: List[str], cache_dir: Path) -> List[Path]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for s in smiles_list:
        h = hashlib.md5(s.encode('utf-8')).hexdigest()[:16]
        paths.append(cache_dir / f'{h}.pt')
    return paths

def _empty_graph():
    return {
        'node_feat': torch.zeros(1, 128),
        'pos': torch.zeros(1, 3),
        'edge_index': torch.zeros(2, 0, dtype=torch.long),
        'N': 1,
    }

def _load_or_build_one_graph(smi: str, cache_path: Path):
    if USE_DRUG_GRAPH_CACHE and cache_path.exists():
        return torch.load(cache_path)
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return _empty_graph()
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = int(RDKit_RANDOM_SEED)
    try:
        code = AllChem.EmbedMolecule(mol, params)
    except Exception:
        code = -1
    if code == -1:
        try:
            Chem.Kekulize(mol, clearAromaticFlags=True)
        except Exception:
            pass
        params.randomSeed = int(RDKit_RANDOM_SEED) + 1
        try:
            code = AllChem.EmbedMolecule(mol, params)
        except Exception:
            code = -1
    if code == -1 or mol.GetNumConformers() == 0:
        return _empty_graph()
    try:
        AllChem.UFFOptimizeMolecule(mol, maxIters=200)
    except Exception:
        pass
    conf = mol.GetConformer()
    N = mol.GetNumAtoms()
    nf = build_atom_feature(mol)
    pos = torch.tensor([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z] for i in range(N)], dtype=torch.float32)
    src, dst = [], []
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            src.append(i); dst.append(j)
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    g = {'node_feat': nf, 'pos': pos, 'edge_index': edge_index, 'N': N}
    if USE_DRUG_GRAPH_CACHE:
        torch.save(g, cache_path)
    return g

def smiles_to_3d_batch(smiles_list: List[str], cache_dir: Path):
    paths = _cache_paths_for_smiles_list(smiles_list, cache_dir)
    graphs = [_load_or_build_one_graph(s, p) for s, p in zip(smiles_list, paths)]
    node_feats, poss, batch_idx = [], [], []
    edge_src, edge_dst = [], []
    offset = 0
    for b, g in enumerate(graphs):
        nf, pos, ei = g['node_feat'], g['pos'], g['edge_index']
        N = int(g.get('N', nf.shape[0]))
        node_feats.append(nf); poss.append(pos)
        batch_idx.append(torch.full((N,), b, dtype=torch.long))
        if ei.numel() > 0:
            edge_src += (ei[0] + offset).tolist()
            edge_dst += (ei[1] + offset).tolist()
        offset += N
    node_feat = torch.cat(node_feats, dim=0)
    pos = torch.cat(poss, dim=0)
    batch_idx = torch.cat(batch_idx, dim=0)
    if len(edge_src) == 0:
        edge_index = torch.zeros(2, 0, dtype=torch.long)
    else:
        edge_index = torch.stack([torch.tensor(edge_src, dtype=torch.long), torch.tensor(edge_dst, dtype=torch.long)], dim=0)
    return {'node_feat': node_feat, 'pos': pos, 'edge_index': edge_index, 'batch_idx': batch_idx}


def load_pathway_df() -> pd.DataFrame:
    df = pd.read_csv(FILE_PATH)
    df.iloc[:, 0] = df.iloc[:, 0].astype(str)
    X = df.iloc[:, 1:].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
    df.iloc[:, 1:] = X
    return df

def load_expression_df(zscore_cols: bool = False) -> pd.DataFrame:
    df = pd.read_csv(FILE_EXP)
    df.iloc[:, 0] = df.iloc[:, 0].astype(str)
    X = df.iloc[:, 1:].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
    if zscore_cols:
        mu = X.mean(0); std = X.std(0).replace(0.0, 1.0)
        X = (X - mu) / std
    df.iloc[:, 1:] = X.astype(np.float32)
    return df

def load_mutation_binary_df() -> pd.DataFrame:
    df = pd.read_csv(FILE_MUT)
    df.iloc[:, 0] = df.iloc[:, 0].astype(str)
    X = (df.iloc[:, 1:].fillna(0.0).values > 0).astype(np.float32)
    df.iloc[:, 1:] = X
    return df

def load_methylation_df(zscore_cols: bool = False) -> pd.DataFrame:
    df = pd.read_csv(FILE_METH)
    df.iloc[:, 0] = df.iloc[:, 0].astype(str)
    X = df.iloc[:, 1:].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
    if zscore_cols:
        mu = X.mean(0); std = X.std(0).replace(0.0, 1.0)
        X = (X - mu) / std
    df.iloc[:, 1:] = X.astype(np.float32)
    return df


def _winsorize(x, p_low=0.005, p_high=0.995):
    lo, hi = np.nanpercentile(x, [p_low * 100, p_high * 100])
    return np.clip(x, lo, hi)

def _mad(x):
    med = np.nanmedian(x)
    return np.nanmedian(np.abs(x - med)) + 1e-6

def _auto_should_log(train_long_df: pd.DataFrame) -> bool:
    y = train_long_df['ic50'].astype(np.float32).values
    if np.any(y <= 0):
        return False
    p90, p10 = np.nanpercentile(y, [90, 10])
    med = np.nanmedian(y)
    spread = (p90 - p10) / (abs(med) + 1e-6)
    return spread > 2.0

def fit_drug_scaler(train_long_df: pd.DataFrame, do_log: Optional[bool] = None) -> Dict:
    df = train_long_df.copy()
    y = df['ic50'].astype(np.float32).values
    if do_log is None:
        do_log = _auto_should_log(df)
    if do_log:
        y = np.log2(y + EPS).astype(np.float32)
    df['ic50'] = y
    y_w = _winsorize(y)
    g_med = float(np.nanmedian(y_w))
    g_mad = float(_mad(y_w))
    g_scale = 1.4826 * g_mad

    per = {}
    for d, grp in df.groupby('drug'):
        vals = grp['ic50'].values.astype(np.float32)
        vals_w = _winsorize(vals)
        med = float(np.nanmedian(vals_w))
        mad = float(_mad(vals_w))
        n = int(len(vals_w))
        scale = 1.4826 * mad
        if n < 8 or scale < 5e-4:
            med = 0.7 * med + 0.3 * g_med
            scale = max(0.7 * scale + 0.3 * g_scale, 1e-3)
        per[str(d)] = {'center': med, 'scale': scale, 'n': n}

    return {'do_log': do_log, 'robust': True, 'global_center': g_med, 'global_scale': g_scale, 'per_drug': per}

def standardize_with_drug_scaler(long_df: pd.DataFrame, scaler: Dict, inverse: bool = False) -> pd.DataFrame:
    df = long_df.copy()
    y = df['ic50'].astype(np.float32).values
    per = scaler['per_drug']
    g_c = scaler['global_center']
    g_s = scaler['global_scale']

    if not inverse:
        if scaler.get('do_log', False):
            y = np.log2(y + EPS).astype(np.float32)
        c = np.array([per.get(str(d), {'center': g_c})['center'] for d in df['drug'].astype(str).values], np.float32)
        s = np.array([per.get(str(d), {'scale': g_s})['scale'] for d in df['drug'].astype(str).values], np.float32)
        df['ic50'] = ((y - c) / (s + 1e-6)).astype(np.float32)
        return df

    c = np.array([per.get(str(d), {'center': g_c})['center'] for d in df['drug'].astype(str).values], np.float32)
    s = np.array([per.get(str(d), {'scale': g_s})['scale'] for d in df['drug'].astype(str).values], np.float32)
    y_inv = y * (s + 1e-6) + c
    if scaler.get('do_log', False):
        y_inv = np.maximum(2.0 ** y_inv - EPS, 0.0).astype(np.float32)
    df['ic50'] = y_inv.astype(np.float32)
    return df

def save_drug_scaler(scaler: Dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(scaler, f, indent=2)

def load_drug_scaler(path: Path) -> Dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def make_bipartite_graph(train_pairs_raw: pd.DataFrame, cell_ids: List[str], drug_ids: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
    cell_index = {c: i for i, c in enumerate(cell_ids)}
    drug_index = {d: i for i, d in enumerate(drug_ids)}
    N = len(cell_ids) + len(drug_ids)
    A = np.zeros((N, N), dtype=np.float32)

    yy = train_pairs_raw['ic50'].values.astype(np.float32)
    med = float(np.nanmedian(yy))
    scale = float(1.4826 * _mad(yy))
    z = (-(yy - med) / (scale + 1e-6)).astype(np.float32)
    z = (z - z.min()) / (z.max() - z.min() + 1e-6)
    w = np.clip(z, 0.05, 0.95)

    tmp = train_pairs_raw.copy()
    tmp['w'] = w
    for _, r in tmp.iterrows():
        ci = cell_index.get(str(r['cell']))
        dj = drug_index.get(str(r['drug']))
        if ci is None or dj is None:
            continue
        i = ci
        j = len(cell_ids) + dj
        A[i, j] = r['w']
        A[j, i] = r['w']

    A = A + np.eye(N, dtype=np.float32) * np.float32(1e-3)
    D = np.sum(A, axis=1, keepdims=True).astype(np.float32) + np.float32(1e-6)
    A_norm = A / np.sqrt(D @ D.T + np.float32(1e-12))

    X0 = np.zeros((N, 4), dtype=np.float32)
    X0[:len(cell_ids), 0] = 1.0
    X0[len(cell_ids):, 1] = 1.0
    return torch.tensor(A_norm, dtype=torch.float32), torch.tensor(X0, dtype=torch.float32)


def S_from_H(H_sparse: torch.Tensor) -> torch.Tensor:
    H = H_sparse.coalesce()
    Nc, Ne = H.size()
    Dv = torch.sparse.sum(H, dim=1).to_dense().clamp_min(1e-6)
    De = torch.sparse.sum(H, dim=0).to_dense().clamp_min(1.0)
    inv_sqrt_Dv = Dv.pow(-0.5)
    inv_De = De.pow(-1.0)
    vals = H.values()
    cols = H.indices()[1]
    vals_scaled = vals * inv_De[cols]
    A = torch.sparse_coo_tensor(H.indices(), vals_scaled, size=(Nc, Ne)).coalesce()
    S = torch.sparse.mm(A, H.transpose(0, 1))
    dL = inv_sqrt_Dv.unsqueeze(1)
    dR = inv_sqrt_Dv.unsqueeze(0)
    S_dense = S.to_dense()
    S_dense = dL * S_dense * dR
    return S_dense.contiguous()

def _align_df_to_cells(df: pd.DataFrame, cell_ids: List[str], fill_value: float = 0.0) -> pd.DataFrame:
    out = df.copy()
    out.iloc[:, 0] = out.iloc[:, 0].astype(str)
    out = out.drop_duplicates(subset=out.columns[0], keep='first')
    out = out.set_index(out.columns[0]).reindex(cell_ids)
    feat = out.iloc[:, :].replace([np.inf, -np.inf], np.nan).fillna(fill_value).astype(np.float32)
    feat.insert(0, 'cell', cell_ids)
    return feat.reset_index(drop=True)

def _fit_standardizer_from_train(mat_train: np.ndarray):
    mu = np.nanmean(mat_train, axis=0, keepdims=True).astype(np.float32)
    std = np.nanstd(mat_train, axis=0, keepdims=True).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return mu, std

def _apply_standardizer(mat: np.ndarray, mu: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((mat - mu) / std).astype(np.float32)

def _cosine_row_normalize(X: np.ndarray) -> np.ndarray:
    Xc = X - X.mean(axis=1, keepdims=True)
    denom = np.linalg.norm(Xc, axis=1, keepdims=True) + 1e-6
    return (Xc / denom).astype(np.float32)

def _softmax_select(sim: np.ndarray, topk: int, tau: float):
    k = min(int(topk), int(sim.shape[0]))
    if k <= 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float32)
    idx = np.argpartition(-sim, k - 1)[:k]
    idx = idx[np.isfinite(sim[idx])]
    if idx.size == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float32)
    s = sim[idx].astype(np.float32)
    w = np.exp(s / max(float(tau), 1e-6))
    w = w / (w.sum() + 1e-6)
    return idx.astype(np.int64), w.astype(np.float32)

def _coo_from_entries(rows: List[np.ndarray], cols: List[np.ndarray], vals: List[np.ndarray], n_rows: int, n_cols: int):
    if len(rows) == 0:
        indices = torch.zeros((2, 1), dtype=torch.long)
        values = torch.ones((1,), dtype=torch.float32)
        return torch.sparse_coo_tensor(indices, values, size=(n_rows, 1)).coalesce().float()
    r = np.concatenate(rows).astype(np.int64)
    c = np.concatenate(cols).astype(np.int64)
    v = np.concatenate(vals).astype(np.float32)
    H = torch.sparse_coo_tensor(torch.tensor(np.vstack([r, c]), dtype=torch.long),
                                torch.tensor(v, dtype=torch.float32),
                                size=(n_rows, n_cols)).coalesce()
    return H.float()

def build_train_defined_pathway_hypergraph(path_df_full: pd.DataFrame, train_cell_ids: List[str], topk_per_edge: int = 50):
    cell_ids = path_df_full.iloc[:, 0].astype(str).tolist()
    train_set = set(train_cell_ids)
    train_idx = np.array([i for i, c in enumerate(cell_ids) if c in train_set], dtype=np.int64)
    X = path_df_full.iloc[:, 1:].values.astype(np.float32)
    n_all, n_path = X.shape
    if train_idx.size == 0:
        raise ValueError('No training cells available for pathway hypergraph construction.')
    rows_list, cols_list, vals_list = [], [], []
    X_train = X[train_idx]
    for j in range(n_path):
        col_train = np.abs(X_train[:, j])
        k = min(int(topk_per_edge), int(train_idx.size))
        if k <= 0:
            continue
        top_pos = np.argpartition(-col_train, k - 1)[:k]
        threshold = float(np.min(col_train[top_pos]))
        col_full = np.abs(X[:, j])
        members = np.where(np.isfinite(col_full) & (col_full >= threshold))[0]
        if members.size == 0:
            members = train_idx[top_pos]
        rows_list.append(members.astype(np.int64))
        cols_list.append(np.full(members.shape[0], j, dtype=np.int64))
        vals_list.append(np.ones(members.shape[0], dtype=np.float32))
    H = _coo_from_entries(rows_list, cols_list, vals_list, n_all, n_path)
    col_sum = torch.sparse.sum(H, dim=0).to_dense().clamp_min(1.0)
    inv_col = 1.0 / col_sum
    coo = H.coalesce()
    H = torch.sparse_coo_tensor(coo.indices(), coo.values() * inv_col[coo.indices()[1]], size=H.shape).coalesce().float()
    return H

def build_train_anchor_knn_hypergraph(full_df: pd.DataFrame, train_cell_ids: List[str], k: int, tau: float, mutual: bool = False):
    if mutual:
        raise NotImplementedError('strict_inductive builder currently assumes KNN_MUTUAL=False.')
    cell_ids = full_df.iloc[:, 0].astype(str).tolist()
    train_set = set(train_cell_ids)
    train_idx = np.array([i for i, c in enumerate(cell_ids) if c in train_set], dtype=np.int64)
    X = full_df.iloc[:, 1:].values.astype(np.float32)
    n_all = X.shape[0]
    n_train = train_idx.size
    if n_train == 0:
        raise ValueError('No training cells available for anchor hypergraph construction.')
    Xn = _cosine_row_normalize(X)
    X_train = Xn[train_idx]
    train_train_cos = (X_train @ X_train.T).astype(np.float32)

    rows_list, cols_list, vals_list = [], [], []
    for col_anchor, _ in enumerate(train_idx):
        sim = train_train_cos[:, col_anchor].copy()
        idx_local, w = _softmax_select(sim, k, tau)
        if idx_local.size == 0:
            idx_local = np.array([col_anchor], dtype=np.int64)
            w = np.array([1.0], dtype=np.float32)
        global_rows = train_idx[idx_local]
        rows_list.append(global_rows.astype(np.int64))
        cols_list.append(np.full(global_rows.shape[0], col_anchor, dtype=np.int64))
        vals_list.append(w.astype(np.float32))

    nontrain_idx = np.array([i for i, c in enumerate(cell_ids) if c not in train_set], dtype=np.int64)
    if nontrain_idx.size > 0:
        cos_to_train = (Xn[nontrain_idx] @ X_train.T).astype(np.float32)
        for row_local, global_row in enumerate(nontrain_idx):
            idx_local, w = _softmax_select(cos_to_train[row_local], k, tau)
            if idx_local.size == 0:
                continue
            rows_list.append(np.full(idx_local.shape[0], global_row, dtype=np.int64))
            cols_list.append(idx_local.astype(np.int64))
            vals_list.append(w.astype(np.float32))

    H = _coo_from_entries(rows_list, cols_list, vals_list, n_all, n_train)
    col_sum = torch.sparse.sum(H, dim=0).to_dense().clamp_min(1e-6)
    inv_col = 1.0 / col_sum
    coo = H.coalesce()
    H = torch.sparse_coo_tensor(coo.indices(), coo.values() * inv_col[coo.indices()[1]], size=H.shape).coalesce().float()
    return H

def _write_preprocessing_meta(ctx: dict, split_meta: dict, train_cells: List[str], train_drugs: List[str], cell_ids: List[str], drug_ids: List[str]):
    meta = {
        'protocol': split_meta['protocol'],
        'experiment_name': ctx['experiment_name'],
        'cell_graph_mode': CELL_GRAPH_MODE,
        'preprocessing_scope': {
            'drug_response_scaler': 'fit_on_train_only',
            'motif_vocab_idf': 'fit_on_train_drugs_only',
            'bipartite_cell_drug_graph': 'built_from_train_pairs_only',
            'omics_column_standardization': 'fit_on_train_cells_only',
            'pathway_hyperedges': 'defined_from_train_cells_only',
            'expr_mut_meth_hyperedges': 'defined_from_training_anchor_cells_only',
            'val_test_cell_projection': 'project_to_training_defined_hyperedges_only',
        },
        'n_train_cells': int(len(train_cells)),
        'n_train_drugs': int(len(train_drugs)),
        'n_protocol_cells_total': int(len(cell_ids)),
        'n_protocol_drugs_total': int(len(drug_ids)),
        'train_cell_ids': list(sorted(map(str, train_cells))),
        'train_drug_ids': list(sorted(map(str, train_drugs))),
        'split_meta_path': str(ctx['split_meta_path']),
        'scaler_path': str(ctx['scaler_path']),
        'motif_vocab_path': str(ctx['motif_vocab_path']),
        'graph_cache_dir': str(ctx['graph_cache_dir']),
    }
    with open(ctx['preprocessing_meta_path'], 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


class PairDS(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame, cell_ids: List[str], drug_ids: List[str], drug_smiles: dict):
        self.df = df.reset_index(drop=True)
        self.cidx = {c: i for i, c in enumerate(cell_ids)}
        self.didx = {d: i for i, d in enumerate(drug_ids)}
        self.drug_smiles = drug_smiles

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        c = self.cidx[str(r['cell'])]
        d = self.didx[str(r['drug'])]
        y = float(r['ic50'])
        s = self.drug_smiles.get(str(r['drug']), '')
        return {'cell_idx': c, 'drug_idx': d, 'ic50': y, 'drug_smiles': s}

def pair_collate_fn(samples: List[dict], cache_dir: Path):
    cell_idx = torch.tensor([s['cell_idx'] for s in samples], dtype=torch.long)
    drug_idx = torch.tensor([s['drug_idx'] for s in samples], dtype=torch.long)
    y = torch.tensor([s['ic50'] for s in samples], dtype=torch.float32)
    smiles = [s['drug_smiles'] for s in samples]
    graph_batch = smiles_to_3d_batch(smiles, cache_dir=cache_dir)
    return {
        'cell_idx': cell_idx,
        'drug_idx': drug_idx,
        'ic50': y,
        'drug_smiles': smiles,
        'drug_graph_batch': graph_batch,
    }


def prepare_dataloaders(split_mode: str = 'pair_level', experiment_name: Optional[str] = None, overwrite_splits: bool = False):
    split_mode = normalize_split_mode(split_mode)
    profile = get_protocol_profile(split_mode)
    train_raw, val_raw, test_raw, split_meta, ctx = ensure_protocol_split_files(split_mode, experiment_name, overwrite=overwrite_splits)

    scaler = fit_drug_scaler(train_raw, do_log=None)
    save_drug_scaler(scaler, ctx['scaler_path'])
    train_df = standardize_with_drug_scaler(train_raw, scaler, inverse=False)
    val_df = standardize_with_drug_scaler(val_raw, scaler, inverse=False)
    test_df = standardize_with_drug_scaler(test_raw, scaler, inverse=False)

    protocol_pairs = pd.concat([train_raw, val_raw, test_raw], axis=0, ignore_index=True)
    cell_ids = sorted(protocol_pairs['cell'].astype(str).unique().tolist())
    drug_ids = sorted(protocol_pairs['drug'].astype(str).unique().tolist())
    train_cell_ids = sorted(train_raw['cell'].astype(str).unique().tolist())
    train_drug_ids = sorted(train_raw['drug'].astype(str).unique().tolist())

    drug_smiles_map = load_drug_smiles_map(FILE_DRUG_INFO)
    for d in drug_ids:
        drug_smiles_map.setdefault(str(d), '')

    path_df_raw = _align_df_to_cells(load_pathway_df(), cell_ids)
    expr_df_raw = _align_df_to_cells(load_expression_df(zscore_cols=False), cell_ids)
    mut_df_raw = _align_df_to_cells(load_mutation_binary_df(), cell_ids)
    meth_df_raw = _align_df_to_cells(load_methylation_df(zscore_cols=False), cell_ids)

    train_mask = np.array([c in set(train_cell_ids) for c in cell_ids], dtype=bool)
    if train_mask.sum() == 0:
        raise ValueError('Training split contains no cells; cannot fit protocol-specific preprocessing.')

    path_mat_raw = path_df_raw.iloc[:, 1:].values.astype(np.float32)
    expr_mat_raw = expr_df_raw.iloc[:, 1:].values.astype(np.float32)
    meth_mat_raw = meth_df_raw.iloc[:, 1:].values.astype(np.float32)
    mut_mat = (mut_df_raw.iloc[:, 1:].values.astype(np.float32) > 0).astype(np.float32)

    path_mu, path_std = _fit_standardizer_from_train(path_mat_raw[train_mask])
    expr_mu, expr_std = _fit_standardizer_from_train(expr_mat_raw[train_mask])
    meth_mu, meth_std = _fit_standardizer_from_train(meth_mat_raw[train_mask])

    path_df = path_df_raw.copy(); path_df.iloc[:, 1:] = _apply_standardizer(path_mat_raw, path_mu, path_std)
    expr_df = expr_df_raw.copy(); expr_df.iloc[:, 1:] = _apply_standardizer(expr_mat_raw, expr_mu, expr_std)
    mut_df = mut_df_raw.copy(); mut_df.iloc[:, 1:] = mut_mat
    meth_df = meth_df_raw.copy(); meth_df.iloc[:, 1:] = _apply_standardizer(meth_mat_raw, meth_mu, meth_std)

    A_norm, X0 = make_bipartite_graph(train_raw, cell_ids, drug_ids)

    if CELL_GRAPH_MODE != 'strict_inductive':
        raise ValueError(f'Unsupported CELL_GRAPH_MODE={CELL_GRAPH_MODE}.')
    H_path = build_train_defined_pathway_hypergraph(path_df, train_cell_ids, topk_per_edge=PATH_TOPK)
    H_expr = build_train_anchor_knn_hypergraph(expr_df, train_cell_ids, KNN_K, KNN_TAU, KNN_MUTUAL)
    H_mut = build_train_anchor_knn_hypergraph(mut_df, train_cell_ids, KNN_K, KNN_TAU, KNN_MUTUAL)
    H_meth = build_train_anchor_knn_hypergraph(meth_df, train_cell_ids, KNN_K, KNN_TAU, KNN_MUTUAL)

    S_path = S_from_H(H_path)
    S_expr = S_from_H(H_expr)
    S_mut = S_from_H(H_mut)
    S_meth = S_from_H(H_meth)

    F_path = torch.tensor(path_df.iloc[:, 1:].values, dtype=torch.float32)
    F_expr = torch.tensor(expr_df.iloc[:, 1:].values, dtype=torch.float32)
    F_mut = torch.tensor(mut_df.iloc[:, 1:].values, dtype=torch.float32)
    F_meth = torch.tensor(meth_df.iloc[:, 1:].values, dtype=torch.float32)

    train_ds = PairDS(train_df, cell_ids, drug_ids, drug_smiles_map)
    val_ds = PairDS(val_df, cell_ids, drug_ids, drug_smiles_map)
    test_ds = PairDS(test_df, cell_ids, drug_ids, drug_smiles_map)

    batch_size = int(profile['batch_size'])
    collate = lambda samples: pair_collate_fn(samples, cache_dir=ctx['graph_cache_dir'])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False, collate_fn=collate)

    drug_weights = {}
    for d, meta in scaler['per_drug'].items():
        n = max(int(meta.get('n', 1)), 1)
        drug_weights[str(d)] = float(np.sqrt(n))
    mean_w = np.mean(list(drug_weights.values())) if drug_weights else 1.0
    for k in drug_weights:
        drug_weights[k] /= (mean_w + 1e-6)

    _write_preprocessing_meta(ctx, split_meta, train_cell_ids, train_drug_ids, cell_ids, drug_ids)

    return {
        'train': train_loader, 'val': val_loader, 'test': test_loader,
        'cell_ids': cell_ids, 'drug_ids': drug_ids,
        'F_path': F_path, 'F_expr': F_expr, 'F_mut': F_mut, 'F_meth': F_meth,
        'S_path': S_path, 'S_expr': S_expr, 'S_mut': S_mut, 'S_meth': S_meth,
        'A_norm': A_norm, 'X0': X0,
        'path_df': path_df,
        'scaler_path': ctx['scaler_path'],
        'drug_weights': drug_weights,
        'drug_smiles_map': drug_smiles_map,
        'graph_cache_dir': ctx['graph_cache_dir'],
        'split_meta_path': ctx['split_meta_path'],
        'split_meta': split_meta,
        'context': ctx,
        'preprocessing_meta_path': ctx['preprocessing_meta_path'],
    }


import json
from pathlib import Path

import pandas as pd

from config import (
    FILE_GDSC_IC50, CELL_ID_CANDIDATES, DRUG_ID_CANDIDATES, STANDARD_CELL_ID,
    SEED, VAL_RATIO, TEST_RATIO, get_experiment_context, normalize_split_mode,
)


def read_ic50_long(csv_path: Path = FILE_GDSC_IC50,
                   cell_col_candidates=CELL_ID_CANDIDATES,
                   drug_col_candidates=DRUG_ID_CANDIDATES) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    cols_lower = [c.lower() for c in df.columns]
    if df.shape[1] > 3 and (STANDARD_CELL_ID in df.columns or df.columns[0].lower() in cols_lower):
        cell_col = STANDARD_CELL_ID if STANDARD_CELL_ID in df.columns else df.columns[0]
        rows = []
        for col in df.columns:
            if col == cell_col:
                continue
            cvals = df[[cell_col, col]].dropna()
            for _, r in cvals.iterrows():
                rows.append((str(r[cell_col]), str(col), float(r[col])))
        long_df = pd.DataFrame(rows, columns=['cell', 'drug', 'ic50'])
    else:
        cell_col = next((c for c in cell_col_candidates if c in df.columns), df.columns[0])
        drug_col = next((c for c in drug_col_candidates if c in df.columns), df.columns[1])
        ic50_col = next((c for c in ['ic50', 'IC50', 'lnIC50', 'LNIC50', 'auc', 'AUC', 'value', 'Value'] if c in df.columns), df.columns[-1])
        long_df = df[[cell_col, drug_col, ic50_col]].copy()
        long_df.columns = ['cell', 'drug', 'ic50']
    long_df = long_df.dropna(subset=['ic50']).copy()
    long_df['cell'] = long_df['cell'].astype(str)
    long_df['drug'] = long_df['drug'].astype(str)
    return long_df


def _split_pairs(pairs: pd.DataFrame, seed: int, val_ratio: float, test_ratio: float):
    pairs = pairs.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n = len(pairs)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)
    test_df = pairs.iloc[:n_test]
    val_df = pairs.iloc[n_test:n_test + n_val]
    train_df = pairs.iloc[n_test + n_val:]
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def _split_by_entity(pairs: pd.DataFrame, entity_col: str, seed: int, val_ratio: float, test_ratio: float):
    entities = pd.Series(sorted(pairs[entity_col].astype(str).unique())).sample(frac=1.0, random_state=seed).tolist()
    n = len(entities)
    n_test = max(1, int(n * test_ratio))
    n_val = max(1, int(n * val_ratio))
    test_entities = set(entities[:n_test])
    val_entities = set(entities[n_test:n_test + n_val])
    train_entities = set(entities[n_test + n_val:])
    train_df = pairs[pairs[entity_col].isin(train_entities)].copy()
    val_df = pairs[pairs[entity_col].isin(val_entities)].copy()
    test_df = pairs[pairs[entity_col].isin(test_entities)].copy()
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True), train_entities, val_entities, test_entities


def make_protocol_splits(split_mode: str, seed: int = SEED, val_ratio: float = VAL_RATIO, test_ratio: float = TEST_RATIO):
    split_mode = normalize_split_mode(split_mode)
    pairs = read_ic50_long(FILE_GDSC_IC50)
    meta = {
        'protocol': split_mode,
        'seed': int(seed),
        'val_ratio': float(val_ratio),
        'test_ratio': float(test_ratio),
        'source_file': str(FILE_GDSC_IC50),
    }
    if split_mode == 'pair_level':
        train_df, val_df, test_df = _split_pairs(pairs, seed, val_ratio, test_ratio)
    elif split_mode == 'drug_coldstart':
        train_df, val_df, test_df, train_drugs, val_drugs, test_drugs = _split_by_entity(pairs, 'drug', seed, val_ratio, test_ratio)
        meta.update({'train_drug_ids': sorted(train_drugs), 'val_drug_ids': sorted(val_drugs), 'test_drug_ids': sorted(test_drugs)})
    elif split_mode == 'cell_coldstart':
        train_df, val_df, test_df, train_cells, val_cells, test_cells = _split_by_entity(pairs, 'cell', seed, val_ratio, test_ratio)
        meta.update({'train_cell_ids': sorted(train_cells), 'val_cell_ids': sorted(val_cells), 'test_cell_ids': sorted(test_cells)})
    else:
        raise ValueError(split_mode)

    meta.update({
        'n_total_pairs': int(len(pairs)),
        'n_train_pairs': int(len(train_df)),
        'n_val_pairs': int(len(val_df)),
        'n_test_pairs': int(len(test_df)),
        'n_train_drugs': int(train_df['drug'].nunique()),
        'n_val_drugs': int(val_df['drug'].nunique()),
        'n_test_drugs': int(test_df['drug'].nunique()),
        'n_train_cells': int(train_df['cell'].nunique()),
        'n_val_cells': int(val_df['cell'].nunique()),
        'n_test_cells': int(test_df['cell'].nunique()),
    })
    return train_df, val_df, test_df, meta


def ensure_protocol_split_files(split_mode: str, experiment_name=None, overwrite: bool = False):
    ctx = get_experiment_context(split_mode, experiment_name)
    if (not overwrite and ctx['split_meta_path'].exists() and ctx['train_pairs_path'].exists() and
            ctx['val_pairs_path'].exists() and ctx['test_pairs_path'].exists()):
        train_df = pd.read_csv(ctx['train_pairs_path'])
        val_df = pd.read_csv(ctx['val_pairs_path'])
        test_df = pd.read_csv(ctx['test_pairs_path'])
        with open(ctx['split_meta_path'], 'r', encoding='utf-8') as f:
            meta = json.load(f)
        return train_df, val_df, test_df, meta, ctx

    train_df, val_df, test_df, meta = make_protocol_splits(split_mode)
    train_df.to_csv(ctx['train_pairs_path'], index=False)
    val_df.to_csv(ctx['val_pairs_path'], index=False)
    test_df.to_csv(ctx['test_pairs_path'], index=False)
    with open(ctx['split_meta_path'], 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return train_df, val_df, test_df, meta, ctx

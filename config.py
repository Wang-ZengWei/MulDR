
from pathlib import Path

DATA_ROOT = Path('./data')
CELL_DIR = DATA_ROOT / 'CELL'
INFO_DIR = DATA_ROOT / 'drug_info'

FILE_EXP = CELL_DIR / 'geo_expression_cosmic.csv'
FILE_METH = CELL_DIR / 'geo_methylation_cosmic.csv'
FILE_MUT = CELL_DIR / 'geo_mutation_cosmic.csv'
FILE_PATH = CELL_DIR / 'pathway_cosmic.csv'

FILE_GDSC_IC50 = DATA_ROOT / 'GDSC2_IC50.csv'

FILE_DRUG_INFO = INFO_DIR / 'drug_info.csv'
FG_SMARTS_PATH = INFO_DIR / 'new_fg_groups.txt'

CELL_ID_CANDIDATES = ('COSMIC_ID', 'cell_id', 'cell_line', 'Cell', 'cell_line_name', 'CCLE_Name')
DRUG_ID_CANDIDATES = ('pubchem_id', 'PubChem_ID', 'DRUG_ID', 'drug_id', 'drug_name')
STANDARD_CELL_ID = 'COSMIC_ID'
STANDARD_DRUG_ID = 'pubchem_id'

SEED = 22
RDKit_RANDOM_SEED = 2025
DEVICE = 'cuda'

VAL_RATIO = 0.1
TEST_RATIO = 0.1

RUNS_ROOT = Path('./runs')
PREPARED_ROOT = Path('./prepared')
SPLITS_ROOT = PREPARED_ROOT / 'protocol_splits'

SUPPORTED_SPLIT_MODES = ('pair_level', 'drug_coldstart', 'cell_coldstart')

PATH_TOPK = 50
KNN_K = 30
KNN_TAU = 0.20
KNN_MUTUAL = False

CELL_GRAPH_MODE = 'strict_inductive'

MOTIF_MIN_FREQ = 3
USE_DRUG_GRAPH_CACHE = True

GNN3D_ATOM_FEAT_DIM = 128
GNN3D_HIDDEN_DIM = 128
GNN3D_LAYERS = 4
GNN3D_NUM_RBF = 32
GNN3D_RBF_MAX = 10.0


def normalize_split_mode(split_mode: str) -> str:
    s = str(split_mode).strip().lower()
    if s not in SUPPORTED_SPLIT_MODES:
        raise ValueError(f'Unsupported split_mode={split_mode}. Supported: {SUPPORTED_SPLIT_MODES}')
    return s


def experiment_name_for(split_mode: str) -> str:
    sm = normalize_split_mode(split_mode)
    if sm == 'pair_level':
        return 'main'
    return sm


def get_experiment_context(split_mode: str, experiment_name=None) -> dict:
    sm = normalize_split_mode(split_mode)
    exp = experiment_name or experiment_name_for(sm)
    if experiment_name is None and sm == 'pair_level':
        out_dir = RUNS_ROOT
        prepared_dir = PREPARED_ROOT
        splits_dir = SPLITS_ROOT
    else:
        out_dir = RUNS_ROOT / exp
        prepared_dir = PREPARED_ROOT / exp
        splits_dir = SPLITS_ROOT / exp
    out_dir.mkdir(parents=True, exist_ok=True)
    prepared_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)
    return {
        'split_mode': sm,
        'experiment_name': exp,
        'out_dir': out_dir,
        'prepared_dir': prepared_dir,
        'splits_dir': splits_dir,
        'log_csv_path': out_dir / 'fusion_history.csv',
        'best_model_path': out_dir / 'fusion_best.pt',
        'test_metrics_path': out_dir / 'fusion_test_metrics.csv',
        'final_eval_json_path': out_dir / 'final_eval_summary.json',
        'scaler_path': prepared_dir / 'drug_scaler.json',
        'motif_vocab_path': prepared_dir / 'motif_vocab.json',
        'motif_vocab_meta_path': prepared_dir / 'motif_vocab_meta.json',
        'graph_cache_dir': prepared_dir / 'drug_graph_cache',
        'preprocessing_meta_path': prepared_dir / 'protocol_preprocessing_meta.json',
        'split_meta_path': splits_dir / 'split_meta.json',
        'train_pairs_path': splits_dir / 'train_pairs.csv',
        'val_pairs_path': splits_dir / 'val_pairs.csv',
        'test_pairs_path': splits_dir / 'test_pairs.csv',
    }


def get_protocol_profile(split_mode: str) -> dict:
    sm = normalize_split_mode(split_mode)

    base = {
        'batch_size': 128,
        'lr': 1e-3,
        'weight_decay': 1e-4,
        'epochs': 160,
        'patience': 4,
        'early_stop_min_delta': 1e-4,
        'degrade_stop_patience': 1,
        'degrade_stop_rel': 0.0,
        'degrade_stop_abs': 0.002,
        'min_epochs_before_stop': 8,
        'warmup_epochs': 5,
        'gradient_clip': 1.0,
        'cell_dim': 512,
        'drug_dim': 512,
        'hidden_dim': 512,
        'heads_gat': 8,
        'appnp_T': 5,
        'appnp_alpha': 0.1,
        'hyper_drop': 0.30,
        'mfh_k': 1000,
        'mfh_r': 5,
        'mfh_h': 2,
        'mfh_dropout': 0.10,
        'view_tau': 1.5,
        'min_view_weight': 0.02,
        'view_dropout': 0.05,
        'view_entropy_weight': 3e-4,
        'use_bipartite': True,
        'v_branch_scale': 1.0,
        'prop_loss_weight': 1e-3,
        'fp_recon_weight': 2e-4,
        'motif_recon_weight': 2e-4,
        'backbone_consistency_weight': 2e-4,
        'pred_consistency_weight': 2e-4,
        'cons_weight': 1.0,
        'lap_weight': 0.30,
        'mse_weight': 1.0,
        'huber_weight': 0.20,
        'corr_weight': 0.00,
        'backbone_pred_weight': 0.10,
        'drug_encoder_lr_mult': 0.5,
        'view_gate_weight_decay': 1e-3,
        'freeze_view_gate_after': 6,
        'swa_start_frac': 0.70,
        'inference_backbone_blend': 0.0,
        'use_ema_swa_blend': True,
    }

    if sm == 'pair_level':
        return base

    if sm == 'drug_coldstart':
        prof = dict(base)
        prof.update({
            'lr': 8e-4,
            'weight_decay': 1.5e-4,
            'epochs': 200,
            'patience': 4,
            'cell_dim': 384,
            'drug_dim': 384,
            'hidden_dim': 384,
            'heads_gat': 4,
            'mfh_k': 700,
            'mfh_dropout': 0.18,
            'view_tau': 1.8,
            'min_view_weight': 0.05,
            'view_dropout': 0.10,
            'view_entropy_weight': 1.2e-3,
            'use_bipartite': False,
            'v_branch_scale': 0.0,
            'prop_loss_weight': 5e-3,
            'fp_recon_weight': 2e-3,
            'motif_recon_weight': 2e-3,
            'backbone_consistency_weight': 2e-3,
            'pred_consistency_weight': 1e-3,
            'cons_weight': 1.0,
            'lap_weight': 0.15,
            'mse_weight': 0.90,
            'huber_weight': 0.50,
            'corr_weight': 0.05,
            'backbone_pred_weight': 0.25,
            'freeze_view_gate_after': 10,
            'inference_backbone_blend': 0.20,
        })
        return prof

    if sm == 'cell_coldstart':
        prof = dict(base)
        prof.update({
            'lr': 9e-4,
            'weight_decay': 1.2e-4,
            'epochs': 180,
            'patience': 4,
            'cell_dim': 448,
            'drug_dim': 448,
            'hidden_dim': 448,
            'heads_gat': 8,
            'mfh_k': 850,
            'mfh_dropout': 0.14,
            'view_tau': 1.6,
            'min_view_weight': 0.03,
            'view_dropout': 0.06,
            'view_entropy_weight': 7e-4,
            'use_bipartite': True,
            'v_branch_scale': 0.35,
            'prop_loss_weight': 2e-3,
            'fp_recon_weight': 7e-4,
            'motif_recon_weight': 7e-4,
            'backbone_consistency_weight': 8e-4,
            'pred_consistency_weight': 4e-4,
            'cons_weight': 1.0,
            'lap_weight': 0.25,
            'mse_weight': 1.0,
            'huber_weight': 0.35,
            'corr_weight': 0.02,
            'backbone_pred_weight': 0.15,
            'freeze_view_gate_after': 8,
            'inference_backbone_blend': 0.05,
        })
        return prof

    raise ValueError(sm)


import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.swa_utils import AveragedModel
from tqdm import tqdm

from config import (
    DEVICE, SEED, FG_SMARTS_PATH,
    get_experiment_context, normalize_split_mode, get_protocol_profile,
)
from data import prepare_dataloaders, load_drug_scaler, standardize_with_drug_scaler
from model import MulDR
from metrics import mse, rmse, mae, r2, pcc, scc

torch.manual_seed(SEED)
np.random.seed(SEED)


class EMA:
    def __init__(self, model, decay=0.9995):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}
        self.backup = None

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            self.shadow[k] = self.decay * self.shadow[k] + (1. - self.decay) * v.detach()

    @torch.no_grad()
    def apply_to(self, model):
        self.backup = {k: v.clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.shadow, strict=False)

    @torch.no_grad()
    def restore(self, model):
        if self.backup is not None:
            model.load_state_dict(self.backup, strict=False)
            self.backup = None


def _append_row_csv(path: Path, row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([row])
    if not path.exists():
        df.to_csv(path, index=False)
    else:
        df.to_csv(path, mode='a', header=False, index=False)


def _write_full_csv(path: Path, row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(path, index=False)


def _to_device_graph_batch(graph_batch: dict, device: torch.device) -> dict:
    out = {}
    for k, v in graph_batch.items():
        out[k] = v.to(device) if torch.is_tensor(v) else v
    return out


def batch_to_device(batch: dict, device: torch.device):
    ci = batch['cell_idx']
    dj = batch['drug_idx']
    y = batch['ic50']
    if not torch.is_tensor(ci):
        ci = torch.tensor(ci, dtype=torch.long)
    if not torch.is_tensor(dj):
        dj = torch.tensor(dj, dtype=torch.long)
    if not torch.is_tensor(y):
        y = torch.tensor(y, dtype=torch.float32)
    ci = ci.to(device)
    dj = dj.to(device)
    y = y.to(device)
    smiles = batch['drug_smiles']
    graphb = _to_device_graph_batch(batch['drug_graph_batch'], device)
    return ci, dj, y, smiles, graphb


def _verify_protocol_artifacts(ctx: dict, split_mode: str):
    if not Path(ctx['motif_vocab_path']).exists():
        raise FileNotFoundError(
            f"Protocol-specific motif vocabulary not found: {ctx['motif_vocab_path']}\n"
            f"Run first: python build_motif_vocab_and_cache.py --split_mode {split_mode}"
        )
    if not Path(ctx['motif_vocab_meta_path']).exists():
        raise FileNotFoundError(f'motif_vocab_meta.json not found: {ctx["motif_vocab_meta_path"]}')
    with open(ctx['motif_vocab_meta_path'], 'r', encoding='utf-8') as f:
        vocab_meta = json.load(f)
    if vocab_meta.get('protocol') != split_mode:
        raise ValueError(
            f'motif_vocab_meta protocol={vocab_meta.get("protocol")} does not match split_mode={split_mode}.'
        )


def batch_corr_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = pred.float()
    target = target.float()
    pred_c = pred - pred.mean()
    target_c = target - target.mean()
    denom = pred_c.std(unbiased=False) * target_c.std(unbiased=False)
    if float(denom.detach().cpu().item()) < 1e-8:
        return pred.new_tensor(0.0)
    corr = (pred_c * target_c).mean() / (denom + 1e-6)
    return 1.0 - corr.clamp(-1.0, 1.0)


def compute_loss(outputs: dict, aux: dict, y_std: torch.Tensor, profile: dict) -> tuple[torch.Tensor, dict]:
    pred_main = outputs['pred_main']
    pred_backbone = outputs['pred_backbone']

    reg_main = (
        float(profile['mse_weight']) * F.mse_loss(pred_main, y_std) +
        float(profile['huber_weight']) * F.smooth_l1_loss(pred_main, y_std)
    )
    if float(profile['corr_weight']) > 0:
        reg_main = reg_main + float(profile['corr_weight']) * batch_corr_loss(pred_main, y_std)

    reg_backbone = (
        F.mse_loss(pred_backbone, y_std) +
        0.5 * F.smooth_l1_loss(pred_backbone, y_std)
    ) * float(profile['backbone_pred_weight'])

    pred_cons = float(profile['pred_consistency_weight']) * F.smooth_l1_loss(pred_main, pred_backbone)

    structure = (
        float(profile['cons_weight']) * aux['cons'] +
        float(profile['lap_weight']) * aux['lap'] +
        float(profile['view_entropy_weight']) * aux['view_entropy_reg'] +
        float(profile['backbone_consistency_weight']) * aux['backbone_consistency']
    )

    pp = aux['drug_prop']
    prop_true = pp['prop_true']
    prop_pred = pp['prop_pred']
    if prop_true.numel() > 0:
        pt = (prop_true - prop_true.mean(0, keepdim=True)) / (prop_true.std(0, keepdim=True, unbiased=False) + 1e-6)
        ppn = (prop_pred - prop_pred.mean(0, keepdim=True)) / (prop_pred.std(0, keepdim=True, unbiased=False) + 1e-6)
        prop_loss = F.mse_loss(ppn, pt)
    else:
        prop_loss = pred_main.new_tensor(0.0)

    fp_loss = F.binary_cross_entropy_with_logits(pp['fp_recon_logits'], pp['fp_target'])
    motif_loss = F.mse_loss(pp['motif_recon'], pp['motif_target'])

    loss = (
        reg_main +
        reg_backbone +
        pred_cons +
        structure +
        float(profile['prop_loss_weight']) * prop_loss +
        float(profile['fp_recon_weight']) * fp_loss +
        float(profile['motif_recon_weight']) * motif_loss
    )

    loss_items = {
        'reg_main': float(reg_main.detach().cpu().item()),
        'reg_backbone': float(reg_backbone.detach().cpu().item()),
        'pred_cons': float(pred_cons.detach().cpu().item()),
        'structure': float(structure.detach().cpu().item()),
        'prop': float(prop_loss.detach().cpu().item()),
        'fp': float(fp_loss.detach().cpu().item()),
        'motif': float(motif_loss.detach().cpu().item()),
    }
    return loss, loss_items


def collect_predictions(model, loader, device, F_path, F_expr, F_mut, F_meth, S_path, S_expr, S_mut, S_meth, X0, A_norm):
    model.eval()
    ys = []
    yps = []
    djs = []
    with torch.no_grad():
        for batch in loader:
            ci, dj, y_std, drug_smiles, drug_graph_batch = batch_to_device(batch, device)
            outputs, _ = model(ci, drug_smiles, drug_graph_batch, F_path, F_expr, F_mut, F_meth, S_path, S_expr, S_mut, S_meth, X0, A_norm)
            ys.append(y_std.cpu().numpy())
            yps.append(outputs['pred'].cpu().numpy())
            djs.append(dj.cpu().numpy())
    return np.concatenate(ys), np.concatenate(yps), np.concatenate(djs)


def inverse_metric_dict(y_std, yp_std, dj_idx, drug_ids, scaler):
    drug_keys = [drug_ids[i] for i in dj_idx.tolist()]
    df_true = pd.DataFrame({'cell': [None] * len(y_std), 'drug': drug_keys, 'ic50': y_std.astype(np.float32)})
    df_pred = pd.DataFrame({'cell': [None] * len(y_std), 'drug': drug_keys, 'ic50': yp_std.astype(np.float32)})
    y_true = standardize_with_drug_scaler(df_true, scaler, inverse=True)['ic50'].values
    y_pred = standardize_with_drug_scaler(df_pred, scaler, inverse=True)['ic50'].values
    return {
        'MSE': mse(y_true, y_pred),
        'RMSE': rmse(y_true, y_pred),
        'MAE': mae(y_true, y_pred),
        'R2': r2(y_true, y_pred),
        'PCC': pcc(y_true, y_pred),
        'SCC': scc(y_true, y_pred),
        'y_true': y_true,
        'y_pred': y_pred,
    }


def run(split_mode: str = 'pair_level', experiment_name=None):
    split_mode = normalize_split_mode(split_mode)
    profile = get_protocol_profile(split_mode)
    ctx = get_experiment_context(split_mode, experiment_name)
    _verify_protocol_artifacts(ctx, split_mode)

    loaders = prepare_dataloaders(split_mode=split_mode, experiment_name=experiment_name)
    cell_ids = loaders['cell_ids']
    drug_ids = loaders['drug_ids']
    device = torch.device(DEVICE if torch.cuda.is_available() else 'cpu')

    F_path = loaders['F_path'].to(device)
    F_expr = loaders['F_expr'].to(device)
    F_mut = loaders['F_mut'].to(device)
    F_meth = loaders['F_meth'].to(device)
    S_path = loaders['S_path'].to(device)
    S_expr = loaders['S_expr'].to(device)
    S_mut = loaders['S_mut'].to(device)
    S_meth = loaders['S_meth'].to(device)
    A_norm = loaders['A_norm'].clone().detach().float().to(device)
    X0 = loaders['X0'].clone().detach().float().to(device)

    feat_dims = {
        'path': F_path.size(1),
        'expr': F_expr.size(1),
        'mut': F_mut.size(1),
        'meth': F_meth.size(1),
    }
    model = MulDR(
        n_cells=len(cell_ids),
        n_drugs=len(drug_ids),
        feat_dims=feat_dims,
        profile=profile,
        motif_vocab_path=str(ctx['motif_vocab_path']),
        fg_smarts_path=str(FG_SMARTS_PATH),
    ).to(device)

    gate_params = list(model.view_gate.parameters())
    drug_params = list(model.drug_encoder.parameters())
    other_params = [p for n, p in model.named_parameters() if (not n.startswith('view_gate.')) and (not n.startswith('drug_encoder.'))]
    opt = optim.AdamW([
        {'name': 'others', 'params': other_params, 'lr': float(profile['lr']), 'weight_decay': float(profile['weight_decay'])},
        {'name': 'drug_encoder', 'params': drug_params, 'lr': float(profile['lr']) * float(profile['drug_encoder_lr_mult']), 'weight_decay': float(profile['weight_decay'])},
        {'name': 'view_gate', 'params': gate_params, 'lr': float(profile['lr']), 'weight_decay': float(profile['view_gate_weight_decay'])},
    ], lr=float(profile['lr']))

    warmup_epochs = int(profile['warmup_epochs'])
    total_epochs = int(profile['epochs'])

    def lr_lambda(cur_epoch):
        if cur_epoch < warmup_epochs:
            return (cur_epoch + 1) / max(1, warmup_epochs)
        progress = (cur_epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        cosine = 0.5 * (1 + np.cos(np.pi * progress))
        return 0.1 + 0.9 * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=4, min_lr=1e-6)
    ema = EMA(model, decay=0.9995)
    swa_start = max(12, int(float(profile['swa_start_frac']) * total_epochs))
    swa_model = AveragedModel(model)
    swa_n = 0

    scaler = load_drug_scaler(loaders['scaler_path'])
    train_loader = loaders['train']
    val_loader = loaders['val']
    test_loader = loaders['test']

    best_val = np.inf
    best_epoch = -1
    best_state = None
    patience_left = int(profile['patience'])
    degrade_epochs = 0
    early_stop_min_delta = float(profile.get('early_stop_min_delta', 0.0))
    degrade_stop_patience = int(profile.get('degrade_stop_patience', 0))
    degrade_stop_rel = float(profile.get('degrade_stop_rel', 0.0))
    degrade_stop_abs = float(profile.get('degrade_stop_abs', 0.0))
    min_epochs_before_stop = int(profile.get('min_epochs_before_stop', 0))
    freeze_done = False

    for epoch in range(1, total_epochs + 1):
        model.train()
        tr_losses = []
        tr_items = {'reg_main': [], 'reg_backbone': [], 'pred_cons': [], 'structure': [], 'prop': [], 'fp': [], 'motif': []}
        tr_y_std = []
        tr_yp_std = []
        tr_dj = []

        t0 = time.time()
        pbar = tqdm(train_loader, total=len(train_loader), desc=f'[train:{split_mode}] epoch {epoch}', ncols=120)
        for batch in pbar:
            ci, dj, y_std, drug_smiles, drug_graph_batch = batch_to_device(batch, device)
            tr_dj.append(dj.detach().cpu().numpy())

            opt.zero_grad(set_to_none=True)
            outputs, aux = model(ci, drug_smiles, drug_graph_batch, F_path, F_expr, F_mut, F_meth, S_path, S_expr, S_mut, S_meth, X0, A_norm)
            loss, loss_items = compute_loss(outputs, aux, y_std, profile)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(profile['gradient_clip']))
            opt.step()
            ema.update(model)

            tr_losses.append(float(loss.item()))
            for k, v in loss_items.items():
                tr_items[k].append(v)
            tr_y_std.append(y_std.detach().cpu().numpy())
            tr_yp_std.append(outputs['pred'].detach().cpu().numpy())
            pbar.set_postfix(loss=f'{np.mean(tr_losses):.4f}')

        scheduler.step()

        tr_y_std = np.concatenate(tr_y_std)
        tr_yp_std = np.concatenate(tr_yp_std)
        tr_dj = np.concatenate(tr_dj)
        tr_metrics = inverse_metric_dict(tr_y_std, tr_yp_std, tr_dj, drug_ids, scaler)

        ema.apply_to(model)
        va_y_std, va_yp_std, va_dj = collect_predictions(model, val_loader, device, F_path, F_expr, F_mut, F_meth, S_path, S_expr, S_mut, S_meth, X0, A_norm)
        ema.restore(model)
        va_metrics = inverse_metric_dict(va_y_std, va_yp_std, va_dj, drug_ids, scaler)

        elapsed = time.time() - t0
        plateau.step(va_metrics['RMSE'])

        summary = {
            'epoch': epoch,
            'train_MSE': float(tr_metrics['MSE']),
            'train_RMSE': float(tr_metrics['RMSE']),
            'train_MAE': float(tr_metrics['MAE']),
            'train_R2': float(tr_metrics['R2']),
            'train_PCC': float(tr_metrics['PCC']),
            'train_SCC': float(tr_metrics['SCC']),
            'val_MSE': float(va_metrics['MSE']),
            'val_RMSE': float(va_metrics['RMSE']),
            'val_MAE': float(va_metrics['MAE']),
            'val_R2': float(va_metrics['R2']),
            'val_PCC': float(va_metrics['PCC']),
            'val_SCC': float(va_metrics['SCC']),
            'train_loss_mean': float(np.mean(tr_losses)),
            'time_sec': float(elapsed),
        }
        for k, arr in tr_items.items():
            summary[f'loss_{k}'] = float(np.mean(arr)) if arr else 0.0

        print(
            f"\nEpoch {epoch:03d}/{total_epochs} | "
            f"Val RMSE {va_metrics['RMSE']:.4f} | MAE {va_metrics['MAE']:.4f} | R2 {va_metrics['R2']:.4f} | "
            f"time {elapsed:.1f}s"
        )
        _append_row_csv(ctx['log_csv_path'], summary)

        if epoch >= swa_start:
            ema.apply_to(model)
            swa_model.update_parameters(model)
            ema.restore(model)
            swa_n += 1

        val_rmse = float(va_metrics['RMSE'])
        improved = val_rmse < (best_val - early_stop_min_delta)
        if improved:
            best_val = val_rmse
            best_epoch = epoch
            ema.apply_to(model)
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            torch.save(best_state, ctx['best_model_path'])
            ema.restore(model)
            patience_left = int(profile['patience'])
            degrade_epochs = 0
        else:
            patience_left -= 1
            if np.isfinite(best_val) and best_epoch > 0 and epoch > best_epoch:
                degrade_margin = max(degrade_stop_abs, abs(best_val) * degrade_stop_rel)
                if val_rmse > best_val + degrade_margin:
                    degrade_epochs += 1
                else:
                    degrade_epochs = 0
            else:
                degrade_epochs = 0

        if (epoch - best_epoch) >= int(profile['freeze_view_gate_after']) and (not freeze_done):
            for p in model.view_gate.parameters():
                p.requires_grad = False
            for g in opt.param_groups:
                if g.get('name', '') == 'view_gate':
                    g['lr'] = 0.0
                    g['weight_decay'] = 0.0
            freeze_done = True
            print(f'[Freeze] view_gate frozen at epoch {epoch}.')

        can_stop = epoch >= min_epochs_before_stop
        if can_stop and degrade_stop_patience > 0 and degrade_epochs >= degrade_stop_patience:
            print('Early stop.')
            break

        if can_stop and patience_left <= 0:
            print('Early stop.')
            break

    if best_state is not None:
        model.load_state_dict(best_state, strict=False)
    model.eval()
    val_ema_std, val_ema_pred_std, val_ema_dj = collect_predictions(model, val_loader, device, F_path, F_expr, F_mut, F_meth, S_path, S_expr, S_mut, S_meth, X0, A_norm)
    val_ema_metrics = inverse_metric_dict(val_ema_std, val_ema_pred_std, val_ema_dj, drug_ids, scaler)
    test_ema_std, test_ema_pred_std, test_ema_dj = collect_predictions(model, test_loader, device, F_path, F_expr, F_mut, F_meth, S_path, S_expr, S_mut, S_meth, X0, A_norm)
    test_ema_metrics = inverse_metric_dict(test_ema_std, test_ema_pred_std, test_ema_dj, drug_ids, scaler)

    candidates = {
        'best_ema': {
            'val': val_ema_metrics,
            'test': test_ema_metrics,
        }
    }

    if swa_n > 0:
        tmp_state = {k: v.clone() for k, v in model.state_dict().items()}
        model.load_state_dict(swa_model.module.state_dict(), strict=False)
        val_swa_std, val_swa_pred_std, val_swa_dj = collect_predictions(model, val_loader, device, F_path, F_expr, F_mut, F_meth, S_path, S_expr, S_mut, S_meth, X0, A_norm)
        val_swa_metrics = inverse_metric_dict(val_swa_std, val_swa_pred_std, val_swa_dj, drug_ids, scaler)
        test_swa_std, test_swa_pred_std, test_swa_dj = collect_predictions(model, test_loader, device, F_path, F_expr, F_mut, F_meth, S_path, S_expr, S_mut, S_meth, X0, A_norm)
        test_swa_metrics = inverse_metric_dict(test_swa_std, test_swa_pred_std, test_swa_dj, drug_ids, scaler)
        candidates['swa'] = {'val': val_swa_metrics, 'test': test_swa_metrics}

        if profile.get('use_ema_swa_blend', False):
            val_blend_pred_std = 0.5 * (val_ema_pred_std + val_swa_pred_std)
            test_blend_pred_std = 0.5 * (test_ema_pred_std + test_swa_pred_std)
            val_blend_metrics = inverse_metric_dict(val_ema_std, val_blend_pred_std, val_ema_dj, drug_ids, scaler)
            test_blend_metrics = inverse_metric_dict(test_ema_std, test_blend_pred_std, test_ema_dj, drug_ids, scaler)
            candidates['ema_swa_blend'] = {'val': val_blend_metrics, 'test': test_blend_metrics}
        model.load_state_dict(tmp_state, strict=False)

    best_source = min(candidates.items(), key=lambda kv: kv[1]['val']['RMSE'])[0]
    final = candidates[best_source]
    final_metrics = {
        'MSE': float(final['test']['MSE']),
        'RMSE': float(final['test']['RMSE']),
        'MAE': float(final['test']['MAE']),
        'R2': float(final['test']['R2']),
        'PCC': float(final['test']['PCC']),
        'SCC': float(final['test']['SCC']),
        'epoch': int(best_epoch),
        'selection_source': best_source,
    }
    _write_full_csv(ctx['test_metrics_path'], final_metrics)

    summary_json = {
        'split_mode': split_mode,
        'experiment_name': ctx['experiment_name'],
        'profile': profile,
        'best_epoch': int(best_epoch),
        'selection_source': best_source,
        'candidates': {
            name: {
                'val_RMSE': float(pack['val']['RMSE']),
                'test_RMSE': float(pack['test']['RMSE']),
                'test_MAE': float(pack['test']['MAE']),
                'test_R2': float(pack['test']['R2']),
                'test_PCC': float(pack['test']['PCC']),
                'test_SCC': float(pack['test']['SCC']),
            }
            for name, pack in candidates.items()
        }
    }
    with open(ctx['final_eval_json_path'], 'w', encoding='utf-8') as f:
        json.dump(summary_json, f, ensure_ascii=False, indent=2)

    print(f'[Final] split={split_mode} | source={best_source} | Test RMSE={final_metrics["RMSE"]:.4f} | MAE={final_metrics["MAE"]:.4f}')
    return final_metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split_mode', type=str, default='pair_level', choices=['pair_level', 'drug_coldstart', 'cell_coldstart'])
    parser.add_argument('--experiment_name', type=str, default=None)
    args = parser.parse_args()
    run(split_mode=args.split_mode, experiment_name=args.experiment_name)

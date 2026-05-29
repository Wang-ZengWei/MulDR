
from typing import Dict, List, Optional, Tuple
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from drug_features.motif_vocab import MotifVocab
from drug_features.drug_encoder import DrugEncoder
from mfh import MFHFusionHead
from config import (
    FG_SMARTS_PATH,
    GNN3D_ATOM_FEAT_DIM, GNN3D_HIDDEN_DIM, GNN3D_LAYERS, GNN3D_NUM_RBF, GNN3D_RBF_MAX,
)


class DenseGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, heads=8, attn_drop=0.20, feat_drop=0.20, negative_slope=0.2):
        super().__init__()
        self.h = heads
        self.fc = nn.Linear(in_dim, out_dim * heads, bias=False)
        self.attn_l = nn.Parameter(torch.Tensor(heads, out_dim))
        self.attn_r = nn.Parameter(torch.Tensor(heads, out_dim))
        self.leaky = nn.LeakyReLU(negative_slope)
        self.adrop = nn.Dropout(attn_drop)
        self.fdrop = nn.Dropout(feat_drop)
        self.ln = nn.LayerNorm(out_dim * heads)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.xavier_uniform_(self.attn_l)
        nn.init.xavier_uniform_(self.attn_r)

    def forward(self, X, S):
        Nc = X.size(0)
        H = self.fc(self.fdrop(X)).view(Nc, self.h, -1)
        el = (H * self.attn_l).sum(-1)
        er = (H * self.attn_r).sum(-1)
        e = self.leaky(el.unsqueeze(1) + er.unsqueeze(0))
        mask = (S > 0).unsqueeze(-1)
        e = e.masked_fill(~mask, float('-inf'))
        a = torch.softmax(e, dim=1)
        a = self.adrop(a) * S.unsqueeze(-1)
        out = torch.einsum('ijh,jhf->ihf', a, H)
        out = out.reshape(Nc, -1)
        return self.ln(out + X if out.size(1) == X.size(1) else out)


class HyperEncoderView(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, heads=8, appnp_T=5, appnp_alpha=0.1, drop=0.3):
        super().__init__()
        self.gat1 = DenseGATLayer(in_dim, hid_dim // heads, heads=heads, attn_drop=0.1, feat_drop=0.1)
        self.gat2 = DenseGATLayer(hid_dim, out_dim // heads, heads=heads, attn_drop=0.1, feat_drop=0.1)
        self.appnp_T = appnp_T
        self.appnp_alpha = appnp_alpha
        self.lin_skip = nn.Linear(in_dim, out_dim)
        self.ln = nn.LayerNorm(out_dim)
        self.drop = nn.Dropout(drop)

    def appnp(self, Z0, S):
        Z = Z0
        a = self.appnp_alpha
        for _ in range(self.appnp_T):
            Z = (1 - a) * (S @ Z) + a * Z0
        return Z

    def forward(self, X, S):
        H1 = self.gat1(X, S)
        H2 = self.gat2(H1, S)
        Z0 = self.drop(H2)
        Z = self.appnp(Z0, S)
        return self.ln(Z + self.lin_skip(X))


class MulDR(nn.Module):
    def __init__(
        self,
        n_cells: int,
        n_drugs: int,
        feat_dims: Dict[str, int],
        profile: Dict,
        motif_vocab_path: Optional[str] = None,
        fg_smarts_path: Optional[str] = None,
    ):
        super().__init__()
        self.profile = dict(profile)
        self.n_cells = n_cells
        self.n_drugs = n_drugs
        self.C = int(profile['cell_dim'])
        self.D = int(profile['drug_dim'])
        self.H = int(profile['hidden_dim'])
        heads_gat = int(profile['heads_gat'])
        appnp_T = int(profile['appnp_T'])
        appnp_alpha = float(profile['appnp_alpha'])
        hyper_drop = float(profile['hyper_drop'])

        self.enc_path = HyperEncoderView(feat_dims['path'], self.C, self.C, heads=heads_gat, appnp_T=appnp_T, appnp_alpha=appnp_alpha, drop=hyper_drop)
        self.enc_expr = HyperEncoderView(feat_dims['expr'], self.C, self.C, heads=heads_gat, appnp_T=appnp_T, appnp_alpha=appnp_alpha, drop=hyper_drop)
        self.enc_mut = HyperEncoderView(feat_dims['mut'], self.C, self.C, heads=heads_gat, appnp_T=appnp_T, appnp_alpha=appnp_alpha, drop=hyper_drop)
        self.enc_meth = HyperEncoderView(feat_dims['meth'], self.C, self.C, heads=heads_gat, appnp_T=appnp_T, appnp_alpha=appnp_alpha, drop=hyper_drop)

        self.use_bipartite = bool(profile['use_bipartite'])
        self.v_branch_scale = float(profile['v_branch_scale'])
        self.inference_backbone_blend = float(profile['inference_backbone_blend'])

        self.gcn1 = nn.Linear(4, self.C)
        self.gcn2 = nn.Linear(self.C, self.C)
        self.ln_g = nn.LayerNorm(self.C)

        vocab_file = str(motif_vocab_path)
        fg_file = str(fg_smarts_path or FG_SMARTS_PATH)
        if not vocab_file or not os.path.exists(vocab_file):
            raise FileNotFoundError(f'Motif vocabulary file not found: {vocab_file}')
        if not os.path.exists(fg_file):
            raise FileNotFoundError(f'Functional group SMARTS file not found: {fg_file}')

        self.motif_vocab = MotifVocab(vocab_file)
        with open(fg_file, 'r', encoding='utf-8') as f:
            self.fg_smarts = [line.strip() for line in f if line.strip()]
        self.drug_encoder = DrugEncoder(
            motif_vocab=self.motif_vocab,
            fg_smarts=self.fg_smarts,
            profile=self.profile,
            atom_feat_dim=GNN3D_ATOM_FEAT_DIM,
            gnn_hidden_dim=GNN3D_HIDDEN_DIM,
            num_layers=GNN3D_LAYERS,
            num_rbf=GNN3D_NUM_RBF,
            rbf_max=GNN3D_RBF_MAX,
            drug_dim=self.D,
        )

        self.view_gate = nn.Sequential(
            nn.Linear(self.D, max(128, self.D // 2)),
            nn.ReLU(),
            nn.LayerNorm(max(128, self.D // 2)),
            nn.Linear(max(128, self.D // 2), 4),
        )
        self.gate_u = nn.Linear(self.C, self.C)
        self.gate_v = nn.Linear(self.C, self.C)
        self.mfh = MFHFusionHead(
            cell_dim=self.C,
            drug_dim=self.D,
            k=int(profile['mfh_k']),
            r=int(profile['mfh_r']),
            h=int(profile['mfh_h']),
            dropout=float(profile['mfh_dropout']),
            out_hidden=self.H,
        )
        self.residual_head = nn.Sequential(
            nn.Linear(self.C + self.D, self.H),
            nn.GELU(),
            nn.Dropout(float(profile['mfh_dropout'])),
            nn.Linear(self.H, 1),
        )
        self.lap_lambda = 1e-4
        self.cons_lambda = 5e-4

    def forward_cell_V(self, X0, A_norm):
        H = F.relu(self.gcn1(X0))
        H = A_norm @ H
        H = F.relu(self.gcn2(H))
        H = A_norm @ H
        return self.ln_g(H)[:self.n_cells]

    def _predict_from_pair(self, Z_cell: torch.Tensor, D: torch.Tensor) -> torch.Tensor:
        return self.mfh(Z_cell, D) + 0.10 * self.residual_head(torch.cat([Z_cell, D], dim=-1)).squeeze(-1)

    def forward(
        self,
        ci: torch.Tensor,
        drug_smiles: List[str],
        drug_graph_batch: Dict[str, torch.Tensor],
        F_path: torch.Tensor,
        F_expr: torch.Tensor,
        F_mut: torch.Tensor,
        F_meth: torch.Tensor,
        S_path: torch.Tensor,
        S_expr: torch.Tensor,
        S_mut: torch.Tensor,
        S_meth: torch.Tensor,
        X0: torch.Tensor,
        A_norm: torch.Tensor,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        U_path = self.enc_path(F_path, S_path)
        U_expr = self.enc_expr(F_expr, S_expr)
        U_mut = self.enc_mut(F_mut, S_mut)
        U_meth = self.enc_meth(F_meth, S_meth)

        V_all = self.forward_cell_V(X0, A_norm) if self.use_bipartite else None

        D, D_backbone, drug_aux = self.drug_encoder(smiles=drug_smiles, graph_batch=drug_graph_batch)

        logits = self.view_gate(D) / max(float(self.profile['view_tau']), 1e-6)
        w = torch.softmax(logits, dim=-1)
        w = torch.clamp(w, min=float(self.profile['min_view_weight']))
        if self.training:
            mask = (torch.rand_like(w) > float(self.profile['view_dropout'])).float()
            if mask.sum(dim=-1).min().item() <= 0:
                mask[:, 0] = 1.0
            w = w * mask
            w = w / (w.sum(dim=-1, keepdim=True) + 1e-6)
        else:
            w = w / (w.sum(dim=-1, keepdim=True) + 1e-6)

        U_stack = torch.stack([U_path[ci], U_expr[ci], U_mut[ci], U_meth[ci]], dim=1)
        U_fused = (w.unsqueeze(-1) * U_stack).sum(dim=1)

        if self.use_bipartite:
            V_cell = V_all[ci]
            alpha = torch.sigmoid(self.gate_u(U_fused) + self.gate_v(V_cell))
            Z_cell = U_fused + self.v_branch_scale * (1.0 - alpha) * (V_cell - U_fused)
        else:
            V_cell = torch.zeros_like(U_fused)
            alpha = torch.ones_like(U_fused)
            Z_cell = U_fused

        pred_main = self._predict_from_pair(Z_cell, D)
        pred_backbone = self._predict_from_pair(Z_cell, D_backbone)
        pred = (1.0 - self.inference_backbone_blend) * pred_main + self.inference_backbone_blend * pred_backbone

        U_norm = [F.normalize(U[ci], dim=-1) for U in [U_path, U_expr, U_mut, U_meth]]
        cons = 0.0
        for i in range(4):
            for j in range(i + 1, 4):
                cons = cons + F.mse_loss(U_norm[i], U_norm[j])
        cons = (cons / 6.0) * self.cons_lambda

        with torch.no_grad():
            L = torch.eye(S_path.size(0), device=S_path.device, dtype=S_path.dtype) - S_path
        z_full = torch.zeros_like(U_path)
        z_full[ci] = Z_cell
        lap = self.lap_lambda * torch.trace(z_full.T @ L @ z_full)

        w_entropy = -(w * (w.clamp_min(1e-9).log())).sum(dim=-1).mean()
        backbone_cons = 1.0 - F.cosine_similarity(drug_aux['drug_vec'], drug_aux['backbone_vec'], dim=-1).mean()

        aux = {
            'cons': cons,
            'lap': lap,
            'view_entropy_reg': -w_entropy,
            'backbone_consistency': backbone_cons,
            'drug_prop': drug_aux,
            'view_weights': w,
            'alpha': alpha,
        }
        outputs = {
            'pred': pred,
            'pred_main': pred_main,
            'pred_backbone': pred_backbone,
            'drug_vec': drug_aux['drug_vec'],
            'drug_vec_backbone': drug_aux['backbone_vec'],
        }
        return outputs, aux

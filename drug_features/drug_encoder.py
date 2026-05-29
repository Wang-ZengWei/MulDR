
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors


from .motif_vocab import MotifVocab
from .motif_extractor import tfidf_vector


class RBF(nn.Module):
    def __init__(self, num_rbf: int = 32, rbf_max: float = 8.0):
        super().__init__()
        centers = torch.linspace(0.0, float(rbf_max), int(num_rbf))
        self.register_buffer('centers', centers)
        self.gamma = 1.0 / ((centers[1] - centers[0] + 1e-6) ** 2)

    def forward(self, d: torch.Tensor) -> torch.Tensor:
        return torch.exp(-self.gamma * (d - self.centers) ** 2)


class AttentionPool(nn.Module):
    def __init__(self, dim: int, hidden: int = 256):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor, batch_idx: torch.Tensor) -> torch.Tensor:
        B = int(batch_idx.max().item()) + 1 if batch_idx.numel() > 0 else 0
        if B == 0:
            return x.new_zeros(0, x.size(-1))
        raw = self.score(x).squeeze(-1)
        w = F.relu(raw) + 1e-6
        denom = torch.zeros(B, device=x.device, dtype=x.dtype)
        denom.index_add_(0, batch_idx, w)
        w_norm = w / (denom[batch_idx] + 1e-6)
        out = torch.zeros(B, x.size(1), device=x.device, dtype=x.dtype)
        out.index_add_(0, batch_idx, w_norm.unsqueeze(-1) * x)
        return out


class DrugEncoder(nn.Module):
    def __init__(
        self,
        motif_vocab: MotifVocab,
        fg_smarts: List[str],
        profile: Dict,
        atom_feat_dim: int = 128,
        gnn_hidden_dim: int = 128,
        num_layers: int = 4,
        num_rbf: int = 32,
        rbf_max: float = 10.0,
        drug_dim: int = 512,
    ):
        super().__init__()
        self.vocab = motif_vocab
        self.fg_smarts = fg_smarts
        self.profile = dict(profile)
        self.drug_dim = int(drug_dim)

        self.rbf = RBF(num_rbf=num_rbf, rbf_max=rbf_max)
        self.node_in = nn.Linear(atom_feat_dim, gnn_hidden_dim)
        self.edge_mlp = nn.Sequential(
            nn.Linear(num_rbf, gnn_hidden_dim),
            nn.ReLU(),
            nn.Linear(gnn_hidden_dim, gnn_hidden_dim),
        )
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(gnn_hidden_dim, gnn_hidden_dim),
                nn.ReLU(),
                nn.Linear(gnn_hidden_dim, gnn_hidden_dim),
            )
            for _ in range(num_layers)
        ])
        self.readout = AttentionPool(gnn_hidden_dim)
        self.gnn_proj = nn.Sequential(
            nn.Linear(gnn_hidden_dim, drug_dim),
            nn.GELU(),
            nn.LayerNorm(drug_dim),
        )

        vocab_size = len(getattr(self.vocab, 'id2token', []))
        self.vocab_size = max(1, int(vocab_size))
        self.motif_proj = nn.Sequential(
            nn.Linear(self.vocab_size, 1024),
            nn.ReLU(),
            nn.LayerNorm(1024),
            nn.Linear(1024, drug_dim),
        )

        self.morgan_proj = nn.Sequential(
            nn.Linear(2048, drug_dim),
            nn.GELU(),
            nn.LayerNorm(drug_dim),
        )

        self.backbone_fuse = nn.Sequential(
            nn.Linear(drug_dim * 2, drug_dim),
            nn.GELU(),
            nn.LayerNorm(drug_dim),
        )
        self.final_fuse = nn.Sequential(
            nn.Linear(drug_dim * 2, drug_dim),
            nn.GELU(),
            nn.LayerNorm(drug_dim),
        )

        self.prop_head = nn.Linear(drug_dim, 6)
        self.fp_recon_head = nn.Linear(drug_dim, 2048)
        self.motif_recon_head = nn.Linear(drug_dim, self.vocab_size)

    @torch.no_grad()
    def _mol_props(self, smi: str) -> torch.Tensor:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return torch.zeros(6)
        return torch.tensor([
            Descriptors.MolWt(m),
            Descriptors.MolLogP(m),
            Descriptors.TPSA(m),
            Descriptors.NumHDonors(m),
            Descriptors.NumHAcceptors(m),
            Descriptors.RingCount(m),
        ], dtype=torch.float32)

    def _motif_vector(self, smi: str):
        if hasattr(self.vocab, 'encode_counts'):
            vec = self.vocab.encode_counts(smi)
        else:
            vec = tfidf_vector(smi, self.vocab, self.fg_smarts)
        arr = np.asarray(vec, dtype=np.float32)
        if arr.ndim == 0:
            arr = np.zeros((self.vocab_size,), dtype=np.float32)
        if arr.shape[0] < self.vocab_size:
            pad = np.zeros((self.vocab_size - arr.shape[0],), dtype=np.float32)
            arr = np.concatenate([arr, pad], axis=0)
        elif arr.shape[0] > self.vocab_size:
            arr = arr[:self.vocab_size]
        return torch.tensor(arr, dtype=torch.float32)

    def _morgan_bits(self, smi: str) -> np.ndarray:
        m = Chem.MolFromSmiles(smi)
        arr = np.zeros((2048,), dtype=np.float32)
        if m is None:
            return arr
        bv = AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=2048)
        DataStructs.ConvertToNumpyArray(bv, arr)
        return arr

    def forward(self, smiles: List[str], graph_batch: Dict[str, torch.Tensor]):
        device = graph_batch['node_feat'].device
        x = graph_batch['node_feat']
        pos = graph_batch['pos']
        ei = graph_batch['edge_index']
        batch_idx = graph_batch['batch_idx']

        h = self.node_in(x)
        src, dst = ei[0], ei[1]
        dist = (pos[src] - pos[dst]).pow(2).sum(-1).sqrt().unsqueeze(-1)
        e = self.edge_mlp(self.rbf(dist))
        for layer in self.layers:
            m = torch.zeros_like(h)
            m.index_add_(0, dst, e * h[src])
            h = h + layer(m)
        gnn_vec = self.readout(h, batch_idx)
        gnn_emb = self.gnn_proj(gnn_vec)

        if len(smiles) == 0:
            motif_vec = torch.zeros(0, self.vocab_size, device=device)
            morgan_bits = torch.zeros(0, 2048, device=device)
        else:
            motif_vec = torch.stack([self._motif_vector(s) for s in smiles], 0).to(device)
            morgan_bits = torch.from_numpy(np.stack([self._morgan_bits(s) for s in smiles], axis=0)).to(device)

        motif_emb = self.motif_proj(motif_vec)
        morgan_emb = self.morgan_proj(morgan_bits)

        backbone_vec = self.backbone_fuse(torch.cat([motif_emb, gnn_emb], dim=-1))
        drug_vec = self.final_fuse(torch.cat([backbone_vec, morgan_emb], dim=-1))

        props = torch.stack([self._mol_props(s) for s in smiles], 0).to(device) if len(smiles) > 0 else torch.zeros(0, 6, device=device)
        aux = {
            'prop_pred': self.prop_head(backbone_vec),
            'prop_true': props,
            'fp_recon_logits': self.fp_recon_head(drug_vec),
            'fp_target': morgan_bits,
            'motif_recon': self.motif_recon_head(drug_vec),
            'motif_target': motif_vec,
            'drug_vec': drug_vec,
            'backbone_vec': backbone_vec,
            'motif_emb': motif_emb,
            'morgan_emb': morgan_emb,
            'gnn_emb': gnn_emb,
        }
        return drug_vec, backbone_vec, aux

import torch
import torch.nn as nn
import torch.nn.functional as F

class MFHFusionHead(nn.Module):
    def __init__(
        self,
        cell_dim: int,
        drug_dim: int,
        k: int = 1000,
        r: int = 5,
        h: int = 2,
        proj_hidden: int = None,
        dropout: float = 0.1,
        out_hidden: int = 512
    ):
        super().__init__()
        self.k = k
        self.r = r
        self.h = h

        def proj(in_dim):
            if proj_hidden is None:
                return nn.Linear(in_dim, k * r)
            else:
                return nn.Sequential(
                    nn.Linear(in_dim, proj_hidden),
                    nn.ReLU(),
                    nn.LayerNorm(proj_hidden),
                    nn.Linear(proj_hidden, k * r)
                )

        self.z_linears = nn.ModuleList([proj(cell_dim) for _ in range(h)])
        self.d_linears = nn.ModuleList([proj(drug_dim) for _ in range(h)])

        self.drop = nn.Dropout(dropout)

        self.out = nn.Sequential(
            nn.Linear(k * h, out_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_hidden, out_hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_hidden // 2, 1)
        )

        self.ln_z = nn.LayerNorm(cell_dim)
        self.ln_d = nn.LayerNorm(drug_dim)

    @staticmethod
    def _signed_sqrt_l2norm(x: torch.Tensor, eps: float = 1e-8):
        x = torch.sign(x) * torch.sqrt(torch.clamp(x.abs(), min=eps))
        x = F.normalize(x, p=2, dim=-1)
        return x

    def forward(self, Z: torch.Tensor, D: torch.Tensor) -> torch.Tensor:
        B = Z.size(0)
        Zn = self.ln_z(Z)
        Dn = self.ln_d(D)

        outs = []
        z_in = Zn
        d_in = Dn
        for i in range(self.h):
            z_proj = self.z_linears[i](z_in)
            d_proj = self.d_linears[i](d_in)

            z_proj = z_proj.view(B, self.k, self.r)
            d_proj = d_proj.view(B, self.k, self.r)

            zdh = z_proj * d_proj

            pooled = zdh.sum(dim=-1)
            pooled = self._signed_sqrt_l2norm(pooled)

            outs.append(pooled)

            z_in = Zn
            d_in = Dn

        fused = torch.cat(outs, dim=-1)
        fused = self.drop(fused)
        y = self.out(fused).squeeze(-1)
        return y

"""PatchCore-style memory bank on the GPU (torch): greedy k-center coreset +
chunked nearest-neighbour distances. Falls back to CPU (torch / faiss) when CUDA is
unavailable."""
from __future__ import annotations

import numpy as np


def _device(prefer: str = "cuda"):
    import torch

    return torch.device("cuda" if (prefer == "cuda" and torch.cuda.is_available()) else "cpu")


def greedy_coreset(X, m: int, proj_dim: int = 128, seed: int = 0):
    """Indices of an approximate k-center coreset (PatchCore, random projection)."""
    import torch

    n, d = X.shape
    if m >= n:
        return torch.arange(n, device=X.device)
    g = torch.Generator(device="cpu").manual_seed(seed)
    if d > proj_dim:
        P = torch.randn(d, proj_dim, generator=g).to(X.device) / np.sqrt(proj_dim)
        Z = X @ P
    else:
        Z = X
    sel = torch.empty(m, dtype=torch.long, device=X.device)
    start = int(torch.randint(0, n, (1,), generator=g))
    sel[0] = start
    mind = torch.linalg.vector_norm(Z - Z[start], dim=1)
    for i in range(1, m):
        j = torch.argmax(mind)
        sel[i] = j
        mind = torch.minimum(mind, torch.linalg.vector_norm(Z - Z[j], dim=1))
    return sel


class MemoryBank:
    """``fit``: random pre-subsampling to ``presample`` features, then greedy k-center
    coreset of ``min(coreset_max, coreset_ratio * N)``. The pre-subsampling keeps the
    sequential greedy loop cheap (each iteration touches ``presample`` rows) and the GPU
    footprint small (the Isaac worker shares the 12 GB card); the coreset still spans
    the normal feature distribution (k-center on a 200k uniform subsample)."""

    def __init__(self, coreset_ratio: float = 0.03, coreset_max: int = 10000, device: str = "cuda", seed: int = 0,
                 presample: int = 200_000):
        self.ratio = coreset_ratio
        self.cmax = coreset_max
        self.device = _device(device)
        self.seed = seed
        self.presample = presample
        self.bank = None

    def fit(self, feats: list[np.ndarray] | np.ndarray) -> "MemoryBank":
        import torch

        X = np.concatenate(feats) if isinstance(feats, list) else feats
        n = len(X)
        m = int(min(self.cmax, max(1, round(self.ratio * n))))
        if n > self.presample:
            rng = np.random.default_rng(self.seed)
            X = X[rng.choice(n, self.presample, replace=False)]
        Xt = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32)).to(self.device)
        idx = greedy_coreset(Xt, m, seed=self.seed)
        self.bank = Xt[idx].contiguous()
        self.bank_sq = (self.bank * self.bank).sum(dim=1)
        del Xt
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return self

    def score(self, feats: np.ndarray, chunk: int = 8192) -> np.ndarray:
        import torch

        out = np.empty(len(feats), np.float32)
        B, Bsq = self.bank, self.bank_sq
        for s in range(0, len(feats), chunk):
            q = torch.from_numpy(np.ascontiguousarray(feats[s:s + chunk], dtype=np.float32)).to(self.device)
            d2 = (q * q).sum(dim=1, keepdim=True) + Bsq[None, :] - 2.0 * (q @ B.T)
            out[s:s + chunk] = torch.sqrt(torch.clamp(d2.min(dim=1).values, min=0)).cpu().numpy()
        return out

    @property
    def size(self) -> int:
        return 0 if self.bank is None else int(self.bank.shape[0])

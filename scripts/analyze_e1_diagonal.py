"""Exploratory re-read of the cross-application experiment (E1): is a part's own optimum the best on it?

M[i][j] = report FNR on target part j with the optimum found for part i (delta[i][j] = M[i][j] - M[j][j]).
"""
import itertools
import json
import sys

import numpy as np

S = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "runs/exp/e1/summary.json"))
P = ["tless_13", "tless_18", "tless_29", "tless_30", "mulsen_nut", "mulsen_flat_pad"]
M = np.array(S["fnr_matrix_report"], dtype=float)
D = np.array(S["delta_matrix"], dtype=float)
assert abs(M[0, 1] - M[1, 1] - D[0, 1]) < 1e-9          # orientation check
N_DEF = 36


def column_view(idx):
    out = []
    for j in idx:
        col = {P[i]: M[i, j] for i in idx}
        self_v = M[j, j]
        better = [P[i] for i in idx if i != j and M[i, j] < self_v - 1e-9]
        ties = [P[i] for i in idx if i != j and abs(M[i, j] - self_v) < 1e-9]
        rank = 1 + len(better) + 0.5 * len(ties)
        adv = np.mean([M[i, j] for i in idx if i != j]) - self_v
        out.append(dict(part=P[j], self=self_v, rank=rank, n=len(idx), better={b: round((self_v - M[P.index(b), j]) * N_DEF) for b in better},
                        ties=ties, mean_adv_pp=100 * adv))
    return out


def perm_test(idx, n_iter=200000, seed=0):
    """H0: which config is 'own' in each column is arbitrary. Statistic: mean over columns of
    (mean of the other configs - own). Exact enumeration when small, else Monte Carlo."""
    idx = list(idx)
    sub = M[np.ix_(idx, idx)]
    k = len(idx)

    def stat(choice):
        return np.mean([(sub[:, j].sum() - sub[c, j]) / (k - 1) - sub[c, j] for j, c in enumerate(choice)])

    obs = stat(range(k))
    if k ** k <= 2_000_000:
        vals = np.array([stat(c) for c in itertools.product(range(k), repeat=k)])
    else:
        rng = np.random.default_rng(seed)
        vals = np.array([stat(rng.integers(0, k, k)) for _ in range(n_iter)])
    return obs, float((vals >= obs - 1e-12).mean())


def first_place_p(idx):
    """P(own config is strictly or jointly first in >= observed columns) under 1/k per column."""
    from math import comb
    k = len(idx)
    firsts = sum(1 for j in idx if all(M[j, j] <= M[i, j] + 1e-9 for i in idx))
    p = sum(comb(k, m) * (1 / k) ** m * (1 - 1 / k) ** (k - m) for m in range(firsts, k + 1))
    return firsts, p


for name, idx in [("all 6", range(6)), ("flat pad excluded (5x5)", range(5))]:
    idx = list(idx)
    print(f"== {name}")
    for r in column_view(idx):
        print(f"  {r['part']:16s} own FNR {r['self']:.3f}  rank {r['rank']}/{r['n']}  own better than mean of others by {r['mean_adv_pp']:+.1f}pp"
              f"  beaten by {r['better']} (defects)  ties {r['ties']}")
    obs, p = perm_test(idx)
    f, pf = first_place_p(idx)
    print(f"  mean own advantage {100 * obs:+.1f}pp, permutation p = {p:.4f};  own config first in {f}/{len(idx)} columns, binomial p = {pf:.3f}")
print("universal (one config on all six) mean FNR:", {k: round(v, 3) for k, v in S["universal_config_mean_fnr"].items()})
print("mean of own optima:", round(float(np.mean(np.diag(M))), 3))

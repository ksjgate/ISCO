"""Coverage anomaly ("material that is always seen on normal parts is missing").

For template surface samples t (nominal part, part frame): ``observed_i(t)`` = a merged
point of instance i lies within ``radius``. ``p_obs(t)`` = mean over bank instances.
Missing score ``m(t) = p_obs(t) * (1 - observed(t))``, smoothed over ``smooth_radius``;
image score = max over t. Catches holes / cuts / chips whose removed surface would
otherwise just be "not there" for a feature-only detector.
"""
from __future__ import annotations

import numpy as np


class CoverageScorer:
    def __init__(self, template_pts: np.ndarray, radius: float = 0.6e-3, smooth_radius: float = 1.0e-3):
        from scipy.spatial import cKDTree

        self.t = np.asarray(template_pts, dtype=np.float64)
        self.radius = radius
        tree = cKDTree(self.t)
        nb = tree.query_ball_point(self.t, r=smooth_radius)
        self.nb_idx = np.concatenate([np.asarray(x, np.int64) for x in nb])
        self.nb_ptr = np.cumsum([0] + [len(x) for x in nb])
        self.p_obs = None

    def observed(self, xyz: np.ndarray) -> np.ndarray:
        from scipy.spatial import cKDTree

        if len(xyz) == 0:
            return np.zeros(len(self.t), bool)
        d, _ = cKDTree(xyz).query(self.t, k=1, distance_upper_bound=self.radius)
        return np.isfinite(d)

    def fit(self, observed_list: list[np.ndarray]) -> "CoverageScorer":
        self.p_obs = np.mean(np.stack(observed_list).astype(np.float64), axis=0)
        return self

    def point_scores(self, observed: np.ndarray) -> np.ndarray:
        m = self.p_obs * (1.0 - observed.astype(np.float64))
        sums = np.add.reduceat(m[self.nb_idx], self.nb_ptr[:-1])
        cnt = np.diff(self.nb_ptr)
        return sums / np.maximum(cnt, 1)

    def image_score(self, observed: np.ndarray) -> float:
        return float(self.point_scores(observed).max())

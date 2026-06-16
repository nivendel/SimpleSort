"""Online clustering for L2-normalised features with adaptive gating.

Input:  x (D,)  L2-normalised feature
Output: cluster_id (int)
State:  match_centers (K, D)  L2-normalised active centres

Copyright (C) 2026 Nivendel
With assistance from Claude Code and deepseek-v4-pro[1m]
SPDX-License-Identifier: AGPL-3.0-or-later
"""

import numpy as np


class KEMA:
    """Online clustering with EMA updates and adaptive cosine-distance gating.

    Inspired by K-means, but uses EMA-based incremental centre updates
    (no iterative Lloyd's algorithm) and adaptive split decisions based on
    pooled within-cluster statistics — no hand-tuned absolute threshold.
    """

    def __init__(
        self,
        min_hits: int = 3,
        n_std: float = 1.0,
        alpha: float = 0.02,
        split_thresh: float | None = 0.008,
        stale_frames: int = 500,
        stale_max_hits: int = 10,
    ):
        self.min_hits = min_hits
        self.n_std = n_std
        self.alpha = alpha
        self.split_thresh = split_thresh
        self.stale_frames = stale_frames
        self.stale_max_hits = stale_max_hits

        self._centers: np.ndarray = np.empty((0, 0), dtype=np.float32)
        self._hits: np.ndarray = np.empty(0, dtype=np.int32)
        self._active: np.ndarray = np.empty(0, dtype=bool)
        self._recovering: np.ndarray = np.empty(0, dtype=bool)
        self._dist_sum: np.ndarray = np.empty(0, dtype=np.float64)
        self._dist_sum_sq: np.ndarray = np.empty(0, dtype=np.float64)
        self._frames_since_hit: np.ndarray = np.empty(0, dtype=np.int32)

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _normalize(v: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return v / norms

    # -- statistics ---------------------------------------------------------

    def _cluster_mean(self, k: int) -> float | None:
        """Mean cosine distance of samples merged into cluster *k*."""
        n = self._hits[k] - 1
        if n < 1:
            return None
        return float(self._dist_sum[k] / n)

    def _pooled_std(self) -> float | None:
        """Weighted average of per-cluster std (cosine distance)."""
        total_ss = 0.0
        total_df = 0
        for i in range(len(self._hits)):
            n = self._hits[i] - 1
            if n >= 2:
                mean_i = self._dist_sum[i] / n
                ss_i = self._dist_sum_sq[i] - self._dist_sum[i] * mean_i
                total_ss += max(0.0, ss_i)
                total_df += n - 1
        if total_df < 1:
            return None
        return float(np.sqrt(total_ss / total_df))

    # -- properties ---------------------------------------------------------

    @property
    def n_active(self) -> int:
        return int(self._active.sum())

    @property
    def is_ready(self) -> bool:
        return self._active.any()

    @property
    def match_centers(self) -> np.ndarray:
        if self._active.any():
            return self._normalize(self._centers[self._active])
        if self._hits.size == 0:
            return np.empty((0, self._centers.shape[1]), dtype=np.float32)
        best = int(np.argmax(self._hits))
        return self._normalize(self._centers[best : best + 1])

    # -- cluster creation ----------------------------------------------------

    def _create_cluster(self, x: np.ndarray, recovering: bool = False) -> int:
        """Add a new cluster and return its index."""
        if self._centers.size == 0:
            self._centers = x.reshape(1, -1).copy()
            self._hits = np.array([1], dtype=np.int32)
            self._active = np.array([False])
            self._recovering = np.array([recovering])
            self._dist_sum = np.array([0.0], dtype=np.float64)
            self._dist_sum_sq = np.array([0.0], dtype=np.float64)
            self._frames_since_hit = np.array([0], dtype=np.int32)
            return 0

        self._centers = np.vstack([self._centers, x.reshape(1, -1)])
        self._hits = np.append(self._hits, 1)
        self._active = np.append(self._active, False)
        self._recovering = np.append(self._recovering, recovering)
        self._dist_sum = np.append(self._dist_sum, 0.0)
        self._dist_sum_sq = np.append(self._dist_sum_sq, 0.0)
        self._frames_since_hit = np.append(self._frames_since_hit, 0)
        return self._centers.shape[0] - 1

    # -- main API -----------------------------------------------------------

    def partial_fit(self, feature: np.ndarray, recovering: bool = False) -> int:
        feature = np.asarray(feature, dtype=np.float32).copy()

        if self._centers.size == 0:
            return self._create_cluster(feature, recovering)

        # increment idle counter for all centres
        self._frames_since_hit += 1

        centers = self._normalize(self._centers)
        similarities = feature @ centers.T
        best = int(np.argmax(similarities))
        best_dist = float(1.0 - similarities[best])

        if self.split_thresh is not None:
            should_split = best_dist > self.split_thresh
        else:
            mean_k = self._cluster_mean(best)
            pooled_std = self._pooled_std()
            if mean_k is not None and pooled_std is not None:
                should_split = best_dist > mean_k + self.n_std * pooled_std
            else:
                should_split = False

        if should_split:
            hit = self._create_cluster(feature, recovering)
        else:
            self._centers[best] = (
                (1.0 - self.alpha) * self._centers[best] + self.alpha * feature
            )
            self._hits[best] += 1
            self._dist_sum[best] += best_dist
            self._dist_sum_sq[best] += best_dist * best_dist
            self._frames_since_hit[best] = 0
            required = 2 * self.min_hits if self._recovering[best] else self.min_hits
            if self._hits[best] >= required:
                self._active[best] = True
            hit = best

        # decay inactive clusters
        mask = ~self._active
        mask[hit] = False
        self._hits[mask] -= 1

        # prune: inactive with hits <= 0, or stale active (idle too long + low total hits)
        stale = self._active & (self._frames_since_hit >= self.stale_frames) & (self._hits < self.stale_max_hits)
        keep = (self._hits > 0) & ~stale
        self._centers = self._centers[keep]
        self._hits = self._hits[keep]
        self._active = self._active[keep]
        self._recovering = self._recovering[keep]
        self._dist_sum = self._dist_sum[keep]
        self._dist_sum_sq = self._dist_sum_sq[keep]
        self._frames_since_hit = self._frames_since_hit[keep]

        return hit

    def __call__(self, feature: np.ndarray, recovering: bool = False) -> int:
        return self.partial_fit(feature, recovering)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Assign each row of *X* to the nearest cluster."""
        X = np.asarray(X, dtype=np.float32)
        if self._centers.shape[0] == 0:
            raise RuntimeError("No clusters fitted – call partial_fit first")
        similarities = X @ self._normalize(self._centers).T
        return np.argmax(similarities, axis=1)

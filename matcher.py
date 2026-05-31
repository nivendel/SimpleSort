"""Matching — IoU and appearance-based association.

Every matcher returns (matches, unmatched_tracks, unmatched_dets):
    matches           : list[tuple[int, int]]  — (track_idx, det_idx)
    unmatched_tracks  : list[int]
    unmatched_dets    : list[int]

Copyright (C) 2026 Nivendel, College of Civil Engineering, Tongji University
With assistance from Claude Code and deepseek-v4-pro[1m]
SPDX-License-Identifier: AGPL-3.0-or-later
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from .detector import Detection
from .track import Track


# -- distance helpers -------------------------------------------------------

def _cosine_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise cosine distance: 1 - cos_sim.  Inputs must be L2-normalised.

    a : (M, D)  gallery
    b : (N, D)  queries
    →  (M, N)
    """
    return 1.0 - np.dot(a, b.T)


def _nn_cosine_distance(gallery: np.ndarray, queries: np.ndarray) -> np.ndarray:
    """Min cosine distance from each query to any gallery sample.

    gallery : (K, D)  L2-normalised
    queries : (N, D)  L2-normalised
    →  (N,)
    """
    return _cosine_distance(gallery, queries).min(axis=0)


# -- cost helpers -----------------------------------------------------------

def iou_cost(tlwhs1: list, tlwhs2: list) -> np.ndarray:
    """1 - IoU cost matrix (0 = perfect overlap).

    tlwhs1 : list of [x, y, w, h]
    tlwhs2 : list of [x, y, w, h]
    →  (len(tlwhs1), len(tlwhs2))
    """
    n, m = len(tlwhs1), len(tlwhs2)
    cost = np.ones((n, m))
    for i, (x1, y1, w1, h1) in enumerate(tlwhs1):
        area1 = w1 * h1
        if area1 <= 0:
            continue
        for j, (x2, y2, w2, h2) in enumerate(tlwhs2):
            area2 = w2 * h2
            if area2 <= 0:
                continue
            xi1, yi1 = max(x1, x2), max(y1, y2)
            xi2, yi2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
            inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
            union = area1 + area2 - inter
            if union > 0:
                cost[i, j] = 1.0 - inter / union
    return cost


def linear_assignment(
    cost_matrix: np.ndarray, thresh: float = float("inf")
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Hungarian algorithm with cost threshold.

    cost_matrix : (N, M)
    thresh : only keep assignments with cost ≤ thresh.
    →  (matches, unmatched_rows, unmatched_cols)
    """
    n_rows, n_cols = cost_matrix.shape
    if cost_matrix.size == 0:
        return [], list(range(n_rows)), list(range(n_cols))

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    mask = cost_matrix[row_ind, col_ind] <= thresh
    matched_rows = row_ind[mask]
    matched_cols = col_ind[mask]
    matches = list(zip(matched_rows, matched_cols))

    unmatched_rows = [i for i in range(n_rows) if i not in matched_rows]
    unmatched_cols = [j for j in range(n_cols) if j not in matched_cols]

    return matches, unmatched_rows, unmatched_cols


INFTY_COST = 1e5


# -- Matcher ----------------------------------------------------------------

class Matcher:
    """Multi-stage association: strict IoU → loose IoU → appearance NN.
    IoU stages are gated by appearance (cosine distance > appearance_thresh → rejected).
    """

    def __init__(
        self,
        appearance_thresh: float = 0.4,
        strict_iou_thresh: float = 0.5,
        loose_iou_thresh: float = 0.3,
    ):
        self.appearance_thresh = appearance_thresh
        self.strict_iou_thresh = strict_iou_thresh
        self.loose_iou_thresh = loose_iou_thresh

    # ── appearance cost ─────────────────────────────────────────────────

    def _appearance_cost(
        self, tracks: list[Track], features: np.ndarray,
    ) -> np.ndarray:
        """NN cosine-distance cost matrix.

        tracks   : list[Track]
        features : (N, D)  L2-normalised detection features
        →  (len(tracks), N)   min cosine distance to each track's KEMA centres
        """
        cost = np.ones((len(tracks), len(features)))
        for i, t in enumerate(tracks):
            centers = t.kema.match_centers
            if centers.shape[0] > 0:
                cost[i, :] = _nn_cosine_distance(centers, features)
        return cost

    # ── IoU match ───────────────────────────────────────────────────────

    def iou_match(
        self,
        tracks: list[Track],
        dets: list[Detection],
        iou_thresh: float | None = None,
        gate_appearance: bool = False,
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        """IoU match using Kalman-predicted boxes.

        If *gate_appearance*, reject matches where the NN cosine distance
        from the track's KEMA gallery to the detection exceeds
        ``appearance_thresh`` (like StrongSort's Mahalanobis gating, but
        in reverse — appearance gates position).
        """
        if iou_thresh is None:
            iou_thresh = self.strict_iou_thresh

        n_tracks, n_dets = len(tracks), len(dets)
        if n_tracks == 0 or n_dets == 0:
            return [], list(range(n_tracks)), list(range(n_dets))

        tlwhs_track = [t.tlwh for t in tracks]
        tlwhs_det = [d.tlwh for d in dets]
        cost = iou_cost(tlwhs_track, tlwhs_det)

        if gate_appearance:
            det_feats = np.stack([d.feat for d in dets])
            for i, t in enumerate(tracks):
                centers = t.kema.match_centers
                if centers.shape[0] > 0:
                    nn_dist = _nn_cosine_distance(centers, det_feats)
                    cost[i, nn_dist > self.appearance_thresh] = INFTY_COST

        return linear_assignment(cost, iou_thresh)

    # ── appearance NN match ─────────────────────────────────────────────

    def nn_match(
        self,
        tracks: list[Track],
        dets: list[Detection],
        app_thresh: float | None = None,
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        """Cosine-distance NN match (no Mahalanobis gating).

        Suitable for recovering LOST tracks.  Detections must have ``feat``.
        """
        if app_thresh is None:
            app_thresh = self.appearance_thresh

        n_tracks, n_dets = len(tracks), len(dets)
        if n_tracks == 0 or n_dets == 0:
            return [], list(range(n_tracks)), list(range(n_dets))

        det_feats = np.stack([d.feat for d in dets])
        cost = self._appearance_cost(tracks, det_feats)
        return linear_assignment(cost, app_thresh)

"""Kalman filter — 8-dim constant-velocity model.

State: (x, y, a, h, vx, vy, va, vh)

xyah : np.ndarray (4,)  — [center_x, center_y, aspect_ratio, height]
mean : np.ndarray (8,)  — full state
cov  : np.ndarray (8,8) — state covariance

Copyright (C) 2026 Nivendel
With assistance from Claude Code and deepseek-v4-pro[1m]
SPDX-License-Identifier: AGPL-3.0-or-later
"""

import numpy as np
import scipy.linalg


class KalmanFilter:
    """Stateless 8-dim Kalman filter — all methods are classmethods."""

    _ndim = 4
    _dt = 1.0
    _motion_mat = np.eye(2 * _ndim, 2 * _ndim)
    for i in range(_ndim):
        _motion_mat[i, _ndim + i] = _dt
    _update_mat = np.eye(_ndim, 2 * _ndim)
    _std_weight_position = 1.0 / 20
    _std_weight_velocity = 1.0 / 160

    @classmethod
    def initiate(cls, xyah: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Create a new track state from a detection."""
        mean = np.hstack([xyah, np.zeros(4)])
        h = xyah[3]
        pos_std = 2 * cls._std_weight_position * h
        vel_std = 10 * cls._std_weight_velocity * h
        std = [pos_std, pos_std, 1e-2, pos_std, vel_std, vel_std, 1e-5, vel_std]
        cov = np.diag(np.square(std))
        return mean, cov

    @classmethod
    def predict(cls, mean: np.ndarray, cov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Advance the state one time-step."""
        h = mean[3]
        pos_std = cls._std_weight_position * h
        vel_std = cls._std_weight_velocity * h
        std = [pos_std, pos_std, 1e-2, pos_std, vel_std, vel_std, 1e-5, vel_std]
        motion_cov = np.diag(np.square(std))
        mean = cls._motion_mat @ mean
        cov = cls._motion_mat @ cov @ cls._motion_mat.T + motion_cov
        return mean, cov

    @classmethod
    def project(cls, mean: np.ndarray, cov: np.ndarray, confidence: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
        """Project state to measurement space."""
        pos_std = cls._std_weight_position * mean[3]
        std = [(1 - confidence) * s for s in [pos_std, pos_std, 1e-1, pos_std]]
        innovation_cov = np.diag(np.square(std))
        mean_proj = cls._update_mat @ mean
        cov_proj = cls._update_mat @ cov @ cls._update_mat.T + innovation_cov
        return mean_proj, cov_proj

    @classmethod
    def update(
        cls,
        mean: np.ndarray,
        cov: np.ndarray,
        xyah: np.ndarray,
        confidence: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Kalman update with a matched detection."""
        mean_proj, cov_proj = cls.project(mean, cov, confidence)
        chol, low = scipy.linalg.cho_factor(cov_proj, lower=True, check_finite=False)
        K_gain = scipy.linalg.cho_solve(
            (chol, low), (cov @ cls._update_mat.T).T, check_finite=False
        ).T
        innovation = xyah - mean_proj
        mean_new = mean + innovation @ K_gain.T
        cov_new = cov - K_gain @ cov_proj @ K_gain.T
        return mean_new, cov_new

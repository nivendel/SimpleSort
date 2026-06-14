"""Track — single-object trajectory with Kalman state and appearance model.

Copyright (C) 2026 Nivendel, College of Civil Engineering, Tongji University
With assistance from Claude Code and deepseek-v4-pro[1m]
SPDX-License-Identifier: AGPL-3.0-or-later
"""

import numpy as np
from .detector import Detection
from .kalmanfilter import KalmanFilter as kf
from .kema import KEMA

TENTATIVE = 0    # newly created, not yet confirmed
CONFIRMED = 1    # active, matched to detections
LOST = 2         # confirmed but unmatched for too long
DELETED = 3      # removed from the active set
RECOVERING = 4   # re-acquired after LOST, probation before CONFIRMED


class Track:
    """Single-object trajectory.

    Parameters
    ----------
    det : Detection
        Seed detection.
    n_confirm : int = 3
        Hits needed to go TENTATIVE → CONFIRMED.
    max_age : int = 10
        Frames unmatched before CONFIRMED → LOST.
    max_tentative_misses : int = 2
        Misses before TENTATIVE → DELETED.
    kema_min_hits, kema_n_std, kema_alpha :
        Passed to :class:`KEMA`.
    """

    _id_counter = 0

    def __init__(
        self,
        det: Detection,
        n_confirm: int = 3,
        max_age: int = 10,
        max_tentative_misses: int = 2,
        kema_min_hits: int = 3,
        kema_n_std: float = 3.0,
        kema_alpha: float = 0.1,
        kema_split_thresh: float | None = None,
    ):
        mean, cov = kf.initiate(det.xyah)
        self.mean = np.asarray(mean, dtype=np.float64)
        self.cov = np.asarray(cov, dtype=np.float64)
        self.id = Track._id_counter
        Track._id_counter += 1

        self.state = TENTATIVE
        self.time_since_update = 0
        self.hits = 0
        self.misses = 0
        self.age = 0
        self.last_tlwh = list(det.tlwh)

        self.kema = KEMA(kema_min_hits, kema_n_std, kema_alpha, kema_split_thresh)
        if det.feat is not None:
            self.kema(det.feat)

        self._n_confirm = n_confirm
        self._max_age = max_age
        self._max_tentative_misses = max_tentative_misses

    # -- properties --------------------------------------------------------

    @property
    def tlwh(self) -> list[float]:
        """Kalman-predicted position as [x, y, w, h]."""
        x, y, a, h = self.mean[:4]
        return [x, y, a * h, h]

    @property
    def is_confirmed(self) -> bool:
        return self.state == CONFIRMED

    @property
    def is_tentative(self) -> bool:
        return self.state == TENTATIVE

    @property
    def is_lost(self) -> bool:
        return self.state == LOST

    @property
    def is_deleted(self) -> bool:
        return self.state == DELETED

    @property
    def is_recovering(self) -> bool:
        return self.state == RECOVERING

    # -- lifecycle ---------------------------------------------------------

    def predict(self) -> None:
        """Advance Kalman filter one frame."""
        self.mean, self.cov = kf.predict(self.mean, self.cov)
        self.age += 1
        self.time_since_update += 1

    def update(self, det: Detection) -> None:
        """Update with a matched detection.  Re-initialises if LOST."""
        if self.state == LOST:
            mean, cov = kf.initiate(det.xyah)
            self.reactivate(mean, cov, det.feat, det.tlwh)
            return

        self.mean, self.cov = kf.update(self.mean, self.cov, det.xyah, det.conf)

        self.kema(det.feat, recovering=self.is_recovering)
        self.last_tlwh = list(det.tlwh)

        self.hits += 1
        self.misses = 0
        self.time_since_update = 0

        if self.state == TENTATIVE and self.hits >= self._n_confirm:
            self.state = CONFIRMED
        elif self.state == RECOVERING and self.hits >= 2 * self._n_confirm:
            self.state = CONFIRMED

    def mark_missed(self) -> None:
        """Advance miss counters — may transition to DELETED or LOST."""
        if self.state == TENTATIVE:
            self.misses += 1
            if self.misses > self._max_tentative_misses:
                self.state = DELETED
        elif self.state == RECOVERING:
            self.state = LOST
        elif self.state == CONFIRMED and self.time_since_update > self._max_age:
            self.state = LOST

    def reactivate(
        self,
        mean: np.ndarray,
        cov: np.ndarray,
        feat: np.ndarray,
        tlwh: list[float],
    ) -> None:
        """Re-activate a LOST track, entering RECOVERING probation."""
        self.mean = mean
        self.cov = cov
        self.kema(feat, recovering=True)
        self.last_tlwh = list(tlwh)
        self.state = RECOVERING
        self.hits = 0
        self.time_since_update = 0
        self.age = 0

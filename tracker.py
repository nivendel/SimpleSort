"""Multi-object tracker — 3-stage association pipeline.

1. Strict IoU  — KF-predicted, time_since_update ≤ 1, excl. RECOVERING
2. Loose IoU  — remaining (excl. LOST, RECOVERING)
3. Appearance NN — cosine distance on KEMA centres (all remaining)
4. Initiation — unmatched dets → new TENTATIVE tracks
5. Cleanup    — remove DELETED tracks

Input : frame (H, W, 3) BGR  →  np.ndarray
State : self.tracks (list[Track]), self._cur_dets (list[tuple[Detection, int]])

Copyright (C) 2026 Nivendel, College of Civil Engineering, Tongji University
With assistance from Claude Code and deepseek-v4-pro[1m]
SPDX-License-Identifier: AGPL-3.0-or-later
"""

import numpy as np
import cv2
from .detector import Detection, Detector
from .reid import ReID
from .track import Track
from .matcher import Matcher

_NUM_COLORS = 32
_GOLDEN = 0.618033988749895
_hues = (np.arange(_NUM_COLORS) * _GOLDEN * 180) % 180
_HSV_HUES = _hues.astype(np.uint8)
_HSV_SAT = np.full(_NUM_COLORS, 255, dtype=np.uint8)
_HSV_VAL = np.full(_NUM_COLORS, 255, dtype=np.uint8)
_HSV = np.stack([_HSV_HUES, _HSV_SAT, _HSV_VAL], axis=1).reshape(-1, 1, 3)
_PALETTE = cv2.cvtColor(_HSV, cv2.COLOR_HSV2BGR).reshape(-1, 3)


class Tracker:
    """Multi-object tracker: YOLO + Kalman + ReID.

    Parameters
    ----------
    detector : Detector
    reid : ReID
    matcher : Matcher
    n_confirm : int = 3
    max_age : int = 10
    max_tentative_misses : int = 2
    kema_min_hits, kema_n_std, kema_alpha : passed to :class:`Track`.
    """

    def __init__(
        self,
        detector: Detector,
        reid: ReID,
        matcher: Matcher,
        n_confirm: int = 3,
        max_age: int = 10,
        max_tentative_misses: int = 2,
        kema_min_hits: int = 3,
        kema_n_std: float = 3.0,
        kema_alpha: float = 0.1,
        kema_split_thresh: float | None = None,
    ):
        self.detector = detector
        self.reid = reid
        self.matcher = matcher
        self._n_confirm = n_confirm
        self._max_age = max_age
        self._max_tentative_misses = max_tentative_misses
        self._kema_min_hits = kema_min_hits
        self._kema_n_std = kema_n_std
        self._kema_alpha = kema_alpha
        self._kema_split_thresh = kema_split_thresh
        self.tracks: list[Track] = []
        self._cur_dets: list[tuple[Detection, int]] = []

    # -- helpers ------------------------------------------------------------

    def _commit_deletions(self) -> None:
        self.tracks = [t for t in self.tracks if not t.is_deleted]

    def _make_track(self, det: Detection) -> Track:
        return Track(
            det, self._n_confirm, self._max_age, self._max_tentative_misses,
            self._kema_min_hits, self._kema_n_std, self._kema_alpha,
            self._kema_split_thresh,
        )

    # -- main loop ----------------------------------------------------------

    def update(self, frame: np.ndarray) -> None:
        """Predict → detect → extract features → match → update state."""
        for t in self.tracks:
            t.predict()

        dets = self.detector(frame)
        if not dets:
            self._cur_dets = []
            for t in self.tracks:
                t.mark_missed()
            self._commit_deletions()
            return

        n_dets = len(dets)

        tlwhs = [det.tlwh for det in dets]
        feats = self.reid(frame, tlwhs)
        for det, feat in zip(dets, feats):
            det.set_feat(feat)

        matched_track_ids: set[int] = set()
        matched_det_ids: set[int] = set()
        all_matches: list[tuple[int, int]] = []

        def _unmatched_dets() -> tuple[list[int], list[Detection]]:
            idx = [j for j in range(n_dets) if j not in matched_det_ids]
            return idx, [dets[j] for j in idx]

        # stage 1: strict IoU (tsu ≤ 1, not recovering)
        s1_idx = [i for i, t in enumerate(self.tracks)
                  if t.time_since_update <= 1 and not t.is_recovering]
        if s1_idx:
            s1_tracks = [self.tracks[i] for i in s1_idx]
            m, _, _ = self.matcher.iou_match(s1_tracks, dets, self.matcher.strict_iou_thresh,
                                               gate_appearance=True)
            for r, c in m:
                ti, di = s1_idx[r], c
                all_matches.append((ti, di))
                matched_track_ids.add(ti)
                matched_det_ids.add(di)

        # stage 2: loose IoU (remaining, not LOST, not recovering)
        rem_det_idx, rem_dets = _unmatched_dets()
        s2_idx = [i for i in range(len(self.tracks))
                  if i not in matched_track_ids
                  and not self.tracks[i].is_lost
                  and not self.tracks[i].is_recovering]
        if s2_idx and rem_dets:
            s2_tracks = [self.tracks[i] for i in s2_idx]
            m, _, _ = self.matcher.iou_match(s2_tracks, rem_dets, self.matcher.loose_iou_thresh,
                                               gate_appearance=True)
            for r, c in m:
                ti, di = s2_idx[r], rem_det_idx[c]
                all_matches.append((ti, di))
                matched_track_ids.add(ti)
                matched_det_ids.add(di)

        # stage 3: appearance NN (all remaining, incl. LOST)
        rem_det_idx, rem_dets = _unmatched_dets()
        s3_idx = [i for i in range(len(self.tracks)) if i not in matched_track_ids]
        if s3_idx and rem_dets:
            s3_tracks = [self.tracks[i] for i in s3_idx]
            m, _, _ = self.matcher.nn_match(s3_tracks, rem_dets)
            for r, c in m:
                ti, di = s3_idx[r], rem_det_idx[c]
                all_matches.append((ti, di))
                matched_track_ids.add(ti)
                matched_det_ids.add(di)

        # apply updates
        for ti, di in all_matches:
            self.tracks[ti].update(dets[di])

        # mark missed
        for i in range(len(self.tracks)):
            if i not in matched_track_ids:
                self.tracks[i].mark_missed()

        # initiate new tracks
        new_track_ids: dict[int, int] = {}
        for j in range(n_dets):
            if j not in matched_det_ids:
                t = self._make_track(dets[j])
                self.tracks.append(t)
                new_track_ids[j] = t.id

        # record for visual
        self._cur_dets = []
        for ti, di in all_matches:
            self._cur_dets.append((dets[di], self.tracks[ti].id))
        for j, tid in new_track_ids.items():
            self._cur_dets.append((dets[j], tid))

        self._commit_deletions()

    # -- visualisation ------------------------------------------------------

    def visual(self, frame: np.ndarray) -> np.ndarray:
        """Draw detection boxes with track IDs (RGB)."""
        out = frame.copy()
        for det, tid in self._cur_dets:
            x, y, w, h = map(int, det.tlwh)
            c = tuple(int(v) for v in _PALETTE[tid % len(_PALETTE)])
            cv2.rectangle(out, (x, y), (x + w, y + h), c, 2)
            cv2.putText(out, f"ID:{tid}", (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1)
        return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)

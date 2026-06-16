#!/usr/bin/env python3
"""Quick-start: track a video, produce output video + FPS plot, print ID stats.

Usage::

    cd SimpleSort
    python run.py /path/to/video.mp4 [--yolo yolo26x.pt] [--reid msmt_sbs_R101-ibn.pth]

Or from anywhere using the package::

    PYTHONPATH=/path/to/SimpleSort/parent python -c "from SimpleSort.run import main; main()" -- /path/to/video.mp4

Copyright (C) 2026 Nivendel
With assistance from Claude Code and deepseek-v4-pro[1m]
SPDX-License-Identifier: AGPL-3.0-or-later
"""

import argparse, sys, time, threading, queue
from pathlib import Path

# Make the parent directory importable so ``from SimpleSort import …`` works
_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

from SimpleSort import Detector, ReID, Matcher, Tracker, VideoFrameReader

import cv2
import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SimpleSort — track a video, produce output + FPS plot.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("video", type=str, help="Path to input video")
    p.add_argument("--yolo", type=str, default="yolo26x.pt", help="YOLO weights path")
    p.add_argument("--reid", type=str, default="msmt_sbs_R101-ibn.pth", help="ReID weights path")
    p.add_argument("--output", "-o", type=str, default="output/run", help="Output directory")
    p.add_argument("--no-video", action="store_true", help="Skip writing tracked video")
    p.add_argument("--n-confirm", type=int, default=None)
    p.add_argument("--max-age", type=int, default=None)
    p.add_argument("--max-tentative-misses", type=int, default=None)
    p.add_argument("--appearance-thresh", type=float, default=None)
    p.add_argument("--strict-iou", type=float, default=None)
    p.add_argument("--loose-iou", type=float, default=None)
    p.add_argument("--det-conf", type=float, default=None)
    p.add_argument("--det-iou", type=float, default=None)
    p.add_argument("--kema-min-hits", type=int, default=None)
    p.add_argument("--kema-n-std", type=float, default=None)
    p.add_argument("--kema-alpha", type=float, default=None)
    p.add_argument("--kema-split-thresh", type=float, default=None)
    p.add_argument("--kema-stale-frames", type=int, default=None)
    p.add_argument("--kema-stale-max-hits", type=int, default=None)
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    video_out = out_dir / "tracked.mp4"
    fps_png = out_dir / "fps.png"

    # Only pass args that were explicitly set; let classes use their defaults otherwise
    det_kwargs: dict = {}
    if args.det_conf is not None:
        det_kwargs["conf"] = args.det_conf
    if args.det_iou is not None:
        det_kwargs["iou"] = args.det_iou

    mat_kwargs: dict = {}
    if args.appearance_thresh is not None:
        mat_kwargs["appearance_thresh"] = args.appearance_thresh
    if args.strict_iou is not None:
        mat_kwargs["strict_iou_thresh"] = args.strict_iou
    if args.loose_iou is not None:
        mat_kwargs["loose_iou_thresh"] = args.loose_iou

    trk_kwargs: dict = {}
    for k in ("n_confirm", "max_age", "max_tentative_misses",
              "kema_min_hits", "kema_n_std", "kema_alpha",
              "kema_split_thresh", "kema_stale_frames", "kema_stale_max_hits"):
        v = getattr(args, k)
        if v is not None:
            trk_kwargs[k] = v

    tracker = Tracker(
        detector=Detector(args.yolo, **det_kwargs),
        reid=ReID(args.reid),
        matcher=Matcher(**mat_kwargs),
        **trk_kwargs,
    )

    reader = VideoFrameReader(args.video)
    print(reader)
    print(f"  Output:    {out_dir.resolve()}")

    # -- background writer thread -------------------------------------------
    _NUM_COLORS = 32
    _GOLDEN = 0.618033988749895
    _hues = (np.arange(_NUM_COLORS) * _GOLDEN * 180) % 180
    _HSV = np.stack([_hues.astype(np.uint8), np.full(_NUM_COLORS, 255, np.uint8),
                     np.full(_NUM_COLORS, 255, np.uint8)], axis=1).reshape(-1, 1, 3)
    _PALETTE = cv2.cvtColor(_HSV, cv2.COLOR_HSV2BGR).reshape(-1, 3)

    vw = None
    write_queue: queue.Queue = queue.Queue(maxsize=120)
    _stop_sentinel = object()

    def _writer() -> None:
        while True:
            item = write_queue.get()
            if item is _stop_sentinel:
                break
            frame, cur_dets = item
            out = frame.copy()
            for det, tid in cur_dets:
                x, y, w, h = map(int, det.tlwh)
                c = tuple(int(v) for v in _PALETTE[tid % len(_PALETTE)])
                cv2.rectangle(out, (x, y), (x + w, y + h), c, 2)
                cv2.putText(out, f"ID:{tid}", (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1)
            vw.write(out)

    if not args.no_video:
        vw = cv2.VideoWriter(
            str(video_out),
            cv2.VideoWriter_fourcc(*"mp4v"),
            reader.fps,
            (reader.width, reader.height),
        )
        writer_thread = threading.Thread(target=_writer, daemon=True)
        writer_thread.start()

    # -- tracking loop -------------------------------------------------------
    fps_history: list[float] = []
    t_start = time.perf_counter()

    pbar = tqdm(total=reader.total_frames, desc="Tracking", unit="fr")
    for _fi, frame in reader:
        t0 = time.perf_counter()
        tracker.update(frame)
        fps = 1.0 / (time.perf_counter() - t0)
        fps_history.append(fps)
        pbar.set_postfix(fps=f"{fps:.1f}")
        pbar.update()
        if vw is not None:
            write_queue.put((frame.copy(), list(tracker._cur_dets)))
    pbar.close()

    wall_time = time.perf_counter() - t_start
    reader.release()

    if vw is not None:
        write_queue.put(_stop_sentinel)
        writer_thread.join()
        vw.release()

    fps_arr = np.array(fps_history)
    N = len(fps_arr)

    # -- FPS plot ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 5))
    frames = np.arange(N)
    ax.plot(frames, fps_arr, "b-", alpha=0.3, lw=0.5, label="per-frame")
    if N >= 20:
        smooth = np.convolve(fps_arr, np.ones(20) / 20, mode="valid")
        ax.plot(frames[19:], smooth, "b-", lw=2, label="20-frame smooth")
    ax.axhline(fps_arr.mean(), color="r", ls="--", lw=1,
               label=f"mean: {fps_arr.mean():.1f} fps")
    ax.set_xlabel("Frame")
    ax.set_ylabel("FPS")
    ax.set_title(f"{Path(args.video).name}  —  {N} frames, {wall_time:.1f}s wall")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fps_png, dpi=150)
    plt.close(fig)

    # -- ID statistics -------------------------------------------------------
    tracks = tracker.tracks
    state_names = {0: "TENTATIVE", 1: "CONFIRMED", 2: "LOST", 3: "DELETED", 4: "RECOVERING"}
    state_counts: dict[str, int] = {}
    confirmed_ids: list[int] = []
    for t in tracks:
        label = state_names.get(t.state, f"UNKNOWN({t.state})")
        state_counts[label] = state_counts.get(label, 0) + 1
        if t.is_confirmed:
            confirmed_ids.append(t.id)

    # exclude first frame (model warmup) from per-frame stats
    fps_steady = fps_arr[1:] if len(fps_arr) > 1 else fps_arr

    print(f"\n── Tracking Summary " + "─" * 40)
    print(f"  Video:       {Path(args.video).name}")
    print(f"  Frames:      {N}")
    print(f"  Mean FPS:    {fps_steady.mean():.1f}  (min {fps_steady.min():.1f} / max {fps_steady.max():.1f})")
    print(f"\n── Track IDs  —  {len(tracks)} total")
    for label in ("CONFIRMED", "TENTATIVE", "LOST", "RECOVERING", "DELETED"):
        c = state_counts.get(label, 0)
        if c:
            print(f"  {label:<13s} {c:>5d}")
    print(f"\n  Confirmed IDs:  {sorted(confirmed_ids)}")
    if confirmed_ids:
        print(f"  Count:          {len(confirmed_ids)}")

    print(f"\n── Per-ID Detail " + "─" * 43)
    print(f"  {'ID':>4s}  {'state':>11s}  {'hits':>5s}  {'missed':>6s}  "
          f"{'age':>5s}  {'centres':>7s}  {'active':>6s}")
    for t in sorted(tracks, key=lambda t: t.id):
        label = state_names.get(t.state, "?")
        n_centers = t.kema._centers.shape[0]
        n_active = int(t.kema._active.sum())
        total_missed = t.age - t.hits
        print(f"  {t.id:>4d}  {label:>11s}  {t.hits:>5d}  {total_missed:>6d}  "
              f"{t.age:>5d}  {n_centers:>7d}  {n_active:>6d}")

    print(f"\n── Output " + "─" * 45)
    if vw is not None:
        print(f"  Video:      {video_out.resolve()}")
    print(f"  FPS plot:   {fps_png.resolve()}")


if __name__ == "__main__":
    main()

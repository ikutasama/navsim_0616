"""
Create NAVSIM continuous-scene GIFs for Alpamayo1.5 outputs.

This is different from visualize_cot_trajectory_cases.py --make_case_gifs:
- case GIF: one token, fixed current image, reveals that token's 4s future trajectory.
- continuous scene GIF: one log_name, consecutive NAVSIM tokens sorted by start_time.
  Each GIF frame uses the current front camera image for that timestep and overlays the
  Alpamayo prediction / GT future trajectory for that same timestep.

Typical usage:
  python visualize_continuous_navsim_scene.py \
    --scores_csv /path/to/alpamayo_pdm_scores_merged.csv \
    --analysis_csv /path/to/cot_failure_pattern_enriched.csv \
    --metric_cache_path $OPENSCENE_DATA_ROOT/metric_cache \
    --navsim_log_path $OPENSCENE_DATA_ROOT/navsim_logs/mini \
    --sensor_blobs_path $OPENSCENE_DATA_ROOT/sensor_blobs/mini \
    --output_dir $OPENSCENE_DATA_ROOT/exp/continuous_scene_gifs \
    --select weakest --top_k 10 --frames 12

The output GIF is a short continuous driving clip for one NAVSIM log, useful for judging:
- whether the front-view scene is actually complex over time;
- whether per-timestep CoT/meta_action matches the visible situation;
- whether predicted ego futures agree with human/GT futures frame by frame.
"""

import argparse
import math
import re
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from visualize_cot_trajectory_cases import (
    add_basic_ranking_columns,
    bool_mark,
    compute_plot_bounds,
    draw_agent_future_tracks,
    draw_static_objects,
    extract_gt_traj_and_objects,
    get_front_camera_image,
    infer_data_root_from_scores,
    load_and_merge,
    load_metric_cache,
    parse_json_list,
    safe_str,
    wrap_text,
)


def _to_float(x: Any, default: float = float("nan")) -> float:
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def predicted_traj_from_row(row: pd.Series) -> Optional[np.ndarray]:
    xs = parse_json_list(row.get("pred_traj_x", "[]"))
    ys = parse_json_list(row.get("pred_traj_y", "[]"))
    hs = parse_json_list(row.get("pred_traj_heading", "[]"))
    if not xs or not ys:
        return None
    n = min(len(xs), len(ys))
    if len(hs) < n:
        hs = hs + [0.0] * (n - len(hs))
    return np.asarray(list(zip(xs[:n], ys[:n], hs[:n])), dtype=float)


def require_time_and_log_columns(df: pd.DataFrame) -> None:
    missing = [c for c in ["token", "log_name", "start_time"] if c not in df.columns]
    if missing:
        raise ValueError(
            "continuous-scene GIF needs columns token, log_name, start_time in scores_csv. "
            f"Missing: {missing}. Re-run run_alpamayo_eval.py from this repo; it saves log_name/start_time."
        )


def select_anchor_rows(df: pd.DataFrame, mode: str, top_k: int, tokens: Optional[List[str]]) -> pd.DataFrame:
    if tokens:
        token_set = set(tokens)
        return df[df["token"].astype(str).isin(token_set)].copy()
    if mode == "all":
        return df.head(top_k).copy()
    if mode == "low_pdm":
        return df.sort_values("pdm_score", ascending=True, na_position="last").head(top_k).copy()
    if mode == "inconsistent":
        return df.sort_values("cot_traj_consistency", ascending=True, na_position="last").head(top_k).copy()
    return df.sort_values("weak_case_score", ascending=False, na_position="last").head(top_k).copy()


def contiguous_segment_for_anchor(df: pd.DataFrame, anchor_row: pd.Series, frames: int, pre_frames: int) -> pd.DataFrame:
    log_name = safe_str(anchor_row.get("log_name", ""))
    token = safe_str(anchor_row.get("token", ""))
    if not log_name:
        raise ValueError(f"anchor token={token} has empty log_name")

    group = df[df["log_name"].astype(str) == log_name].copy()
    group["start_time_num"] = pd.to_numeric(group["start_time"], errors="coerce")
    group = group.sort_values(["start_time_num", "token"], kind="stable").reset_index(drop=True)
    matches = group.index[group["token"].astype(str) == token].tolist()
    if not matches:
        # Fallback to nearest start_time within the same log.
        t = _to_float(anchor_row.get("start_time"))
        if math.isfinite(t) and group["start_time_num"].notna().any():
            idx = int((group["start_time_num"] - t).abs().idxmin())
        else:
            idx = 0
    else:
        idx = matches[0]

    start = max(0, idx - max(pre_frames, 0))
    end = min(len(group), start + max(frames, 1))
    # If close to end, shift left so the segment still has requested length when possible.
    start = max(0, end - max(frames, 1))
    return group.iloc[start:end].copy()


def stable_bounds_for_segment(segment: pd.DataFrame, metric_cache_path: Optional[str], auto_bounds: bool) -> Tuple[float, float, float, float]:
    if not auto_bounds:
        # Fixed local ego-frame window makes frame-to-frame comparison easier.
        return -12.0, 62.0, -28.0, 28.0

    all_pts: List[List[float]] = [[0.0, 0.0]]
    for _, row in segment.iterrows():
        pred = predicted_traj_from_row(row)
        token = safe_str(row.get("token", ""))
        metric_cache = load_metric_cache(metric_cache_path, token) if metric_cache_path else None
        gt, objects, agent_tracks = extract_gt_traj_and_objects(metric_cache)
        xmin, xmax, ymin, ymax = compute_plot_bounds(pred, gt, objects, agent_tracks)
        all_pts.extend([[xmin, ymin], [xmax, ymax]])
    pts = np.asarray(all_pts, dtype=float)
    return float(pts[:, 0].min()), float(pts[:, 0].max()), float(pts[:, 1].min()), float(pts[:, 1].max())


def draw_one_timestep(
    row: pd.Series,
    frame_idx: int,
    n_frames: int,
    anchor_token: str,
    bounds: Tuple[float, float, float, float],
    metric_cache_path: Optional[str],
    navsim_log_path: Optional[str],
    sensor_blobs_path: Optional[str],
):
    token = safe_str(row.get("token", ""))
    log_name = safe_str(row.get("log_name", ""))
    start_time = _to_float(row.get("start_time"))
    pred = predicted_traj_from_row(row)

    metric_cache = load_metric_cache(metric_cache_path, token) if metric_cache_path else None
    gt, objects, agent_tracks = extract_gt_traj_and_objects(metric_cache)
    front_img = get_front_camera_image(token, navsim_log_path, sensor_blobs_path)

    pdm = _to_float(row.get("pdm_score"))
    cons = _to_float(row.get("cot_traj_consistency"))
    weak = _to_float(row.get("weak_case_score"))
    meta = wrap_text(safe_str(row.get("meta_action", "")), width=58, max_lines=3)
    cot = wrap_text(safe_str(row.get("cot", "")), width=58, max_lines=9)
    answer = wrap_text(safe_str(row.get("answer", "")), width=58, max_lines=3)

    fig = plt.figure(figsize=(16.5, 8.8), dpi=115)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.05, 1.45], height_ratios=[0.78, 1.0], wspace=0.18, hspace=0.18)
    ax_cam = fig.add_subplot(gs[0, 0])
    ax_text = fig.add_subplot(gs[1, 0])
    ax = fig.add_subplot(gs[:, 1])

    anchor_mark = "ANCHOR" if token == anchor_token else ""
    t_str = f"{start_time:.2f}s" if math.isfinite(start_time) else "?s"
    fig.suptitle(
        f"Continuous NAVSIM log={log_name[:42]}  frame {frame_idx + 1}/{n_frames}  t={t_str}  {anchor_mark}\n"
        f"token={token[:22]}  PDM={pdm:.3f}  CoT-Traj={cons:.3f}  weak={weak:.3f}",
        fontsize=12.5,
        fontweight="bold",
    )

    ax_cam.set_title("Front camera cam_f0 at this timestep")
    ax_cam.axis("off")
    if front_img is not None:
        ax_cam.imshow(front_img)
    else:
        ax_cam.text(0.5, 0.5, "front camera unavailable", ha="center", va="center")

    ax_text.axis("off")
    text_lines = ["META_ACTION", meta, "", "COT", cot, ""]
    if answer and answer != "<empty>":
        text_lines += ["ANSWER", answer, ""]
    text_lines += ["QUANT", f"  pdm_score: {pdm:.4f}", f"  cot_traj_consistency: {cons:.4f}", f"  weak_case_score: {weak:.4f}"]
    for flag in ["mentions_object_any", "mentions_other_agent_intent", "mentions_risk_or_conflict", "mentions_spatial_relation"]:
        if flag in row.index:
            text_lines.append(f"  {flag}: {bool_mark(row.get(flag))}")
    ax_text.text(0.0, 1.0, "\n".join(text_lines), va="top", ha="left", fontsize=8.3, family="monospace")

    ax.set_title("Local ego-frame future at this timestep: prediction vs human/GT")
    ax.axhline(0, color="0.86", lw=1)
    ax.axvline(0, color="0.86", lw=1)
    ax.scatter([0], [0], c="black", s=70, marker="*", label="ego@now", zorder=5)
    draw_static_objects(ax, objects)
    draw_agent_future_tracks(ax, agent_tracks, reveal_fraction=1.0, alpha=0.55)

    if gt is not None and len(gt) >= 2:
        ax.plot(gt[:, 0], gt[:, 1], "-o", color="tab:green", lw=2.8, ms=4.5, label="GT/human future")
        ax.scatter([gt[-1, 0]], [gt[-1, 1]], c="tab:green", s=65, marker="x", zorder=6)
    if pred is not None and len(pred) >= 2:
        ax.plot(pred[:, 0], pred[:, 1], "-o", color="tab:blue", lw=2.8, ms=4.5, label="Alpamayo pred future")
        ax.scatter([pred[-1, 0]], [pred[-1, 1]], c="tab:blue", s=65, marker="x", zorder=6)

    xmin, xmax, ymin, ymax = bounds
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x forward / meters")
    ax.set_ylabel("y left / meters")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    return fig


def make_continuous_scene_gif(
    segment: pd.DataFrame,
    anchor_token: str,
    gif_path: Path,
    metric_cache_path: Optional[str],
    navsim_log_path: Optional[str],
    sensor_blobs_path: Optional[str],
    duration_ms: int,
    auto_bounds: bool,
) -> None:
    try:
        from PIL import Image
    except Exception as e:
        raise RuntimeError(f"Pillow is required for GIF creation: {e}")

    bounds = stable_bounds_for_segment(segment, metric_cache_path, auto_bounds=auto_bounds)
    frames = []
    n = len(segment)
    for i, (_, row) in enumerate(segment.iterrows()):
        fig = draw_one_timestep(
            row=row,
            frame_idx=i,
            n_frames=n,
            anchor_token=anchor_token,
            bounds=bounds,
            metric_cache_path=metric_cache_path,
            navsim_log_path=navsim_log_path,
            sensor_blobs_path=sensor_blobs_path,
        )
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        frames.append(Image.open(buf).convert("RGB"))

    if not frames:
        raise RuntimeError("segment has no frames")
    # Hold final frame briefly for reading.
    frames.extend([frames[-1]] * 2)
    gif_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=duration_ms, loop=0)


def main():
    parser = argparse.ArgumentParser(description="Make continuous NAVSIM log GIFs from Alpamayo per-token outputs")
    parser.add_argument("--scores_csv", required=True, help="alpamayo_pdm_scores_merged.csv with token/log_name/start_time")
    parser.add_argument("--analysis_csv", default=None, help="optional cot_failure_pattern_enriched.csv or cot_consistency_analysis.csv")
    parser.add_argument("--metric_cache_path", default=None)
    parser.add_argument("--navsim_log_path", default=None)
    parser.add_argument("--sensor_blobs_path", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--select", choices=["weakest", "low_pdm", "inconsistent", "all"], default="weakest")
    parser.add_argument("--top_k", type=int, default=10, help="number of anchor scenes/log clips to create")
    parser.add_argument("--tokens", nargs="*", default=None, help="anchor tokens; overrides --select")
    parser.add_argument("--frames", type=int, default=12, help="number of consecutive NAVSIM tokens per GIF")
    parser.add_argument("--pre_frames", type=int, default=3, help="frames before anchor token inside the same log")
    parser.add_argument("--gif_duration_ms", type=int, default=500)
    parser.add_argument("--auto_bounds", action="store_true", help="auto-fit BEV bounds per segment instead of using fixed local bounds")
    parser.add_argument("--scene_log_name", default=None, help="optional exact log_name filter")
    parser.add_argument("--start_time", type=float, default=None, help="optional anchor start_time within --scene_log_name")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    data_root = infer_data_root_from_scores(args.scores_csv)
    if args.metric_cache_path is None and data_root is not None:
        args.metric_cache_path = str(data_root / "metric_cache")
    if args.navsim_log_path is None and data_root is not None:
        args.navsim_log_path = str(data_root / "navsim_logs" / "mini")
    if args.sensor_blobs_path is None and data_root is not None:
        args.sensor_blobs_path = str(data_root / "sensor_blobs" / "mini")

    df = load_and_merge(args.scores_csv, args.analysis_csv)
    df = add_basic_ranking_columns(df)
    require_time_and_log_columns(df)
    df["start_time"] = pd.to_numeric(df["start_time"], errors="coerce")
    df = df[df["valid"] == True].copy() if "valid" in df.columns else df.copy()

    if args.scene_log_name:
        scene_df = df[df["log_name"].astype(str) == args.scene_log_name].copy()
        if scene_df.empty:
            raise RuntimeError(f"No rows for scene_log_name={args.scene_log_name}")
        scene_df = scene_df.sort_values("start_time", kind="stable")
        if args.start_time is not None:
            idx = int((scene_df["start_time"] - args.start_time).abs().idxmin())
            anchors = scene_df.iloc[[idx]].copy()
        else:
            anchors = scene_df.head(1).copy()
    else:
        anchors = select_anchor_rows(df, args.select, args.top_k, args.tokens)

    if anchors.empty:
        raise RuntimeError("No anchor rows selected. Check token filters / CSV paths.")

    manifest_rows: List[Dict[str, Any]] = []
    used = set()
    made = 0
    for rank, (_, anchor) in enumerate(anchors.iterrows(), start=1):
        anchor_token = safe_str(anchor.get("token", ""))
        segment = contiguous_segment_for_anchor(df, anchor, frames=args.frames, pre_frames=args.pre_frames)
        if segment.empty:
            print(f"[continuous-viz] skip empty segment for token={anchor_token}", flush=True)
            continue
        log_name = safe_str(segment.iloc[0].get("log_name", ""))
        first_t = _to_float(segment.iloc[0].get("start_time"), 0.0)
        last_t = _to_float(segment.iloc[-1].get("start_time"), first_t)
        # Avoid producing duplicate clips when several weak anchors are in the same local window.
        dedupe_key = (log_name, round(first_t, 2), round(last_t, 2))
        if dedupe_key in used:
            continue
        used.add(dedupe_key)
        made += 1

        safe_log = re.sub(r"[^A-Za-z0-9_.-]+", "_", log_name)[:60]
        safe_tok = re.sub(r"[^A-Za-z0-9_.-]+", "_", anchor_token)[:30]
        gif_path = output_dir / f"continuous_{made:03d}_{safe_log}_t{first_t:.1f}-{last_t:.1f}_{safe_tok}.gif"
        seg_csv = output_dir / f"continuous_{made:03d}_{safe_log}_segment.csv"
        print(f"[continuous-viz] {made}: log={log_name} frames={len(segment)} anchor={anchor_token} -> {gif_path}", flush=True)
        segment.drop(columns=[c for c in ["start_time_num"] if c in segment.columns]).to_csv(seg_csv, index=False)
        make_continuous_scene_gif(
            segment=segment,
            anchor_token=anchor_token,
            gif_path=gif_path,
            metric_cache_path=args.metric_cache_path,
            navsim_log_path=args.navsim_log_path,
            sensor_blobs_path=args.sensor_blobs_path,
            duration_ms=args.gif_duration_ms,
            auto_bounds=args.auto_bounds,
        )
        manifest_rows.append({
            "gif_path": str(gif_path),
            "segment_csv": str(seg_csv),
            "anchor_token": anchor_token,
            "log_name": log_name,
            "first_start_time": first_t,
            "last_start_time": last_t,
            "num_frames": len(segment),
            "anchor_pdm_score": _to_float(anchor.get("pdm_score")),
            "anchor_cot_traj_consistency": _to_float(anchor.get("cot_traj_consistency")),
            "anchor_weak_case_score": _to_float(anchor.get("weak_case_score")),
        })

    manifest = output_dir / "continuous_scene_gif_manifest.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest, index=False)
    print(f"[continuous-viz] manifest: {manifest}", flush=True)
    print(f"[continuous-viz] output_dir: {output_dir}", flush=True)


if __name__ == "__main__":
    main()

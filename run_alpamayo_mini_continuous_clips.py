"""
Run Alpamayo1.5 on NAVSIM mini as true continuous sliding-window clips and make CoT GIFs.

This script is for qualitative/diagnostic analysis, not official NAVSIM PDM scoring.
It differs from run_alpamayo_eval.py:
- uses SceneFilter(frame_interval=1) so consecutive GIF frames are consecutive NAVSIM frames;
- runs Alpamayo inference at every timestep in each mini log/clip;
- writes one GIF per clip with changing front camera + per-timestep CoT + pred-vs-GT future;
- computes clip-level scene complexity and CoT/trajectory inconsistency statistics.

Example:
  cd /data/mnt_m181/zhn/navsim_0616 && python run_alpamayo_mini_continuous_clips.py \
    --gpu 1 --max_clips 0 --max_frames_per_clip 0 --gif_every_n_clips 1 \
    --output_dir /data/mnt_m181/zhn/navsim_0616/navsim/navsim_dataset/exp/mini_continuous_alpamayo

Notes:
- --gpu sets CUDA_VISIBLE_DEVICES before loading torch/model; inside process --device defaults to cuda:0.
- max_clips=0 and max_frames_per_clip=0 mean all available mini clips / sliding windows.
"""

import argparse
import json
import math
import os
import re
import time
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from analyze_cot_consistency import consistency_score, text_intent, traj_features
from visualize_cot_trajectory_cases import safe_str, wrap_text


NAVSIM_DT = 0.5


def infer_data_root(repo_dir: Path) -> Path:
    env_root = os.environ.get("OPENSCENE_DATA_ROOT", "").strip()
    if env_root:
        return Path(env_root).expanduser().resolve()
    for cand in [repo_dir / "navsim" / "navsim_dataset", repo_dir / "navsim_dataset", repo_dir / "dataset"]:
        if (cand / "navsim_logs" / "mini").exists() and (cand / "sensor_blobs" / "mini").exists():
            return cand.resolve()
    return (repo_dir / "navsim" / "navsim_dataset").resolve()


def finite_float(x: Any, default: float = float("nan")) -> float:
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def trajectory_to_lists(traj) -> Tuple[List[float], List[float], List[float]]:
    poses = np.asarray(traj.poses, dtype=float)
    if poses.ndim != 2 or poses.shape[1] < 3:
        return [], [], []
    return poses[:, 0].tolist(), poses[:, 1].tolist(), poses[:, 2].tolist()


def ego_pose_from_raw_frame(frame: Dict[str, Any]) -> np.ndarray:
    from pyquaternion import Quaternion
    trans = frame["ego2global_translation"]
    quat = Quaternion(*frame["ego2global_rotation"])
    return np.asarray([trans[0], trans[1], quat.yaw_pitch_roll[0]], dtype=np.float64)


def gt_trajectory_from_frame_list(frame_list: List[Dict[str, Any]], num_history_frames: int, n_poses: int = 8) -> Optional[np.ndarray]:
    """Human future trajectory in current ego frame, without loading map/MetricCache."""
    try:
        from nuplan.common.actor_state.state_representation import StateSE2
        from navsim.planning.simulation.planner.pdm_planner.utils.pdm_geometry_utils import convert_absolute_to_relative_se2_array
        start_idx = num_history_frames - 1
        end_idx = min(len(frame_list), start_idx + n_poses + 1)
        if end_idx - start_idx < 3:
            return None
        global_poses = np.asarray([ego_pose_from_raw_frame(frame_list[i]) for i in range(start_idx, end_idx)], dtype=np.float64)
        local = convert_absolute_to_relative_se2_array(StateSE2(*global_poses[0]), global_poses[1:])
        if len(local) >= 2:
            return np.asarray(local[:, :3], dtype=float)
    except Exception:
        return None
    return None


def frame_timestamp_s(scene_frame: Dict[str, Any], fallback_idx: int) -> float:
    for key in ["timestamp", "timestamp_us", "time_us"]:
        if key in scene_frame:
            try:
                v = float(scene_frame[key])
                # nuPlan/nuscenes timestamps are often us; small values may already be seconds.
                return v / 1e6 if v > 1e5 else v
            except Exception:
                pass
    return fallback_idx * NAVSIM_DT


def annotation_objects_from_frame(scene_frame: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract current-frame object boxes from raw NAVSIM frame annotations.

    NAVSIM raw annotation boxes are in ego/lidar coordinates for the frame; use x,y directly.
    Expected box layout is [x,y,z,l,w,h,yaw] for most OpenScene/NuPlan exports.
    """
    objs: List[Dict[str, Any]] = []
    try:
        anns = scene_frame.get("anns", {})
        boxes = np.asarray(anns.get("gt_boxes", []), dtype=float)
        names = list(anns.get("gt_names", []))
        track_tokens = list(anns.get("track_tokens", [""] * len(boxes)))
        for i, box in enumerate(boxes):
            if len(box) < 2:
                continue
            x = float(box[0])
            y = float(box[1])
            length = float(box[3]) if len(box) > 3 and np.isfinite(box[3]) else 4.5
            width = float(box[4]) if len(box) > 4 and np.isfinite(box[4]) else 1.8
            heading = float(box[6]) if len(box) > 6 and np.isfinite(box[6]) else 0.0
            typ = str(names[i]).lower() if i < len(names) else "object"
            tok = str(track_tokens[i]) if i < len(track_tokens) else f"obj_{i}"
            dist = float(math.sqrt(x * x + y * y))
            objs.append({"x": x, "y": y, "length": length, "width": width, "heading": heading, "type": typ, "track_token": tok, "dist": dist})
    except Exception:
        pass
    return objs


def complexity_from_objects_and_motion(n10_values: List[int], min_dist_values: List[float], curvature_values: List[float]) -> Dict[str, Any]:
    arr = np.asarray(n10_values, dtype=float) if n10_values else np.asarray([], dtype=float)
    peak_n10 = int(np.nanmax(arr)) if len(arr) else 0
    avg_n10 = float(np.nanmean(arr)) if len(arr) else 0.0
    finite_dist = np.asarray([d for d in min_dist_values if np.isfinite(d)], dtype=float)
    min_dist = float(np.min(finite_dist)) if len(finite_dist) else np.nan
    finite_curv = np.asarray([c for c in curvature_values if np.isfinite(c)], dtype=float)
    peak_curv = float(np.max(finite_curv)) if len(finite_curv) else 0.0

    # 0/1/2: intentionally simple and interpretable.
    # 2 = many close objects or sustained nearby interactions or sharp maneuver.
    # 1 = some nearby objects / moderate maneuver.
    # 0 = mostly empty/simple.
    if peak_n10 >= 4 or avg_n10 >= 2.0 or (np.isfinite(min_dist) and min_dist < 3.0) or peak_curv > 0.18:
        level = 2
    elif peak_n10 >= 1 or avg_n10 >= 0.4 or (np.isfinite(min_dist) and min_dist < 10.0) or peak_curv > 0.08:
        level = 1
    else:
        level = 0

    return {
        "complexity_level": level,
        "complexity_label": ["simple", "medium", "complex"][level],
        "peak_objects_within_10m": peak_n10,
        "avg_objects_within_10m": avg_n10,
        "min_object_dist": min_dist,
        "peak_gt_curvature_proxy": peak_curv,
    }


def pred_vs_gt_metrics(pred: Optional[np.ndarray], gt: Optional[np.ndarray]) -> Dict[str, Any]:
    if pred is None or gt is None or len(pred) < 2 or len(gt) < 2:
        return {
            "ade": np.nan, "fde": np.nan, "heading_abs_err": np.nan,
            "turn_direction_mismatch": False, "lateral_final_err": np.nan,
            "progress_err": np.nan, "traj_bad": False,
        }
    n = min(len(pred), len(gt))
    p = np.asarray(pred[:n, :3], dtype=float)
    g = np.asarray(gt[:n, :3], dtype=float)
    d = np.sqrt((p[:, 0] - g[:, 0]) ** 2 + (p[:, 1] - g[:, 1]) ** 2)
    ade = float(np.mean(d))
    fde = float(d[-1])
    heading_err = float(abs(np.unwrap(p[:, 2])[-1] - np.unwrap(g[:, 2])[-1]))
    lateral_err = float(abs(p[-1, 1] - g[-1, 1]))
    p_prog = float(math.sqrt((p[-1, 0] - p[0, 0]) ** 2 + (p[-1, 1] - p[0, 1]) ** 2))
    g_prog = float(math.sqrt((g[-1, 0] - g[0, 0]) ** 2 + (g[-1, 1] - g[0, 1]) ** 2))
    progress_err = abs(p_prog - g_prog)
    p_turn = np.sign(p[-1, 1]) if abs(p[-1, 1]) > 1.0 else 0.0
    g_turn = np.sign(g[-1, 1]) if abs(g[-1, 1]) > 1.0 else 0.0
    turn_mismatch = bool(p_turn != 0 and g_turn != 0 and p_turn != g_turn)
    traj_bad = bool(fde > 4.0 or ade > 2.0 or heading_err > 0.6 or lateral_err > 3.0 or progress_err > 7.0 or turn_mismatch)
    return {
        "ade": ade, "fde": fde, "heading_abs_err": heading_err,
        "turn_direction_mismatch": turn_mismatch, "lateral_final_err": lateral_err,
        "progress_err": float(progress_err), "traj_bad": traj_bad,
    }


def intent_vs_gt_consistency(cot: str, meta: str, answer: str, gt: Optional[np.ndarray]) -> float:
    if gt is None or len(gt) < 2:
        return np.nan
    xs, ys, hs = gt[:, 0].tolist(), gt[:, 1].tolist(), gt[:, 2].tolist()
    return consistency_score(text_intent(cot, meta, answer), traj_features(xs, ys, hs))


def text_pred_consistency(cot: str, meta: str, answer: str, pred: Optional[np.ndarray]) -> float:
    if pred is None or len(pred) < 2:
        return np.nan
    xs, ys, hs = pred[:, 0].tolist(), pred[:, 1].tolist(), pred[:, 2].tolist()
    return consistency_score(text_intent(cot, meta, answer), traj_features(xs, ys, hs))


def draw_object_boxes(ax, objects: List[Dict[str, Any]]) -> None:
    for obj in objects:
        x, y = obj["x"], obj["y"]
        if abs(x) > 60 or abs(y) > 40:
            continue
        typ = obj.get("type", "")
        color = "tab:orange" if any(k in typ for k in ["vehicle", "car", "truck", "bus"]) else "tab:red" if any(k in typ for k in ["ped", "cycl", "bike", "bicycle"]) else "tab:gray"
        rect = Rectangle((x - obj.get("length", 4.5) / 2, y - obj.get("width", 1.8) / 2), obj.get("length", 4.5), obj.get("width", 1.8), color=color, alpha=0.22)
        ax.add_patch(rect)


def make_clip_gif(clip_rows: pd.DataFrame, gif_path: Path, duration_ms: int = 420, max_text_lines: int = 8) -> None:
    from PIL import Image

    frames = []
    # fixed bounds across clip based on pred/gt/object extent
    pts = [[0.0, 0.0], [55.0, 0.0], [-8.0, -22.0], [-8.0, 22.0]]
    for _, r in clip_rows.iterrows():
        for prefix in ["pred", "gt"]:
            xs = json.loads(r[f"{prefix}_traj_x"]) if safe_str(r.get(f"{prefix}_traj_x", "")) else []
            ys = json.loads(r[f"{prefix}_traj_y"]) if safe_str(r.get(f"{prefix}_traj_y", "")) else []
            pts.extend(list(zip(xs, ys)))
        try:
            for obj in json.loads(r.get("objects_json", "[]"))[:80]:
                pts.append([obj.get("x", 0.0), obj.get("y", 0.0)])
        except Exception:
            pass
    arr = np.asarray(pts, dtype=float)
    arr = arr[np.isfinite(arr).all(axis=1)]
    xmin, ymin = arr.min(axis=0)
    xmax, ymax = arr.max(axis=0)
    bounds = (min(-10, xmin - 6), max(35, xmax + 6), min(-22, ymin - 6), max(22, ymax + 6))

    total = len(clip_rows)
    for i, (_, r) in enumerate(clip_rows.iterrows(), start=1):
        fig = plt.figure(figsize=(16.5, 8.8), dpi=110)
        gs = fig.add_gridspec(2, 2, width_ratios=[1.1, 1.45], height_ratios=[0.78, 1.0], wspace=0.18, hspace=0.18)
        ax_cam = fig.add_subplot(gs[0, 0])
        ax_text = fig.add_subplot(gs[1, 0])
        ax = fig.add_subplot(gs[:, 1])

        fig.suptitle(
            f"clip #{int(r['clip_index']):04d} frame {i}/{total}  log={safe_str(r['log_name'])[:40]}  t={finite_float(r['time_s']):.2f}s\n"
            f"complexity={int(r['clip_complexity_level'])}({r['clip_complexity_label']})  n10={int(r['objects_within_10m'])}  "
            f"text-pred={finite_float(r['text_pred_consistency']):.2f} text-gt={finite_float(r['text_gt_consistency']):.2f} FDE={finite_float(r['fde']):.2f}",
            fontsize=12,
            fontweight="bold",
        )

        ax_cam.axis("off")
        ax_cam.set_title("continuous front camera cam_f0")
        image_path = safe_str(r.get("front_image_path", ""))
        if image_path and Path(image_path).exists():
            ax_cam.imshow(Image.open(image_path))
        else:
            ax_cam.text(0.5, 0.5, "image unavailable", ha="center", va="center")

        ax_text.axis("off")
        text = [
            "META_ACTION", wrap_text(r.get("meta_action", ""), width=58, max_lines=3), "",
            "COT", wrap_text(r.get("cot", ""), width=58, max_lines=max_text_lines), "",
            "FLAGS",
            f"  cot_traj_inconsistent: {bool(r.get('cot_traj_inconsistent', False))}",
            f"  cot_gt_inconsistent: {bool(r.get('cot_gt_inconsistent', False))}",
            f"  traj_bad: {bool(r.get('traj_bad', False))}",
            f"  turn_mismatch: {bool(r.get('turn_direction_mismatch', False))}",
        ]
        ax_text.text(0, 1, "\n".join(text), va="top", ha="left", fontsize=8.2, family="monospace")

        ax.set_title("per-timestep future: Alpamayo pred vs human/GT")
        ax.axhline(0, color="0.86", lw=1)
        ax.axvline(0, color="0.86", lw=1)
        ax.scatter([0], [0], c="black", marker="*", s=70, label="ego@now", zorder=5)
        try:
            draw_object_boxes(ax, json.loads(r.get("objects_json", "[]")))
        except Exception:
            pass
        for prefix, color, label in [("gt", "tab:green", "GT future"), ("pred", "tab:blue", "Alpamayo pred")]:
            xs = json.loads(r[f"{prefix}_traj_x"]) if safe_str(r.get(f"{prefix}_traj_x", "")) else []
            ys = json.loads(r[f"{prefix}_traj_y"]) if safe_str(r.get(f"{prefix}_traj_y", "")) else []
            if len(xs) >= 2 and len(ys) >= 2:
                ax.plot(xs, ys, "-o", color=color, lw=2.7, ms=4, label=label)
                ax.scatter([xs[-1]], [ys[-1]], c=color, marker="x", s=70, zorder=6)
        ax.set_xlim(bounds[0], bounds[1])
        ax.set_ylim(bounds[2], bounds[3])
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25)
        ax.set_xlabel("x forward / m")
        ax.set_ylabel("y left / m")
        ax.legend(loc="best", fontsize=8)

        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        frames.append(Image.open(buf).convert("RGB"))

    if frames:
        frames.extend([frames[-1]] * 2)
        gif_path.parent.mkdir(parents=True, exist_ok=True)
        frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=duration_ms, loop=0)


def cuda_memory_report(label: str) -> None:
    try:
        import torch
        if not torch.cuda.is_available():
            print(f"[continuous-mini][cuda] {label}: cuda not available", flush=True)
            return
        idx = torch.cuda.current_device()
        allocated = torch.cuda.memory_allocated(idx) / 1024**3
        reserved = torch.cuda.memory_reserved(idx) / 1024**3
        max_alloc = torch.cuda.max_memory_allocated(idx) / 1024**3
        free_b, total_b = torch.cuda.mem_get_info(idx)
        print(
            f"[continuous-mini][cuda] {label}: device={idx} name={torch.cuda.get_device_name(idx)} "
            f"allocated={allocated:.2f}GiB reserved={reserved:.2f}GiB max_alloc={max_alloc:.2f}GiB "
            f"free={free_b/1024**3:.2f}GiB total={total_b/1024**3:.2f}GiB",
            flush=True,
        )
    except Exception as e:
        print(f"[continuous-mini][cuda] {label}: failed to query cuda memory: {e}", flush=True)


def model_residency_report(model: Any) -> None:
    """Best-effort parameter count by device/dtype to catch accidental CPU/offload loading."""
    try:
        counts: Dict[str, int] = {}
        total = 0
        for p in model.parameters():
            n = int(p.numel())
            total += n
            key = f"{str(p.device)}|{str(p.dtype)}"
            counts[key] = counts.get(key, 0) + n
        print(f"[continuous-mini][model] parameters_total={total/1e9:.3f}B", flush=True)
        for key, n in sorted(counts.items(), key=lambda kv: kv[1], reverse=True):
            print(f"[continuous-mini][model]   {key}: {n/1e9:.3f}B params", flush=True)
    except Exception as e:
        print(f"[continuous-mini][model] parameter residency report failed: {e}", flush=True)


def analyze_and_write_outputs(frame_df: pd.DataFrame, clip_df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_df.to_csv(output_dir / "continuous_frame_results.csv", index=False)
    clip_df.to_csv(output_dir / "continuous_clip_summary.csv", index=False)

    # Complexity-level summary.
    if len(clip_df):
        summary = clip_df.groupby("complexity_level").agg(
            clips=("clip_index", "count"),
            frames=("num_frames", "sum"),
            cot_traj_inconsistency_rate=("cot_traj_inconsistency_rate", "mean"),
            cot_gt_inconsistency_rate=("cot_gt_inconsistency_rate", "mean"),
            traj_bad_rate=("traj_bad_rate", "mean"),
            turn_mismatch_rate=("turn_mismatch_rate", "mean"),
            mean_ade=("mean_ade", "mean"),
            mean_fde=("mean_fde", "mean"),
            avg_objects_within_10m=("avg_objects_within_10m", "mean"),
            peak_objects_within_10m=("peak_objects_within_10m", "mean"),
        ).reset_index()
        summary.to_csv(output_dir / "complexity_vs_cot_failure_summary.csv", index=False)
    else:
        pd.DataFrame().to_csv(output_dir / "complexity_vs_cot_failure_summary.csv", index=False)

    # Correlations for continuous features.
    numeric_cols = [
        "complexity_level", "peak_objects_within_10m", "avg_objects_within_10m", "min_object_dist", "peak_gt_curvature_proxy",
        "cot_traj_inconsistency_rate", "cot_gt_inconsistency_rate", "traj_bad_rate", "turn_mismatch_rate", "mean_ade", "mean_fde",
    ]
    existing = [c for c in numeric_cols if c in clip_df.columns]
    if len(clip_df) and len(existing) >= 2:
        clip_df[existing].corr(numeric_only=True).to_csv(output_dir / "clip_feature_failure_correlation.csv")

    # Human-readable report.
    lines = []
    lines.append("Alpamayo1.5 NAVSIM mini continuous-clip CoT/trajectory analysis")
    lines.append("============================================================")
    lines.append(f"clips: {len(clip_df)}")
    lines.append(f"frames: {len(frame_df)}")
    if len(clip_df):
        lines.append("")
        lines.append("By complexity level (0=simple, 1=medium, 2=complex):")
        for _, r in clip_df.groupby("complexity_level").agg(
            clips=("clip_index", "count"),
            cot_traj=("cot_traj_inconsistency_rate", "mean"),
            cot_gt=("cot_gt_inconsistency_rate", "mean"),
            bad=("traj_bad_rate", "mean"),
            fde=("mean_fde", "mean"),
            n10=("avg_objects_within_10m", "mean"),
        ).reset_index().iterrows():
            lines.append(
                f"  level {int(r['complexity_level'])}: clips={int(r['clips'])}, "
                f"cot-vs-pred incons={r['cot_traj']:.3f}, cot-vs-gt incons={r['cot_gt']:.3f}, "
                f"traj_bad={r['bad']:.3f}, mean_fde={r['fde']:.2f}, avg_n10={r['n10']:.2f}"
            )
        lines.append("")
        lines.append("Top clips by CoT-vs-pred inconsistency:")
        cols = ["clip_index", "log_name", "complexity_level", "cot_traj_inconsistency_rate", "cot_gt_inconsistency_rate", "traj_bad_rate", "mean_fde", "gif_path"]
        for _, r in clip_df.sort_values("cot_traj_inconsistency_rate", ascending=False).head(20).iterrows():
            lines.append(
                f"  #{int(r['clip_index']):04d} level={int(r['complexity_level'])} "
                f"cot_pred={r['cot_traj_inconsistency_rate']:.3f} cot_gt={r['cot_gt_inconsistency_rate']:.3f} "
                f"traj_bad={r['traj_bad_rate']:.3f} fde={r['mean_fde']:.2f} log={r['log_name']} gif={r.get('gif_path','')}"
            )
    (output_dir / "continuous_analysis_report.txt").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Run Alpamayo continuous sliding-window inference on all NAVSIM mini clips and make CoT GIFs")
    repo_dir = Path(__file__).resolve().parent
    data_root = infer_data_root(repo_dir)
    parser.add_argument("--data_root", default=str(data_root))
    parser.add_argument("--navsim_log_path", default=None)
    parser.add_argument("--sensor_blobs_path", default=None)
    parser.add_argument("--model_path", default="/data/mnt_m181/z59900495/workspace/model/Alpamayo-1.5-10B")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--gpu", default=None, help="physical GPU id; sets CUDA_VISIBLE_DEVICES before model load")
    parser.add_argument("--device", default="cuda:0", help="logical device inside process; keep cuda:0 when --gpu is set")
    parser.add_argument("--max_clips", type=int, default=0, help="0=all mini logs")
    parser.add_argument("--max_frames_per_clip", type=int, default=0, help="0=all sliding windows in each clip")
    parser.add_argument("--start_clip", type=int, default=0, help="skip first N sorted logs, useful for manual sharding")
    parser.add_argument("--clip_stride", type=int, default=1, help="process every Nth clip after start_clip, useful for manual sharding")
    parser.add_argument("--num_history_frames", type=int, default=4)
    parser.add_argument("--num_future_frames", type=int, default=10)
    parser.add_argument("--gif_every_n_clips", type=int, default=1, help="1=gif for every clip; 0 disables GIF writing")
    parser.add_argument("--gif_duration_ms", type=int, default=420)
    parser.add_argument("--save_every_frames", type=int, default=20, help="periodically flush frame CSV")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.98)
    parser.add_argument("--max_generation_length", type=int, default=256)
    args = parser.parse_args()

    if args.gpu is not None and str(args.gpu).strip() != "":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    data_root = Path(args.data_root).expanduser().resolve()
    navsim_log_path = Path(args.navsim_log_path).expanduser().resolve() if args.navsim_log_path else data_root / "navsim_logs" / "mini"
    sensor_blobs_path = Path(args.sensor_blobs_path).expanduser().resolve() if args.sensor_blobs_path else data_root / "sensor_blobs" / "mini"
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else data_root / "exp" / "mini_continuous_alpamayo" / datetime.now().strftime("run_%Y%m%d_%H%M%S")
    gifs_dir = output_dir / "gifs"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[continuous-mini] ===== resolved paths =====", flush=True)
    print(f"[continuous-mini] data_root: {data_root}", flush=True)
    print(f"[continuous-mini] navsim_log_path: {navsim_log_path}", flush=True)
    print(f"[continuous-mini] sensor_blobs_path: {sensor_blobs_path}", flush=True)
    print(f"[continuous-mini] model_path: {args.model_path}", flush=True)
    print(f"[continuous-mini] output_dir: {output_dir}", flush=True)
    print(f"[continuous-mini] CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', '')}", flush=True)

    from navsim.agents.alpamayo_agent.alpamayo_agent import AlpamayoAgent
    from navsim.common.dataclasses import SceneFilter
    from navsim.common.dataloader import SceneLoader
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

    sensor_config = AlpamayoAgent(model_path=args.model_path, device=args.device).get_sensor_config()
    # Need a real agent instance after sensor_config; avoid loading model twice by constructing final agent now.
    agent = AlpamayoAgent(
        trajectory_sampling=TrajectorySampling(time_horizon=4, interval_length=0.5),
        model_path=args.model_path,
        device=args.device,
        temperature=args.temperature,
        top_p=args.top_p,
        max_generation_length=args.max_generation_length,
    )
    agent.initialize()
    print("[continuous-mini] Alpamayo model loaded", flush=True)
    model_residency_report(getattr(agent, "_model", None))
    cuda_memory_report("after model load")

    log_files = sorted(navsim_log_path.glob("*.pkl"))
    selected_logs = log_files[args.start_clip::max(args.clip_stride, 1)]
    if args.max_clips > 0:
        selected_logs = selected_logs[:args.max_clips]
    if not selected_logs:
        raise RuntimeError(f"No mini log pkl files found/selected under {navsim_log_path}")
    print(f"[continuous-mini] selected clips: {len(selected_logs)} / total logs {len(log_files)}", flush=True)

    all_frame_rows: List[Dict[str, Any]] = []
    clip_rows: List[Dict[str, Any]] = []

    for clip_idx, log_path in enumerate(selected_logs, start=args.start_clip + 1):
        t_clip = time.time()
        log_name = log_path.stem
        print(f"[continuous-mini] clip #{clip_idx:04d}/{args.start_clip + len(selected_logs):04d}: {log_name}", flush=True)
        scene_filter = SceneFilter(
            num_history_frames=args.num_history_frames,
            num_future_frames=args.num_future_frames,
            frame_interval=1,
            log_names=[log_name],
            has_route=True,
        )
        scene_loader = SceneLoader(
            data_path=navsim_log_path,
            synthetic_sensor_path=sensor_blobs_path,
            original_sensor_path=sensor_blobs_path,
            synthetic_scenes_path=navsim_log_path,
            scene_filter=scene_filter,
            sensor_config=sensor_config,
        )
        tokens = list(scene_loader.tokens_stage_one)
        if args.max_frames_per_clip > 0:
            tokens = tokens[:args.max_frames_per_clip]
        if not tokens:
            print(f"[continuous-mini][warn] no valid sliding-window tokens for log {log_name}", flush=True)
            continue

        frame_rows: List[Dict[str, Any]] = []
        n10_values: List[int] = []
        min_dist_values: List[float] = []
        curv_values: List[float] = []

        for frame_i, token in enumerate(tokens):
            t0 = time.time()
            try:
                frame_list = scene_loader.scene_frames_dicts[token]
                agent_input = scene_loader.get_agent_input_from_token(token)
                pred_traj = agent.compute_trajectory(agent_input)
                pred = np.asarray(pred_traj.poses, dtype=float)
                gt = gt_trajectory_from_frame_list(frame_list, args.num_history_frames, n_poses=pred.shape[0] if pred.ndim == 2 else 8)
                current_raw = frame_list[args.num_history_frames - 1]
                objs = annotation_objects_from_frame(current_raw)
                n10 = int(sum(1 for o in objs if o.get("dist", 1e9) <= 10.0))
                min_dist = float(min([o.get("dist", np.inf) for o in objs], default=np.nan))
                n10_values.append(n10)
                min_dist_values.append(min_dist)
                if gt is not None:
                    gtf = traj_features(gt[:, 0], gt[:, 1], gt[:, 2])
                    curv_values.append(float(gtf.get("pred_curvature_proxy", np.nan)))
                else:
                    curv_values.append(np.nan)

                cot = getattr(agent, "_last_cot_text", "")
                meta = getattr(agent, "_last_meta_action_text", "")
                answer = getattr(agent, "_last_answer_text", "")
                text_pred = text_pred_consistency(cot, meta, answer, pred)
                text_gt = intent_vs_gt_consistency(cot, meta, answer, gt)
                pg = pred_vs_gt_metrics(pred, gt)
                pred_x, pred_y, pred_h = pred[:, 0].tolist(), pred[:, 1].tolist(), pred[:, 2].tolist()
                if gt is not None:
                    gt_x, gt_y, gt_h = gt[:, 0].tolist(), gt[:, 1].tolist(), gt[:, 2].tolist()
                else:
                    gt_x, gt_y, gt_h = [], [], []
                front_image_path = ""
                try:
                    cam_path = agent_input.cameras[-1].cam_f0.camera_path
                    if cam_path is not None:
                        front_image_path = str(sensor_blobs_path / cam_path)
                except Exception:
                    pass
                row = {
                    "clip_index": clip_idx,
                    "log_name": log_name,
                    "frame_index": frame_i,
                    "token": token,
                    "time_s": frame_timestamp_s(current_raw, frame_i),
                    "front_image_path": front_image_path,
                    "cot": cot,
                    "meta_action": meta,
                    "answer": answer,
                    "pred_traj_x": json.dumps(pred_x),
                    "pred_traj_y": json.dumps(pred_y),
                    "pred_traj_heading": json.dumps(pred_h),
                    "gt_traj_x": json.dumps(gt_x),
                    "gt_traj_y": json.dumps(gt_y),
                    "gt_traj_heading": json.dumps(gt_h),
                    "text_pred_consistency": text_pred,
                    "text_gt_consistency": text_gt,
                    "cot_traj_inconsistent": bool(np.isfinite(text_pred) and text_pred < 0.5),
                    "cot_gt_inconsistent": bool(np.isfinite(text_gt) and text_gt < 0.5),
                    "objects_within_10m": n10,
                    "n_objects": len(objs),
                    "min_object_dist": min_dist,
                    "objects_json": json.dumps(objs[:120]),
                    "runtime_s": time.time() - t0,
                    "valid": True,
                }
                row.update(pg)
                frame_rows.append(row)
                print(f"  [{frame_i+1:03d}/{len(tokens):03d}] token={token[:10]} n10={n10} text_pred={finite_float(text_pred):.2f} text_gt={finite_float(text_gt):.2f} fde={finite_float(pg['fde']):.2f} {time.time()-t0:.1f}s", flush=True)
                if frame_i == 0 or ((frame_i + 1) % 20 == 0):
                    cuda_memory_report(f"after frame {frame_i+1}")
            except Exception as e:
                print(f"  [{frame_i+1:03d}/{len(tokens):03d}] token={token[:10]} FAILED: {e}", flush=True)
                frame_rows.append({"clip_index": clip_idx, "log_name": log_name, "frame_index": frame_i, "token": token, "valid": False, "error": str(e)})

            if args.save_every_frames > 0 and len(frame_rows) % args.save_every_frames == 0:
                pd.DataFrame(all_frame_rows + frame_rows).to_csv(output_dir / "continuous_frame_results.partial.csv", index=False)

        valid_rows = pd.DataFrame([r for r in frame_rows if r.get("valid", False)])
        comp = complexity_from_objects_and_motion(n10_values, min_dist_values, curv_values)
        gif_path = ""
        if len(valid_rows) and args.gif_every_n_clips > 0 and ((len(clip_rows) % args.gif_every_n_clips) == 0):
            safe_log = re.sub(r"[^A-Za-z0-9_.-]+", "_", log_name)[:80]
            gif_file = gifs_dir / f"clip_{clip_idx:04d}_complexity{comp['complexity_level']}_{safe_log}.gif"
            tmp_rows = valid_rows.copy()
            for k, v in comp.items():
                tmp_rows[f"clip_{k}"] = v
            make_clip_gif(tmp_rows, gif_file, duration_ms=args.gif_duration_ms)
            gif_path = str(gif_file)

        if len(valid_rows):
            clip_summary = {
                "clip_index": clip_idx,
                "log_name": log_name,
                "num_frames": len(valid_rows),
                "num_failed_frames": len(frame_rows) - len(valid_rows),
                "gif_path": gif_path,
                "cot_traj_inconsistency_rate": float(valid_rows["cot_traj_inconsistent"].mean()),
                "cot_gt_inconsistency_rate": float(valid_rows["cot_gt_inconsistent"].mean()),
                "traj_bad_rate": float(valid_rows["traj_bad"].mean()),
                "turn_mismatch_rate": float(valid_rows["turn_direction_mismatch"].mean()),
                "mean_ade": float(valid_rows["ade"].mean()),
                "mean_fde": float(valid_rows["fde"].mean()),
                "runtime_s": time.time() - t_clip,
            }
            clip_summary.update(comp)
            clip_rows.append(clip_summary)
            for r in frame_rows:
                if r.get("valid", False):
                    for k, v in comp.items():
                        r[f"clip_{k}"] = v
                    r["gif_path"] = gif_path
        all_frame_rows.extend(frame_rows)
        analyze_and_write_outputs(pd.DataFrame(all_frame_rows), pd.DataFrame(clip_rows), output_dir)
        print(f"[continuous-mini] clip done #{clip_idx:04d}: complexity={comp['complexity_level']} frames={len(valid_rows)} gif={gif_path} elapsed={time.time()-t_clip:.1f}s", flush=True)

    analyze_and_write_outputs(pd.DataFrame(all_frame_rows), pd.DataFrame(clip_rows), output_dir)
    print("[continuous-mini] ===== DONE =====", flush=True)
    print(f"[continuous-mini] output_dir: {output_dir}", flush=True)
    print(f"[continuous-mini] frame_csv: {output_dir / 'continuous_frame_results.csv'}", flush=True)
    print(f"[continuous-mini] clip_csv: {output_dir / 'continuous_clip_summary.csv'}", flush=True)
    print(f"[continuous-mini] report: {output_dir / 'continuous_analysis_report.txt'}", flush=True)
    print(f"[continuous-mini] gifs_dir: {gifs_dir}", flush=True)


if __name__ == "__main__":
    main()

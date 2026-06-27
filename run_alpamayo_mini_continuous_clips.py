"""
Run Alpamayo1.5 on NAVSIM mini as true continuous sliding-window clips and make CoT MP4 videos.

This script is for qualitative/diagnostic analysis, not official NAVSIM PDM scoring.
It differs from run_alpamayo_eval.py:
- uses SceneFilter(frame_interval=1) so consecutive video frames are consecutive NAVSIM frames;
- runs Alpamayo inference at every timestep in each mini log/clip;
- writes one MP4 video per clip with changing front camera + per-timestep CoT + pred-vs-GT future;
- computes clip-level scene complexity and CoT/trajectory inconsistency statistics.

Example:
  cd /data/mnt_m181/zhn/navsim_0616 && python run_alpamayo_mini_continuous_clips.py \
    --gpu 1 --max_clips 0 --max_frames_per_clip 0 --video_every_n_clips 1 \
    --output_dir /data/mnt_m181/zhn/navsim_0616/navsim/navsim_dataset/exp/mini_continuous_alpamayo

Notes:
- --gpu sets CUDA_VISIBLE_DEVICES before loading torch/model; inside process --device defaults to cuda:0.
- max_clips=0 and max_frames_per_clip=0 mean all available mini clips / sliding windows.
"""

import argparse
import concurrent.futures
import json
import math
import os
import re
import shutil
import subprocess
import sys
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
            "traj_bad_reasons": "GT/pred missing",
            "turn_mismatch_reason": "GT/pred missing",
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

    reasons: List[str] = []
    if fde > 4.0:
        reasons.append(f"FDE {fde:.2f}>4.0m")
    if ade > 2.0:
        reasons.append(f"ADE {ade:.2f}>2.0m")
    if heading_err > 0.6:
        reasons.append(f"heading_err {heading_err:.2f}>0.60rad")
    if lateral_err > 3.0:
        reasons.append(f"lateral_final_err {lateral_err:.2f}>3.0m")
    if progress_err > 7.0:
        reasons.append(f"progress_err {progress_err:.2f}>7.0m (pred={p_prog:.1f}, gt={g_prog:.1f})")
    if turn_mismatch:
        reasons.append("turn direction mismatch")
    traj_bad = bool(reasons)

    p_turn_label = "left" if p_turn > 0 else "right" if p_turn < 0 else "straight/weak"
    g_turn_label = "left" if g_turn > 0 else "right" if g_turn < 0 else "straight/weak"
    turn_reason = f"pred={p_turn_label} final_y={p[-1,1]:.2f}; gt={g_turn_label} final_y={g[-1,1]:.2f}; trigger=both |y|>1m and signs differ" if turn_mismatch else f"no trigger: pred_y={p[-1,1]:.2f}, gt_y={g[-1,1]:.2f}"

    return {
        "ade": ade, "fde": fde, "heading_abs_err": heading_err,
        "turn_direction_mismatch": turn_mismatch, "lateral_final_err": lateral_err,
        "progress_err": float(progress_err), "traj_bad": traj_bad,
        "traj_bad_reasons": "; ".join(reasons) if reasons else "none",
        "turn_mismatch_reason": turn_reason,
    }


def _intent_summary(intent: Dict[str, Any]) -> str:
    names = [
        k.replace("intent_", "")
        for k, v in intent.items()
        if k.startswith("intent_") and bool(v)
    ]
    return ",".join(names) if names else "none"


def consistency_score_with_reasons(cot: str, meta: str, answer: str, traj: Optional[np.ndarray], traj_name: str) -> Tuple[float, str]:
    if traj is None or len(traj) < 2:
        return np.nan, f"{traj_name} trajectory missing"
    xs, ys, hs = traj[:, 0].tolist(), traj[:, 1].tolist(), traj[:, 2].tolist()
    intent = text_intent(cot, meta, answer)
    tf = traj_features(xs, ys, hs)
    checks: List[Tuple[bool, str]] = []
    progress = float(tf["pred_progress"])
    lat = float(tf["pred_lateral_range"])
    heading = float(tf["pred_final_heading"])
    heading_change = float(tf["pred_heading_change"])
    final_y = float(ys[-1])

    if intent["intent_stop"]:
        checks.append((progress < 4.0, f"stop expects progress<4.0m; got {progress:.1f}m"))
    if intent["intent_slow"]:
        checks.append((progress < 14.0, f"slow/yield/brake expects progress<14.0m; got {progress:.1f}m"))
    if intent["intent_keep"] and not (intent["intent_left"] or intent["intent_right"]):
        checks.append((lat < 1.5 and heading_change < 0.35, f"keep/straight expects lat<1.5m and heading_change<0.35rad; got lat={lat:.1f}, dhead={heading_change:.2f}"))
    if intent["intent_left"] and not intent["intent_right"]:
        checks.append((heading > 0.05 or final_y > 0.5, f"left expects heading>0.05rad or final_y>0.5m; got heading={heading:.2f}, final_y={final_y:.1f}"))
    if intent["intent_right"] and not intent["intent_left"]:
        checks.append((heading < -0.05 or final_y < -0.5, f"right expects heading<-0.05rad or final_y<-0.5m; got heading={heading:.2f}, final_y={final_y:.1f}"))
    if intent["intent_overtake_or_nudge"]:
        checks.append((lat > 0.4, f"overtake/nudge/avoid expects lateral_range>0.4m; got {lat:.1f}m"))

    if not checks:
        return np.nan, f"no text intent trigger; extracted_intents={_intent_summary(intent)}"
    passed = [ok for ok, _ in checks]
    failed_reasons = [reason for ok, reason in checks if not ok]
    score = float(np.mean([1.0 if ok else 0.0 for ok in passed]))
    if failed_reasons:
        return score, "; ".join(failed_reasons)
    return score, f"all checks passed; extracted_intents={_intent_summary(intent)}"


def intent_vs_gt_consistency(cot: str, meta: str, answer: str, gt: Optional[np.ndarray]) -> float:
    return consistency_score_with_reasons(cot, meta, answer, gt, "GT")[0]


def text_pred_consistency(cot: str, meta: str, answer: str, pred: Optional[np.ndarray]) -> float:
    return consistency_score_with_reasons(cot, meta, answer, pred, "pred")[0]


def draw_object_boxes(ax, objects: List[Dict[str, Any]]) -> None:
    for obj in objects:
        x, y = obj["x"], obj["y"]
        if abs(x) > 60 or abs(y) > 40:
            continue
        typ = obj.get("type", "")
        color = "tab:orange" if any(k in typ for k in ["vehicle", "car", "truck", "bus"]) else "tab:red" if any(k in typ for k in ["ped", "cycl", "bike", "bicycle"]) else "tab:gray"
        rect = Rectangle((x - obj.get("length", 4.5) / 2, y - obj.get("width", 1.8) / 2), obj.get("length", 4.5), obj.get("width", 1.8), color=color, alpha=0.22)
        ax.add_patch(rect)


def write_mp4_with_ffmpeg(frames: List[np.ndarray], video_path: Path, fps: float) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        try:
            import imageio_ffmpeg
            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            ffmpeg = None
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to write MP4 videos. Install ffmpeg or imageio-ffmpeg, then rerun.")
    if not frames:
        return
    h, w = frames[0].shape[:2]
    # H.264 yuv420p requires even dimensions. Crop one pixel if needed.
    h2, w2 = h - (h % 2), w - (w % 2)
    frames = [np.ascontiguousarray(f[:h2, :w2, :3], dtype=np.uint8) for f in frames]
    video_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg, "-y", "-loglevel", "error",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-pix_fmt", "rgb24", "-s", f"{w2}x{h2}", "-r", f"{fps:.3f}", "-i", "-",
        "-an", "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(video_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    assert proc.stdin is not None
    for frame in frames:
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    err = proc.stderr.read() if proc.stderr is not None else b""
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"ffmpeg failed to write {video_path}: {err.decode('utf-8', errors='replace')}")


def make_clip_video(clip_rows: pd.DataFrame, video_path: Path, duration_ms: int = 420, max_text_lines: int = 8) -> None:
    from PIL import Image

    frames: List[np.ndarray] = []
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
        reason_lines = [
            f"  cot_traj_inconsistent: {bool(r.get('cot_traj_inconsistent', False))}",
            "    " + wrap_text(r.get("cot_traj_reason", ""), width=70, max_lines=2),
            f"  cot_gt_inconsistent: {bool(r.get('cot_gt_inconsistent', False))}",
            "    " + wrap_text(r.get("cot_gt_reason", ""), width=70, max_lines=2),
            f"  traj_bad: {bool(r.get('traj_bad', False))}",
            "    " + wrap_text(r.get("traj_bad_reasons", ""), width=70, max_lines=2),
            f"  turn_mismatch: {bool(r.get('turn_direction_mismatch', False))}",
            "    " + wrap_text(r.get("turn_mismatch_reason", ""), width=70, max_lines=2),
        ]
        text = [
            "META_ACTION", wrap_text(r.get("meta_action", ""), width=58, max_lines=2), "",
            "COT", wrap_text(r.get("cot", ""), width=58, max_lines=max_text_lines), "",
            "FLAGS + TRIGGER REASONS",
            *reason_lines,
        ]
        ax_text.text(0, 1, "\n".join(text), va="top", ha="left", fontsize=7.2, family="monospace")

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
        fig.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)
        frames.append(np.asarray(Image.open(buf).convert("RGB")))

    if frames:
        frames.extend([frames[-1]] * 2)
        fps = max(1.0, 1000.0 / max(float(duration_ms), 1.0))
        write_mp4_with_ffmpeg(frames, video_path, fps=fps)


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
        cols = ["clip_index", "log_name", "complexity_level", "cot_traj_inconsistency_rate", "cot_gt_inconsistency_rate", "traj_bad_rate", "mean_fde", "video_path"]
        for _, r in clip_df.sort_values("cot_traj_inconsistency_rate", ascending=False).head(20).iterrows():
            lines.append(
                f"  #{int(r['clip_index']):04d} level={int(r['complexity_level'])} "
                f"cot_pred={r['cot_traj_inconsistency_rate']:.3f} cot_gt={r['cot_gt_inconsistency_rate']:.3f} "
                f"traj_bad={r['traj_bad_rate']:.3f} fde={r['mean_fde']:.2f} log={r['log_name']} video={r.get('video_path', r.get('gif_path',''))}"
            )
    (output_dir / "continuous_analysis_report.txt").write_text("\n".join(lines), encoding="utf-8")


def parse_gpu_spec(spec: Optional[str]) -> List[str]:
    """Parse GPU spec like '0', '0-3', '0,1,2,3', or '0 1' (if shell quoted)."""
    if spec is None:
        return []
    s = str(spec).strip()
    if not s:
        return []
    parts = re.split(r"[,+\s]+", s)
    gpus: List[str] = []
    for part in parts:
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            start, end = int(a), int(b)
            step = 1 if end >= start else -1
            gpus.extend([str(i) for i in range(start, end + step, step)])
        else:
            gpus.append(str(int(part)))
    out: List[str] = []
    seen = set()
    for g in gpus:
        if g not in seen:
            seen.add(g)
            out.append(g)
    return out


def merge_shard_outputs(output_dir: Path, shard_dirs: List[Path]) -> None:
    """Merge shard CSVs/reports and collect MP4 videos into one user-facing output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    clips = []
    videos_dir = output_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    for shard_dir in shard_dirs:
        f_csv = shard_dir / "continuous_frame_results.csv"
        c_csv = shard_dir / "continuous_clip_summary.csv"
        if f_csv.exists():
            frames.append(pd.read_csv(f_csv))
        if c_csv.exists():
            cdf = pd.read_csv(c_csv)
            new_paths = []
            path_col = "video_path" if "video_path" in cdf.columns else "gif_path"
            for p in cdf.get(path_col, pd.Series([""] * len(cdf))).fillna("").astype(str).tolist():
                if p and Path(p).exists():
                    dst = videos_dir / Path(p).name
                    if not dst.exists():
                        shutil.copy2(p, dst)
                    new_paths.append(str(dst))
                else:
                    new_paths.append(p)
            cdf["video_path"] = new_paths
            cdf["gif_path"] = new_paths  # backward-compatible alias for older analysis code
            clips.append(cdf)
        for video in (shard_dir / "videos").glob("*.mp4"):
            dst = videos_dir / video.name
            if not dst.exists():
                shutil.copy2(video, dst)

    frame_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    clip_df = pd.concat(clips, ignore_index=True) if clips else pd.DataFrame()
    if len(frame_df) and "clip_index" in frame_df.columns:
        frame_df = frame_df.sort_values(["clip_index", "frame_index"], kind="stable")
    if len(clip_df) and "clip_index" in clip_df.columns:
        clip_df = clip_df.sort_values("clip_index", kind="stable")
    if len(frame_df) and len(clip_df) and "gif_path" in frame_df.columns and "gif_path" in clip_df.columns:
        gif_map = dict(zip(clip_df["clip_index"], clip_df["gif_path"])); frame_df["gif_path"] = frame_df["clip_index"].map(gif_map).fillna(frame_df["gif_path"])
    analyze_and_write_outputs(frame_df, clip_df, output_dir)
    print(f"[continuous-mini][launcher] merged frame rows={len(frame_df)} clip rows={len(clip_df)}", flush=True)
    print(f"[continuous-mini][launcher] merged output_dir: {output_dir}", flush=True)
    print(f"[continuous-mini][launcher] merged videos_dir: {videos_dir}", flush=True)


def run_multi_gpu_launcher(args, repo_dir: Path, gpu_ids: List[str]) -> None:
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else infer_data_root(repo_dir) / "exp" / "mini_continuous_alpamayo" / datetime.now().strftime("run_%Y%m%d_%H%M%S")
    logs_dir = output_dir / "logs"
    shards_dir = output_dir / "shards"
    logs_dir.mkdir(parents=True, exist_ok=True)
    shards_dir.mkdir(parents=True, exist_ok=True)
    print(f"[continuous-mini][launcher] GPUs: {gpu_ids}", flush=True)
    print(f"[continuous-mini][launcher] output_dir: {output_dir}", flush=True)

    procs = []
    shard_dirs: List[Path] = []
    base_argv = sys.argv[1:]

    def strip_arg(argv: List[str], value_names: List[str], flag_names: List[str]) -> List[str]:
        out = []
        skip = False
        for item in argv:
            if skip:
                skip = False
                continue
            if item in value_names:
                skip = True
                continue
            if item in flag_names:
                continue
            if any(item.startswith(n + "=") for n in value_names + flag_names):
                continue
            out.append(item)
        return out

    common = strip_arg(base_argv, ["--gpu", "--output_dir", "--start_clip", "--clip_stride"], ["--worker"])
    for rank, gpu in enumerate(gpu_ids):
        shard_dir = shards_dir / f"gpu{gpu}_rank{rank}"
        shard_dirs.append(shard_dir)
        cmd = [
            sys.executable, str(Path(__file__).resolve()),
            *common,
            "--worker",
            "--gpu", str(gpu),
            "--start_clip", str(rank),
            "--clip_stride", str(len(gpu_ids)),
            "--output_dir", str(shard_dir),
        ]
        log_path = logs_dir / f"gpu{gpu}_rank{rank}.log"
        print(f"[continuous-mini][launcher] start gpu={gpu} rank={rank}: {' '.join(cmd)}", flush=True)
        f = open(log_path, "w", encoding="utf-8")
        proc = subprocess.Popen(cmd, cwd=str(repo_dir), stdout=f, stderr=subprocess.STDOUT)
        procs.append((proc, f, log_path, gpu, rank))
        stagger_sec = max(float(getattr(args, "launcher_stagger_sec", 0.0)), 0.0)
        if stagger_sec > 0 and rank < len(gpu_ids) - 1:
            print(f"[continuous-mini][launcher] stagger next worker by {stagger_sec:.0f}s to avoid concurrent 10B model load", flush=True)
            time.sleep(stagger_sec)

    failed = []
    remaining = {(gpu, rank): (proc, f, log_path) for proc, f, log_path, gpu, rank in procs}
    log_offsets: Dict[Tuple[str, int], int] = {(gpu, rank): 0 for _, _, _, gpu, rank in procs}
    poll_sec = max(float(getattr(args, "launcher_poll_sec", 20.0)), 1.0)
    print(f"[continuous-mini][launcher] child logs are under: {logs_dir}", flush=True)
    print(f"[continuous-mini][launcher] polling child logs every {poll_sec:.0f}s; model loading can take several minutes", flush=True)
    while remaining:
        for key, (proc, f, log_path) in list(remaining.items()):
            gpu, rank = key
            try:
                if log_path.exists():
                    with open(log_path, "r", encoding="utf-8", errors="replace") as lf:
                        lf.seek(log_offsets.get(key, 0))
                        new_text = lf.read()
                        log_offsets[key] = lf.tell()
                    if new_text.strip():
                        lines = new_text.splitlines()
                        for line in lines[-40:]:
                            print(f"[gpu{gpu}/rank{rank}] {line}", flush=True)
            except Exception as e:
                print(f"[continuous-mini][launcher][warn] cannot read log {log_path}: {e}", flush=True)

            rc = proc.poll()
            if rc is not None:
                # Drain remaining log text once more before reporting done.
                try:
                    if log_path.exists():
                        with open(log_path, "r", encoding="utf-8", errors="replace") as lf:
                            lf.seek(log_offsets.get(key, 0))
                            tail_text = lf.read()
                            log_offsets[key] = lf.tell()
                        if tail_text.strip():
                            for line in tail_text.splitlines()[-80:]:
                                print(f"[gpu{gpu}/rank{rank}] {line}", flush=True)
                except Exception:
                    pass
                f.close()
                if rc != 0:
                    failed.append((gpu, rank, rc, log_path))
                print(f"[continuous-mini][launcher] done gpu={gpu} rank={rank} exit={rc} log={log_path}", flush=True)
                del remaining[key]
        if remaining:
            alive = ", ".join([f"gpu{gpu}/rank{rank}" for gpu, rank in remaining.keys()])
            print(f"[continuous-mini][launcher] still running: {alive}", flush=True)
            time.sleep(poll_sec)

    if failed:
        for gpu, rank, rc, log_path in failed:
            print(f"[continuous-mini][launcher][FAILED] gpu={gpu} rank={rank} exit={rc} tail {log_path}", flush=True)
            try:
                lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
                print("\n".join(lines[-120:]), flush=True)
            except Exception as e:
                print(f"cannot read log: {e}", flush=True)
        raise SystemExit(1)

    merge_shard_outputs(output_dir, shard_dirs)


def main():
    parser = argparse.ArgumentParser(description="Run Alpamayo continuous sliding-window inference on all NAVSIM mini clips and make CoT MP4 videos")
    repo_dir = Path(__file__).resolve().parent
    data_root = infer_data_root(repo_dir)
    parser.add_argument("--data_root", default=str(data_root))
    parser.add_argument("--navsim_log_path", default=None)
    parser.add_argument("--sensor_blobs_path", default=None)
    parser.add_argument("--model_path", default="/data/mnt_m181/z59900495/workspace/model/Alpamayo-1.5-10B")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--gpu", default=None, help="GPU id/spec. Single worker: '1'. Launcher: '0-3' or '0,1,2,3' starts one process per GPU")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--device", default="cuda:0", help="logical device inside process; keep cuda:0 when --gpu is set")
    parser.add_argument("--max_clips", type=int, default=0, help="0=all mini logs")
    parser.add_argument("--max_frames_per_clip", type=int, default=0, help="0=all sliding windows in each clip")
    parser.add_argument("--start_clip", type=int, default=0, help="skip first N sorted logs, useful for manual sharding")
    parser.add_argument("--clip_stride", type=int, default=1, help="process every Nth clip after start_clip, useful for manual sharding")
    parser.add_argument("--num_history_frames", type=int, default=4)
    parser.add_argument("--num_future_frames", type=int, default=10)
    parser.add_argument("--video_every_n_clips", type=int, default=None, help="1=MP4 for every clip; 0 disables video writing")
    parser.add_argument("--gif_every_n_clips", type=int, default=None, help="Deprecated alias for --video_every_n_clips")
    parser.add_argument("--video_duration_ms", type=int, default=None, help="Per-frame video duration in ms")
    parser.add_argument("--gif_duration_ms", type=int, default=420, help="Deprecated alias for --video_duration_ms")
    parser.add_argument("--save_every_frames", type=int, default=20, help="periodically flush frame CSV")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.98)
    parser.add_argument("--max_generation_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8, help="number of frames per batch inference call; 1 disables batching")
    parser.add_argument("--launcher_poll_sec", type=float, default=20.0, help="launcher mode: seconds between child-log progress polls")
    parser.add_argument("--launcher_stagger_sec", type=float, default=0.0, help="launcher mode: seconds to wait between starting GPU workers; use 60-90 if concurrent model load hangs")
    args = parser.parse_args()

    gpu_ids = parse_gpu_spec(args.gpu)
    if len(gpu_ids) > 1 and not args.worker:
        run_multi_gpu_launcher(args, repo_dir, gpu_ids)
        return

    if len(gpu_ids) == 1:
        args.gpu = gpu_ids[0]
    if args.gpu is not None and str(args.gpu).strip() != "":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    data_root = Path(args.data_root).expanduser().resolve()
    navsim_log_path = Path(args.navsim_log_path).expanduser().resolve() if args.navsim_log_path else data_root / "navsim_logs" / "mini"
    sensor_blobs_path = Path(args.sensor_blobs_path).expanduser().resolve() if args.sensor_blobs_path else data_root / "sensor_blobs" / "mini"
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else data_root / "exp" / "mini_continuous_alpamayo" / datetime.now().strftime("run_%Y%m%d_%H%M%S")
    videos_dir = output_dir / "videos"
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.video_every_n_clips is None:
        args.video_every_n_clips = args.gif_every_n_clips if args.gif_every_n_clips is not None else 1
    if args.video_duration_ms is None:
        args.video_duration_ms = args.gif_duration_ms

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
    stride = max(args.clip_stride, 1)
    start = max(args.start_clip, 0)
    selected_pairs = [(i, p) for i, p in enumerate(log_files) if i >= start and ((i - start) % stride == 0)]
    if args.max_clips > 0:
        selected_pairs = selected_pairs[:args.max_clips]
    if not selected_pairs:
        raise RuntimeError(f"No mini log pkl files found/selected under {navsim_log_path}")
    print(f"[continuous-mini] selected clips: {len(selected_pairs)} / total logs {len(log_files)}", flush=True)

    all_frame_rows: List[Dict[str, Any]] = []
    clip_rows: List[Dict[str, Any]] = []

    for orig_idx, log_path in selected_pairs:
        clip_idx = orig_idx + 1
        t_clip = time.time()
        log_name = log_path.stem
        print(f"[continuous-mini] clip #{clip_idx:04d}/{len(log_files):04d}: {log_name}", flush=True)
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

        batch_size = max(args.batch_size, 1)
        for batch_start in range(0, len(tokens), batch_size):
            batch_tokens = tokens[batch_start:batch_start + batch_size]
            batch_actual = len(batch_tokens)
            t_batch = time.time()

            # Phase 1: load data for all frames in batch (CPU, overlaps with GPU)
            batch_meta = []
            for bidx, token in enumerate(batch_tokens):
                frame_i = batch_start + bidx
                try:
                    frame_list = scene_loader.scene_frames_dicts[token]
                    agent_input = scene_loader.get_agent_input_from_token(token)
                    batch_meta.append((frame_i, token, frame_list, agent_input))
                except Exception as e:
                    print(f"  [{frame_i+1:03d}/{len(tokens):03d}] token={token[:10]} data load FAILED: {e}", flush=True)
                    batch_meta.append((frame_i, token, None, None))

            valid_inputs = [m[3] for m in batch_meta if m[3] is not None]
            results: List[Optional[dict]] = [None] * batch_actual

            # Phase 2: batched GPU inference
            if valid_inputs:
                try:
                    batch_results = agent.compute_trajectory_batch(valid_inputs)
                    vi = 0
                    for bidx in range(batch_actual):
                        if batch_meta[bidx][3] is not None:
                            results[bidx] = batch_results[vi]
                            vi += 1
                except Exception as e:
                    print(f"  batch inference failed ({batch_actual} frames): {e}; falling back to single-frame", flush=True)
                    for bidx in range(batch_actual):
                        if batch_meta[bidx][3] is not None:
                            try:
                                traj = agent.compute_trajectory(batch_meta[bidx][3])
                                results[bidx] = {
                                    "trajectory": traj,
                                    "cot": getattr(agent, "_last_cot_text", ""),
                                    "meta_action": getattr(agent, "_last_meta_action_text", ""),
                                    "answer": getattr(agent, "_last_answer_text", ""),
                                }
                            except Exception as e2:
                                print(f"  single-frame fallback also failed: {e2}", flush=True)

            batch_elapsed = time.time() - t_batch
            per_frame = batch_elapsed / batch_actual if batch_actual else 0
            print(f"  batch [{batch_start+1:03d}-{batch_start+batch_actual:03d}/{len(tokens):03d}] {batch_actual} frames in {batch_elapsed:.1f}s ({per_frame:.1f}s/frame)", flush=True)

            # Phase 3: post-process each frame's result
            for bidx in range(batch_actual):
                frame_i, token, frame_list, agent_input = batch_meta[bidx]
                t0 = time.time()
                result = results[bidx]
                if result is None or frame_list is None:
                    frame_rows.append({"clip_index": clip_idx, "log_name": log_name, "frame_index": frame_i, "token": token, "valid": False, "error": "inference failed"})
                    continue
                try:
                    pred_traj = result["trajectory"]
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

                    cot = result["cot"]
                    meta = result["meta_action"]
                    answer = result["answer"]
                    text_pred, text_pred_reason = consistency_score_with_reasons(cot, meta, answer, pred, "pred")
                    text_gt, text_gt_reason = consistency_score_with_reasons(cot, meta, answer, gt, "GT")
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
                        "cot_traj_reason": text_pred_reason,
                        "cot_gt_reason": text_gt_reason,
                        "objects_within_10m": n10,
                        "n_objects": len(objs),
                        "min_object_dist": min_dist,
                        "objects_json": json.dumps(objs[:120]),
                        "runtime_s": time.time() - t0,
                        "valid": True,
                    }
                    row.update(pg)
                    frame_rows.append(row)
                    print(f"  [{frame_i+1:03d}/{len(tokens):03d}] token={token[:10]} n10={n10} text_pred={finite_float(text_pred):.2f} text_gt={finite_float(text_gt):.2f} fde={finite_float(pg['fde']):.2f}", flush=True)
                    if frame_i == 0 or ((frame_i + 1) % 20 == 0):
                        cuda_memory_report(f"after frame {frame_i+1}")
                except Exception as e:
                    print(f"  [{frame_i+1:03d}/{len(tokens):03d}] token={token[:10]} post-process FAILED: {e}", flush=True)
                    frame_rows.append({"clip_index": clip_idx, "log_name": log_name, "frame_index": frame_i, "token": token, "valid": False, "error": str(e)})

            if args.save_every_frames > 0 and len(frame_rows) % args.save_every_frames == 0:
                pd.DataFrame(all_frame_rows + frame_rows).to_csv(output_dir / "continuous_frame_results.partial.csv", index=False)

        valid_rows = pd.DataFrame([r for r in frame_rows if r.get("valid", False)])
        comp = complexity_from_objects_and_motion(n10_values, min_dist_values, curv_values)
        video_path = ""
        if len(valid_rows) and args.video_every_n_clips > 0 and ((len(clip_rows) % args.video_every_n_clips) == 0):
            safe_log = re.sub(r"[^A-Za-z0-9_.-]+", "_", log_name)[:80]
            video_file = videos_dir / f"clip_{clip_idx:04d}_complexity{comp['complexity_level']}_{safe_log}.mp4"
            tmp_rows = valid_rows.copy()
            for k, v in comp.items():
                tmp_rows[f"clip_{k}"] = v
            make_clip_video(tmp_rows, video_file, duration_ms=args.video_duration_ms)
            video_path = str(video_file)

        if len(valid_rows):
            clip_summary = {
                "clip_index": clip_idx,
                "log_name": log_name,
                "num_frames": len(valid_rows),
                "num_failed_frames": len(frame_rows) - len(valid_rows),
                "video_path": video_path,
                "gif_path": video_path,  # backward-compatible alias
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
                    r["video_path"] = video_path
                    r["gif_path"] = video_path  # backward-compatible alias
        all_frame_rows.extend(frame_rows)
        analyze_and_write_outputs(pd.DataFrame(all_frame_rows), pd.DataFrame(clip_rows), output_dir)
        print(f"[continuous-mini] clip done #{clip_idx:04d}: complexity={comp['complexity_level']} frames={len(valid_rows)} video={video_path} elapsed={time.time()-t_clip:.1f}s", flush=True)

    analyze_and_write_outputs(pd.DataFrame(all_frame_rows), pd.DataFrame(clip_rows), output_dir)
    print("[continuous-mini] ===== DONE =====", flush=True)
    print(f"[continuous-mini] output_dir: {output_dir}", flush=True)
    print(f"[continuous-mini] frame_csv: {output_dir / 'continuous_frame_results.csv'}", flush=True)
    print(f"[continuous-mini] clip_csv: {output_dir / 'continuous_clip_summary.csv'}", flush=True)
    print(f"[continuous-mini] report: {output_dir / 'continuous_analysis_report.txt'}", flush=True)
    print(f"[continuous-mini] videos_dir: {videos_dir}", flush=True)


if __name__ == "__main__":
    main()

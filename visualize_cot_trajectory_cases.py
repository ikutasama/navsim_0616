"""
Visualize Alpamayo1.5 NavSim CoT, predicted trajectory, GT trajectory, and quantitative metrics.

Typical usage:
  python visualize_cot_trajectory_cases.py \
    --scores_csv /path/to/alpamayo_pdm_scores_merged.csv \
    --analysis_csv /path/to/cot_failure_pattern_enriched.csv \
    --metric_cache_path $OPENSCENE_DATA_ROOT/metric_cache \
    --output_dir $OPENSCENE_DATA_ROOT/exp/visualizations/weak_cases \
    --select weakest --top_k 30 --make_case_gifs

Important: each NavSim token is one planning scene/current timestep, not a full continuous rollout.
The per-case GIF animates the 4-second predicted/GT future trajectory being revealed step-by-step.
Different scenes are NOT stitched into one GIF.

If --analysis_csv is omitted, the script still plots CoT + pred trajectory + PDM metrics from scores_csv.
If --metric_cache_path is omitted or unavailable, GT trajectory / object context are skipped.
"""

import argparse
import json
import math
import os
import re
import textwrap
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Force non-interactive backend before importing pyplot.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle


_SCENE_LOADER = None


PDM_METRICS = [
    "pdm_score",
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "driving_direction_compliance",
    "traffic_light_compliance",
    "ego_progress",
    "time_to_collision_within_bound",
    "lane_keeping",
    "history_comfort",
]


def safe_str(x: Any) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return str(x)


def parse_json_list(x: Any) -> List[float]:
    if isinstance(x, list):
        return [float(v) for v in x]
    if x is None:
        return []
    try:
        if pd.isna(x):
            return []
    except Exception:
        pass
    s = str(x).strip()
    if not s:
        return []
    try:
        obj = json.loads(s)
        if isinstance(obj, list):
            return [float(v) for v in obj]
    except Exception:
        pass
    try:
        return [float(v) for v in s.strip("[]").split(",") if v.strip()]
    except Exception:
        return []


def trajectory_features(xs: List[float], ys: List[float], hs: List[float]) -> Dict[str, float]:
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    hs = np.asarray(hs, dtype=float)
    if len(xs) < 2 or len(ys) < 2:
        return {
            "progress": np.nan,
            "lateral_range": np.nan,
            "heading_change": np.nan,
            "arc_length": np.nan,
            "final_x": np.nan,
            "final_y": np.nan,
        }
    hs_unwrap = np.unwrap(hs) if len(hs) == len(xs) else np.zeros_like(xs)
    dx = xs[-1] - xs[0]
    dy = ys[-1] - ys[0]
    arc = float(np.sqrt(np.diff(xs) ** 2 + np.diff(ys) ** 2).sum())
    return {
        "progress": float(math.sqrt(dx * dx + dy * dy)),
        "lateral_range": float(np.max(ys) - np.min(ys)),
        "heading_change": float(abs(hs_unwrap[-1] - hs_unwrap[0])) if len(hs_unwrap) else np.nan,
        "arc_length": arc,
        "final_x": float(xs[-1]),
        "final_y": float(ys[-1]),
    }


def load_metric_cache(metric_cache_path: Optional[str], token: str):
    if not metric_cache_path:
        return None
    try:
        from navsim.common.dataloader import MetricCacheLoader
        loader = MetricCacheLoader(Path(metric_cache_path))
        return loader.get_from_token(token)
    except Exception as e:
        print(f"[warn] metric_cache load failed for token={token}: {e}", flush=True)
        return None


def infer_data_root_from_scores(scores_csv: Optional[str] = None) -> Optional[Path]:
    env_root = os.environ.get("OPENSCENE_DATA_ROOT", "").strip()
    if env_root:
        return Path(env_root).expanduser().resolve()
    if scores_csv:
        p = Path(scores_csv).expanduser().resolve()
        for parent in [p] + list(p.parents):
            if (parent / "navsim_logs").exists() and (parent / "sensor_blobs").exists():
                return parent
    cwd = Path.cwd()
    for cand in [cwd / "navsim" / "navsim_dataset", cwd / "navsim_dataset", cwd / "dataset"]:
        if (cand / "navsim_logs").exists() and (cand / "sensor_blobs").exists():
            return cand.resolve()
    return None


def get_front_camera_image(token: str, navsim_log_path: Optional[str], sensor_blobs_path: Optional[str]):
    """Load current front camera image for a token using SceneLoader.

    Returns RGB uint8 image or None. Uses one cached SceneLoader per process.
    """
    global _SCENE_LOADER
    if not navsim_log_path or not sensor_blobs_path:
        return None
    try:
        from navsim.common.dataclasses import SceneFilter, SensorConfig
        from navsim.common.dataloader import SceneLoader
        if _SCENE_LOADER is None:
            sensor_config = SensorConfig(
                cam_f0=[3],
                cam_l0=False, cam_l1=False, cam_l2=False,
                cam_r0=False, cam_r1=False, cam_r2=False,
                cam_b0=False,
                lidar_pc=False,
            )
            scene_filter = SceneFilter(num_history_frames=4, num_future_frames=10)
            _SCENE_LOADER = SceneLoader(
                data_path=Path(navsim_log_path),
                synthetic_sensor_path=Path(sensor_blobs_path),
                original_sensor_path=Path(sensor_blobs_path),
                synthetic_scenes_path=Path(navsim_log_path),
                scene_filter=scene_filter,
                sensor_config=sensor_config,
            )
        if token not in _SCENE_LOADER.tokens:
            return None
        agent_input = _SCENE_LOADER.get_agent_input_from_token(token)
        if not agent_input.cameras:
            return None
        img = agent_input.cameras[-1].cam_f0.image
        return img
    except Exception as e:
        print(f"[warn] front camera load failed for token={token}: {e}", flush=True)
        return None


def extract_gt_traj_and_objects(metric_cache) -> Tuple[Optional[np.ndarray], List[Dict[str, Any]], List[Dict[str, Any]]]:
    gt = None
    objects: List[Dict[str, Any]] = []
    agent_future_tracks: List[Dict[str, Any]] = []
    if metric_cache is None:
        return gt, objects, agent_future_tracks

    try:
        if getattr(metric_cache, "human_trajectory", None) is not None:
            poses = metric_cache.human_trajectory.poses
            if poses is not None and len(poses):
                gt = np.asarray(poses, dtype=float)
    except Exception:
        gt = None

    det = None
    try:
        if getattr(metric_cache, "current_tracked_objects", None):
            det = metric_cache.current_tracked_objects[0]
        elif getattr(metric_cache, "observation", None) is not None:
            tracks = metric_cache.observation.detections_tracks
            if len(tracks):
                det = tracks[0]
    except Exception:
        det = None

    try:
        if det is not None:
            for obj in det.tracked_objects.tracked_objects:
                typ = str(obj.tracked_object_type).split(".")[-1].lower()
                box = obj.box
                center = box.center
                heading = float(getattr(center, "heading", 0.0))
                x = float(getattr(center, "x", 0.0))
                y = float(getattr(center, "y", 0.0))
                width = float(getattr(box, "width", 1.8))
                length = float(getattr(box, "length", 4.5))
                token = safe_str(getattr(getattr(obj, "metadata", None), "track_token", getattr(obj, "track_token", "")))
                objects.append({"type": typ, "x": x, "y": y, "heading": heading, "width": width, "length": length, "track_token": token})
    except Exception:
        objects = []

    # Background-agent future trajectories from current + future detection tracks.
    # These are the tracks PDM uses to judge interactions/collisions, so plotting them is
    # critical for checking whether CoT mentions the relevant actors and intentions.
    try:
        det_frames = []
        if getattr(metric_cache, "current_tracked_objects", None):
            det_frames.extend(metric_cache.current_tracked_objects)
        if getattr(metric_cache, "future_tracked_objects", None):
            det_frames.extend(metric_cache.future_tracked_objects)
        tracks: Dict[str, Dict[str, Any]] = {}
        for t_idx, frame_det in enumerate(det_frames):
            for obj in frame_det.tracked_objects.tracked_objects:
                typ = str(obj.tracked_object_type).split(".")[-1].lower()
                box = obj.box
                center = box.center
                x = float(getattr(center, "x", 0.0))
                y = float(getattr(center, "y", 0.0))
                if not (abs(x) <= 80 and abs(y) <= 50):
                    continue
                token = safe_str(getattr(getattr(obj, "metadata", None), "track_token", getattr(obj, "track_token", "")))
                if not token:
                    token = f"obj_{len(tracks)}_{typ}"
                rec = tracks.setdefault(token, {"track_token": token, "type": typ, "points": []})
                rec["points"].append((t_idx, x, y))
        for rec in tracks.values():
            pts = sorted(rec["points"], key=lambda p: p[0])
            if len(pts) >= 2:
                arr = np.asarray([[p[1], p[2]] for p in pts], dtype=float)
                min_dist = float(np.min(np.sqrt(arr[:, 0] ** 2 + arr[:, 1] ** 2)))
                rec["xy"] = arr
                rec["min_dist"] = min_dist
                agent_future_tracks.append(rec)
        agent_future_tracks = sorted(agent_future_tracks, key=lambda r: r.get("min_dist", 1e9))[:25]
    except Exception:
        agent_future_tracks = []

    return gt, objects, agent_future_tracks


def object_patch(obj: Dict[str, Any]) -> Rectangle:
    # Approximate oriented boxes by axis-aligned boxes for robustness/readability.
    x = obj.get("x", 0.0)
    y = obj.get("y", 0.0)
    length = obj.get("length", 4.5)
    width = obj.get("width", 1.8)
    return Rectangle((x - length / 2.0, y - width / 2.0), length, width)


def wrap_text(text: str, width: int = 82, max_lines: int = 18) -> str:
    text = re.sub(r"\s+", " ", safe_str(text)).strip()
    if not text:
        return "<empty>"
    lines = textwrap.wrap(text, width=width)
    if len(lines) > max_lines:
        lines = lines[:max_lines] + ["... <truncated>"]
    return "\n".join(lines)


def bool_mark(x: Any) -> str:
    try:
        return "Y" if bool(x) else "N"
    except Exception:
        return "?"


def build_case_title(row: pd.Series, idx: int) -> str:
    token = safe_str(row.get("token", ""))
    pdm = row.get("pdm_score", np.nan)
    cons = row.get("cot_traj_consistency", np.nan)
    weak = row.get("weak_case_score", np.nan)
    return f"#{idx:03d} token={token[:18]}  PDM={pdm:.3f}  CoT-Traj={cons:.3f}  weak={weak:.3f}"


def plot_case(row: pd.Series, idx: int, output_path: Path, metric_cache_path: Optional[str] = None, navsim_log_path: Optional[str] = None, sensor_blobs_path: Optional[str] = None):
    token = safe_str(row.get("token", ""))
    xs = parse_json_list(row.get("pred_traj_x", "[]"))
    ys = parse_json_list(row.get("pred_traj_y", "[]"))
    hs = parse_json_list(row.get("pred_traj_heading", "[]"))
    pred = np.asarray(list(zip(xs, ys, hs if len(hs) == len(xs) else [0.0] * len(xs))), dtype=float) if xs and ys else None
    pred_feat = trajectory_features(xs, ys, hs)

    metric_cache = load_metric_cache(metric_cache_path, token) if metric_cache_path else None
    gt, objects, agent_tracks = extract_gt_traj_and_objects(metric_cache)
    front_img = get_front_camera_image(token, navsim_log_path, sensor_blobs_path)

    fig = plt.figure(figsize=(20, 11), dpi=150)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.25, 1.15, 1.35], height_ratios=[1.05, 0.85], wspace=0.25, hspace=0.28)
    ax_text = fig.add_subplot(gs[:, 0])
    ax_cam = fig.add_subplot(gs[0, 1])
    ax_bev = fig.add_subplot(gs[0, 2])
    ax_bar = fig.add_subplot(gs[1, 1])
    ax_feat = fig.add_subplot(gs[1, 2])

    fig.suptitle(build_case_title(row, idx), fontsize=14, fontweight="bold")

    # Text panel.
    ax_text.axis("off")
    meta = safe_str(row.get("meta_action", ""))
    cot = safe_str(row.get("cot", ""))
    answer = safe_str(row.get("answer", ""))
    text_blocks = []
    text_blocks.append("METRICS")
    for m in ["pdm_score", "cot_traj_consistency", "weak_case_score", "complexity_score", "n_objects", "n_vehicles", "n_pedestrians"]:
        if m in row.index:
            v = row.get(m)
            if isinstance(v, (float, int, np.floating, np.integer)):
                text_blocks.append(f"  {m}: {v:.4f}")
            else:
                text_blocks.append(f"  {m}: {safe_str(v)}")
    text_blocks.append("")
    text_blocks.append("COT PATTERN FLAGS")
    for m in ["mentions_object_any", "mentions_vehicle_obj", "mentions_vulnerable_obj", "mentions_spatial_relation", "mentions_other_agent_intent", "mentions_risk_or_conflict", "mentions_traffic_rule", "has_causal_connector"]:
        if m in row.index:
            text_blocks.append(f"  {m}: {bool_mark(row.get(m))}")
    if "cot_len_words_raw" in row.index:
        text_blocks.append(f"  cot_len_words: {row.get('cot_len_words_raw')}")
    elif "cot_len_words" in row.index:
        text_blocks.append(f"  cot_len_words: {row.get('cot_len_words')}")
    text_blocks.append("")
    text_blocks.append("PRED TRAJ FEATURES")
    for k, v in pred_feat.items():
        text_blocks.append(f"  pred_{k}: {v:.3f}" if np.isfinite(v) else f"  pred_{k}: nan")
    text_blocks.append("")
    text_blocks.append("META_ACTION")
    text_blocks.append(wrap_text(meta, width=72, max_lines=4))
    text_blocks.append("")
    text_blocks.append("COT")
    text_blocks.append(wrap_text(cot, width=72, max_lines=14))
    if answer:
        text_blocks.append("")
        text_blocks.append("ANSWER")
        text_blocks.append(wrap_text(answer, width=72, max_lines=4))
    ax_text.text(0.0, 1.0, "\n".join(text_blocks), va="top", ha="left", fontsize=8.5, family="monospace")

    # Front camera panel.
    ax_cam.set_title("Current front camera (cam_f0)")
    ax_cam.axis("off")
    if front_img is not None:
        ax_cam.imshow(front_img)
    else:
        ax_cam.text(0.5, 0.5, "Front camera unavailable\n(pass --navsim_log_path and --sensor_blobs_path,\nor set OPENSCENE_DATA_ROOT)", ha="center", va="center", fontsize=10)

    # BEV plot.
    ax_bev.set_title("BEV trajectory: predicted vs GT (ego frame)")
    ax_bev.axhline(0, color="0.85", lw=1)
    ax_bev.axvline(0, color="0.85", lw=1)
    ax_bev.scatter([0], [0], c="black", s=55, marker="*", label="ego@now", zorder=5)

    draw_static_objects(ax_bev, objects)
    draw_agent_future_tracks(ax_bev, agent_tracks, reveal_fraction=1.0, alpha=0.65)

    if gt is not None and len(gt) >= 2:
        ax_bev.plot(gt[:, 0], gt[:, 1], "-o", color="tab:green", lw=2.5, ms=4, label="GT/human")
        ax_bev.scatter([gt[-1, 0]], [gt[-1, 1]], c="tab:green", s=55, marker="x")
    if pred is not None and len(pred) >= 2:
        ax_bev.plot(pred[:, 0], pred[:, 1], "-o", color="tab:blue", lw=2.5, ms=4, label="Alpamayo pred")
        ax_bev.scatter([pred[-1, 0]], [pred[-1, 1]], c="tab:blue", s=55, marker="x")

    xmin, xmax, ymin, ymax = compute_plot_bounds(pred, gt, objects, agent_tracks)
    ax_bev.set_xlim(xmin, xmax)
    ax_bev.set_ylim(ymin, ymax)
    ax_bev.set_aspect("equal", adjustable="box")
    ax_bev.set_xlabel("x forward / meters")
    ax_bev.set_ylabel("y left / meters")
    ax_bev.grid(True, alpha=0.25)
    ax_bev.legend(loc="best", fontsize=9)

    # PDM bars.
    metrics = [m for m in PDM_METRICS if m in row.index]
    vals = []
    labels = []
    for m in metrics:
        try:
            vals.append(float(row.get(m)))
            labels.append(m.replace("_", "\n"))
        except Exception:
            pass
    ax_bar.set_title("PDM / submetrics")
    if vals:
        colors = ["tab:red" if v < 0.5 else "tab:green" for v in vals]
        ax_bar.bar(range(len(vals)), vals, color=colors, alpha=0.85)
        ax_bar.set_ylim(0, 1.05)
        ax_bar.set_xticks(range(len(vals)))
        ax_bar.set_xticklabels(labels, rotation=55, ha="right", fontsize=7)
        ax_bar.grid(axis="y", alpha=0.25)
    else:
        ax_bar.text(0.5, 0.5, "No PDM metrics", ha="center", va="center")
        ax_bar.axis("off")

    # Feature comparison bars.
    feat_names = []
    feat_vals = []
    for col, label in [
        ("cot_specificity_score", "CoT\nspecific"),
        ("object_aware_score", "object\naware"),
        ("cot_traj_consistency", "CoT-traj\nconsistent"),
        ("complexity_score", "complexity\n(raw/4)"),
    ]:
        if col in row.index:
            try:
                val = float(row.get(col))
                if col == "complexity_score":
                    val = min(val / 4.0, 1.0)
                feat_names.append(label)
                feat_vals.append(val)
            except Exception:
                pass
    ax_feat.set_title("CoT quality proxies")
    if feat_vals:
        ax_feat.bar(range(len(feat_vals)), feat_vals, color="tab:purple", alpha=0.78)
        ax_feat.set_ylim(0, 1.05)
        ax_feat.set_xticks(range(len(feat_vals)))
        ax_feat.set_xticklabels(feat_names, fontsize=8)
        ax_feat.grid(axis="y", alpha=0.25)
    else:
        ax_feat.text(0.5, 0.5, "No CoT feature columns\n(run analyze_cot_failure_patterns.py first)", ha="center", va="center")
        ax_feat.axis("off")

    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def load_and_merge(scores_csv: str, analysis_csv: Optional[str]) -> pd.DataFrame:
    scores = pd.read_csv(scores_csv)
    if "valid" in scores.columns:
        scores = scores[scores["valid"] == True].copy()
    if analysis_csv:
        analysis = pd.read_csv(analysis_csv)
        if "valid" in analysis.columns:
            analysis = analysis[analysis["valid"] == True].copy()
        # Prefer analysis columns, but keep trajectory columns from scores if missing.
        if "token" not in analysis.columns or "token" not in scores.columns:
            raise ValueError("Both scores_csv and analysis_csv must contain token column")
        missing_cols = [c for c in scores.columns if c not in analysis.columns]
        df = analysis.merge(scores[["token"] + missing_cols], on="token", how="left")
    else:
        df = scores
    return df


def add_basic_ranking_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "pdm_score" in df.columns:
        df["pdm_score"] = pd.to_numeric(df["pdm_score"], errors="coerce")
    if "cot_traj_consistency" in df.columns:
        df["cot_traj_consistency"] = pd.to_numeric(df["cot_traj_consistency"], errors="coerce")
    else:
        df["cot_traj_consistency"] = np.nan
    if "weak_case_score" not in df.columns:
        pdm_bad = 1.0 - df.get("pdm_score", pd.Series(0.5, index=df.index)).clip(0, 1).fillna(0.5)
        cons_bad = 1.0 - df["cot_traj_consistency"].clip(0, 1).fillna(0.5)
        df["weak_case_score"] = 0.6 * pdm_bad + 0.4 * cons_bad
    return df


def select_cases(df: pd.DataFrame, mode: str, top_k: int, tokens: Optional[List[str]]) -> pd.DataFrame:
    if tokens:
        token_set = set(tokens)
        return df[df["token"].astype(str).isin(token_set)].copy()
    if mode == "all":
        return df.head(top_k).copy()
    if mode == "low_pdm":
        return df.sort_values("pdm_score", ascending=True).head(top_k).copy()
    if mode == "inconsistent":
        return df.sort_values("cot_traj_consistency", ascending=True, na_position="last").head(top_k).copy()
    # Default: weak score high first.
    return df.sort_values("weak_case_score", ascending=False).head(top_k).copy()


def compute_plot_bounds(pred: Optional[np.ndarray], gt: Optional[np.ndarray], objects: List[Dict[str, Any]], agent_tracks: Optional[List[Dict[str, Any]]] = None) -> Tuple[float, float, float, float]:
    pts = [[0.0, 0.0]]
    if pred is not None and len(pred):
        pts.extend(pred[:, :2].tolist())
    if gt is not None and len(gt):
        pts.extend(gt[:, :2].tolist())
    for obj in objects[:80]:
        pts.append([obj.get("x", 0.0), obj.get("y", 0.0)])
    for tr in (agent_tracks or [])[:25]:
        xy = tr.get("xy")
        if xy is not None:
            pts.extend(np.asarray(xy, dtype=float).tolist())
    pts_arr = np.asarray(pts, dtype=float)
    finite = pts_arr[np.isfinite(pts_arr).all(axis=1)]
    if len(finite):
        xmin, ymin = finite.min(axis=0)
        xmax, ymax = finite.max(axis=0)
        return min(-10, xmin - 8), max(30, xmax + 8), min(-20, ymin - 8), max(20, ymax + 8)
    return -10, 30, -20, 20


def draw_static_objects(ax, objects: List[Dict[str, Any]]):
    if not objects:
        return
    patches = []
    colors = []
    for obj in objects:
        x, y = obj.get("x", 0.0), obj.get("y", 0.0)
        if abs(x) <= 80 and abs(y) <= 50:
            patches.append(object_patch(obj))
            typ = obj.get("type", "")
            if "ped" in typ or "bicycle" in typ or "cycl" in typ:
                colors.append("tab:red")
            elif "vehicle" in typ:
                colors.append("tab:orange")
            else:
                colors.append("tab:gray")
    if patches:
        pc = PatchCollection(patches, facecolor=colors, edgecolor="none", alpha=0.25, label="objects@now")
        ax.add_collection(pc)


def draw_agent_future_tracks(ax, agent_tracks: List[Dict[str, Any]], reveal_fraction: float = 1.0, alpha: float = 0.65):
    """Draw other agents' future tracks in BEV."""
    if not agent_tracks:
        return
    for i, tr in enumerate(agent_tracks):
        xy = tr.get("xy")
        if xy is None or len(xy) < 2:
            continue
        xy = np.asarray(xy, dtype=float)
        k = max(2, int(math.ceil(len(xy) * max(0.0, min(1.0, reveal_fraction)))))
        xy_show = xy[:k]
        typ = tr.get("type", "")
        if "ped" in typ or "bicycle" in typ or "cycl" in typ:
            color = "tab:red"
            label = "other vulnerable future" if i == 0 else None
        elif "vehicle" in typ:
            color = "tab:orange"
            label = "other vehicle future" if i == 0 else None
        else:
            color = "tab:gray"
            label = "other object future" if i == 0 else None
        ax.plot(xy[:, 0], xy[:, 1], "--", color=color, lw=1.0, alpha=0.18)
        ax.plot(xy_show[:, 0], xy_show[:, 1], "-", color=color, lw=1.7, alpha=alpha, label=label)
        ax.scatter([xy_show[-1, 0]], [xy_show[-1, 1]], c=color, s=18, alpha=alpha, zorder=4)


def make_case_gif(row: pd.Series, idx: int, gif_path: Path, metric_cache_path: Optional[str] = None, duration_ms: int = 450, navsim_log_path: Optional[str] = None, sensor_blobs_path: Optional[str] = None):
    """Create one GIF for one NavSim planning scene.

    NavSim evaluation output has one 4s future trajectory per token. This GIF does not stitch
    different tokens; it reveals this single token's predicted and GT future trajectory step-by-step.
    """
    try:
        from PIL import Image
    except Exception as e:
        print(f"[warn] PIL/Pillow is required for GIF creation: {e}", flush=True)
        return

    token = safe_str(row.get("token", ""))
    xs = parse_json_list(row.get("pred_traj_x", "[]"))
    ys = parse_json_list(row.get("pred_traj_y", "[]"))
    hs = parse_json_list(row.get("pred_traj_heading", "[]"))
    pred = np.asarray(list(zip(xs, ys, hs if len(hs) == len(xs) else [0.0] * len(xs))), dtype=float) if xs and ys else None

    metric_cache = load_metric_cache(metric_cache_path, token) if metric_cache_path else None
    gt, objects, agent_tracks = extract_gt_traj_and_objects(metric_cache)
    front_img = get_front_camera_image(token, navsim_log_path, sensor_blobs_path)
    n_pred = len(pred) if pred is not None else 0
    n_gt = len(gt) if gt is not None else 0
    n_steps = max(n_pred, n_gt, 1)
    xmin, xmax, ymin, ymax = compute_plot_bounds(pred, gt, objects, agent_tracks)

    cot = wrap_text(safe_str(row.get("cot", "")), width=58, max_lines=11)
    meta = wrap_text(safe_str(row.get("meta_action", "")), width=58, max_lines=3)
    answer = wrap_text(safe_str(row.get("answer", "")), width=58, max_lines=3)
    pdm = row.get("pdm_score", np.nan)
    cons = row.get("cot_traj_consistency", np.nan)
    weak = row.get("weak_case_score", np.nan)

    frames = []
    for step in range(n_steps + 1):
        fig = plt.figure(figsize=(15.5, 8.5), dpi=120)
        gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.45], height_ratios=[0.78, 1.0], wspace=0.18, hspace=0.18)
        ax_cam = fig.add_subplot(gs[0, 0])
        ax_text = fig.add_subplot(gs[1, 0])
        ax = fig.add_subplot(gs[:, 1])
        t_sec = min(step, 8) * 0.5
        fig.suptitle(f"Scene #{idx:03d} token={token[:18]}  t=+{t_sec:.1f}s  PDM={pdm:.3f}  CoT-Traj={cons:.3f}", fontsize=13, fontweight="bold")

        ax_cam.set_title("Current front camera (cam_f0)")
        ax_cam.axis("off")
        if front_img is not None:
            ax_cam.imshow(front_img)
        else:
            ax_cam.text(0.5, 0.5, "front camera unavailable", ha="center", va="center")

        ax_text.axis("off")
        text_lines = [
            "META_ACTION", meta, "",
            "COT", cot, "",
        ]
        if answer and answer != "<empty>":
            text_lines += ["ANSWER", answer, ""]
        text_lines += [
            "QUANT", f"  pdm_score: {pdm:.4f}", f"  cot_traj_consistency: {cons:.4f}", f"  weak_case_score: {weak:.4f}",
        ]
        for flag in ["mentions_object_any", "mentions_other_agent_intent", "mentions_risk_or_conflict", "mentions_spatial_relation"]:
            if flag in row.index:
                text_lines.append(f"  {flag}: {bool_mark(row.get(flag))}")
        ax_text.text(0.0, 1.0, "\n".join(text_lines), va="top", ha="left", fontsize=8.5, family="monospace")

        ax.set_title("Single-scene 4s future trajectory reveal")
        ax.axhline(0, color="0.85", lw=1)
        ax.axvline(0, color="0.85", lw=1)
        ax.scatter([0], [0], c="black", s=70, marker="*", label="ego@now", zorder=5)
        draw_static_objects(ax, objects)
        reveal_fraction = step / max(n_steps, 1)
        draw_agent_future_tracks(ax, agent_tracks, reveal_fraction=reveal_fraction, alpha=0.75)

        if gt is not None and len(gt) >= 2:
            ax.plot(gt[:, 0], gt[:, 1], "--", color="tab:green", lw=1.5, alpha=0.25, label="GT full")
            k = min(step, len(gt))
            if k > 0:
                ax.plot(gt[:k, 0], gt[:k, 1], "-o", color="tab:green", lw=3, ms=5, label="GT revealed")
                ax.scatter([gt[k - 1, 0]], [gt[k - 1, 1]], c="tab:green", s=80, marker="x", zorder=6)
        if pred is not None and len(pred) >= 2:
            ax.plot(pred[:, 0], pred[:, 1], "--", color="tab:blue", lw=1.5, alpha=0.25, label="pred full")
            k = min(step, len(pred))
            if k > 0:
                ax.plot(pred[:k, 0], pred[:k, 1], "-o", color="tab:blue", lw=3, ms=5, label="pred revealed")
                ax.scatter([pred[k - 1, 0]], [pred[k - 1, 1]], c="tab:blue", s=80, marker="x", zorder=6)

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x forward / meters")
        ax.set_ylabel("y left / meters")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=8)

        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
        frames.append(img)

    # Hold the final frame a bit longer.
    if frames:
        frames.extend([frames[-1]] * 3)
        frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=duration_ms, loop=0)


def main():
    parser = argparse.ArgumentParser(description="Visualize CoT + predicted trajectory + PDM metrics for Alpamayo NavSim cases")
    parser.add_argument("--scores_csv", required=True, help="alpamayo_pdm_scores_merged.csv")
    parser.add_argument("--analysis_csv", default=None, help="optional cot_failure_pattern_enriched.csv or cot_consistency_analysis.csv")
    parser.add_argument("--metric_cache_path", default=None, help="optional metric_cache directory for GT trajectory, objects, and other-agent futures")
    parser.add_argument("--navsim_log_path", default=None, help="optional navsim_logs/mini path for loading front camera; auto-inferred if omitted")
    parser.add_argument("--sensor_blobs_path", default=None, help="optional sensor_blobs/mini path for loading front camera; auto-inferred if omitted")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--select", choices=["weakest", "low_pdm", "inconsistent", "all"], default="weakest")
    parser.add_argument("--top_k", type=int, default=30)
    parser.add_argument("--tokens", nargs="*", default=None, help="specific tokens to visualize; overrides --select")
    parser.add_argument("--make_case_gifs", action="store_true", help="Create one trajectory-reveal GIF per selected scene/token")
    parser.add_argument("--make_gif", action="store_true", help="Backward-compatible alias for --make_case_gifs; creates per-scene GIFs, not a multi-scene montage")
    parser.add_argument("--gif_duration_ms", type=int, default=450)
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
    chosen = select_cases(df, args.select, args.top_k, args.tokens)

    if len(chosen) == 0:
        raise RuntimeError("No cases selected. Check CSV paths / token filters.")

    # Save selected rows for traceability.
    selected_csv = output_dir / "selected_visualized_cases.csv"
    chosen.to_csv(selected_csv, index=False)

    image_paths: List[Path] = []
    gif_paths: List[Path] = []
    make_case_gifs_flag = args.make_case_gifs or args.make_gif
    for i, (_, row) in enumerate(chosen.iterrows(), start=1):
        token = safe_str(row.get("token", f"case{i}"))
        safe_token = re.sub(r"[^A-Za-z0-9_.-]+", "_", token)[:80]
        out_path = output_dir / f"case_{i:03d}_{safe_token}.png"
        print(f"[viz] {i}/{len(chosen)} token={token} -> {out_path}", flush=True)
        plot_case(row, i, out_path, metric_cache_path=args.metric_cache_path, navsim_log_path=args.navsim_log_path, sensor_blobs_path=args.sensor_blobs_path)
        image_paths.append(out_path)
        if make_case_gifs_flag:
            gif_path = output_dir / f"case_{i:03d}_{safe_token}.gif"
            print(f"[viz] {i}/{len(chosen)} token={token} -> {gif_path}", flush=True)
            make_case_gif(row, i, gif_path, metric_cache_path=args.metric_cache_path, duration_ms=args.gif_duration_ms, navsim_log_path=args.navsim_log_path, sensor_blobs_path=args.sensor_blobs_path)
            gif_paths.append(gif_path)

    print(f"[viz] selected_csv: {selected_csv}", flush=True)
    print(f"[viz] png_dir: {output_dir}", flush=True)
    if gif_paths:
        print(f"[viz] per-scene_gifs: {len(gif_paths)} files in {output_dir}", flush=True)


if __name__ == "__main__":
    main()

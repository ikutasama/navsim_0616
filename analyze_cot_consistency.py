"""
分析 Alpamayo1.5 CoT / meta_action 与预测轨迹、场景复杂度、PDM指标之间的关系。

输入：run_alpamayo_eval.py 产生的合并CSV（包含 cot/meta_action/answer/pred_traj_*）和 metric_cache。
输出：
  - cot_consistency_analysis.csv：每个token的场景特征、轨迹特征、CoT意图、一致性分数、PDM指标
  - cot_consistency_summary.csv：复杂度分桶下的一致性/PDM统计
  - 控制台打印：哪些复杂度特征与低PDM、低一致性最相关

Usage:
  python analyze_cot_consistency.py \
    --scores_csv $OPENSCENE_DATA_ROOT/exp/eval_results/alpamayo_pdm_scores_merged.csv \
    --metric_cache_path $OPENSCENE_DATA_ROOT/metric_cache \
    --output_dir $OPENSCENE_DATA_ROOT/exp/cot_analysis
"""

import argparse
import json
import logging
import math
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _json_list(x):
    if isinstance(x, list):
        return x
    if pd.isna(x):
        return []
    try:
        return json.loads(x)
    except Exception:
        try:
            return [float(v) for v in str(x).strip("[]").split(",") if v.strip()]
        except Exception:
            return []


def traj_features(xs, ys, hs):
    xs, ys, hs = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float), np.asarray(hs, dtype=float)
    if len(xs) < 2:
        return dict(pred_progress=0.0, pred_lateral_range=0.0, pred_heading_change=0.0,
                    pred_final_heading=0.0, pred_curvature_proxy=0.0)
    dx = xs[-1] - xs[0]
    dy = ys[-1] - ys[0]
    progress = float(math.sqrt(dx * dx + dy * dy))
    lateral_range = float(np.max(np.abs(ys - ys[0])))
    hs_unwrap = np.unwrap(hs)
    heading_change = float(abs(hs_unwrap[-1] - hs_unwrap[0]))
    arc = np.sqrt(np.diff(xs) ** 2 + np.diff(ys) ** 2).sum()
    curvature_proxy = float(heading_change / max(arc, 1e-3))
    return dict(
        pred_progress=progress,
        pred_lateral_range=lateral_range,
        pred_heading_change=heading_change,
        pred_final_heading=float(hs_unwrap[-1]),
        pred_curvature_proxy=curvature_proxy,
    )


def text_intent(cot, meta_action, answer):
    text = f"{cot or ''} {meta_action or ''} {answer or ''}".lower()
    # English keywords from Alpamayo style; keep broad to tolerate phrasing variation.
    flags = {
        "intent_stop": bool(re.search(r"\b(stop|stopping|halt|stationary|red light)\b", text)),
        "intent_slow": bool(re.search(r"\b(slow|decelerate|brake|cautious|yield|give way)\b", text)),
        "intent_keep": bool(re.search(r"\b(keep|maintain|straight|continue|lane keeping|lane-keeping)\b", text)),
        "intent_left": bool(re.search(r"\b(left|turn left|nudge left|change lane left)\b", text)),
        "intent_right": bool(re.search(r"\b(right|turn right|nudge right|change lane right)\b", text)),
        "intent_overtake_or_nudge": bool(re.search(r"\b(overtake|nudge|avoid|go around|bypass)\b", text)),
        "mentions_pedestrian": bool(re.search(r"\b(pedestrian|walker|crosswalk|cyclist|bicycle|bike)\b", text)),
        "mentions_vehicle": bool(re.search(r"\b(vehicle|car|truck|bus|lead vehicle|traffic)\b", text)),
        "mentions_light": bool(re.search(r"\b(red light|green light|traffic light|signal)\b", text)),
    }
    flags["cot_len_chars"] = len(text)
    flags["cot_len_words"] = len(text.split())
    return flags


def consistency_score(intent, tf):
    """Rule-based CoT-trajectory consistency score in [0,1].

    This is not a semantic judge; it is a cheap proxy:
    - stop/slow should not have large progress
    - left/right should match heading/lateral sign
    - keep/straight should have small heading/lateral change
    """
    checks = []

    progress = tf["pred_progress"]
    lat = tf["pred_lateral_range"]
    heading = tf["pred_final_heading"]
    heading_change = tf["pred_heading_change"]

    if intent["intent_stop"]:
        checks.append(1.0 if progress < 4.0 else 0.0)
    if intent["intent_slow"]:
        checks.append(1.0 if progress < 14.0 else 0.0)
    if intent["intent_keep"] and not (intent["intent_left"] or intent["intent_right"]):
        checks.append(1.0 if lat < 1.5 and heading_change < 0.35 else 0.0)
    if intent["intent_left"] and not intent["intent_right"]:
        checks.append(1.0 if heading > 0.05 or lat > 0.5 else 0.0)
    if intent["intent_right"] and not intent["intent_left"]:
        checks.append(1.0 if heading < -0.05 or lat > 0.5 else 0.0)
    if intent["intent_overtake_or_nudge"]:
        checks.append(1.0 if lat > 0.4 else 0.0)

    if not checks:
        return np.nan
    return float(np.mean(checks))


def extract_scene_features(metric_cache):
    """Robust scene complexity features from MetricCache."""
    feats = {}
    objects = []
    try:
        # Prefer current tracked objects if present
        det = None
        if getattr(metric_cache, "current_tracked_objects", None):
            det = metric_cache.current_tracked_objects[0]
        elif getattr(metric_cache, "observation", None) is not None:
            tracks = metric_cache.observation.detections_tracks
            if len(tracks):
                det = tracks[0]
        if det is not None:
            objects = det.tracked_objects.tracked_objects
    except Exception:
        objects = []

    feats["n_objects"] = len(objects)
    n_vehicle = n_ped = n_barrier = 0
    min_dist = np.inf
    min_vehicle_dist = np.inf
    try:
        ego_geom = metric_cache.ego_state.car_footprint.oriented_box.geometry
        for obj in objects:
            typ = str(obj.tracked_object_type).lower()
            if "vehicle" in typ:
                n_vehicle += 1
            if "pedestrian" in typ or "bicyclist" in typ or "cyclist" in typ:
                n_ped += 1
            if "barrier" in typ:
                n_barrier += 1
            try:
                d = float(ego_geom.distance(obj.box.geometry))
                min_dist = min(min_dist, d)
                if "vehicle" in typ:
                    min_vehicle_dist = min(min_vehicle_dist, d)
            except Exception:
                pass
    except Exception:
        pass
    feats["n_vehicles"] = n_vehicle
    feats["n_pedestrians"] = n_ped
    feats["n_barriers"] = n_barrier
    feats["min_object_dist"] = float(min_dist) if np.isfinite(min_dist) else np.nan
    feats["min_vehicle_dist"] = float(min_vehicle_dist) if np.isfinite(min_vehicle_dist) else np.nan

    # Human/GT trajectory features if available.
    poses = None
    try:
        if metric_cache.human_trajectory is not None:
            poses = metric_cache.human_trajectory.poses
    except Exception:
        poses = None
    if poses is not None and len(poses) >= 2:
        xs, ys, hs = poses[:, 0], poses[:, 1], poses[:, 2]
        tf = traj_features(xs, ys, hs)
        feats.update({"gt_" + k.replace("pred_", ""): v for k, v in tf.items()})
        speeds = np.sqrt(np.diff(xs) ** 2 + np.diff(ys) ** 2) / 0.5
        feats["gt_speed_var"] = float(np.var(speeds)) if len(speeds) else 0.0
        feats["gt_max_speed"] = float(np.max(speeds)) if len(speeds) else 0.0
    else:
        for k in ["gt_progress", "gt_lateral_range", "gt_heading_change", "gt_final_heading", "gt_curvature_proxy", "gt_speed_var", "gt_max_speed"]:
            feats[k] = np.nan

    # A compact complexity score for binning.
    dist_term = 0.0 if pd.isna(feats["min_object_dist"]) else max(0.0, (20.0 - feats["min_object_dist"]) / 20.0)
    feats["complexity_score"] = (
        0.08 * feats["n_objects"]
        + 0.15 * feats["n_pedestrians"]
        + 1.0 * dist_term
        + 1.5 * (0.0 if pd.isna(feats["gt_curvature_proxy"]) else min(feats["gt_curvature_proxy"], 1.0))
        + 0.2 * (0.0 if pd.isna(feats["gt_speed_var"]) else min(feats["gt_speed_var"], 10.0))
    )
    return feats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores_csv", required=True)
    parser.add_argument("--metric_cache_path", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    from navsim.common.dataloader import MetricCacheLoader

    os.makedirs(args.output_dir, exist_ok=True)
    df = pd.read_csv(args.scores_csv)
    df = df[df["valid"] == True].copy() if "valid" in df.columns else df.copy()
    logger.info(f"loaded scores rows={len(df)} from {args.scores_csv}")

    loader = MetricCacheLoader(Path(args.metric_cache_path))
    rows = []
    for _, row in df.iterrows():
        token = row["token"]
        try:
            mc = loader.get_from_token(token)
            scene_feat = extract_scene_features(mc)
        except Exception as e:
            logger.warning(f"metric cache failed token={token}: {e}")
            scene_feat = {}

        xs = _json_list(row.get("pred_traj_x", "[]"))
        ys = _json_list(row.get("pred_traj_y", "[]"))
        hs = _json_list(row.get("pred_traj_heading", "[]"))
        pred_feat = traj_features(xs, ys, hs)
        intent = text_intent(row.get("cot", ""), row.get("meta_action", ""), row.get("answer", ""))
        cons = consistency_score(intent, pred_feat)

        out = row.to_dict()
        out.update(scene_feat)
        out.update(pred_feat)
        out.update(intent)
        out["cot_traj_consistency"] = cons
        rows.append(out)

    out_df = pd.DataFrame(rows)
    out_csv = os.path.join(args.output_dir, "cot_consistency_analysis.csv")
    out_df.to_csv(out_csv, index=False)
    logger.info(f"saved {out_csv}")

    # Correlations with PDM and consistency.
    feature_cols = [
        "n_objects", "n_vehicles", "n_pedestrians", "min_object_dist", "min_vehicle_dist",
        "gt_curvature_proxy", "gt_heading_change", "gt_lateral_range", "gt_speed_var",
        "complexity_score", "cot_len_words", "cot_traj_consistency",
    ]
    targets = [c for c in ["pdm_score", "drivable_area_compliance", "lane_keeping", "driving_direction_compliance", "cot_traj_consistency"] if c in out_df.columns]
    corr_rows = []
    for feat in feature_cols:
        if feat not in out_df.columns:
            continue
        for target in targets:
            # If feat == target, out_df[[feat, target]] creates duplicate column names;
            # valid[feat] then returns a DataFrame and pandas.corr() crashes. Skip self-corr.
            if feat == target:
                continue
            pair = out_df[[feat, target]].copy()
            pair[feat] = pd.to_numeric(pair[feat], errors="coerce")
            pair[target] = pd.to_numeric(pair[target], errors="coerce")
            valid = pair.dropna()
            if len(valid) > 5 and valid[feat].nunique() > 1 and valid[target].nunique() > 1:
                corr_rows.append({"feature": feat, "target": target, "corr": valid[feat].corr(valid[target]), "n": len(valid)})
    corr_df = pd.DataFrame(corr_rows)
    if len(corr_df):
        corr_df = corr_df.sort_values("corr", key=lambda s: s.abs(), ascending=False)
    corr_csv = os.path.join(args.output_dir, "cot_complexity_correlations.csv")
    corr_df.to_csv(corr_csv, index=False)
    logger.info(f"saved {corr_csv}")
    logger.info("top correlations:\n" + str(corr_df.head(20)))

    # Complexity bins.
    if "complexity_score" in out_df.columns and len(out_df) >= 4:
        try:
            out_df["complexity_bin"] = pd.qcut(out_df["complexity_score"].rank(method="first"), q=4, labels=["low", "mid", "high", "very_high"])
            agg_cols = [c for c in ["pdm_score", "cot_traj_consistency", "n_objects", "gt_curvature_proxy", "gt_speed_var"] if c in out_df.columns]
            summary = out_df.groupby("complexity_bin", observed=False)[agg_cols].mean().reset_index()
            summary["count"] = out_df.groupby("complexity_bin", observed=False).size().values
            summary_csv = os.path.join(args.output_dir, "cot_consistency_by_complexity.csv")
            summary.to_csv(summary_csv, index=False)
            logger.info(f"saved {summary_csv}")
            logger.info("complexity summary:\n" + str(summary))
        except Exception as e:
            logger.warning(f"complexity bin summary skipped: {e}")


if __name__ == "__main__":
    main()

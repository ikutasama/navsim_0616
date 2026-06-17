"""
弱势场景分析脚本：找出Alpamayo1.5在NavSim上表现差的场景共同特征

流程：
1. 加载PDM score结果（从run_pdm_score.py输出的CSV）
2. 加载metric_cache提取每个场景的特征
3. 计算特征与PDM score的相关性
4. 聚类分析弱势场景，找出共同特征

特征维度：
- 场景复杂度：检测目标数量、目标密度
- 目标接近度：最近目标距离、最近大车距离
- 真值轨迹动态：速度变化率、最大曲率、heading变化总量、横向位移
- 道路类型：是否有交通灯、交叉口、弯道
- 车辆初始状态：初始速度、加速度

Usage:
  # 先跑全量PDM评估，输出scores CSV
  python navsim/planning/script/run_pdm_score.py agent=alpamayo_agent train_test_split=navmini
  # 然后分析
  python analyze_weak_scenarios.py \
    --scores_csv /path/to/scores.csv \
    --metric_cache_path /path/to/metric_cache \
    --navsim_log_path /path/to/navsim_logs/mini \
    --sensor_blobs_path /path/to/sensor_blobs/mini
"""

import argparse
import sys
import os
import logging
import pickle
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def extract_scene_features(metric_cache, token):
    """从metric_cache中提取一个场景的特征。

    返回dict，包含：
    - n_objects: 初始帧检测目标数量
    - n_vehicles: 初始帧车辆数量
    - n_pedestrians: 初始帧行人数量
    - min_object_dist: 最近目标距离(米)
    - closest_vehicle_dist: 最近车辆距离
    - ego_initial_speed: ego初始速度(m/s)
    - ego_initial_accel: ego初始加速度(m/s^2)
    - ego_max_speed: ego轨迹中最大速度
    - ego_speed_change_rate: 速度变化率
    - gt_max_curvature: 真值轨迹最大曲率
    - gt_total_heading_change: heading变化总量(rad)
    - gt_lateral_range: 横向位移范围(米)
    - gt_progress: ego前进距离(米)
    - has_traffic_light: 是否有交通灯
    - scene_length: 场景持续时间(秒)
    """
    from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType

    features = {}

    try:
        # --- 目标数量和类型 ---
        detections = metric_cache.observation.detections_tracks
        if len(detections) > 0:
            first_frame = detections[0]
            objects = first_frame.tracked_objects.tracked_objects
            features["n_objects"] = len(objects)

            n_vehicles = 0
            n_pedestrians = 0
            n_barriers = 0
            for obj in objects:
                t = obj.tracked_object_type
                if t in (TrackedObjectType.VEHICLE, TrackedObjectType.EGO_VEHICLE):
                    n_vehicles += 1
                elif t in (TrackedObjectType.PEDESTRIAN, TrackedObjectType.BICYCLIST, TrackedObjectType.MOTORCYCLIST):
                    n_pedestrians += 1
                elif t == TrackedObjectType.BARRIER:
                    n_barriers += 1
            features["n_vehicles"] = n_vehicles
            features["n_pedestrians"] = n_pedestrians
            features["n_barriers"] = n_barriers

            # --- 最近目标距离 ---
            ego_box = metric_cache.ego_state.car_footprint.oriented_box.geometry
            min_dist = float("inf")
            closest_vehicle_dist = float("inf")
            for obj in objects:
                dist = ego_box.distance(obj.box.geometry)
                if dist < min_dist:
                    min_dist = dist
                if obj.tracked_object_type in (TrackedObjectType.VEHICLE, TrackedObjectType.EGO_VEHICLE):
                    if dist < closest_vehicle_dist:
                        closest_vehicle_dist = dist

            features["min_object_dist"] = min_dist if min_dist != float("inf") else -1.0
            features["closest_vehicle_dist"] = closest_vehicle_dist if closest_vehicle_dist != float("inf") else -1.0

            # 检查交通灯
            has_tl = any(
                obj.tracked_object_type == TrackedObjectType.TRAFFIC_LIGHT
                for obj in objects
            )
            features["has_traffic_light"] = int(has_tl)
        else:
            features["n_objects"] = 0
            features["n_vehicles"] = 0
            features["n_pedestrians"] = 0
            features["n_barriers"] = 0
            features["min_object_dist"] = -1.0
            features["closest_vehicle_dist"] = -1.0
            features["has_traffic_light"] = 0

        # --- Ego状态特征 ---
        ego_states = metric_cache.ego_state
        # ego_pose是SE2 [x, y, heading]
        ego_velocity = metric_cache.ego_state.velocity
        ego_accel = metric_cache.ego_state.acceleration

        # velocity和acceleration可能是数组或标量
        if hasattr(ego_velocity, "__len__"):
            features["ego_initial_speed"] = float(np.linalg.norm(ego_velocity[0]))
            features["ego_max_speed"] = float(np.max(np.linalg.norm(ego_velocity, axis=-1)))
            features["ego_speed_change_rate"] = float(
                abs(np.linalg.norm(ego_velocity[-1]) - np.linalg.norm(ego_velocity[0]))
                / max(len(ego_velocity) * 0.5, 1.0)
            )
        else:
            features["ego_initial_speed"] = float(np.linalg.norm(ego_velocity))
            features["ego_max_speed"] = float(np.linalg.norm(ego_velocity))
            features["ego_speed_change_rate"] = 0.0

        if hasattr(ego_accel, "__len__"):
            features["ego_initial_accel"] = float(np.linalg.norm(ego_accel[0]))
        else:
            features["ego_initial_accel"] = float(np.linalg.norm(ego_accel))

        # --- GT轨迹特征 ---
        # 从metric_cache的expert_trajectory提取
        expert_traj = metric_cache.ego_state.expert_trajectory
        if expert_traj is not None:
            poses = expert_traj.poses  # (N, 3) [x, y, heading]
            if len(poses) > 1:
                # 前进距离
                dx = poses[-1, 0] - poses[0, 0]
                dy = poses[-1, 1] - poses[0, 1]
                features["gt_progress"] = float(np.sqrt(dx**2 + dy**2))

                # heading变化总量
                heading_changes = np.abs(np.diff(poses[:, 2]))
                # unwrap
                heading_changes = np.minimum(heading_changes, np.abs(heading_changes - 2 * np.pi))
                features["gt_total_heading_change"] = float(np.sum(heading_changes))

                # 最大曲率
                if len(poses) >= 3:
                    # 曲率 = |heading_change / arc_length|
                    arc_lengths = np.sqrt(np.diff(poses[:, 0])**2 + np.diff(poses[:, 1])**2)
                    valid_mask = arc_lengths > 0.01
                    curvatures = heading_changes[valid_mask] / arc_lengths[valid_mask]
                    features["gt_max_curvature"] = float(np.max(curvatures)) if len(curvatures) > 0 else 0.0
                else:
                    features["gt_max_curvature"] = 0.0

                # 横向位移范围
                lateral_disp = poses[:, 1] - poses[0, 1]
                features["gt_lateral_range"] = float(np.max(np.abs(lateral_disp)))

                # 速度变化率
                speeds = np.sqrt(np.diff(poses[:, 0])**2 + np.diff(poses[:, 1])**2) / 0.5
                if len(speeds) > 1:
                    features["gt_speed_variance"] = float(np.var(speeds))
                else:
                    features["gt_speed_variance"] = 0.0
            else:
                features["gt_progress"] = 0.0
                features["gt_total_heading_change"] = 0.0
                features["gt_max_curvature"] = 0.0
                features["gt_lateral_range"] = 0.0
                features["gt_speed_variance"] = 0.0
        else:
            features["gt_progress"] = 0.0
            features["gt_total_heading_change"] = 0.0
            features["gt_max_curvature"] = 0.0
            features["gt_lateral_range"] = 0.0
            features["gt_speed_variance"] = 0.0

        features["token"] = token

    except Exception as e:
        logger.warning(f"  提取token={token}特征失败: {e}")
        return None

    return features


def run_analysis(scores_csv, metric_cache_path, output_dir):
    """主分析流程"""
    import pandas as pd
    from navsim.common.dataloader import MetricCacheLoader

    logger.info("=== 弱势场景分析 ===")

    # 1. 加载PDM scores
    logger.info(f"  加载scores: {scores_csv}")
    if not os.path.exists(scores_csv):
        logger.error(f"  scores文件不存在: {scores_csv}")
        logger.info("  先跑: python navsim/planning/script/run_pdm_score.py agent=alpamayo_agent train_test_split=navmini")
        return None

    scores_df = pd.read_csv(scores_csv)
    logger.info(f"  scores: {len(scores_df)} 行, 列: {list(scores_df.columns)}")

    # 2. 加载metric cache
    logger.info(f"  加载metric_cache: {metric_cache_path}")
    metric_cache_loader = MetricCacheLoader(Path(metric_cache_path))
    cache_tokens = set(metric_cache_loader.tokens)
    logger.info(f"  metric_cache有 {len(cache_tokens)} 个tokens")

    # 3. 提取每个场景的特征
    logger.info("  提取场景特征...")
    all_features = []

    # 只处理有score的tokens
    score_tokens = set()
    if "token" in scores_df.columns:
        score_tokens = set(scores_df["token"].values)
    elif "initial_token" in scores_df.columns:
        score_tokens = set(scores_df["initial_token"].values)
    else:
        # 尝试用所有metric_cache tokens
        score_tokens = cache_tokens

    common_tokens = list(score_tokens & cache_tokens)
    logger.info(f"  共有 {len(common_tokens)} 个场景同时有score和metric_cache")

    for i, token in enumerate(common_tokens):
        if i % 50 == 0:
            logger.info(f"    进度: {i}/{len(common_tokens)}")
        try:
            mc = metric_cache_loader.get_from_token(token)
            feat = extract_scene_features(mc, token)
            if feat is not None:
                all_features.append(feat)
        except Exception as e:
            logger.warning(f"  跳过token={token}: {e}")
            continue

    if len(all_features) == 0:
        logger.error("  没有提取到任何特征!")
        return None

    features_df = pd.DataFrame(all_features)
    logger.info(f"  特征DataFrame: {features_df.shape}")

    # 4. 合并scores和features
    # 根据score_df的token列名来merge
    token_col = "token" if "token" in scores_df.columns else "initial_token"
    merged_df = pd.merge(scores_df, features_df, on=token_col, how="inner")
    logger.info(f"  合并后: {merged_df.shape}")

    # 5. 相关性分析
    logger.info("\n=== 特征与PDM Score相关性 ===")

    score_col = "pdm_score"
    if score_col not in merged_df.columns:
        # 尝试其他列名
        score_candidates = [c for c in merged_df.columns if "score" in c.lower()]
        if score_candidates:
            score_col = score_candidates[0]
        else:
            logger.error("  找不到score列!")
            return None

    feature_cols = [
        "n_objects", "n_vehicles", "n_pedestrians", "n_barriers",
        "min_object_dist", "closest_vehicle_dist",
        "ego_initial_speed", "ego_speed_change_rate",
        "gt_total_heading_change", "gt_max_curvature",
        "gt_lateral_range", "gt_progress", "gt_speed_variance",
        "has_traffic_light",
    ]
    # 只分析存在的列
    feature_cols = [c for c in feature_cols if c in merged_df.columns]

    correlations = {}
    for feat in feature_cols:
        if feat in merged_df.columns and score_col in merged_df.columns:
            # 只用非NaN、非-1的数据
            valid = merged_df[merged_df[feat] != -1.0].dropna(subset=[feat, score_col])
            if len(valid) > 5:
                corr = valid[feat].corr(valid[score_col])
                correlations[feat] = corr
                logger.info(f"    {feat}: corr={corr:.4f} (n={len(valid)})")

    # 6. 弱势场景分析：PDM score < 阈值的场景
    logger.info("\n=== 弱势场景特征 ===")

    weak_threshold = merged_df[score_col].quantile(0.25)  # 25%分位数
    strong_threshold = merged_df[score_col].quantile(0.75)  # 75%分位数

    weak_df = merged_df[merged_df[score_col] <= weak_threshold]
    strong_df = merged_df[merged_df[score_col] >= strong_threshold]

    logger.info(f"  PDM score分布: mean={merged_df[score_col].mean():.4f}, "
                f"median={merged_df[score_col].median():.4f}, "
                f"std={merged_df[score_col].std():.4f}")
    logger.info(f"  弱势阈值(Q25): {weak_threshold:.4f}, 弱势场景数: {len(weak_df)}")
    logger.info(f"  强势阈值(Q75): {strong_threshold:.4f}, 强势场景数: {len(strong_df)}")

    # 弱势 vs 强势特征对比
    logger.info("\n  特征对比 (弱势 vs 强势):")
    for feat in feature_cols:
        if feat in merged_df.columns:
            weak_mean = weak_df[feat].mean() if len(weak_df) > 0 else "N/A"
            strong_mean = strong_df[feat].mean() if len(strong_df) > 0 else "N/A"
            all_mean = merged_df[feat].mean()

            # 格式化
            def fmt(v):
                return f"{v:.3f}" if isinstance(v, float) else str(v)

            logger.info(f"    {feat}: 全体={fmt(all_mean)}, "
                        f"弱势={fmt(weak_mean)}, "
                        f"强势={fmt(strong_mean)}, "
                        f"corr={fmt(correlations.get(feat, 0.0))}")

    # 7. 分指标分析：哪些子指标最常失败
    logger.info("\n=== PDM子指标失败率 ===")
    sub_metrics = [
        "no_at_fault_collisions", "drivable_area_compliance",
        "driving_direction_compliance", "traffic_light_compliance",
        "ego_progress", "time_to_collision_within_bound",
        "lane_keeping", "history_comfort",
    ]
    for m in sub_metrics:
        if m in merged_df.columns:
            fail_rate = (merged_df[m] == 0).sum() / len(merged_df)
            # 在弱势场景中的fail rate
            weak_fail = (weak_df[m] == 0).sum() / max(len(weak_df), 1)
            logger.info(f"    {m}: 全体fail={fail_rate:.3f}, 弱势fail={weak_fail:.3f}")

    # 8. 弱势场景聚类特征
    logger.info("\n=== 弱势场景类型分布 ===")
    if len(weak_df) > 0:
        # 高目标密度
        high_density = weak_df[weak_df["n_objects"] > merged_df["n_objects"].median()]
        logger.info(f"  高目标密度(n_obj>median): {len(high_density)}/{len(weak_df)} = {len(high_density)/max(len(weak_df),1):.2%}")

        # 近距离目标
        close_obj = weak_df[(weak_df["min_object_dist"] != -1.0) & (weak_df["min_object_dist"] < 10.0)]
        logger.info(f"  近距离目标(<10m): {len(close_obj)}/{len(weak_df)} = {len(close_obj)/max(len(weak_df),1):.2%}")

        # 大曲率
        high_curv = weak_df[weak_df["gt_max_curvature"] > merged_df["gt_max_curvature"].median()]
        logger.info(f"  大曲率(>median): {len(high_curv)}/{len(weak_df)} = {len(high_curv)/max(len(weak_df),1):.2%}")

        # 大速度变化
        high_speed_change = weak_df[weak_df["gt_speed_variance"] > merged_df["gt_speed_variance"].median()]
        logger.info(f"  高速度方差(>median): {len(high_speed_change)}/{len(weak_df)} = {len(high_speed_change)/max(len(weak_df),1):.2%}")

        # 有交通灯
        with_tl = weak_df[weak_df["has_traffic_light"] == 1]
        logger.info(f"  有交通灯: {len(with_tl)}/{len(weak_df)} = {len(with_tl)/max(len(weak_df),1):.2%}")

    # 9. 保存结果
    os.makedirs(output_dir, exist_ok=True)
    merged_df.to_csv(os.path.join(output_dir, "weak_scenario_analysis.csv"), index=False)
    logger.info(f"\n  分析结果保存到: {output_dir}/weak_scenario_analysis.csv")

    # 保存top最弱场景的tokens
    if len(weak_df) > 0:
        top_weak = weak_df.sort_values(score_col, ascending=True).head(20)
        top_weak_tokens = top_weak[token_col].tolist()
        with open(os.path.join(output_dir, "top_weak_tokens.txt"), "w") as f:
            for t in top_weak_tokens:
                f.write(t + "\n")
        logger.info(f"  Top 20最弱场景tokens保存到: {output_dir}/top_weak_tokens.txt")

    # 保存相关性汇总
    corr_summary = pd.DataFrame([
        {"feature": k, "correlation_with_pdm": v}
        for k, v in sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    ])
    corr_summary.to_csv(os.path.join(output_dir, "feature_correlations.csv"), index=False)
    logger.info(f"  相关性汇总保存到: {output_dir}/feature_correlations.csv")

    logger.info("\n=== 分析完成 ===")
    return merged_df


def main():
    parser = argparse.ArgumentParser(description="分析Alpamayo1.5在NavSim上的弱势场景")
    parser.add_argument("--scores_csv", type=str, required=True,
                        help="PDM score结果CSV路径（从run_pdm_score.py输出）")
    parser.add_argument("--metric_cache_path", type=str, required=True,
                        help="metric_cache目录路径")
    parser.add_argument("--output_dir", type=str,
                        default=os.environ.get("OPENSCENE_DATA_ROOT", "") + "/weak_analysis",
                        help="分析结果输出目录")
    args = parser.parse_args()

    run_analysis(args.scores_csv, args.metric_cache_path, args.output_dir)


if __name__ == "__main__":
    main()

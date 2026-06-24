"""
Alpamayo1.5 全量PDM评估脚本（支持多GPU分片）

核心逻辑：先加载全部场景找metric_cache交集token，再按GPU分片评估。
--max_eval_tokens 控制评估数量（不限制SceneLoader加载量）。
--gpu_id / --total_gpus 支持多卡并行，每卡跑自己的分片。

Usage:
  # 单卡跑全部396个场景
  python run_alpamayo_eval.py

  # 先跑10个场景快速测试
  python run_alpamayo_eval.py --max_eval_tokens 10

  # 指定GPU（8卡服务器）
  CUDA_VISIBLE_DEVICES=0 python run_alpamayo_eval.py --gpu_id 0 --total_gpus 8

  # 多卡并行（分别在8个终端跑）
  CUDA_VISIBLE_DEVICES=0 python run_alpamayo_eval.py --gpu_id 0 --total_gpus 8 &
  CUDA_VISIBLE_DEVICES=1 python run_alpamayo_eval.py --gpu_id 1 --total_gpus 8 &
  ...
  CUDA_VISIBLE_DEVICES=7 python run_alpamayo_eval.py --gpu_id 7 --total_gpus 8 &
  # 跑完后合并：
  python merge_eval_results.py --input_dir $OUTPUT_DIR
"""

import argparse
import sys
import os
import logging
import time
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_eval(navsim_log_path, sensor_blobs_path, metric_cache_path,
             model_path, output_dir, max_eval_tokens, gpu_id, total_gpus, save_cot):
    """逐场景评估Alpamayo1.5，支持多GPU分片"""
    import torch
    from navsim.common.dataloader import SceneLoader, SceneFilter, MetricCacheLoader
    from navsim.common.dataclasses import SensorConfig, PDMResults
    from navsim.agents.alpamayo_agent.alpamayo_agent import AlpamayoAgent
    from navsim.evaluate.pdm_score import pdm_score
    from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
    from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer, PDMScorerConfig
    from navsim.traffic_agents_policies.log_replay_traffic_agents import LogReplayTrafficAgents
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

    logger.info(f"=== Alpamayo1.5 评估 (gpu_id={gpu_id}/{total_gpus}) ===")

    proposal_sampling = TrajectorySampling(time_horizon=4, interval_length=0.5)

    # 1. 加载metric cache
    logger.info("  加载metric cache...")
    metric_cache_loader = MetricCacheLoader(Path(metric_cache_path))
    cache_tokens = set(metric_cache_loader.tokens)
    logger.info(f"  metric_cache: {len(cache_tokens)} tokens")

    # 2. 加载全部NavSim场景（不限制max_scenes，确保找到交集）
    logger.info("  加载NavSim场景数据...")
    sensor_config = SensorConfig(
        cam_f0=[0, 1, 2, 3], cam_l0=[0, 1, 2, 3], cam_l1=False, cam_l2=False,
        cam_r0=[0, 1, 2, 3], cam_r1=False, cam_r2=False, cam_b0=False, lidar_pc=False,
    )
    scene_filter = SceneFilter(num_history_frames=4, num_future_frames=10)

    scene_loader = SceneLoader(
        data_path=Path(navsim_log_path),
        synthetic_sensor_path=Path(sensor_blobs_path),
        original_sensor_path=Path(sensor_blobs_path),
        synthetic_scenes_path=Path(navsim_log_path),
        scene_filter=scene_filter,
        sensor_config=sensor_config,
    )
    scene_tokens = set(scene_loader.tokens_stage_one)
    logger.info(f"  SceneLoader: {len(scene_tokens)} tokens")

    # 3. 找交集token，再限制评估数量
    common_tokens = sorted(list(cache_tokens & scene_tokens))
    logger.info(f"  可评估场景: {len(common_tokens)} (metric_cache ∩ scene_loader)")

    if len(common_tokens) == 0:
        logger.error("  没有可评估场景! 检查数据路径.")
        logger.info(f"  metric_cache tokens示例: {list(cache_tokens)[:5]}")
        logger.info(f"  scene_loader tokens示例: {list(scene_tokens)[:5]}")
        return None

    # 限制评估数量（只取前N个token，不影响SceneLoader加载）
    if max_eval_tokens > 0 and max_eval_tokens < len(common_tokens):
        common_tokens = common_tokens[:max_eval_tokens]
        logger.info(f"  限制评估: 只跑前 {max_eval_tokens} 个场景")

    # 4. 多GPU分片：每卡只跑自己的那部分token
    if total_gpus > 1:
        shard_size = len(common_tokens) // total_gpus
        remainder = len(common_tokens) % total_gpus
        start_idx = gpu_id * shard_size + min(gpu_id, remainder)
        end_idx = start_idx + shard_size + (1 if gpu_id < remainder else 0)
        my_tokens = common_tokens[start_idx:end_idx]
        logger.info(f"  GPU分片: total={len(common_tokens)}, gpu{gpu_id}处理 {len(my_tokens)} 个 (idx {start_idx}-{end_idx-1})")
        common_tokens = my_tokens

    # 5. 加载模型（只加载一次）
    logger.info(f"  加载Alpamayo模型: {model_path}")
    gpu_name = torch.cuda.get_device_name(gpu_id) if torch.cuda.is_available() else "N/A"
    vram = torch.cuda.get_device_properties(gpu_id).total_memory / 1e9 if torch.cuda.is_available() else 0
    logger.info(f"  GPU: {gpu_name}, VRAM: {vram:.1f}GB")

    device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"

    agent = AlpamayoAgent(
        trajectory_sampling=proposal_sampling,
        model_path=model_path,
        device=device,
    )
    agent.initialize()
    logger.info("  模型加载完成!")

    # 6. 逐场景推理+评分
    simulator = PDMSimulator(proposal_sampling=proposal_sampling)
    scorer = PDMScorer(proposal_sampling=proposal_sampling, config=PDMScorerConfig())
    traffic_policy = LogReplayTrafficAgents(proposal_sampling)

    all_results = []
    cot_outputs = {} if save_cot else None
    num_success = 0
    num_fail = 0

    logger.info(f"  开始评估 {len(common_tokens)} 个场景...")

    for idx, token in enumerate(common_tokens):
        t0 = time.time()

        try:
            agent_input = scene_loader.get_agent_input_from_token(token)
            trajectory = agent.compute_trajectory(agent_input)

            metric_cache = metric_cache_loader.get_from_token(token)
            score_row, ego_simulated_states = pdm_score(
                metric_cache=metric_cache,
                model_trajectory=trajectory,
                future_sampling=proposal_sampling,
                simulator=simulator,
                scorer=scorer,
                traffic_agents_policy=traffic_policy,
            )

            score_row["valid"] = True
            score_row["token"] = token
            score_row["log_name"] = metric_cache.log_name
            score_row["start_time"] = metric_cache.timepoint.time_s
            all_results.append(score_row)

            if save_cot and hasattr(agent, "_last_cot_text"):
                cot_outputs[token] = agent._last_cot_text

            num_success += 1
            dt = time.time() - t0
            pdm_val = score_row["pdm_score"].iloc[0] if "pdm_score" in score_row.columns else "N/A"
            logger.info(f"    [{idx+1}/{len(common_tokens)}] token={token[:12]}... "
                        f"pdm={pdm_val:.4f} time={dt:.1f}s")

        except Exception as e:
            logger.warning(f"    [{idx+1}/{len(common_tokens)}] token={token} FAILED: {e}")
            empty_row = pd.DataFrame([PDMResults.get_empty_results()])
            empty_row["valid"] = False
            empty_row["token"] = token
            all_results.append(empty_row)
            num_fail += 1

    # 7. 汇总结果
    results_df = pd.concat(all_results, ignore_index=True)
    logger.info(f"\n  评估完成: 成功={num_success}, 失败={num_fail}")

    valid_df = results_df[results_df["valid"] == True]
    if len(valid_df) > 0 and "pdm_score" in valid_df.columns:
        avg_pdm = valid_df["pdm_score"].mean()
        logger.info(f"  平均PDM Score (stage-1): {avg_pdm:.4f}")

        sub_metrics = [
            "no_at_fault_collisions", "drivable_area_compliance",
            "driving_direction_compliance", "traffic_light_compliance",
            "ego_progress", "time_to_collision_within_bound",
            "lane_keeping", "history_comfort",
        ]
        logger.info("  各子指标平均:")
        for m in sub_metrics:
            if m in valid_df.columns:
                val = valid_df[m].mean()
                fail_rate = (valid_df[m] == 0).sum() / len(valid_df)
                logger.info(f"    {m}: mean={val:.4f}, fail_rate={fail_rate:.2%}")

    # 8. 保存结果
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
    suffix = f"_gpu{gpu_id}" if total_gpus > 1 else ""
    csv_path = os.path.join(output_dir, f"alpamayo_pdm_scores_{timestamp}{suffix}.csv")

    save_cols = [c for c in results_df.columns
                 if c not in {"weighted_metrics", "weighted_metrics_array", "ego_simulated_states"}]
    results_df[save_cols].to_csv(csv_path, index=False)
    logger.info(f"  结果保存到: {csv_path}")

    if save_cot and cot_outputs:
        cot_path = os.path.join(output_dir, f"alpamayo_cot_outputs_{timestamp}{suffix}.json")
        with open(cot_path, "w") as f:
            json.dump(cot_outputs, f, indent=2)
        logger.info(f"  CoT输出保存到: {cot_path}")

    logger.info("\n=== 评估完成 ===")
    return results_df


def main():
    parser = argparse.ArgumentParser(description="Alpamayo1.5 全量PDM评估（支持多GPU）")
    parser.add_argument("--navsim_log_path", type=str,
                        default=os.environ.get("OPENSCENE_DATA_ROOT", "") + "/navsim_logs/mini")
    parser.add_argument("--sensor_blobs_path", type=str,
                        default=os.environ.get("OPENSCENE_DATA_ROOT", "") + "/sensor_blobs/mini")
    parser.add_argument("--metric_cache_path", type=str,
                        default=os.environ.get("OPENSCENE_DATA_ROOT", "") + "/metric_cache")
    parser.add_argument("--model_path", type=str,
                        default="/data/mnt_m181/z59900495/workspace/model/Alpamayo-1.5-10B")
    parser.add_argument("--output_dir", type=str,
                        default=os.environ.get("OPENSCENE_DATA_ROOT", "") + "/exp/eval_results")
    parser.add_argument("--max_eval_tokens", type=int, default=0,
                        help="最多评估多少个场景（0=全部，>0快速测试）")
    parser.add_argument("--gpu_id", type=int, default=0,
                        help="当前GPU编号（多卡并行时每卡不同）")
    parser.add_argument("--total_gpus", type=int, default=1,
                        help="总共使用几张GPU（用于分片）")
    parser.add_argument("--save_cot", action="store_true",
                        help="保存CoT文本输出")
    args = parser.parse_args()

    run_eval(
        navsim_log_path=args.navsim_log_path,
        sensor_blobs_path=args.sensor_blobs_path,
        metric_cache_path=args.metric_cache_path,
        model_path=args.model_path,
        output_dir=args.output_dir,
        max_eval_tokens=args.max_eval_tokens,
        gpu_id=args.gpu_id,
        total_gpus=args.total_gpus,
        save_cot=args.save_cot,
    )


if __name__ == "__main__":
    main()

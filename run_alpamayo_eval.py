"""
Alpamayo1.5 全量PDM评估脚本（支持多GPU分片）

关键点：
1. 先加载全部SceneLoader tokens，与metric_cache tokens取交集，避免max_scenes导致交集为0。
2. --max_eval_tokens 只限制“评估token数量”，不限制SceneLoader加载。
3. 多GPU并行时，--shard_id/--total_shards负责切分mini场景；--device负责当前进程实际使用哪张逻辑GPU。
   如果用 CUDA_VISIBLE_DEVICES=5 启动，那么进程内可见GPU通常是 cuda:0，所以 --device 默认 cuda:0。
4. 保存PDM子指标、预测轨迹、Alpamayo生成的cot/meta_action/answer，供后续一致性分析。
"""

import argparse
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def infer_data_root() -> Path:
    env_root = os.environ.get("OPENSCENE_DATA_ROOT", "").strip()
    if env_root:
        return Path(env_root).expanduser().resolve()
    repo_dir = Path(__file__).resolve().parent
    candidates = [repo_dir / "navsim" / "navsim_dataset", repo_dir / "navsim_dataset", repo_dir / "dataset"]
    for cand in candidates:
        if (cand / "navsim_logs").exists() or (cand / "metric_cache").exists() or (cand / "sensor_blobs").exists():
            return cand.resolve()
    return (repo_dir / "navsim" / "navsim_dataset").resolve()


def run_eval(
    navsim_log_path: str,
    sensor_blobs_path: str,
    metric_cache_path: str,
    model_path: str,
    output_dir: str,
    max_eval_tokens: int,
    shard_id: int,
    total_shards: int,
    device: str,
    save_cot_json: bool,
):
    import torch
    from navsim.agents.alpamayo_agent.alpamayo_agent import AlpamayoAgent
    from navsim.common.dataclasses import PDMResults, SensorConfig
    from navsim.common.dataloader import MetricCacheLoader, SceneFilter, SceneLoader
    from navsim.evaluate.pdm_score import pdm_score
    from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer, PDMScorerConfig
    from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
    from navsim.traffic_agents_policies.log_replay_traffic_agents import LogReplayTrafficAgents
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

    logger.info(f"=== Alpamayo1.5 NavSim eval shard={shard_id}/{total_shards}, device={device} ===")

    if torch.cuda.is_available():
        logger.info(f"  torch visible cuda devices: {torch.cuda.device_count()}")
        try:
            visible_idx = int(device.split(":")[-1]) if device.startswith("cuda:") else 0
            logger.info(
                f"  using {device}: {torch.cuda.get_device_name(visible_idx)}, "
                f"VRAM={torch.cuda.get_device_properties(visible_idx).total_memory / 1e9:.1f}GB"
            )
        except Exception as e:
            logger.warning(f"  cannot query cuda device {device}: {e}")

    proposal_sampling = TrajectorySampling(time_horizon=4, interval_length=0.5)

    # 1. metric cache tokens
    logger.info("  加载metric cache...")
    metric_cache_loader = MetricCacheLoader(Path(metric_cache_path))
    cache_tokens = set(metric_cache_loader.tokens)
    logger.info(f"  metric_cache tokens: {len(cache_tokens)}")

    # 2. scene tokens：必须加载全部，不要max_scenes
    logger.info("  加载NavSim场景数据...")
    sensor_config = SensorConfig(
        cam_f0=[0, 1, 2, 3],
        cam_l0=[0, 1, 2, 3],
        cam_l1=False,
        cam_l2=False,
        cam_r0=[0, 1, 2, 3],
        cam_r1=False,
        cam_r2=False,
        cam_b0=False,
        lidar_pc=False,
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
    logger.info(f"  SceneLoader stage-1 tokens: {len(scene_tokens)}")

    # 3. intersection + optional small test limit
    all_tokens = sorted(cache_tokens & scene_tokens)
    logger.info(f"  可评估tokens: {len(all_tokens)} = metric_cache ∩ SceneLoader")
    if not all_tokens:
        logger.error("  没有可评估场景，请检查navsim_log_path/sensor_blobs_path/metric_cache是否同一split")
        logger.info(f"  metric_cache token sample: {list(cache_tokens)[:5]}")
        logger.info(f"  scene_loader token sample: {list(scene_tokens)[:5]}")
        return None

    if max_eval_tokens > 0:
        all_tokens = all_tokens[:max_eval_tokens]
        logger.info(f"  max_eval_tokens={max_eval_tokens}, 本次只评估前 {len(all_tokens)} 个token")

    # 4. shard split
    if total_shards < 1:
        raise ValueError("total_shards must be >= 1")
    if not (0 <= shard_id < total_shards):
        raise ValueError(f"shard_id must be in [0, {total_shards}), got {shard_id}")

    shard_tokens = all_tokens[shard_id::total_shards]
    logger.info(f"  分片结果: shard {shard_id}/{total_shards} 处理 {len(shard_tokens)} / {len(all_tokens)} 个token")

    # 5. load model once
    logger.info(f"  加载Alpamayo模型: {model_path}")
    agent = AlpamayoAgent(
        trajectory_sampling=proposal_sampling,
        model_path=model_path,
        device=device,
    )
    agent.initialize()
    logger.info("  模型加载完成")

    simulator = PDMSimulator(proposal_sampling=proposal_sampling)
    scorer = PDMScorer(proposal_sampling=proposal_sampling, config=PDMScorerConfig())
    traffic_policy = LogReplayTrafficAgents(proposal_sampling)

    rows = []
    cot_json = {}
    n_ok, n_fail = 0, 0

    for i, token in enumerate(shard_tokens):
        t0 = time.time()
        try:
            agent_input = scene_loader.get_agent_input_from_token(token)
            trajectory = agent.compute_trajectory(agent_input)
            metric_cache = metric_cache_loader.get_from_token(token)
            score_row, _ = pdm_score(
                metric_cache=metric_cache,
                model_trajectory=trajectory,
                future_sampling=proposal_sampling,
                simulator=simulator,
                scorer=scorer,
                traffic_agents_policy=traffic_policy,
            )

            # metadata + generated text + predicted trajectory
            score_row["valid"] = True
            score_row["token"] = token
            score_row["log_name"] = metric_cache.log_name
            score_row["start_time"] = metric_cache.timepoint.time_s
            score_row["cot"] = getattr(agent, "_last_cot_text", "")
            score_row["meta_action"] = getattr(agent, "_last_meta_action_text", "")
            score_row["answer"] = getattr(agent, "_last_answer_text", "")
            score_row["pred_traj_x"] = json.dumps([float(x) for x in trajectory.poses[:, 0]])
            score_row["pred_traj_y"] = json.dumps([float(y) for y in trajectory.poses[:, 1]])
            score_row["pred_traj_heading"] = json.dumps([float(h) for h in trajectory.poses[:, 2]])
            rows.append(score_row)

            if save_cot_json:
                cot_json[token] = {
                    "cot": getattr(agent, "_last_cot_text", ""),
                    "meta_action": getattr(agent, "_last_meta_action_text", ""),
                    "answer": getattr(agent, "_last_answer_text", ""),
                }

            n_ok += 1
            pdm_val = float(score_row["pdm_score"].iloc[0]) if "pdm_score" in score_row.columns else float("nan")
            logger.info(f"    [{i+1}/{len(shard_tokens)}] {token[:12]} pdm={pdm_val:.4f} time={time.time()-t0:.1f}s")

        except Exception as e:
            logger.warning(f"    [{i+1}/{len(shard_tokens)}] {token} FAILED: {e}")
            empty = pd.DataFrame([PDMResults.get_empty_results()])
            empty["valid"] = False
            empty["token"] = token
            empty["cot"] = ""
            empty["meta_action"] = ""
            empty["answer"] = ""
            rows.append(empty)
            n_fail += 1

    if not rows:
        logger.error("  本分片没有结果")
        return None

    df = pd.concat(rows, ignore_index=True)
    valid = df[df["valid"] == True]
    logger.info(f"  完成: ok={n_ok}, fail={n_fail}, valid_rows={len(valid)}")
    if len(valid) and "pdm_score" in valid.columns:
        logger.info(f"  mean pdm_score={valid['pdm_score'].mean():.4f}")

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
    suffix = f"shard{shard_id}-of-{total_shards}"
    csv_path = os.path.join(output_dir, f"alpamayo_pdm_scores_{timestamp}_{suffix}.csv")
    drop_cols = {"weighted_metrics", "weighted_metrics_array", "ego_simulated_states"}
    save_cols = [c for c in df.columns if c not in drop_cols]
    df[save_cols].to_csv(csv_path, index=False)
    logger.info(f"  CSV保存到: {csv_path}")

    if save_cot_json:
        cot_path = os.path.join(output_dir, f"alpamayo_cot_outputs_{timestamp}_{suffix}.json")
        with open(cot_path, "w", encoding="utf-8") as f:
            json.dump(cot_json, f, ensure_ascii=False, indent=2)
        logger.info(f"  CoT JSON保存到: {cot_path}")

    return df


def main():
    data_root = infer_data_root()
    parser = argparse.ArgumentParser(description="Alpamayo1.5 NavSim eval with multi-GPU sharding")
    parser.add_argument("--navsim_log_path", default=str(data_root / "navsim_logs" / "mini"))
    parser.add_argument("--sensor_blobs_path", default=str(data_root / "sensor_blobs" / "mini"))
    parser.add_argument("--metric_cache_path", default=str(data_root / "metric_cache"))
    parser.add_argument("--model_path", default="/data/mnt_m181/z59900495/workspace/model/Alpamayo-1.5-10B")
    parser.add_argument("--output_dir", default=str(data_root / "exp" / "eval_results"))
    parser.add_argument("--max_eval_tokens", type=int, default=0, help="0=全部；>0只评估前N个可评估token")
    parser.add_argument("--shard_id", type=int, default=0, help="当前分片编号，0..total_shards-1")
    parser.add_argument("--total_shards", type=int, default=1, help="总分片数；4卡就设4")
    parser.add_argument("--device", default="cuda:0", help="当前进程使用的逻辑device。配合CUDA_VISIBLE_DEVICES时通常保持cuda:0")
    parser.add_argument("--save_cot_json", action="store_true", help="额外保存cot/meta_action/answer到json")
    args = parser.parse_args()

    run_eval(
        navsim_log_path=args.navsim_log_path,
        sensor_blobs_path=args.sensor_blobs_path,
        metric_cache_path=args.metric_cache_path,
        model_path=args.model_path,
        output_dir=args.output_dir,
        max_eval_tokens=args.max_eval_tokens,
        shard_id=args.shard_id,
        total_shards=args.total_shards,
        device=args.device,
        save_cot_json=args.save_cot_json,
    )


if __name__ == "__main__":
    main()

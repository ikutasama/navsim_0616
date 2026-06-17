"""
分步测试脚本：在navsim上测试alpamayo1.5

在远程GPU服务器上运行，逐步验证每个环节。

Usage:
  python test_alpamayo_navsim.py --step 1
  python test_alpamayo_navsim.py --step 2 --navsim_log_path PATH --sensor_blobs_path PATH
  python test_alpamayo_navsim.py --step 3 --navsim_log_path PATH --sensor_blobs_path PATH
  python test_alpamayo_navsim.py --step 4 --navsim_log_path PATH --sensor_blobs_path PATH --model_path PATH
  python test_alpamayo_navsim.py --step 5 --navsim_log_path PATH --sensor_blobs_path PATH --metric_cache_path PATH --model_path PATH

Step 5 runs independently: loads data, inference, then PDM scoring.
No need to rerun step 2/4 before step 5.
"""

import argparse
import sys
import os
import logging

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def step1_imports():
    """验证所有Python包能正常import"""
    logger.info("=== STEP 1: 验证imports ===")

    ok = True
    checks = [
        ("torch", lambda: f"version={__import__('torch').__version__}, cuda={__import__('torch').cuda.is_available()}"),
        ("transformers", lambda: f"version={__import__('transformers').__version__}"),
        ("navsim", lambda: "installed"),
        ("hydra", lambda: f"version={__import__('hydra').__version__}"),
        ("scipy", lambda: f"version={__import__('scipy').__version__}"),
        ("einops", lambda: "installed"),
        ("numpy", lambda: f"version={__import__('numpy').__version__}"),
    ]

    import importlib

    all_checks = checks + [
        ("flash_attn", lambda: f"version={__import__('flash_attn').__version__}"),
        ("alpamayo1_5.models.alpamayo1_5", lambda: "Alpamayo1_5 class OK"),
        ("alpamayo1_5.helper", lambda: "helper module OK"),
    ]

    for mod, desc_fn in all_checks:
        try:
            m = importlib.import_module(mod)
            logger.info(f"  {mod}: {desc_fn()}")
        except ImportError as e:
            logger.error(f"  {mod}: FAIL - {e}")
            ok = False

    if ok:
        logger.info("STEP 1: ALL IMPORTS OK")
    else:
        logger.error("STEP 1: SOME IMPORTS FAILED - fix before proceeding")
    return ok


def step2_navsim_data(navsim_log_path, sensor_blobs_path):
    """验证NavSim数据能加载，拿到一个AgentInput"""
    logger.info("=== STEP 2: 验证NavSim数据 ===")

    from pathlib import Path
    from navsim.common.dataloader import SceneLoader, SceneFilter
    from navsim.common.dataclasses import SensorConfig

    sensor_config = SensorConfig(
        cam_f0=[0, 1, 2, 3], cam_l0=[0, 1, 2, 3], cam_l1=False, cam_l2=False,
        cam_r0=[0, 1, 2, 3], cam_r1=False, cam_r2=False, cam_b0=False, lidar_pc=False,
    )
    scene_filter = SceneFilter(num_history_frames=4, num_future_frames=10, max_scenes=2)

    try:
        scene_loader = SceneLoader(
            data_path=Path(navsim_log_path),
            synthetic_sensor_path=Path(sensor_blobs_path),
            original_sensor_path=Path(sensor_blobs_path),
            synthetic_scenes_path=Path(navsim_log_path),
            scene_filter=scene_filter,
            sensor_config=sensor_config,
        )
        tokens = scene_loader.tokens_stage_one
        logger.info(f"  stage-1 tokens数量: {len(tokens)}")

        if len(tokens) == 0:
            logger.error("  没有找到tokens! 检查数据路径")
            return None, None

        token = tokens[0]
        logger.info(f"  使用token: {token}")

        agent_input = scene_loader.get_agent_input_from_token(token)
        scene = scene_loader.get_scene_from_token(token)

        logger.info(f"  ego_statuses数量: {len(agent_input.ego_statuses)}")
        logger.info(f"  cameras数量: {len(agent_input.cameras)}")

        cam = agent_input.cameras[-1]
        logger.info(f"  cam_f0.image: {cam.cam_f0.image.shape if cam.cam_f0.image is not None else 'None'}")
        logger.info(f"  cam_l0.image: {cam.cam_l0.image.shape if cam.cam_l0.image is not None else 'None'}")
        logger.info(f"  cam_r0.image: {cam.cam_r0.image.shape if cam.cam_r0.image is not None else 'None'}")

        for i, es in enumerate(agent_input.ego_statuses):
            logger.info(f"  ego[{i}]: pose={es.ego_pose}, vel={es.ego_velocity}")

        gt = scene.get_future_trajectory()
        logger.info(f"  GT trajectory: shape={gt.poses.shape}, time_horizon={gt.trajectory_sampling.time_horizon}")

        logger.info("STEP 2: DATA OK")
        return agent_input, scene
    except Exception as e:
        logger.error(f"STEP 2: FAIL - {e}")
        import traceback; traceback.print_exc()
        return None, None


def step3_conversion(agent_input):
    """验证NavSim→Alpamayo格式转换"""
    logger.info("=== STEP 3: 验证格式转换 ===")

    from navsim.agents.alpamayo_agent.alpamayo_agent import AlpamayoAgent
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

    try:
        agent = AlpamayoAgent(trajectory_sampling=TrajectorySampling(time_horizon=4, interval_length=0.5))

        image_frames, camera_indices = agent._prepare_images(agent_input)
        logger.info(f"  image_frames: {image_frames.shape} (期望 N_cameras, 4, 3, H, W)")
        logger.info(f"  camera_indices: {camera_indices.tolist()}")

        ego_xyz, ego_rot = agent._prepare_ego_history(agent_input)
        logger.info(f"  ego_history_xyz: {ego_xyz.shape} (期望 1,1,16,3)")
        logger.info(f"  ego_history_rot: {ego_rot.shape} (期望 1,1,16,3,3)")

        logger.info(f"  xyz范围: x=[{ego_xyz[0,0,:,0].min():.3f},{ego_xyz[0,0,:,0].max():.3f}], "
                     f"y=[{ego_xyz[0,0,:,1].min():.3f},{ego_xyz[0,0,:,1].max():.3f}]")

        logger.info("STEP 3: CONVERSION OK")
        return True
    except Exception as e:
        logger.error(f"STEP 3: FAIL - {e}")
        import traceback; traceback.print_exc()
        return False


def step4_inference(agent_input, model_path="/data/mnt_m181/z59900495/workspace/model/Alpamayo-1.5-10B"):
    """GPU推理：加载模型并跑一个场景"""
    logger.info("=== STEP 4: GPU推理 ===")

    import torch
    from navsim.agents.alpamayo_agent.alpamayo_agent import AlpamayoAgent
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

    if not torch.cuda.is_available():
        logger.error("CUDA不可用!")
        return None

    logger.info(f"  GPU: {torch.cuda.get_device_name(0)}, "
                f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")

    try:
        agent = AlpamayoAgent(
            trajectory_sampling=TrajectorySampling(time_horizon=4, interval_length=0.5),
            model_path=model_path,
        )
        logger.info("  加载模型（可能需要几分钟下载22GB权重）...")
        agent.initialize()
        logger.info("  模型加载成功!")

        logger.info("  运行推理...")
        trajectory = agent.compute_trajectory(agent_input)

        logger.info(f"  输出trajectory: shape={trajectory.poses.shape}")
        for i in range(len(trajectory.poses)):
            p = trajectory.poses[i]
            logger.info(f"    pose[{i}]: x={p[0]:.3f} y={p[1]:.3f} heading={p[2]:.3f}")

        logger.info("STEP 4: INFERENCE OK")
        return trajectory
    except Exception as e:
        logger.error(f"STEP 4: FAIL - {e}")
        import traceback; traceback.print_exc()
        return None


def step5_pdm_score(navsim_log_path, sensor_blobs_path, metric_cache_path, model_path):
    """独立计算PDM分数：加载数据→推理→评分，无需先跑step 2/4"""
    logger.info("=== STEP 5: PDM评分 ===")

    import torch
    from pathlib import Path
    from navsim.common.dataloader import SceneLoader, SceneFilter, MetricCacheLoader
    from navsim.common.dataclasses import SensorConfig
    from navsim.agents.alpamayo_agent.alpamayo_agent import AlpamayoAgent
    from navsim.evaluate.pdm_score import pdm_score
    from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
    from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer, PDMScorerConfig
    from navsim.traffic_agents_policies.log_replay_traffic_agents import LogReplayTrafficAgents
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

    if not metric_cache_path:
        logger.error("需要metric_cache_path! 先运行metric caching脚本.")
        logger.info("  python navsim/navsim/planning/script/run_metric_caching.py train_test_split=navmini")
        return None

    try:
        proposal_sampling = TrajectorySampling(time_horizon=4, interval_length=0.5)

        # 1. 加载metric cache，拿一个token
        logger.info("  加载metric cache...")
        metric_cache_loader = MetricCacheLoader(Path(metric_cache_path))
        cache_tokens = list(metric_cache_loader.tokens)
        logger.info(f"  metric cache中有 {len(cache_tokens)} 个tokens")

        if len(cache_tokens) == 0:
            logger.error("  metric cache为空! 先运行metric caching.")
            return None

        token = cache_tokens[0]
        logger.info(f"  使用token: {token}")

        # 2. 加载NavSim数据拿到agent_input和scene
        logger.info("  加载NavSim场景数据...")
        sensor_config = SensorConfig(
            cam_f0=[0, 1, 2, 3], cam_l0=[0, 1, 2, 3], cam_l1=False, cam_l2=False,
            cam_r0=[0, 1, 2, 3], cam_r1=False, cam_r2=False, cam_b0=False, lidar_pc=False,
        )
        scene_filter = SceneFilter(num_history_frames=4, num_future_frames=10, max_scenes=2)

        scene_loader = SceneLoader(
            data_path=Path(navsim_log_path),
            synthetic_sensor_path=Path(sensor_blobs_path),
            original_sensor_path=Path(sensor_blobs_path),
            synthetic_scenes_path=Path(navsim_log_path),
            scene_filter=scene_filter,
            sensor_config=sensor_config,
        )

        # 用metric cache的token在scene_loader里查找对应的agent_input
        if token not in scene_loader.tokens_stage_one:
            logger.warning(f"  token {token} 不在stage-1 tokens中，使用第一个可用token")
            token = scene_loader.tokens_stage_one[0]

        agent_input = scene_loader.get_agent_input_from_token(token)
        scene = scene_loader.get_scene_from_token(token)

        # 3. 加载模型并推理
        logger.info("  加载Alpamayo模型并推理...")
        if not torch.cuda.is_available():
            logger.error("CUDA不可用!")
            return None

        agent = AlpamayoAgent(
            trajectory_sampling=proposal_sampling,
            model_path=model_path,
        )
        agent.initialize()
        trajectory = agent.compute_trajectory(agent_input)

        logger.info(f"  推理输出: shape={trajectory.poses.shape}")

        # 4. 从metric cache拿对应的数据并计算PDM score
        logger.info("  计算PDM score...")
        metric_cache = metric_cache_loader.get_from_token(token)

        simulator = PDMSimulator(proposal_sampling=proposal_sampling)
        scorer_config = PDMScorerConfig(proposal_sampling=proposal_sampling)
        scorer = PDMScorer(config=scorer_config)
        traffic_policy = LogReplayTrafficAgents(proposal_sampling)

        score_row, _ = pdm_score(
            metric_cache=metric_cache,
            model_trajectory=trajectory,
            future_sampling=proposal_sampling,
            simulator=simulator,
            scorer=scorer,
            traffic_agents_policy=traffic_policy,
        )

        logger.info("  PDM评分结果:")
        score_val = score_row["pdm_score"].iloc[0] if "pdm_score" in score_row.columns else "N/A"
        logger.info(f"  PDM Score: {score_val}")

        for col in score_row.columns:
            if col not in {"weighted_metrics", "weighted_metrics_array", "ego_simulated_states"}:
                logger.info(f"    {col}: {score_row[col].iloc[0]}")

        logger.info("STEP 5: PDM SCORE OK")
        return score_row
    except Exception as e:
        logger.error(f"STEP 5: FAIL - {e}")
        import traceback; traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description="分步测试Alpamayo1.5在NavSim上的表现")
    parser.add_argument("--step", type=int, required=True, choices=[1,2,3,4,5],
                        help="运行哪一步测试")
    parser.add_argument("--navsim_log_path", type=str,
                        default=os.environ.get("OPENSCENE_DATA_ROOT", ""))
    parser.add_argument("--sensor_blobs_path", type=str,
                        default=os.environ.get("OPENSCENE_DATA_ROOT", ""))
    parser.add_argument("--metric_cache_path", type=str,
                        default=os.environ.get("OPENSCENE_DATA_ROOT", "") + "/metric_cache")
    parser.add_argument("--model_path", type=str, default="/data/mnt_m181/z59900495/workspace/model/Alpamayo-1.5-10B")
    args = parser.parse_args()

    if args.step == 1:
        step1_imports()
    elif args.step == 2:
        if not args.navsim_log_path:
            logger.error("需要 --navsim_log_path 或设置 OPENSCENE_DATA_ROOT")
            sys.exit(1)
        step2_navsim_data(args.navsim_log_path, args.sensor_blobs_path)
    elif args.step == 3:
        if not args.navsim_log_path:
            logger.error("需要数据路径")
            sys.exit(1)
        agent_input, scene = step2_navsim_data(args.navsim_log_path, args.sensor_blobs_path)
        if agent_input:
            step3_conversion(agent_input)
    elif args.step == 4:
        if not args.navsim_log_path:
            logger.error("需要数据路径")
            sys.exit(1)
        agent_input, scene = step2_navsim_data(args.navsim_log_path, args.sensor_blobs_path)
        if agent_input:
            step4_inference(agent_input, args.model_path)
    elif args.step == 5:
        if not args.navsim_log_path:
            logger.error("需要 --navsim_log_path 或设置 OPENSCENE_DATA_ROOT")
            sys.exit(1)
        step5_pdm_score(
            navsim_log_path=args.navsim_log_path,
            sensor_blobs_path=args.sensor_blobs_path,
            metric_cache_path=args.metric_cache_path,
            model_path=args.model_path,
        )


if __name__ == "__main__":
    main()

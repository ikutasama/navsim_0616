"""
Standalone inference script: run Alpamayo1.5 on NavSim data without Hydra.

This script demonstrates the complete pipeline:
1. Load NavSim scene data from pickle/metadata
2. Convert to Alpamayo input format
3. Run inference
4. Convert output to NavSim Trajectory format
5. Compute minADE against ground truth

Usage:
    python run_alpamayo_on_navsim.py \
        --navsim_log_path /path/to/navsim_logs \
        --sensor_blobs_path /path/to/sensor_blobs \
        --model_path nvidia/Alpamayo-1.5-10B \
        --token <scene_token>

For batch evaluation, use the NavSim Hydra-based evaluation pipeline with
the AlpamayoAgent class instead.
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import scipy.spatial.transform as spt
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_navsim_scene(token: str, navsim_log_path: Path, sensor_blobs_path: Path):
    """Load a NavSim scene and extract AgentInput-like data.

    This is a simplified version that loads directly from pickle files.
    For full NavSim integration, use SceneLoader.
    """
    from navsim.common.dataloader import SceneLoader, SceneFilter
    from navsim.common.dataclasses import SensorConfig

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

    scene_filter = SceneFilter(
        num_history_frames=4,
        num_future_frames=10,
        tokens=[token],
    )

    scene_loader = SceneLoader(
        data_path=navsim_log_path,
        synthetic_sensor_path=sensor_blobs_path,
        original_sensor_path=sensor_blobs_path,
        synthetic_scenes_path=navsim_log_path,
        scene_filter=scene_filter,
        sensor_config=sensor_config,
    )

    agent_input = scene_loader.get_agent_input_from_token(token)
    scene = scene_loader.get_scene_from_token(token)
    gt_trajectory = scene.get_future_trajectory()

    return agent_input, gt_trajectory


def navsim_to_alpamayo(agent_input):
    """Convert NavSim AgentInput to Alpamayo model inputs."""
    from einops import rearrange
    from alpamayo1_5 import helper as alp_helper

    NAVSIM_TO_ALPAMAYO_CAM = {
        "cam_f0": 1,
        "cam_l0": 0,
        "cam_r0": 2,
    }
    ACTIVE_CAMERAS = ["cam_f0", "cam_l0", "cam_r0"]
    num_frames = 4

    # Prepare images
    latest_cameras = agent_input.cameras[-1]
    image_frames_list = []
    camera_indices_list = []

    for cam_name in ACTIVE_CAMERAS:
        cam_obj = getattr(latest_cameras, cam_name)
        if cam_obj.image is None:
            continue
        img_np = cam_obj.image
        img_tensor = torch.from_numpy(img_np.astype(np.float32))
        img_tensor = rearrange(img_tensor, "h w c -> c h w")
        frames_tensor = img_tensor.unsqueeze(0).repeat(num_frames, 1, 1, 1)
        image_frames_list.append(frames_tensor)
        camera_indices_list.append(NAVSIM_TO_ALPAMAYO_CAM[cam_name])

    image_frames = torch.stack(image_frames_list, dim=0)
    camera_indices = torch.tensor(camera_indices_list, dtype=torch.int64)
    sort_order = torch.argsort(camera_indices)
    image_frames = image_frames[sort_order]
    camera_indices = camera_indices[sort_order]

    # Prepare ego history
    num_history_steps = 16
    time_step = 0.1
    history_poses = [es.ego_pose for es in agent_input.ego_statuses]
    navsim_interval = 0.5
    navsim_times = np.array([
        -(len(history_poses) - 1 - i) * navsim_interval
        for i in range(len(history_poses))
    ])

    alp_times = np.arange(
        -(num_history_steps - 1) * time_step,
        time_step / 2,
        time_step,
    )

    xs = np.array([p[0] for p in history_poses])
    ys = np.array([p[1] for p in history_poses])
    headings = np.array([p[2] for p in history_poses])

    interp_x = np.interp(alp_times, navsim_times, xs)
    interp_y = np.interp(alp_times, navsim_times, ys)
    headings_unwrapped = np.unwrap(headings)
    interp_heading = np.interp(alp_times, navsim_times, headings_unwrapped)

    interp_xyz = np.stack([interp_x, interp_y, np.zeros(num_history_steps)], axis=-1)
    interp_rot = spt.Rotation.from_euler('z', interp_heading.reshape(-1, 1)).as_matrix()

    ego_history_xyz = torch.from_numpy(interp_xyz).float().unsqueeze(0).unsqueeze(0)
    ego_history_rot = torch.from_numpy(interp_rot).float().unsqueeze(0).unsqueeze(0)

    # Build messages
    frames_flat = image_frames.flatten(0, 1)
    messages = alp_helper.create_message(
        frames=frames_flat,
        camera_indices=camera_indices,
    )

    processor = None  # Will be set externally
    return {
        "messages": messages,
        "image_frames": image_frames,
        "camera_indices": camera_indices,
        "ego_history_xyz": ego_history_xyz,
        "ego_history_rot": ego_history_rot,
    }


def alpamayo_to_navsim(pred_xyz, pred_rot, time_horizon=4.0, interval_length=0.5):
    """Convert Alpamayo output to NavSim Trajectory poses."""
    xyz = pred_xyz.cpu().numpy()[0, 0, 0]  # (64, 3)
    rot = pred_rot.cpu().numpy()[0, 0, 0]  # (64, 3, 3)

    num_poses = int(time_horizon / interval_length)
    navsim_indices = [
        int(round((i + 1) * interval_length / 0.1)) - 1
        for i in range(num_poses)
    ]
    navsim_indices = [min(max(idx, 0), len(xyz) - 1) for idx in navsim_indices]

    poses = np.zeros((num_poses, 3), dtype=np.float32)
    for i, idx in enumerate(navsim_indices):
        poses[i, 0] = xyz[idx, 0]
        poses[i, 1] = xyz[idx, 1]
        poses[i, 2] = np.arctan2(rot[idx, 1, 0], rot[idx, 0, 0])

    return poses


def main():
    parser = argparse.ArgumentParser(description="Run Alpamayo1.5 on NavSim data")
    parser.add_argument("--navsim_log_path", type=str, required=True)
    parser.add_argument("--sensor_blobs_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, default="nvidia/Alpamayo-1.5-10B")
    parser.add_argument("--token", type=str, required=True, help="NavSim scene token")
    parser.add_argument("--num_traj_samples", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    # Load model
    logger.info("Loading Alpamayo model...")
    from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
    from alpamayo1_5 import helper

    model = Alpamayo1_5.from_pretrained(args.model_path, dtype=torch.bfloat16).to(args.device)
    processor = helper.get_processor(model.tokenizer)

    # Load NavSim data
    logger.info(f"Loading NavSim scene: {args.token}")
    agent_input, gt_trajectory = load_navsim_scene(
        args.token,
        Path(args.navsim_log_path),
        Path(args.sensor_blobs_path),
    )

    # Convert to Alpamayo format
    logger.info("Converting NavSim input to Alpamayo format...")
    alp_data = navsim_to_alpamayo(agent_input)

    # Process through tokenizer
    inputs = processor.apply_chat_template(
        alp_data["messages"],
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=True,
        return_dict=True,
        return_tensors="pt",
    )

    model_inputs = {
        "tokenized_data": inputs,
        "ego_history_xyz": alp_data["ego_history_xyz"],
        "ego_history_rot": alp_data["ego_history_rot"],
    }
    model_inputs = helper.to_device(model_inputs, args.device)

    # Run inference
    logger.info("Running inference...")
    torch.cuda.manual_seed_all(42)
    with torch.autocast(args.device, dtype=torch.bfloat16):
        pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
            data=model_inputs,
            top_p=0.98,
            temperature=0.6,
            num_traj_samples=args.num_traj_samples,
            max_generation_length=256,
            return_extra=True,
        )

    # Print CoT
    logger.info(f"Chain-of-Causation:\n{extra['cot'][0]}")

    # Convert to NavSim poses
    pred_poses = alpamayo_to_navsim(pred_xyz, pred_rot)
    logger.info(f"Predicted NavSim poses (x, y, heading):\n{pred_poses}")

    # Compare with ground truth
    gt_poses = gt_trajectory.poses  # (N, 3) [x, y, heading]
    logger.info(f"Ground truth poses:\n{gt_poses}")

    # Compute ADE (Average Displacement Error) on xy
    min_num = min(len(pred_poses), len(gt_poses))
    ade = np.linalg.norm(pred_poses[:min_num, :2] - gt_poses[:min_num, :2], axis=1).mean()
    logger.info(f"ADE (xy): {ade:.4f} meters")


if __name__ == "__main__":
    main()

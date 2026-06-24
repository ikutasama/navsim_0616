"""
Alpamayo1.5 Agent for NavSim evaluation.

This agent wraps the Alpamayo1.5 VLA model to produce trajectory predictions
that can be evaluated by NavSim's PDM scorer.

Key adaptation points:
1. NavSim provides 8 cameras (cam_f0/l0/l1/l2/r0/r1/r2/b0) at 2Hz (4 history frames).
   Alpamayo expects 4 cameras (cross_left, front_wide, cross_right, front_tele) at 10Hz.
   We map NavSim cameras to Alpamayo's camera indices and use only the most recent frame
   as the "current" observation (repeated 4 times for temporal consistency).

2. NavSim ego status is SE2 (x, y, heading) in local frame.
   Alpamayo expects ego_history_xyz (3D position) and ego_history_rot (3x3 rotation matrix).
   We reconstruct 3D pose from SE2 by setting z=0 and converting heading to rotation matrix.

3. Alpamayo outputs 64 waypoints at 10Hz (6.4s) in 3D (x,y,z) + rotation.
   NavSim needs 8 poses at 0.5Hz (4s) in SE2 (x, y, heading).
   We downsample and convert.

4. NavSim AgentInput provides 4 history ego statuses (2s at 2Hz).
   Alpamayo expects 16 history steps at 10Hz (1.6s).
   We interpolate from NavSim's sparse history to fill 16 steps.
"""

import numpy as np
import torch
import scipy.spatial.transform as spt
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import AgentInput, SensorConfig, Trajectory


# Camera name mapping: NavSim camera -> Alpamayo camera index
# Alpamayo camera indices (from helper.py CAMERA_DISPLAY_NAMES):
#   0: Front left (cross_left_120fov)
#   1: Front (front_wide_120fov)
#   2: Front right (cross_right_120fov)
#   3: Rear left
#   4: Rear
#   5: Rear right
#   6: Front telephoto (front_tele_30fov)
NAVSIM_TO_ALPAMAYO_CAM = {
    "cam_f0": 1,   # front_wide -> Front camera (index 1)
    "cam_l0": 0,   # cross_left -> Front left camera (index 0)
    "cam_l1": 0,   # further left, still map to front left
    "cam_l2": 0,   # even further left
    "cam_r0": 2,   # cross_right -> Front right camera (index 2)
    "cam_r1": 2,   # further right
    "cam_r2": 2,   # even further right
    "cam_b0": 4,   # rear -> Rear camera (index 4)
}

# Which NavSim cameras to actually use (subset that maps well to Alpamayo's training cameras)
ACTIVE_CAMERAS = ["cam_f0", "cam_l0", "cam_r0"]


class AlpamayoAgent(AbstractAgent):
    """Agent that uses Alpamayo1.5 VLA model for NavSim trajectory prediction."""

    requires_scene = False

    def __init__(
        self,
        model_path: str = "/data/mnt_m181/z59900495/workspace/model/Alpamayo-1.5-10B",
        trajectory_sampling: TrajectorySampling = TrajectorySampling(time_horizon=4, interval_length=0.5),
        num_traj_samples: int = 1,
        top_p: float = 0.98,
        temperature: float = 0.6,
        max_generation_length: int = 256,
        device: str = "cuda",
    ):
        super().__init__(trajectory_sampling)
        self._model_path = model_path
        self._num_traj_samples = num_traj_samples
        self._top_p = top_p
        self._temperature = temperature
        self._max_generation_length = max_generation_length
        self._device = device
        self._model = None
        self._processor = None
        self._last_cot_text = ""
        self._last_meta_action_text = ""
        self._last_answer_text = ""
        self._last_extra = None

    def name(self) -> str:
        return "Alpamayo1_5Agent"

    def initialize(self) -> None:
        """Load Alpamayo model and processor."""
        from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
        from alpamayo1_5 import helper

        self._model = Alpamayo1_5.from_pretrained(
            self._model_path, dtype=torch.bfloat16
        ).to(self._device)
        self._processor = helper.get_processor(self._model.tokenizer)

    def get_sensor_config(self) -> SensorConfig:
        """Request cameras for the current frame only (index 3 = most recent)."""
        # NavSim uses 4 history frames (indices 0,1,2,3), we only need the latest
        history_steps = [3]
        return SensorConfig(
            cam_f0=history_steps,
            cam_l0=history_steps,
            cam_l1=False,
            cam_l2=False,
            cam_r0=history_steps,
            cam_r1=False,
            cam_r2=False,
            cam_b0=False,
            lidar_pc=False,
        )

    def compute_trajectory(self, agent_input: AgentInput) -> Trajectory:
        """Run Alpamayo inference and convert output to NavSim Trajectory."""
        from alpamayo1_5 import helper as alp_helper

        # Step 1: Prepare camera images
        image_frames, camera_indices = self._prepare_images(agent_input)

        # Step 2: Prepare ego history
        ego_history_xyz, ego_history_rot = self._prepare_ego_history(agent_input)

        # Step 3: Build chat messages
        frames_flat = image_frames.flatten(0, 1)  # (N_cameras * N_frames, C, H, W)
        messages = alp_helper.create_message(
            frames=frames_flat,
            camera_indices=camera_indices,
        )

        # Step 4: Process through tokenizer
        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            continue_final_message=True,
            return_dict=True,
            return_tensors="pt",
        )

        model_inputs = {
            "tokenized_data": inputs,
            "ego_history_xyz": ego_history_xyz,
            "ego_history_rot": ego_history_rot,
        }
        model_inputs = alp_helper.to_device(model_inputs, self._device)

        # Step 5: Run inference
        torch.cuda.manual_seed_all(42)
        with torch.autocast(self._device, dtype=torch.bfloat16):
            pred_xyz, pred_rot, extra = self._model.sample_trajectories_from_data_with_vlm_rollout(
                data=model_inputs,
                top_p=self._top_p,
                temperature=self._temperature,
                num_traj_samples=self._num_traj_samples,
                max_generation_length=self._max_generation_length,
                return_extra=True,
            )

        # Save generated text outputs for downstream analysis.
        # extra is produced by alpamayo1_5.models.token_utils.extract_text_tokens()
        # and usually contains keys: cot, meta_action, answer with shape [B, ns, nj].
        self._last_extra = extra
        self._last_cot_text = self._extract_first_text(extra, "cot")
        self._last_meta_action_text = self._extract_first_text(extra, "meta_action")
        self._last_answer_text = self._extract_first_text(extra, "answer")

        # Step 6: Convert Alpamayo output to NavSim Trajectory
        trajectory = self._convert_output_to_trajectory(pred_xyz, pred_rot)

        return trajectory

    @staticmethod
    def _extract_first_text(extra, key: str) -> str:
        """Best-effort extraction of the first generated text string from Alpamayo extra dict."""
        try:
            if extra is None or key not in extra:
                return ""
            value = extra[key]
            if hasattr(value, "flatten"):
                value = value.flatten()[0]
            elif isinstance(value, (list, tuple)):
                value = value[0]
            return str(value)
        except Exception:
            return ""

    def _prepare_images(self, agent_input: AgentInput) -> tuple:
        """Extract camera images from NavSim AgentInput and format for Alpamayo.

        NavSim provides 4 history frames at 2Hz for each camera.
        Alpamayo expects 4 frames per camera at 10Hz (0.4s window).

        Strategy: Use the 4 NavSim history frames directly. Although their
        temporal spacing (0.5s intervals) differs from Alpamayo's expected
        0.1s intervals, this at least provides real motion information
        instead of repeating a single frame 4 times.

        Returns:
            image_frames: (N_cameras, num_frames, 3, H, W) tensor
            camera_indices: (N_cameras,) tensor with Alpamayo camera indices
        """
        from einops import rearrange

        num_frames = 4  # Alpamayo expects 4 frames per camera
        # NavSim history: agent_input.cameras[0..3] at 2Hz
        # Use the last 4 frames: [-1.5s, -1.0s, -0.5s, 0.0s]
        # Reverse to match Alpamayo's order: oldest first

        image_frames_list = []
        camera_indices_list = []

        for cam_name in ACTIVE_CAMERAS:
            frames_for_cam = []
            has_any_image = False

            for frame_idx in range(len(agent_input.cameras)):
                cam_obj = getattr(agent_input.cameras[frame_idx], cam_name)
                if cam_obj.image is not None:
                    img_np = cam_obj.image
                    img_tensor = torch.from_numpy(img_np.astype(np.float32))
                    img_tensor = rearrange(img_tensor, "h w c -> c h w")
                    frames_for_cam.append(img_tensor)
                    has_any_image = True

            if not has_any_image:
                continue

            # If we have fewer than num_frames frames, repeat the last one
            while len(frames_for_cam) < num_frames:
                frames_for_cam.append(frames_for_cam[-1])

            # Take the last num_frames frames (most recent)
            frames_for_cam = frames_for_cam[-num_frames:]

            # Stack: (num_frames, C, H, W)
            frames_tensor = torch.stack(frames_for_cam, dim=0)

            image_frames_list.append(frames_tensor)
            camera_indices_list.append(NAVSIM_TO_ALPAMAYO_CAM[cam_name])

        # Stack: (N_cameras, num_frames, 3, H, W)
        image_frames = torch.stack(image_frames_list, dim=0)
        camera_indices = torch.tensor(camera_indices_list, dtype=torch.int64)

        # Sort by camera index (Alpamayo expects sorted order)
        sort_order = torch.argsort(camera_indices)
        image_frames = image_frames[sort_order]
        camera_indices = camera_indices[sort_order]

        return image_frames, camera_indices

    def _prepare_ego_history(
        self, agent_input: AgentInput
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert NavSim ego history to Alpamayo format.

        NavSim provides 4 history frames at 2Hz (2 seconds).
        Alpamayo expects 16 history steps at 10Hz (1.6 seconds).

        We:
        1. Take the 4 NavSim SE2 poses (x, y, heading) in local frame
        2. Convert heading to 3x3 rotation matrix, set z=0 for xyz
        3. Interpolate from 4 poses (2s@2Hz) to 16 steps (1.6s@10Hz)

        Returns:
            ego_history_xyz: (1, 1, 16, 3)
            ego_history_rot: (1, 1, 16, 3, 3)
        """
        num_history_steps = 16
        time_step = 0.1  # 10Hz
        total_history_time = num_history_steps * time_step  # 1.6s

        # Collect history poses from NavSim
        # NavSim gives 4 frames covering 2s at 2Hz
        # Each ego_status.ego_pose is [x, y, heading] in local frame
        # (relative to the ego position at the current timestep)

        history_poses = []
        for ego_status in agent_input.ego_statuses:
            pose = ego_status.ego_pose  # [x, y, heading] local frame
            history_poses.append(pose)

        # history_poses has 4 entries: [-1.5s, -1.0s, -0.5s, 0.0s] relative to current
        # NavSim interval is 0.5s, 4 frames covering 2s
        navsim_interval = 0.5
        navsim_times = np.array([
            -(len(history_poses) - 1 - i) * navsim_interval
            for i in range(len(history_poses))
        ])  # e.g. [-1.5, -1.0, -0.5, 0.0]

        # Target times for Alpamayo: [-1.5, -1.4, ..., -0.1, 0.0]
        alp_times = np.arange(
            -(num_history_steps - 1) * time_step,
            time_step / 2,
            time_step,
        )  # 16 steps

        # Interpolate x, y, heading
        xs = np.array([p[0] for p in history_poses])
        ys = np.array([p[1] for p in history_poses])
        headings = np.array([p[2] for p in history_poses])

        # Use numpy interpolation for positions
        interp_x = np.interp(alp_times, navsim_times, xs)
        interp_y = np.interp(alp_times, navsim_times, ys)

        # For heading, need to handle wrap-around
        # Unwrap headings for interpolation, then wrap back
        headings_unwrapped = np.unwrap(headings)
        interp_heading = np.interp(alp_times, navsim_times, headings_unwrapped)
        interp_heading = interp_heading % (2 * np.pi)  # wrap back

        # Build xyz (z=0) and rotation matrices
        interp_xyz = np.stack([interp_x, interp_y, np.zeros(num_history_steps)], axis=-1)
        # from_euler('z', ...) requires last dim = 1 (number of axes)
        interp_rot = spt.Rotation.from_euler('z', interp_heading.reshape(-1, 1)).as_matrix()
        # (16, 3, 3)

        # Convert to tensors with batch dimensions: (1, 1, 16, 3) and (1, 1, 16, 3, 3)
        ego_history_xyz = torch.from_numpy(interp_xyz).float().unsqueeze(0).unsqueeze(0)
        ego_history_rot = torch.from_numpy(interp_rot).float().unsqueeze(0).unsqueeze(0)

        return ego_history_xyz, ego_history_rot

    def _convert_output_to_trajectory(
        self, pred_xyz: torch.Tensor, pred_rot: torch.Tensor
    ) -> Trajectory:
        """Convert Alpamayo's 64-step 10Hz 3D prediction to NavSim Trajectory.

        Alpamayo output:
            pred_xyz: (1, 1, num_traj_samples, 64, 3) - x,y,z in ego frame
            pred_rot: (1, 1, num_traj_samples, 64, 3, 3) - rotation matrices

        NavSim expects:
            Trajectory with poses (N, 3) where N=8, format [x, y, heading]
            at 0.5Hz over 4 seconds

        Steps:
        1. Take the best trajectory (minADE among samples if multiple)
        2. Downsample from 10Hz to 0.5Hz (take every 5th waypoint, 8 total)
        3. Convert rotation matrix to heading angle
        """
        # Take first (and likely only) trajectory sample
        # pred_xyz shape: (1, 1, num_traj_samples, T, 3)
        xyz = pred_xyz.cpu().numpy()[0, 0, 0]  # (64, 3)
        rot = pred_rot.cpu().numpy()[0, 0, 0]  # (64, 3, 3)

        # Alpamayo outputs at 10Hz for 6.4s = 64 waypoints
        # NavSim needs 4s at 0.5Hz = 8 poses
        # Downsample: take every 5th step starting from step 5 (0.5s into future)
        # Steps 5, 10, 15, 20, 25, 30, 35, 40 correspond to 0.5s, 1.0s, ..., 4.0s
        navsim_indices = [5, 10, 15, 20, 25, 30, 35, 40]  # 8 poses at 0.5s intervals

        # But let's also handle variable num_poses based on trajectory_sampling
        num_poses = self._trajectory_sampling.num_poses
        interval_length = self._trajectory_sampling.interval_length
        time_horizon = self._trajectory_sampling.time_horizon

        # Compute which Alpamayo indices to sample
        # Alpamayo step k corresponds to time (k+1)*0.1 seconds into the future
        # NavSim pose i corresponds to time (i+1)*interval_length seconds
        navsim_indices = [
            int(round((i + 1) * interval_length / 0.1)) - 1
            for i in range(num_poses)
        ]
        # Clamp to valid range
        navsim_indices = [min(max(idx, 0), len(xyz) - 1) for idx in navsim_indices]

        poses = np.zeros((num_poses, 3), dtype=np.float32)
        for i, idx in enumerate(navsim_indices):
            poses[i, 0] = xyz[idx, 0]  # x
            poses[i, 1] = xyz[idx, 1]  # y
            # Convert rotation matrix to heading (yaw angle)
            # heading = atan2(R[1,0], R[0,0]) for 2D rotation in xy plane
            poses[i, 2] = float(np.arctan2(rot[idx, 1, 0], rot[idx, 0, 0]))

        return Trajectory(poses, self._trajectory_sampling)

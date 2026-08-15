from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import glfw
import mujoco
import mujoco.viewer
import numpy as np
import yaml
from scipy.spatial.transform import Rotation, Slerp

G1_29DOF_SDK_JOINT_NAMES: list[str] = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

NUM_JOINTS = len(G1_29DOF_SDK_JOINT_NAMES)
WAIST_YAW_SDK_INDEX = 12
WAIST_ROLL_SDK_INDEX = 13
WAIST_PITCH_SDK_INDEX = 14
NPZ_MOTION_SUFFIX = ".npz"
NPZ_PELVIS_BODY_INDEX = 0
# MJLab saves body arrays in MuJoCo body order with the world body removed.
# In the G1 MJCF, torso_link has body id 16, so it is index 15 in the npz arrays.
NPZ_TORSO_BODY_INDEX = 15
DEPLOY_ROOT = Path(__file__).resolve().parent
PROJ_ROOT = DEPLOY_ROOT / "config/g1_29dof"
DEFAULT_CONFIG = "config/g1_29dof/config/policy/velocity/v0/params/deploy.yaml"
DEFAULT_FSM_CONFIG = PROJ_ROOT / "config/config.yaml"
DEFAULT_XML = "g1_description/g1_29dof_rev_1_0.xml"
CMD_STEP = 1.0


# ---------------------------------------------------------------------------
# Config / I/O
# ---------------------------------------------------------------------------


def resolve_policy_dir(policy_dir: Path) -> Path:
    """Match ``param::parser_policy_dir`` in unitree_rl_lab C++ deploy."""
    policy_dir = policy_dir.resolve()
    if not policy_dir.is_dir():
        raise FileNotFoundError(f"Policy directory not found: {policy_dir}")

    if (policy_dir / "exported").is_dir():
        return policy_dir

    subdirs = sorted(p for p in policy_dir.iterdir() if p.is_dir())
    for candidate in reversed(subdirs):
        if (candidate / "exported").is_dir():
            return candidate
    return policy_dir


def load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(path_str: str, deploy_root: Path, base: Path | None = None) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path.resolve()

    candidates: list[Path] = []
    if base is not None:
        candidates.append(base / path)
    candidates.extend([deploy_root / path, path])

    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.exists():
            return resolved
    if base is not None:
        return (base / path).resolve()
    return (deploy_root / path).resolve()


def load_deploy_config(config_path: str, deploy_root: Path) -> tuple[Path, dict[str, Any]]:
    cfg_path = resolve_path(config_path, deploy_root)
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Missing config: {cfg_path}")
    return cfg_path, load_yaml(cfg_path)


def needs_mujoco_attach(deploy_cfg: dict[str, Any]) -> bool:
    observations = deploy_cfg.get("observations") or {}
    return "key_body_pos_b" in observations or "depth" in observations


# ---------------------------------------------------------------------------
# Kinematics helpers
# ---------------------------------------------------------------------------


def root_ang_vel_from_mujoco(root_ang_vel_qvel: np.ndarray) -> np.ndarray:
    """MuJoCo free-joint ``qvel[3:6]`` is body-frame angular velocity (IMU gyro)."""
    return root_ang_vel_qvel.astype(np.float32)


def yaw_quaternion_wxyz(quat_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = quat_wxyz
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    half = yaw * 0.5
    return np.array([np.cos(half), 0.0, 0.0, np.sin(half)], dtype=np.float32)


def projected_gravity_b(quat_wxyz: np.ndarray) -> np.ndarray:
    rot = Rotation.from_quat(np.roll(quat_wxyz, -1))
    return rot.inv().apply(np.array([0.0, 0.0, -1.0], dtype=np.float64)).astype(np.float32)


def torso_quat_wxyz(root_quat_wxyz: np.ndarray, sdk_joint_pos: np.ndarray) -> np.ndarray:
    root = Rotation.from_quat(np.roll(root_quat_wxyz, -1))
    waist = Rotation.from_euler(
        "ZXY",
        [
            sdk_joint_pos[WAIST_YAW_SDK_INDEX],
            sdk_joint_pos[WAIST_ROLL_SDK_INDEX],
            sdk_joint_pos[WAIST_PITCH_SDK_INDEX],
        ],
    )
    return np.roll((root * waist).as_quat(), 1).astype(np.float32)


def root_local_rot_tan_norm(root_quat_wxyz: np.ndarray) -> np.ndarray:
    rotation = Rotation.from_quat(np.roll(root_quat_wxyz, -1))
    euler = rotation.as_euler("xyz")
    matrix = (Rotation.from_euler("z", euler[2]).inv() * rotation).as_matrix()
    return np.hstack((matrix[:, 0], matrix[:, 2])).astype(np.float32)


def key_body_pos_b(
    root_pos: np.ndarray,
    root_quat_wxyz: np.ndarray,
    body_pos_w: np.ndarray,
) -> np.ndarray:
    root_matrix_inv = Rotation.from_quat(np.roll(root_quat_wxyz, -1)).inv().as_matrix()
    return (root_matrix_inv @ (body_pos_w - root_pos)).astype(np.float32)


# ---------------------------------------------------------------------------
# Mimic motion loader (port of C++ ``State_Mimic::MotionLoader_``)
# ---------------------------------------------------------------------------


class MotionLoader:
    def __init__(self, motion_file: Path, fps: float):
        self.dt = 1.0 / fps
        self.anchor_quaternions: np.ndarray | None = None
        self.velocity_commands: np.ndarray | None = None

        if motion_file.suffix.lower() == NPZ_MOTION_SUFFIX:
            self._load_npz_motion(motion_file)
        else:
            self._load_csv_motion(motion_file)

        self.index_0 = 0
        self.index_1 = 0
        self.blend = 0.0
        self.update(0.0)

    def _load_csv_motion(self, motion_file: Path) -> None:
        rows = np.loadtxt(motion_file, delimiter=",", dtype=np.float32)
        if rows.ndim == 1:
            rows = rows[None, :]

        self.num_frames = rows.shape[0]
        self.duration = self.num_frames * self.dt
        self.root_positions = rows[:, :3].astype(np.float32)
        self.root_quaternions = np.stack([rows[:, 6], rows[:, 3], rows[:, 4], rows[:, 5]], axis=1).astype(np.float32)
        self.dof_positions = rows[:, 7:].astype(np.float32)
        self.dof_velocities = self._finite_difference_vel(self.dof_positions)
        self.root_lin_velocities = self._finite_difference_vel(self.root_positions)

    def _load_npz_motion(self, motion_file: Path) -> None:
        data = np.load(motion_file)
        self.dof_positions = data["joint_pos"].astype(np.float32)
        self.dof_velocities = (
            data["joint_vel"].astype(np.float32)
            if "joint_vel" in data
            else self._finite_difference_vel(self.dof_positions)
        )
        self.num_frames = self.dof_positions.shape[0]
        self.duration = self.num_frames * self.dt

        if "body_quat_w" in data:
            body_quat = data["body_quat_w"].astype(np.float32)
            self.root_quaternions = body_quat[:, NPZ_PELVIS_BODY_INDEX]
            if body_quat.shape[1] > NPZ_TORSO_BODY_INDEX:
                self.anchor_quaternions = body_quat[:, NPZ_TORSO_BODY_INDEX]
        elif "base_quat_w" in data:
            self.root_quaternions = data["base_quat_w"].astype(np.float32)
        else:
            self.root_quaternions = np.tile(
                np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                (self.num_frames, 1),
            )
        if "body_pos_w" in data:
            body_pos = data["body_pos_w"].astype(np.float32)
            self.root_positions = body_pos[:, NPZ_PELVIS_BODY_INDEX]
        elif "base_pos_w" in data:
            self.root_positions = data["base_pos_w"].astype(np.float32)
        elif "root_pos" in data:
            self.root_positions = data["root_pos"].astype(np.float32)
        else:
            self.root_positions = np.zeros((self.num_frames, 3), dtype=np.float32)
        self.root_lin_velocities = (
            data["body_lin_vel_w"][:, NPZ_PELVIS_BODY_INDEX].astype(np.float32)
            if "body_lin_vel_w" in data
            else self._finite_difference_vel(self.root_positions)
        )
        self.velocity_commands = (
            data["velocity_command"].astype(np.float32)
            if "velocity_command" in data
            else None
        )

    def _finite_difference_vel(self, positions: np.ndarray) -> np.ndarray:
        if positions.shape[0] <= 1:
            return np.zeros_like(positions)
        vel = np.zeros_like(positions)
        vel[:-1] = (positions[1:] - positions[:-1]) / self.dt
        vel[-1] = vel[-2]
        return vel

    def update(self, time: float) -> None:
        self.current_time = float(np.clip(time, 0.0, self.duration))
        phase = float(np.clip(self.current_time / self.duration, 0.0, 1.0))
        self.index_0 = int(round(phase * (self.num_frames - 1)))
        self.index_1 = min(self.index_0 + 1, self.num_frames - 1)
        # ``round(phase * (n-1))`` and ``time / dt`` can disagree at segment starts
        # (e.g. time_start=2.8), yielding blend outside [0, 1] and breaking Slerp.
        self.blend = float(
            np.clip(round((self.current_time - self.index_0 * self.dt) / self.dt * 1e5) / 1e5, 0.0, 1.0)
        )

    def joint_pos(self) -> np.ndarray:
        return self._interp_array(self.dof_positions)

    def joint_vel(self) -> np.ndarray:
        return self._interp_array(self.dof_velocities)

    def root_position(self) -> np.ndarray:
        return self._interp_array(self.root_positions)

    def root_lin_vel(self) -> np.ndarray:
        return self._interp_array(self.root_lin_velocities)

    def velocity_command(self) -> np.ndarray:
        if self.velocity_commands is None:
            quat_wxyz = self.root_quaternion()
            vel_w = self.root_lin_vel()
            rot = Rotation.from_quat(np.roll(quat_wxyz, -1))
            return rot.inv().apply(vel_w).astype(np.float32)
        # Match Isaac's baked-command lookup: round absolute motion time to the
        # nearest frame instead of interpolating command steps.
        frame_idx = int(np.clip(round(self.current_time / self.dt), 0, self.velocity_commands.shape[0] - 1))
        return self.velocity_commands[frame_idx].copy()

    def root_quaternion(self) -> np.ndarray:
        return self._interp_quat(self.root_quaternions)

    @property
    def has_anchor_quaternion(self) -> bool:
        return self.anchor_quaternions is not None

    def anchor_quaternion(self) -> np.ndarray:
        if self.anchor_quaternions is None:
            return self.root_quaternion()
        return self._interp_quat(self.anchor_quaternions)

    def _interp_array(self, values: np.ndarray) -> np.ndarray:
        return values[self.index_0] * (1.0 - self.blend) + values[self.index_1] * self.blend

    def _interp_quat(self, quats_wxyz: np.ndarray) -> np.ndarray:
        if self.index_0 == self.index_1:
            return quats_wxyz[self.index_0].copy()
        q0 = np.roll(quats_wxyz[self.index_0], -1)
        q1 = np.roll(quats_wxyz[self.index_1], -1)
        rots = Rotation.from_quat(np.vstack([q0, q1]))
        slerp = Slerp([0.0, 1.0], rots)
        return np.roll(slerp(self.blend).as_quat(), 1).astype(np.float32)

    def sample_at(self, time: float) -> tuple[np.ndarray, np.ndarray]:
        """Return (joint_pos_sdk, root_quat_wxyz) at ``time``."""
        self.update(time)
        return self.joint_pos(), self.root_quaternion()

    def root_lin_vel_at(self, time: float) -> np.ndarray:
        state = (self.current_time, self.index_0, self.index_1, self.blend)
        self.update(time)
        value = self.root_lin_vel()
        self.current_time, self.index_0, self.index_1, self.blend = state
        return value

    def root_quaternion_at(self, time: float) -> np.ndarray:
        state = (self.current_time, self.index_0, self.index_1, self.blend)
        self.update(time)
        value = self.root_quaternion()
        self.current_time, self.index_0, self.index_1, self.blend = state
        return value


@dataclass
class MimicRuntime:
    loader: MotionLoader
    time_start: float
    time_end: float
    segment_done: bool = False

    def update_for_step(self, episode_length: int, step_dt: float) -> None:
        motion_time = self.time_start + episode_length * step_dt
        if motion_time > self.time_end:
            if not self.segment_done:
                print(f"Mimic segment finished at t={self.time_end:.2f}s, simulation continues...")
                self.segment_done = True
            motion_time = self.time_end
        self.loader.update(motion_time)


def anchor_quat_from_motion(motion: MotionLoader) -> np.ndarray:
    if motion.has_anchor_quaternion:
        return motion.anchor_quaternion()

    root_quat = motion.root_quaternion()
    joint_pos = motion.joint_pos()
    root = Rotation.from_quat(np.roll(root_quat, -1))
    waist = Rotation.from_euler(
        "ZXY",
        [
            joint_pos[WAIST_YAW_SDK_INDEX],
            joint_pos[WAIST_ROLL_SDK_INDEX],
            joint_pos[WAIST_PITCH_SDK_INDEX],
        ],
    )
    return np.roll((root * waist).as_quat(), 1).astype(np.float32)


# ---------------------------------------------------------------------------
# Robot state
# ---------------------------------------------------------------------------


@dataclass
class RobotState:
    root_quat_wxyz: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    root_ang_vel_b: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    joint_pos_policy: np.ndarray = field(default_factory=lambda: np.zeros(NUM_JOINTS, dtype=np.float32))
    joint_vel_policy: np.ndarray = field(default_factory=lambda: np.zeros(NUM_JOINTS, dtype=np.float32))
    joint_pos_sdk: np.ndarray = field(default_factory=lambda: np.zeros(NUM_JOINTS, dtype=np.float32))
    velocity_command: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))


# ---------------------------------------------------------------------------
# Observation registry
# ---------------------------------------------------------------------------


@dataclass
class ObservationTermCfg:
    name: str
    scale: np.ndarray
    clip: tuple[float, float] | None
    history_length: int
    scale_first: bool
    first_push_fill: bool
    params: dict[str, Any]
    buffer: deque[np.ndarray] = field(default_factory=deque)
    _first_add: bool = True

    def reset(self, value: np.ndarray) -> None:
        self.buffer.clear()
        self._first_add = True
        if self.first_push_fill:
            return
        for _ in range(self.history_length):
            self.add(value.copy())

    def add(self, raw: np.ndarray) -> None:
        obs = raw.astype(np.float32, copy=True)
        for j in range(obs.size):
            if self.scale_first:
                if self.scale.size:
                    obs[j] *= self.scale[0] if self.scale.size == 1 else self.scale[j]
                if self.clip is not None:
                    obs[j] = float(np.clip(obs[j], self.clip[0], self.clip[1]))
            else:
                if self.clip is not None:
                    obs[j] = float(np.clip(obs[j], self.clip[0], self.clip[1]))
                if self.scale.size:
                    obs[j] *= self.scale[0] if self.scale.size == 1 else self.scale[j]
        if self.first_push_fill and self._first_add:
            for _ in range(self.history_length):
                self.buffer.append(obs.copy())
            self._first_add = False
            return
        self._first_add = False
        self.buffer.append(obs)
        while len(self.buffer) > self.history_length:
            self.buffer.popleft()

    def get_flat(self) -> np.ndarray:
        return np.concatenate(list(self.buffer), axis=0)


ObsFunc = Callable[["DeployEnv", dict[str, Any]], np.ndarray]
OBSERVATION_REGISTRY: dict[str, ObsFunc] = {}


def register_observation(name: str):
    def decorator(fn: ObsFunc) -> ObsFunc:
        OBSERVATION_REGISTRY[name] = fn
        return fn

    return decorator


def _velocity_command_obs(env: "DeployEnv") -> np.ndarray:
    cmd = env.robot.velocity_command.copy()
    cmd_ranges = (env.cfg.get("commands") or {}).get("base_velocity", {}).get("ranges", {})
    if cmd_ranges:
        cmd[0] = float(np.clip(cmd[0], cmd_ranges["lin_vel_x"][0], cmd_ranges["lin_vel_x"][1]))
        cmd[1] = float(np.clip(cmd[1], cmd_ranges["lin_vel_y"][0], cmd_ranges["lin_vel_y"][1]))
        cmd[2] = float(np.clip(cmd[2], cmd_ranges["ang_vel_z"][0], cmd_ranges["ang_vel_z"][1]))
    return cmd


@register_observation("base_ang_vel")
def obs_base_ang_vel(env: DeployEnv, params: dict[str, Any]) -> np.ndarray:
    return env.robot.root_ang_vel_b.copy()


@register_observation("base_ang_vel_body")
def obs_base_ang_vel_body(env: DeployEnv, params: dict[str, Any]) -> np.ndarray:
    # MuJoCo free-joint angular velocity is already expressed in the body frame.
    return env.robot.root_ang_vel_b.copy()


@register_observation("projected_gravity")
def obs_projected_gravity(env: DeployEnv, params: dict[str, Any]) -> np.ndarray:
    return projected_gravity_b(env.robot.root_quat_wxyz)


@register_observation("velocity_commands")
def obs_velocity_commands(env: DeployEnv, params: dict[str, Any]) -> np.ndarray:
    return _velocity_command_obs(env)


OBSERVATION_REGISTRY["keyboard_velocity_commands"] = obs_velocity_commands
OBSERVATION_REGISTRY["velocity_command"] = obs_velocity_commands


@register_observation("base_lin_vel_ref_command")
def obs_base_lin_vel_ref_command(env: DeployEnv, params: dict[str, Any]) -> np.ndarray:
    num_frames = int(params.get("num_frames", 10))
    source = str(params.get("source", "motion"))
    if source == "keyboard":
        command = _velocity_command_obs(env)
        command[2] = float(params.get("lin_vel_z", 0.0))
        return np.tile(command, num_frames).astype(np.float32)
    if env.motion is None:
        raise RuntimeError("base_lin_vel_ref_command requires a MotionLoader")
    frame_interval_s = float(params.get("frame_interval_s", 0.1))
    current_time = getattr(env.motion, "current_time", 0.0)
    frames = []
    state = (env.motion.current_time, env.motion.index_0, env.motion.index_1, env.motion.blend)
    for frame_i in range(num_frames):
        env.motion.update(current_time + frame_i * frame_interval_s)
        command = env.motion.velocity_command().copy()
        command[2] = float(params.get("lin_vel_z", command[2]))
        frames.append(command)
    env.motion.current_time, env.motion.index_0, env.motion.index_1, env.motion.blend = state
    return np.asarray(frames, dtype=np.float32).reshape(-1)


def _depth_model_and_data(env: "DeployEnv") -> tuple[mujoco.MjModel, mujoco.MjData]:
    if env._mj_model is None or env._mj_data is None:
        raise RuntimeError("depth requires attach_mujoco() with model and data")
    return env._depth_mj_model or env._mj_model, env._depth_mj_data or env._mj_data


def _sync_depth_model_if_present(env: "DeployEnv") -> None:
    if env._depth_mj_model is None or env._depth_mj_data is None or env._mj_data is None:
        return
    _sync_depth_model_state(
        env._mj_data,
        env._depth_mj_model,
        env._depth_mj_data,
        wrap_x=_optional_tuple2_float(env.cfg.get("depth_wrap_x_range"), "depth_wrap_x_range"),
        wrap_y=_optional_tuple2_float(env.cfg.get("depth_wrap_y_range"), "depth_wrap_y_range"),
    )


@register_observation("joint_pos")
def obs_joint_pos(env: DeployEnv, params: dict[str, Any]) -> np.ndarray:
    return env.robot.joint_pos_policy.copy()


@register_observation("joint_pos_rel")
def obs_joint_pos_rel(env: DeployEnv, params: dict[str, Any]) -> np.ndarray:
    return env.robot.joint_pos_policy - env.default_joint_pos


@register_observation("joint_vel")
def obs_joint_vel(env: DeployEnv, params: dict[str, Any]) -> np.ndarray:
    return env.robot.joint_vel_policy.copy()


@register_observation("joint_vel_rel")
def obs_joint_vel_rel(env: DeployEnv, params: dict[str, Any]) -> np.ndarray:
    return env.robot.joint_vel_policy.copy()


@register_observation("last_action")
def obs_last_action(env: DeployEnv, params: dict[str, Any]) -> np.ndarray:
    return env.raw_action.copy()


@register_observation("gait_phase")
def obs_gait_phase(env: DeployEnv, params: dict[str, Any]) -> np.ndarray:
    period = float(params["period"])
    offset = params.get("offset", 0.5)
    if isinstance(offset, (list, tuple)):
        offset = float(offset[-1])
    phase = ((env.episode_length * env.step_dt) % period) / period
    phase_left = phase
    phase_right = (phase + float(offset)) % 1.0
    return np.array(
        [
            np.sin(2.0 * np.pi * phase_left),
            np.sin(2.0 * np.pi * phase_right),
            np.cos(2.0 * np.pi * phase_left),
            np.cos(2.0 * np.pi * phase_right),
        ],
        dtype=np.float32,
    )


@register_observation("root_local_rot_tan_norm")
def obs_root_local_rot_tan_norm(env: DeployEnv, params: dict[str, Any]) -> np.ndarray:
    return root_local_rot_tan_norm(env.robot.root_quat_wxyz)


@register_observation("key_body_pos_b")
def obs_key_body_pos_b(env: DeployEnv, params: dict[str, Any]) -> np.ndarray:
    if env._mj_data is None:
        raise RuntimeError("key_body_pos_b requires attach_mujoco()")
    key_body_names = params.get("key_body_names") or []
    root_pos = env._mj_data.qpos[:3].astype(np.float32)
    out = np.zeros(len(key_body_names) * 3, dtype=np.float32)
    for i, body_name in enumerate(key_body_names):
        pos_w = env._mj_data.body(body_name).xpos.astype(np.float32)
        out[i * 3 : (i + 1) * 3] = key_body_pos_b(root_pos, env.robot.root_quat_wxyz, pos_w)
    return out


@register_observation("motion_command")
def obs_motion_command(env: DeployEnv, params: dict[str, Any]) -> np.ndarray:
    if env.motion is None:
        raise RuntimeError("motion_command requires a MotionLoader")
    pos_sdk = env.motion.joint_pos()
    vel_sdk = env.motion.joint_vel()
    return np.concatenate([pos_sdk[env.joint_ids_map], vel_sdk[env.joint_ids_map]], axis=0)


@register_observation("motion_joint_pos")
def obs_motion_joint_pos(env: DeployEnv, params: dict[str, Any]) -> np.ndarray:
    if env.motion is None:
        raise RuntimeError("motion_joint_pos requires a MotionLoader")
    return env.motion.joint_pos()[env.joint_ids_map].astype(np.float32)


@register_observation("motion_joint_vel")
def obs_motion_joint_vel(env: DeployEnv, params: dict[str, Any]) -> np.ndarray:
    if env.motion is None:
        raise RuntimeError("motion_joint_vel requires a MotionLoader")
    return env.motion.joint_vel()[env.joint_ids_map].astype(np.float32)


@register_observation("motion_anchor_ori_b")
def obs_motion_anchor_ori_b(env: DeployEnv, params: dict[str, Any]) -> np.ndarray:
    if env.motion is None or env.init_quat is None:
        raise RuntimeError("motion_anchor_ori_b requires motion and init_quat")
    real_quat = torso_quat_wxyz(env.robot.root_quat_wxyz, env.robot.joint_pos_sdk)
    ref_quat = anchor_quat_from_motion(env.motion)
    real_rot = Rotation.from_quat(np.roll(real_quat, -1))
    ref_rot = Rotation.from_quat(np.roll(ref_quat, -1))
    init_rot = Rotation.from_quat(np.roll(env.init_quat, -1))
    rel_rot = (init_rot * ref_rot).inv() * real_rot
    mat = rel_rot.as_matrix().T
    return np.array(
        [mat[0, 0], mat[0, 1], mat[1, 0], mat[1, 1], mat[2, 0], mat[2, 1]],
        dtype=np.float32,
    )


@register_observation("depth")
def obs_depth(env: DeployEnv, params: dict[str, Any]) -> np.ndarray:
    _sync_depth_model_if_present(env)
    depth_model, depth_data = _depth_model_and_data(env)

    depth_type = str(params.get("type", "native"))
    camera_name = str(params.get("camera_name", "depth_cam"))
    if depth_type == "native":
        height = int(params.get("height", 30))
        width = int(params.get("width", 40))
        depth_max = float(params.get("max", 2.0))
        renderer = env.depth_renderer(f"{depth_type}:{camera_name}:{height}x{width}", height, width, depth_model)
        return _depth_image_native(renderer, depth_data, camera_name, depth_max)

    if depth_type == "cropped_normalized":
        cam_id = int(mujoco.mj_name2id(depth_model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name))
        if cam_id < 0:
            raise RuntimeError(f"Camera '{camera_name}' not found in model")
        depth_model.cam_pos[cam_id] = np.asarray(params["camera_pos"], dtype=np.float64)
        depth_model.cam_quat[cam_id] = np.asarray(params["camera_quat_opengl"], dtype=np.float64)
        depth_model.cam_fovy[cam_id] = float(params["camera_fovy"])

        render_h = int(params["render_height"])
        render_w = int(params["render_width"])
        renderer = env.depth_renderer(f"{depth_type}:{camera_name}:{render_h}x{render_w}", render_h, render_w, depth_model)
        return _cropped_normalized_depth_image(
            renderer,
            depth_data,
            camera_name,
            crop_region=_tuple4_int(params["crop_region"], "observations.depth.crop_region"),
            depth_range=_tuple2_float(params["range"], "observations.depth.range"),
            blur_sigma=float(params.get("blur_sigma", 1.0)),
            resize_shape=_optional_tuple2_int(params.get("resize_shape"), "observations.depth.resize_shape"),
            min_distance=float(params.get("min_distance", 0.0)),
        )

    raise ValueError(f"Unsupported depth observation type: {depth_type}")


# ---------------------------------------------------------------------------
# Action terms
# ---------------------------------------------------------------------------


class ActionTerm:
    def __init__(self, cfg: dict[str, Any], env: "DeployEnv"):
        self.cfg = cfg
        self.env = env

    @property
    def action_dim(self) -> int:
        raise NotImplementedError

    @property
    def processed_actions(self) -> np.ndarray:
        raise NotImplementedError

    def process_actions(self, action: np.ndarray) -> None:
        raise NotImplementedError

    def reset(self) -> None:
        pass


class JointPositionAction(ActionTerm):
    def __init__(self, cfg: dict[str, Any], env: "DeployEnv"):
        super().__init__(cfg, env)
        self.scale = np.array(cfg["scale"], dtype=np.float32)
        self.offset = np.array(cfg["offset"], dtype=np.float32)
        clip_raw = cfg.get("clip")
        self.clip = None if clip_raw is None else np.array(clip_raw, dtype=np.float32)
        self._raw = np.zeros(len(self.scale), dtype=np.float32)
        self._processed = self.offset.copy()

    @property
    def action_dim(self) -> int:
        return len(self.scale)

    @property
    def processed_actions(self) -> np.ndarray:
        return self._processed

    def process_actions(self, action: np.ndarray) -> None:
        self._raw[:] = action
        self._processed[:] = self._raw * self.scale + self.offset
        if self.clip is not None:
            for i in range(self.action_dim):
                self._processed[i] = np.clip(
                    self._processed[i],
                    self.clip[i][0],
                    self.clip[i][1],
                )

    def reset(self) -> None:
        self._raw[:] = 0.0
        self._processed[:] = self.offset

    def seed_from_sdk_joint_pos(
        self,
        joint_pos_sdk: np.ndarray,
        joint_ids_map: np.ndarray,
        *,
        seed_raw: bool = True,
    ) -> np.ndarray:
        """Set processed (and optionally raw) actions so PD targets match SDK joint positions."""
        for policy_i, sdk_i in enumerate(joint_ids_map):
            processed = float(joint_pos_sdk[sdk_i])
            self._processed[policy_i] = processed
            if seed_raw:
                self._raw[policy_i] = (processed - self.offset[policy_i]) / self.scale[policy_i]
        return self._raw.copy()


ACTION_REGISTRY: dict[str, type[ActionTerm]] = {
    "JointPositionAction": JointPositionAction,
}


# ---------------------------------------------------------------------------
# Deploy environment (port of C++ ``ManagerBasedRLEnv``)
# ---------------------------------------------------------------------------


class DeployEnv:
    def __init__(
        self,
        deploy_cfg: dict[str, Any],
        *,
        motion_loader: MotionLoader | None = None,
        velocity_command: np.ndarray | None = None,
    ):
        self.cfg = deploy_cfg
        self.step_dt = float(deploy_cfg["step_dt"])
        self.joint_ids_map = np.array(deploy_cfg["joint_ids_map"], dtype=np.int32)
        self.default_joint_pos = np.array(deploy_cfg["default_joint_pos"], dtype=np.float32)
        self.stiffness = np.array(deploy_cfg["stiffness"], dtype=np.float32)
        self.damping = np.array(deploy_cfg["damping"], dtype=np.float32)

        self.robot = RobotState()
        if velocity_command is not None:
            # Share the caller's buffer so keyboard updates are visible immediately.
            self.robot.velocity_command = (
                velocity_command
                if velocity_command.dtype == np.float32
                else velocity_command.astype(np.float32)
            )

        self.motion = motion_loader
        self.init_quat: np.ndarray | None = None
        self.episode_length = 0
        self._mj_data: Any | None = None
        self._mj_model: Any | None = None
        self._depth_mj_data: Any | None = None
        self._depth_mj_model: Any | None = None
        self._depth_renderers: dict[str, mujoco.Renderer] = {}

        self.action_terms = self._prepare_action_terms()
        self.action_dim = sum(term.action_dim for term in self.action_terms)
        self.raw_action = np.zeros(self.action_dim, dtype=np.float32)
        self.sensor_terms = self._prepare_sensor_terms()
        self.obs_terms = self._prepare_observation_terms()

    def attach_mujoco(
        self,
        mj_data: Any,
        mj_model: Any | None = None,
        *,
        depth_mj_data: Any | None = None,
        depth_mj_model: Any | None = None,
    ) -> None:
        self._mj_data = mj_data
        self._mj_model = mj_model
        self._depth_mj_data = depth_mj_data
        self._depth_mj_model = depth_mj_model

    def depth_renderer(self, key: str, height: int, width: int, model: mujoco.MjModel | None = None) -> mujoco.Renderer:
        renderer = self._depth_renderers.get(key)
        if renderer is None:
            render_model = model or self._mj_model
            if render_model is None:
                raise RuntimeError("depth_renderer requires attach_mujoco() with a model")
            renderer = mujoco.Renderer(render_model, height=height, width=width)
            renderer.enable_depth_rendering()
            self._depth_renderers[key] = renderer
        return renderer

    def sync_robot_from_mujoco(
        self,
        qpos: np.ndarray,
        qvel: np.ndarray,
        sdk_qposadr: np.ndarray,
        sdk_qveladr: np.ndarray,
    ) -> None:
        """Update ``robot`` from MuJoCo state (free joint + SDK-indexed joints)."""
        self.robot.root_quat_wxyz = qpos[3:7].astype(np.float32)
        self.robot.root_ang_vel_b = root_ang_vel_from_mujoco(qvel[3:6])
        self.robot.joint_pos_sdk = qpos[sdk_qposadr].astype(np.float32)
        joint_vel_sdk = qvel[sdk_qveladr].astype(np.float32)
        self.robot.joint_pos_policy = self.robot.joint_pos_sdk[self.joint_ids_map]
        self.robot.joint_vel_policy = joint_vel_sdk[self.joint_ids_map]

    def _prepare_action_terms(self) -> list[ActionTerm]:
        terms: list[ActionTerm] = []
        for name, term_cfg in self.cfg["actions"].items():
            if name not in ACTION_REGISTRY:
                available = ", ".join(sorted(ACTION_REGISTRY))
                raise KeyError(f"Action term '{name}' is not registered. Available: {available}")
            terms.append(ACTION_REGISTRY[name](term_cfg, self))
        return terms

    def _prepare_observation_terms(self) -> list[ObservationTermCfg]:
        obs_cfg = self.cfg["observations"]
        scale_first = bool(obs_cfg.get("scale_first", False))
        first_push_history = bool(obs_cfg.get("first_push_history", False))
        terms: list[ObservationTermCfg] = []
        for name, term_yaml in obs_cfg.items():
            if name in ("scale_first", "use_gym_history", "first_push_history", "depth"):
                continue
            if name not in OBSERVATION_REGISTRY:
                available = ", ".join(sorted(OBSERVATION_REGISTRY))
                raise KeyError(f"Observation term '{name}' is not registered. Available: {available}")
            scale = np.atleast_1d(np.array(term_yaml.get("scale") or [], dtype=np.float32))
            clip_raw = term_yaml.get("clip")
            clip = None if clip_raw is None else (float(clip_raw[0]), float(clip_raw[1]))
            history_length = int(term_yaml.get("history_length", 1))
            params = dict(term_yaml.get("params") or {})
            terms.append(
                ObservationTermCfg(
                    name=name,
                    scale=scale,
                    clip=clip,
                    history_length=history_length,
                    scale_first=scale_first,
                    first_push_fill=first_push_history,
                    params=params,
                )
            )
        return terms

    def _prepare_sensor_terms(self) -> dict[str, dict[str, Any]]:
        obs_cfg = self.cfg.get("observations") or {}
        terms: dict[str, dict[str, Any]] = {}
        for name, term_yaml in obs_cfg.items():
            if name not in {"depth"}:
                continue
            if name not in OBSERVATION_REGISTRY:
                available = ", ".join(sorted(OBSERVATION_REGISTRY))
                raise KeyError(f"Sensor observation '{name}' is not registered. Available: {available}")
            terms[name] = dict(term_yaml or {})
        return terms

    def _compute_obs_term(self, term: ObservationTermCfg) -> np.ndarray:
        return OBSERVATION_REGISTRY[term.name](self, term.params)

    def compute_sensor(self, name: str) -> np.ndarray:
        if name not in self.sensor_terms:
            available = ", ".join(sorted(self.sensor_terms))
            raise KeyError(f"Sensor observation '{name}' is not configured. Available: {available}")
        return OBSERVATION_REGISTRY[name](self, self.sensor_terms[name])

    @property
    def processed_action(self) -> np.ndarray:
        for term in self.action_terms:
            if isinstance(term, JointPositionAction):
                return term.processed_actions.copy()
        raise RuntimeError("No JointPositionAction term found in deploy config")

    def reset(self) -> None:
        self.episode_length = 0
        self.raw_action[:] = 0.0
        for term in self.action_terms:
            term.reset()
        for term in self.obs_terms:
            term.reset(self._compute_obs_term(term))

    def prepare_mimic_episode(self, joint_pos_sdk: np.ndarray) -> None:
        """Yaw-align anchor frame and seed PD / last_action from motion reference."""
        if self.motion is None:
            return
        ref_yaw = Rotation.from_quat(np.roll(yaw_quaternion_wxyz(anchor_quat_from_motion(self.motion)), -1))
        robot_yaw = Rotation.from_quat(
            np.roll(yaw_quaternion_wxyz(torso_quat_wxyz(self.robot.root_quat_wxyz, self.robot.joint_pos_sdk)), -1)
        )
        init_rot = robot_yaw * ref_yaw.inv()
        self.init_quat = np.roll(init_rot.as_quat(), 1).astype(np.float32)
        self.reset()
        self._seed_actions_from_sdk_joint_pos(joint_pos_sdk)

    def _seed_actions_from_sdk_joint_pos(self, joint_pos_sdk: np.ndarray) -> None:
        seed_raw = bool((self.cfg.get("motion_initializer") or {}).get("seed_last_action", True))
        idx = 0
        for term in self.action_terms:
            if isinstance(term, JointPositionAction):
                raw = term.seed_from_sdk_joint_pos(joint_pos_sdk, self.joint_ids_map, seed_raw=seed_raw)
                self.raw_action[idx : idx + term.action_dim] = raw
            idx += term.action_dim
        for term in self.obs_terms:
            if term.name == "last_action":
                term.reset(self.raw_action.copy())

    def compute_obs(self) -> dict[str, np.ndarray]:
        for term in self.obs_terms:
            term.add(self._compute_obs_term(term))
        obs_vec = np.concatenate([term.get_flat() for term in self.obs_terms], axis=0)
        return {"obs": obs_vec}

    def process_action(self, action: np.ndarray) -> np.ndarray:
        self.raw_action[:] = action
        idx = 0
        for term in self.action_terms:
            term.process_actions(action[idx : idx + term.action_dim])
            idx += term.action_dim
        return self.processed_action

    def target_sdk_joint_pos(self) -> np.ndarray:
        target = np.zeros(NUM_JOINTS, dtype=np.float32)
        processed = self.processed_action
        for policy_i, sdk_i in enumerate(self.joint_ids_map):
            target[sdk_i] = processed[policy_i]
        return target


# ---------------------------------------------------------------------------
# Policy adapters
# ---------------------------------------------------------------------------


class PolicyAdapterBase:
    def __init__(self, model_path: Path) -> None:
        self.model_path = model_path

    @staticmethod
    def batched_vector(value: np.ndarray) -> np.ndarray:
        return np.asarray(value, dtype=np.float32)[None, :]

    @staticmethod
    def batched_depth(value: np.ndarray) -> np.ndarray:
        return np.asarray(value, dtype=np.float32)[None, None, :, :]

    @staticmethod
    def flat_action(value: np.ndarray) -> np.ndarray:
        return np.asarray(value, dtype=np.float32).reshape(-1)


class OnnxPolicyAdapter(PolicyAdapterBase):
    def __init__(self, model_path: Path):
        super().__init__(model_path)
        import onnxruntime as ort

        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self.input_names = [item.name for item in self.session.get_inputs()]
        self.output_names = [item.name for item in self.session.get_outputs()]

    def run(self, inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
        feed = {}
        for name in self.input_names:
            if name not in inputs:
                raise KeyError(f"ONNX input '{name}' missing. Available: {list(inputs)}")
            feed[name] = np.asarray(inputs[name], dtype=np.float32)
        return [np.asarray(out, dtype=np.float32) for out in self.session.run(self.output_names, feed)]

    def act(self, obs_map: dict[str, np.ndarray]) -> np.ndarray:
        inputs: dict[str, np.ndarray] = {}
        for name in self.input_names:
            if name not in obs_map:
                raise KeyError(f"Policy input '{name}' missing from observations. Available: {list(obs_map)}")
            inputs[name] = self.batched_vector(obs_map[name])
        outputs = self.run(inputs)
        return self.flat_action(outputs[0])


class TorchPolicyAdapter(PolicyAdapterBase):
    def __init__(self, model_path: Path, device: str = "cpu"):
        super().__init__(model_path)
        import torch

        self.torch = torch
        self.device = torch.device(device)
        self.model = torch.jit.load(str(model_path), map_location=self.device)
        self.model.eval()

    def tensor(self, value: Any) -> Any:
        if isinstance(value, self.torch.Tensor):
            return value.to(self.device)
        return self.torch.from_numpy(np.asarray(value, dtype=np.float32)).to(self.device)

    def run(self, *inputs: Any) -> Any:
        with self.torch.inference_mode():
            return self.model(*(self.tensor(value) for value in inputs))

    @staticmethod
    def numpy(value: Any) -> np.ndarray:
        return value.detach().cpu().numpy().astype(np.float32)

    def act(self, obs_map: dict[str, np.ndarray]) -> np.ndarray:
        return self.flat_action(self.numpy(self.run(self.batched_vector(obs_map["obs"]))))


class OnnxHistoryObsPolicyAdapter(OnnxPolicyAdapter):
    def act(self, history: np.ndarray, obs: np.ndarray) -> np.ndarray:  # type: ignore[override]
        inputs = {
            self.input_names[0]: self.batched_vector(history),
            self.input_names[1]: self.batched_vector(obs),
        }
        return self.flat_action(self.run(inputs)[0])


class TorchHistoryObsPolicyAdapter(TorchPolicyAdapter):
    def act(self, history: np.ndarray, obs: np.ndarray) -> np.ndarray:  # type: ignore[override]
        return self.flat_action(self.numpy(self.run(self.batched_vector(history), self.batched_vector(obs))))


class OnnxDepthRecurrentPolicyAdapter(OnnxPolicyAdapter):
    def __init__(self, model_path: Path, hidden_shape: tuple[int, int, int]):
        super().__init__(model_path)
        self.hidden = np.zeros(hidden_shape, dtype=np.float32)

    def act(self, obs: np.ndarray, depth: np.ndarray) -> np.ndarray:  # type: ignore[override]
        inputs = {
            self.input_names[0]: self.batched_vector(obs),
            self.input_names[1]: self.batched_depth(depth),
            self.input_names[2]: self.hidden,
        }
        outputs = self.run(inputs)
        self.hidden = outputs[1].astype(np.float32)
        return self.flat_action(outputs[0])


class TorchDepthRecurrentPolicyAdapter(TorchPolicyAdapter):
    def __init__(self, model_path: Path, device: str, hidden_shape: tuple[int, int, int]):
        super().__init__(model_path, device)
        self.hidden_t = self.torch.zeros(*hidden_shape, device=self.device, dtype=self.torch.float32)

    def act(self, obs: np.ndarray, depth: np.ndarray) -> np.ndarray:  # type: ignore[override]
        out = self.run(self.batched_vector(obs), self.batched_depth(depth), self.hidden_t)
        if not isinstance(out, tuple):
            raise RuntimeError("Expected recurrent depth policy to return (actions, hidden_state)")
        action_t, self.hidden_t = out[0], out[1]
        return self.flat_action(self.numpy(action_t))


def load_policy(policy_dir: Path | None, device: str = "cpu") -> OnnxPolicyAdapter | TorchPolicyAdapter:
    policy_path = exported_policy_path(policy_dir)
    if policy_path.suffix == ".onnx":
        return OnnxPolicyAdapter(policy_path)
    return TorchPolicyAdapter(policy_path, device)


def load_history_obs_policy(policy_dir: Path | None, device: str = "cpu") -> OnnxHistoryObsPolicyAdapter | TorchHistoryObsPolicyAdapter:
    policy_path = exported_policy_path(policy_dir)
    if policy_path.suffix == ".onnx":
        return OnnxHistoryObsPolicyAdapter(policy_path)
    return TorchHistoryObsPolicyAdapter(policy_path, device)


def load_depth_recurrent_policy(
    policy_dir: Path | None,
    device: str,
    hidden_shape: tuple[int, int, int],
) -> OnnxDepthRecurrentPolicyAdapter | TorchDepthRecurrentPolicyAdapter:
    policy_path = exported_policy_path(policy_dir)
    if policy_path.suffix == ".onnx":
        return OnnxDepthRecurrentPolicyAdapter(policy_path, hidden_shape)
    return TorchDepthRecurrentPolicyAdapter(policy_path, device, hidden_shape)


class DepthHistoryBuffer:
    """Depth sensor history with delayed frame sampling."""

    def __init__(self, history_length: int, image_shape: tuple[int, int]) -> None:
        self.buf = np.zeros((history_length, *image_shape), dtype=np.float32)
        self._first_push = True

    def append(self, frame: np.ndarray) -> None:
        frame = np.asarray(frame, dtype=np.float32)
        if self._first_push:
            self.buf[:] = frame[np.newaxis, :, :]
            self._first_push = False
        else:
            self.buf[:-1] = self.buf[1:]
            self.buf[-1] = frame

    def delayed_frames(self, num_frames: int, skip_frames: int, delay_frames: int = 0) -> np.ndarray:
        offsets = np.flip(np.arange(0, num_frames * skip_frames, skip_frames))
        indices = self.buf.shape[0] - offsets - int(delay_frames) - 1
        if np.any(indices < 0):
            raise RuntimeError(f"Depth history is too short for frame indices {indices.tolist()}")
        return self.buf[indices.astype(np.int64)]


# ---------------------------------------------------------------------------
# MuJoCo helpers
# ---------------------------------------------------------------------------


@dataclass
class MujocoSdkMaps:
    qposadr: np.ndarray
    qveladr: np.ndarray
    actuator: np.ndarray


def build_mujoco_sdk_maps(mj_model: Any) -> MujocoSdkMaps:
    qposadr = np.zeros(NUM_JOINTS, dtype=np.int32)
    qveladr = np.zeros(NUM_JOINTS, dtype=np.int32)
    actuator = np.zeros(NUM_JOINTS, dtype=np.int32)
    for i, name in enumerate(G1_29DOF_SDK_JOINT_NAMES):
        jid = mj_model.joint(name).id
        qposadr[i] = mj_model.jnt_qposadr[jid]
        qveladr[i] = mj_model.jnt_dofadr[jid]
        actuator[i] = mj_model.actuator(name).id
    return MujocoSdkMaps(qposadr, qveladr, actuator)


def pd_control(
    target_q: np.ndarray,
    q: np.ndarray,
    kp: np.ndarray,
    target_dq: np.ndarray,
    dq: np.ndarray,
    kd: np.ndarray,
) -> np.ndarray:
    return (target_q - q) * kp + (target_dq - dq) * kd


def pd_gains_from_deploy(deploy_cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Return (kp, kd) indexed by SDK joint id for MuJoCo PD."""
    stiffness = np.array(deploy_cfg["stiffness"], dtype=np.float32)
    damping = np.array(deploy_cfg["damping"], dtype=np.float32)
    return stiffness, damping


def apply_sdk_armature(mj_model: Any, sdk_maps: MujocoSdkMaps, deploy_cfg: dict[str, Any]) -> None:
    """Overwrite hinge-joint armature with Isaac BeyondMimic values when provided."""
    raw = deploy_cfg.get("armature")
    if raw is None:
        return
    armature = np.asarray(raw, dtype=np.float64)
    if armature.shape != (NUM_JOINTS,):
        raise ValueError(f"armature must have length {NUM_JOINTS}, got {armature.shape}")
    for sdk_i in range(NUM_JOINTS):
        mj_model.dof_armature[sdk_maps.qveladr[sdk_i]] = armature[sdk_i]


def clear_sdk_joint_passive(mj_model: Any, sdk_maps: MujocoSdkMaps) -> None:
    """Remove MJCF default hinge damping/frictionloss; Isaac PD kd is applied in ``pd_control``."""
    for sdk_i in range(NUM_JOINTS):
        dof = int(sdk_maps.qveladr[sdk_i])
        mj_model.dof_damping[dof] = 0.0
        mj_model.dof_frictionloss[dof] = 0.0


def build_initial_qpos(
    mj_qpos_template: np.ndarray,
    sdk_maps: MujocoSdkMaps,
    env: DeployEnv,
    *,
    root_height: float | None = None,
    motion: MotionLoader | None = None,
    motion_time: float = 0.0,
) -> np.ndarray:
    qpos = mj_qpos_template.copy()
    if root_height is not None:
        qpos[2] = root_height
    if motion is not None:
        joint_pos, root_quat = motion.sample_at(motion_time)
        qpos[:3] = motion.root_position()
        qpos[3:7] = root_quat
        for sdk_i in range(NUM_JOINTS):
            qpos[sdk_maps.qposadr[sdk_i]] = joint_pos[sdk_i]
    else:
        for policy_i, sdk_i in enumerate(env.joint_ids_map):
            qpos[sdk_maps.qposadr[sdk_i]] = env.default_joint_pos[policy_i]
    return qpos


def build_initial_qvel(
    mj_qvel_template: np.ndarray,
    sdk_maps: MujocoSdkMaps,
    *,
    motion: MotionLoader | None = None,
    motion_time: float = 0.0,
) -> np.ndarray:
    qvel = mj_qvel_template.copy()
    if motion is not None:
        qvel[:3] = motion.root_lin_vel_at(motion_time)
        qvel[3:6] = _motion_root_ang_vel_b_at(motion, motion_time)
        joint_vel = motion.joint_vel()
        for sdk_i in range(NUM_JOINTS):
            qvel[sdk_maps.qveladr[sdk_i]] = joint_vel[sdk_i]
    return qvel


def _motion_root_ang_vel_b_at(motion: MotionLoader, motion_time: float) -> np.ndarray:
    t = float(np.clip(motion_time, 0.0, motion.duration))
    if t + motion.dt <= motion.duration:
        t0, t1 = t, t + motion.dt
    else:
        t0, t1 = max(0.0, t - motion.dt), t
    dt = t1 - t0
    if dt <= 0.0:
        return np.zeros(3, dtype=np.float32)

    quat0 = motion.root_quaternion_at(t0)
    quat1 = motion.root_quaternion_at(t1)
    rot0 = Rotation.from_quat(np.roll(quat0, -1))
    rot1 = Rotation.from_quat(np.roll(quat1, -1))
    return ((rot0.inv() * rot1).as_rotvec() / dt).astype(np.float32)


# ---------------------------------------------------------------------------
# Minimal MuJoCo deploy loop
# ---------------------------------------------------------------------------


@dataclass
class PolicyRuntime:
    act: Callable[[], np.ndarray]
    after_action: Callable[[np.ndarray], None] = lambda _action: None


def _uses_depth_history_interface(interface_type: str) -> bool:
    return interface_type == "depth_encoder_actor"


def policy_dir_from_deploy_yaml(deploy_yaml: Path) -> Path:
    deploy_cfg = load_yaml(deploy_yaml)
    if isinstance(deploy_cfg, dict) and deploy_cfg.get("policy_dir"):
        return resolve_policy_dir(resolve_path(str(deploy_cfg["policy_dir"]), DEPLOY_ROOT, base=deploy_yaml.parent))
    if deploy_yaml.parent.name == "params":
        return resolve_policy_dir(deploy_yaml.parent.parent)
    return resolve_policy_dir(deploy_yaml.parent)


def exported_policy_path(policy_dir: Path | None) -> Path:
    if policy_dir is None:
        raise ValueError("policy_dir is required to load exported/policy.onnx or exported/policy.pt")
    exported_dir = policy_dir / "exported"
    for filename in ("policy.onnx", "policy.pt"):
        policy_path = exported_dir / filename
        if policy_path.is_file():
            return policy_path
    raise FileNotFoundError(f"Missing exported policy: expected {exported_dir / 'policy.onnx'} or {exported_dir / 'policy.pt'}")


def exported_model_dir(policy_dir: Path) -> Path:
    model_dir = policy_dir / "exported"
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Missing exported model dir: {model_dir}")
    return model_dir


class ExportedPolicyNormalizer:
    def __init__(self, model_dir: Path, interface_cfg: dict[str, Any]):
        normalizer_path = model_dir / str(interface_cfg.get("normalizer_filename", "policy_normalizer.npz"))
        data = np.load(normalizer_path, allow_pickle=True)
        self.mean = data["mean"].reshape(-1).astype(np.float32)
        self.std = data["std"].reshape(-1).astype(np.float32)
        self.eps = float(data["eps"]) if "eps" in data and data["eps"] is not None else 1e-8
        # PHP / vault VisionMLP: EmpiricalNormalizationWithPassthrough exports normalize_mask
        # (True = apply running stats; False = leave env-normalized depth unchanged).
        if "normalize_mask" in data.files:
            self.normalize_mask = np.asarray(data["normalize_mask"], dtype=bool).reshape(-1)
            if self.normalize_mask.size != self.mean.size:
                raise ValueError(
                    f"normalize_mask size {self.normalize_mask.size} does not match mean size {self.mean.size}"
                )
        else:
            self.normalize_mask = None
        components = interface_cfg.get("normalizer_components")
        depth_component = str(interface_cfg.get("depth_component", "depth_image"))
        if not components:
            raise ValueError("policy_interface.normalizer_components is required for normalized depth_encoder_actor")

        depth_indices: list[int] = []
        proprio_indices: list[int] = []
        offset = 0
        for component in components:
            name = str(component["name"])
            size = int(component["size"])
            target = depth_indices if name == depth_component else proprio_indices
            target.extend(range(offset, offset + size))
            offset += size
        if offset != self.mean.size:
            raise ValueError(f"Normalizer component size {offset} does not match exported size {self.mean.size}")
        self.depth_indices = np.asarray(depth_indices, dtype=np.int64)
        self.proprio_indices = np.asarray(proprio_indices, dtype=np.int64)

    def normalize(self, proprio: np.ndarray, depth_frames: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        proprio_flat = np.asarray(proprio, dtype=np.float32).reshape(-1)
        depth_flat = np.asarray(depth_frames, dtype=np.float32).reshape(-1)
        if proprio_flat.size != self.proprio_indices.size:
            raise ValueError(f"Proprio obs size {proprio_flat.size} != normalizer proprio size {self.proprio_indices.size}")
        if depth_flat.size != self.depth_indices.size:
            raise ValueError(f"Depth obs size {depth_flat.size} != normalizer depth size {self.depth_indices.size}")

        if self.normalize_mask is not None:
            proprio_mask = self.normalize_mask[self.proprio_indices]
            depth_mask = self.normalize_mask[self.depth_indices]
            proprio_norm = proprio_flat.copy()
            if np.any(proprio_mask):
                idx = self.proprio_indices[proprio_mask]
                proprio_norm[proprio_mask] = (proprio_flat[proprio_mask] - self.mean[idx]) / (
                    self.std[idx] + self.eps
                )
            depth_norm = depth_flat.copy()
            if np.any(depth_mask):
                idx = self.depth_indices[depth_mask]
                depth_norm[depth_mask] = (depth_flat[depth_mask] - self.mean[idx]) / (self.std[idx] + self.eps)
            return proprio_norm.astype(np.float32), depth_norm.reshape(depth_frames.shape).astype(np.float32)

        proprio_norm = (proprio_flat - self.mean[self.proprio_indices]) / (self.std[self.proprio_indices] + self.eps)
        depth_norm = (depth_flat - self.mean[self.depth_indices]) / (self.std[self.depth_indices] + self.eps)
        return proprio_norm.astype(np.float32), depth_norm.reshape(depth_frames.shape).astype(np.float32)


def _deploy_env_cfg(
    deploy_cfg: dict[str, Any],
    *,
    history: bool = False,
    first_push_history: bool = False,
) -> dict[str, Any]:
    cfg = dict(deploy_cfg)
    observations: dict[str, Any] = {}
    for name, term_cfg in (deploy_cfg.get("observations") or {}).items():
        if name in {"scale_first", "use_gym_history", "depth"}:
            observations[name] = term_cfg
            continue
        term_cfg = dict(term_cfg or {})
        if not history:
            term_cfg["history_length"] = 1
        observations[name] = term_cfg
    if first_push_history:
        observations["first_push_history"] = True
    cfg["observations"] = observations
    return cfg


def _deploy_env_cfg_for_interface(deploy_cfg: dict[str, Any], interface_type: str) -> dict[str, Any]:
    if interface_type == "deploy_env":
        return deploy_cfg
    use_depth_history = _uses_depth_history_interface(interface_type)
    return _deploy_env_cfg(
        deploy_cfg,
        history=use_depth_history,
        first_push_history=use_depth_history,
    )


def _load_mimic_runtime(policy_dir: Path, fsm_config: Path) -> MimicRuntime | None:
    if not fsm_config.is_file():
        return None
    policy_dir = policy_dir.resolve()
    fsm = load_yaml(fsm_config).get("FSM", {})
    for name, state in fsm.items():
        if name == "_" or not isinstance(state, dict):
            continue
        state_policy_dir = state.get("policy_dir")
        if not state_policy_dir or "motion_file" not in state:
            continue
        if resolve_policy_dir(PROJ_ROOT / state_policy_dir).resolve() != policy_dir:
            continue
        motion_file = resolve_path(state["motion_file"], DEPLOY_ROOT, base=PROJ_ROOT)
        loader = MotionLoader(motion_file, float(state["fps"]))
        time_start = float(state.get("time_start", 0.0))
        time_end = float(state.get("time_end", loader.duration))
        print(
            f"mimic={name}  motion={motion_file.name}  duration={loader.duration:.2f}s  "
            f"segment=[{time_start:.2f}, {time_end:.2f}]"
        )
        return MimicRuntime(loader, time_start, time_end)
    return None


def _load_config_mimic_runtime(deploy_cfg: dict[str, Any], deploy_yaml: Path) -> MimicRuntime | None:
    motion_cfg = deploy_cfg.get("motion_initializer") or {}
    if not motion_cfg.get("enabled", False):
        return None
    motion_file = resolve_path(str(motion_cfg["motion_file"]), DEPLOY_ROOT, base=deploy_yaml.parent)
    loader = MotionLoader(motion_file, float(motion_cfg["fps"]))
    time_start = float(motion_cfg.get("time_start", 0.0))
    time_end = float(motion_cfg.get("time_end", loader.duration))
    print(
        f"mimic=config  motion={motion_file.name}  duration={loader.duration:.2f}s  "
        f"window=[{time_start:.2f}, {time_end:.2f}]"
    )
    return MimicRuntime(loader=loader, time_start=time_start, time_end=time_end)


def _make_key_callback(
    cmd_ranges: dict[str, list[float]],
    velocity_command: np.ndarray,
) -> Callable[[int], None]:
    vx = tuple(cmd_ranges["lin_vel_x"])
    vy = tuple(cmd_ranges["lin_vel_y"])
    wz = tuple(cmd_ranges["ang_vel_z"])

    def key_callback(keycode: int) -> None:
        if keycode == glfw.KEY_KP_6:
            velocity_command[0] = min(velocity_command[0] + CMD_STEP, vx[1])
        elif keycode == glfw.KEY_KP_4:
            velocity_command[0] = max(velocity_command[0] - CMD_STEP, vx[0])
        elif keycode == glfw.KEY_KP_8:
            velocity_command[1] = min(velocity_command[1] + CMD_STEP, vy[1])
        elif keycode == glfw.KEY_KP_2:
            velocity_command[1] = max(velocity_command[1] - CMD_STEP, vy[0])
        elif keycode == glfw.KEY_KP_9:
            velocity_command[2] = min(velocity_command[2] + CMD_STEP, wz[1])
        elif keycode == glfw.KEY_KP_3:
            velocity_command[2] = max(velocity_command[2] - CMD_STEP, wz[0])
        elif keycode == glfw.KEY_KP_5:
            velocity_command[:] = 0.0
        else:
            return

    return key_callback


def _load_depth_model_if_needed(
    deploy_cfg: dict[str, Any],
    xml_path: Path,
    sim_dt: float,
) -> tuple[mujoco.MjModel | None, mujoco.MjData | None]:
    depth_cfg = (deploy_cfg.get("observations") or {}).get("depth") or {}
    if str(depth_cfg.get("type", "")) != "cropped_normalized":
        return None, None

    depth_xml = _inferred_depth_xml_path(xml_path)
    if depth_xml is None or depth_xml == xml_path:
        return None, None

    depth_model = mujoco.MjModel.from_xml_path(str(depth_xml))
    depth_model.opt.timestep = sim_dt
    print(f"depth_model={depth_xml}")
    return depth_model, mujoco.MjData(depth_model)


def _setup_simulation(
    deploy_cfg: dict[str, Any],
    xml_path: Path,
    sim_dt: float,
    mimic: MimicRuntime | None,
    velocity_command: np.ndarray,
) -> tuple[mujoco.MjModel, mujoco.MjData, DeployEnv, MujocoSdkMaps, np.ndarray, np.ndarray]:
    env = DeployEnv(
        deploy_cfg,
        motion_loader=mimic.loader if mimic else None,
        velocity_command=velocity_command,
    )
    mj_model = mujoco.MjModel.from_xml_path(str(xml_path))
    mj_data = mujoco.MjData(mj_model)
    mj_model.opt.timestep = sim_dt
    if needs_mujoco_attach(deploy_cfg):
        depth_model, depth_data = _load_depth_model_if_needed(deploy_cfg, xml_path, sim_dt)
        env.attach_mujoco(mj_data, mj_model, depth_mj_data=depth_data, depth_mj_model=depth_model)

    sdk_maps = build_mujoco_sdk_maps(mj_model)
    sdk_stiffness, sdk_damping = pd_gains_from_deploy(deploy_cfg)
    apply_sdk_armature(mj_model, sdk_maps, deploy_cfg)
    clear_sdk_joint_passive(mj_model, sdk_maps)
    initial_qpos = build_initial_qpos(
        mj_data.qpos,
        sdk_maps,
        env,
        motion=mimic.loader if mimic else None,
        motion_time=mimic.time_start if mimic else 0.0,
    )
    initial_qvel = build_initial_qvel(
        mj_data.qvel,
        sdk_maps,
        motion=mimic.loader if mimic else None,
        motion_time=mimic.time_start if mimic else 0.0,
    )
    mujoco.mj_setState(mj_model, mj_data, initial_qpos, mujoco.mjtState.mjSTATE_QPOS)
    mj_data.qvel[:] = initial_qvel
    mujoco.mj_forward(mj_model, mj_data)
    _lift_root_out_of_ground(mj_model, mj_data)

    env.sync_robot_from_mujoco(mj_data.qpos, mj_data.qvel, sdk_maps.qposadr, sdk_maps.qveladr)
    if mimic is not None:
        env.prepare_mimic_episode(mimic.loader.joint_pos())
    else:
        env.reset()
    return mj_model, mj_data, env, sdk_maps, sdk_stiffness, sdk_damping


def _lift_root_out_of_ground(
    mj_model: Any,
    mj_data: Any,
    *,
    clearance: float = 0.002,
) -> float:
    """Raise the free-joint root so spawned feet are not penetrating the ground.

    Copying motion ``base_pos_w`` 1:1 can still start a few cm in the floor if
    MuJoCo foot geometry is slightly thicker than Isaac.
    """
    lift = 0.0
    for i in range(int(mj_data.ncon)):
        dist = float(mj_data.contact[i].dist)
        if dist < 0.0:
            lift = max(lift, -dist)
    if lift <= 0.0:
        return 0.0
    lift += float(clearance)
    mj_data.qpos[2] += lift
    mujoco.mj_forward(mj_model, mj_data)
    print(f"lifted root {lift * 100:.1f} cm to clear ground penetration")
    return lift


def _depth_image_native(
    renderer: mujoco.Renderer,
    mj_data: mujoco.MjData,
    camera_name: str,
    depth_max: float,
    scene_option: mujoco.MjvOption | None = None,
) -> np.ndarray:
    if scene_option is None:
        scene_option = _make_terrain_only_depth_option()
    renderer.update_scene(mj_data, camera=camera_name, scene_option=scene_option)
    depth = renderer.render().astype(np.float32)
    depth = np.where(np.isfinite(depth), depth, depth_max)
    return (np.clip(depth, 0.0, depth_max) / depth_max).astype(np.float32)


def _make_terrain_only_depth_option() -> mujoco.MjvOption:
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = 0
    opt.geomgroup[0] = 1
    return opt


def _make_terrain_visual_depth_option() -> mujoco.MjvOption:
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = 0
    opt.geomgroup[0] = 1
    opt.geomgroup[1] = 1
    return opt


def _gaussian_blur_3x3(image: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0.0:
        return image.astype(np.float32, copy=False)

    try:
        import cv2  # type: ignore[import-not-found]

        return cv2.GaussianBlur(image, (3, 3), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT_101)
    except ImportError:
        pass

    axis = np.arange(-1, 2, dtype=np.float32)
    kernel_1d = np.exp(-(axis**2) / (2.0 * sigma * sigma))
    kernel_1d /= kernel_1d.sum()
    kernel = np.outer(kernel_1d, kernel_1d).astype(np.float32)
    padded = np.pad(image, ((1, 1), (1, 1)), mode="reflect")
    out = np.zeros_like(image, dtype=np.float32)
    for r in range(image.shape[0]):
        for c in range(image.shape[1]):
            out[r, c] = np.sum(padded[r : r + 3, c : c + 3] * kernel)
    return out


def _tuple2_float(value: list[float], name: str) -> tuple[float, float]:
    vals = tuple(float(v) for v in value)
    if len(vals) != 2:
        raise ValueError(f"{name} must have 2 values, got {len(vals)}")
    return vals


def _optional_tuple2_float(value: list[float] | None, name: str) -> tuple[float, float] | None:
    if value is None:
        return None
    return _tuple2_float(value, name)


def _tuple2_int(value: list[int], name: str) -> tuple[int, int]:
    vals = tuple(int(v) for v in value)
    if len(vals) != 2:
        raise ValueError(f"{name} must have 2 values, got {len(vals)}")
    return vals


def _optional_tuple2_int(value: list[int] | None, name: str) -> tuple[int, int] | None:
    if value is None:
        return None
    return _tuple2_int(value, name)


def _tuple4_int(value: list[int], name: str) -> tuple[int, int, int, int]:
    vals = tuple(int(v) for v in value)
    if len(vals) != 4:
        raise ValueError(f"{name} must have 4 values, got {len(vals)}")
    return vals


def _wrap_value(value: float, bounds: tuple[float, float]) -> float:
    low, high = bounds
    width = high - low
    if width <= 0.0:
        return value
    return ((value - low) % width) + low


def _sync_depth_model_state(
    src_data: mujoco.MjData,
    depth_model: mujoco.MjModel,
    depth_data: mujoco.MjData,
    *,
    wrap_x: tuple[float, float] | None,
    wrap_y: tuple[float, float] | None,
) -> None:
    if depth_data.qpos.shape != src_data.qpos.shape:
        raise RuntimeError("Depth model qpos shape does not match physics model qpos shape")
    depth_data.qpos[:] = src_data.qpos
    if depth_data.qvel.shape == src_data.qvel.shape:
        depth_data.qvel[:] = src_data.qvel
    if wrap_x is not None:
        depth_data.qpos[0] = _wrap_value(float(src_data.qpos[0]), wrap_x)
    if wrap_y is not None:
        depth_data.qpos[1] = _wrap_value(float(src_data.qpos[1]), wrap_y)
    mujoco.mj_forward(depth_model, depth_data)


def _inferred_depth_xml_path(xml_path: Path) -> Path | None:
    candidate = xml_path.with_name(f"{xml_path.stem}_depth_local{xml_path.suffix}")
    return candidate if candidate.is_file() else None


def _resize_depth_image(depth: np.ndarray, resize_shape: tuple[int, int] | None) -> np.ndarray:
    if resize_shape is None or tuple(depth.shape[:2]) == resize_shape:
        return depth
    try:
        import cv2  # type: ignore[import-not-found]

        return cv2.resize(depth, (resize_shape[1], resize_shape[0]), interpolation=cv2.INTER_AREA).astype(np.float32)
    except ImportError:
        from scipy.ndimage import zoom

        zoom_factors = (resize_shape[0] / depth.shape[0], resize_shape[1] / depth.shape[1])
        return zoom(depth, zoom_factors, order=1).astype(np.float32)


def _cropped_normalized_depth_image(
    renderer: mujoco.Renderer,
    mj_data: mujoco.MjData,
    camera_name: str,
    *,
    crop_region: tuple[int, int, int, int],
    depth_range: tuple[float, float],
    blur_sigma: float,
    resize_shape: tuple[int, int] | None = None,
    scene_option: mujoco.MjvOption | None = None,
    min_distance: float = 0.0,
) -> np.ndarray:
    if scene_option is None:
        scene_option = _make_terrain_visual_depth_option()
    renderer.update_scene(mj_data, camera=camera_name, scene_option=scene_option)
    depth_m = renderer.render().astype(np.float32)
    depth_m = np.where(np.isfinite(depth_m), depth_m, depth_range[1])
    # Isaac camera.min_distance + depth_clipping_behavior=max: near hits become far.
    if min_distance > 0.0:
        depth_m = np.where(depth_m < min_distance, depth_range[1], depth_m)
    up, down, left, right = crop_region
    depth_m = depth_m[up : depth_m.shape[0] - down, left : depth_m.shape[1] - right]
    depth_m = _resize_depth_image(depth_m, resize_shape)
    depth_m = _gaussian_blur_3x3(depth_m, blur_sigma)
    min_depth, max_depth = depth_range
    depth_norm = (np.clip(depth_m, min_depth, max_depth) - min_depth) / (max_depth - min_depth)
    return depth_norm.astype(np.float32)


def _load_depth_encoder_actor_onnx(model_dir: Path, interface_cfg: dict[str, Any]) -> tuple[Any, Any]:
    import onnxruntime as ort

    providers = ort.get_available_providers()
    encoder_filename = str(interface_cfg.get("encoder_filename", "0-depth_encoder.onnx"))
    actor_filename = str(interface_cfg.get("actor_filename", "actor.onnx"))
    encoder = ort.InferenceSession(str(model_dir / encoder_filename), providers=providers)
    actor = ort.InferenceSession(str(model_dir / actor_filename), providers=providers)
    return encoder, actor


def _run_depth_encoder_actor_onnx(encoder: Any, actor: Any, proprio: np.ndarray, depth_frames: np.ndarray) -> np.ndarray:
    depth_latent = encoder.run(None, {encoder.get_inputs()[0].name: depth_frames[np.newaxis].astype(np.float32)})[0]
    actor_input = np.concatenate([proprio[np.newaxis].astype(np.float32), depth_latent], axis=1)
    return actor.run(None, {actor.get_inputs()[0].name: actor_input})[0].reshape(-1).astype(np.float32)


_DEPTH_PREVIEW_CV2_MISSING_WARNED = False
_DEPTH_PREVIEW_WINDOW = "depth"
_DEPTH_PREVIEW_SCALE = 8


def _depth_preview_image(depth_norm: np.ndarray, scale: int) -> np.ndarray:
    """Convert normalized depth [0, 1] to a uint8 image for OpenCV display."""
    depth = np.clip(np.asarray(depth_norm, dtype=np.float32), 0.0, 1.0)
    image = (depth * 255.0).astype(np.uint8)
    if scale > 1:
        import cv2  # type: ignore[import-not-found]

        h, w = image.shape[:2]
        image = cv2.resize(image, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)
    return image


def _show_depth_preview(depth_norm: np.ndarray, scale: int, window_name: str = _DEPTH_PREVIEW_WINDOW) -> None:
    global _DEPTH_PREVIEW_CV2_MISSING_WARNED
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError:
        if not _DEPTH_PREVIEW_CV2_MISSING_WARNED:
            print("[WARN] Depth preview requested but OpenCV is not installed.")
            _DEPTH_PREVIEW_CV2_MISSING_WARNED = True
        return

    cv2.imshow(window_name, _depth_preview_image(depth_norm, max(1, int(scale))))
    cv2.waitKey(1)


def _destroy_depth_preview_windows() -> None:
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError:
        return
    try:
        cv2.destroyWindow(_DEPTH_PREVIEW_WINDOW)
    except Exception:
        pass


def _make_policy_runtime(
    interface_type: str,
    deploy_cfg: dict[str, Any],
    policy_dir: Path | None,
    env: DeployEnv,
    mj_data: mujoco.MjData,
    sdk_maps: MujocoSdkMaps,
    velocity_command: np.ndarray,
    device: str,
) -> PolicyRuntime:
    if interface_type == "deploy_env":
        policy = load_policy(policy_dir, device)
        return PolicyRuntime(act=lambda: policy.act(env.compute_obs()))

    if interface_type == "history_obs":
        policy = load_history_obs_policy(policy_dir, device)
        history_env = DeployEnv(_deploy_env_cfg(deploy_cfg, history=True), velocity_command=velocity_command)
        history_env.sync_robot_from_mujoco(mj_data.qpos, mj_data.qvel, sdk_maps.qposadr, sdk_maps.qveladr)
        history_env.reset()

        def act_history() -> np.ndarray:
            history_env.sync_robot_from_mujoco(mj_data.qpos, mj_data.qvel, sdk_maps.qposadr, sdk_maps.qveladr)
            return policy.act(history_env.compute_obs()["obs"], env.compute_obs()["obs"])

        def after_history(action: np.ndarray) -> None:
            history_env.process_action(action)
            history_env.episode_length += 1

        return PolicyRuntime(act=act_history, after_action=after_history)

    if interface_type == "depth_recurrent":
        interface_cfg = deploy_cfg.get("policy_interface") or {}
        hidden_shape = (
            int(interface_cfg.get("gru_num_layers", 1)),
            1,
            int(interface_cfg.get("gru_hidden_size", 512)),
        )
        policy = load_depth_recurrent_policy(policy_dir, device, hidden_shape)

        def act_depth_recurrent() -> np.ndarray:
            depth = env.compute_sensor("depth")
            _show_depth_preview(depth, _DEPTH_PREVIEW_SCALE)
            return policy.act(env.compute_obs()["obs"], depth)

        return PolicyRuntime(act=act_depth_recurrent)

    if interface_type == "depth_encoder_actor":
        interface_cfg = deploy_cfg.get("policy_interface") or {}
        depth_cfg = deploy_cfg["observations"]["depth"]
        render_h = int(depth_cfg["render_height"])
        render_w = int(depth_cfg["render_width"])
        crop_region = _tuple4_int(depth_cfg["crop_region"], "observations.depth.crop_region")
        cropped_shape = _optional_tuple2_int(depth_cfg.get("resize_shape"), "observations.depth.resize_shape") or (
            render_h - crop_region[0] - crop_region[1],
            render_w - crop_region[2] - crop_region[3],
        )
        depth_hist = DepthHistoryBuffer(int(depth_cfg["history_length"]), cropped_shape)
        depth_num_frames = int(depth_cfg.get("num_output_frames", 8))
        depth_skip_frames = int(depth_cfg.get("skip_frames", 5))
        depth_delay_frames = int(depth_cfg.get("delay_frames", 0))
        if policy_dir is None:
            raise RuntimeError("depth_encoder_actor export directory requires policy_dir")
        model_dir = exported_model_dir(policy_dir)
        encoder, actor = _load_depth_encoder_actor_onnx(model_dir, interface_cfg)
        normalizer = (
            ExportedPolicyNormalizer(model_dir, interface_cfg)
            if bool(interface_cfg.get("use_policy_normalizer", False))
            else None
        )

        def act_depth_encoder_actor() -> np.ndarray:
            depth_frame = env.compute_sensor("depth")
            depth_hist.append(depth_frame)
            depth_frames = depth_hist.delayed_frames(depth_num_frames, depth_skip_frames, depth_delay_frames)
            _show_depth_preview(depth_frame, _DEPTH_PREVIEW_SCALE)
            proprio = env.compute_obs()["obs"]
            if normalizer is not None:
                proprio, depth_frames_norm = normalizer.normalize(proprio, depth_frames)
            else:
                depth_frames_norm = depth_frames
            return _run_depth_encoder_actor_onnx(encoder, actor, proprio, depth_frames_norm)

        return PolicyRuntime(act=act_depth_encoder_actor)

    raise ValueError(f"Unsupported policy_interface.type: {interface_type}")


def run_deploy(config_path: str, args: Any) -> None:
    deploy_yaml, deploy_cfg = load_deploy_config(config_path, DEPLOY_ROOT)
    policy_dir = policy_dir_from_deploy_yaml(deploy_yaml)
    interface_cfg = deploy_cfg.get("policy_interface") or {}
    interface_type = str(interface_cfg.get("type", "deploy_env"))
    env_cfg = _deploy_env_cfg_for_interface(deploy_cfg, interface_type)
    xml_arg = getattr(args, "xml_path", None) or deploy_cfg.get("xml_path") or DEFAULT_XML
    xml_path = resolve_path(str(xml_arg), DEPLOY_ROOT, base=deploy_yaml.parent)
    sim_dt = float(getattr(args, "simulation_dt", None) or deploy_cfg.get("simulation_dt", 0.001))
    decimation = max(1, round(float(env_cfg["step_dt"]) / sim_dt))
    duration = float(getattr(args, "simulation_duration", None) or deploy_cfg.get("simulation_duration", 600.0))
    mimic = _load_config_mimic_runtime(deploy_cfg, deploy_yaml)
    if mimic is None and interface_type == "deploy_env":
        mimic = _load_mimic_runtime(
            policy_dir,
            resolve_path(getattr(args, "fsm_config", str(DEFAULT_FSM_CONFIG)), DEPLOY_ROOT),
        )

    print(f"config={deploy_yaml}  policy_dir={policy_dir}")
    print(f"control: sim_dt={sim_dt}  step_dt={env_cfg['step_dt']}  decimation={decimation}")

    velocity_command = np.zeros(3, dtype=np.float32)
    mj_model, mj_data, env, sdk_maps, sdk_stiffness, sdk_damping = _setup_simulation(
        env_cfg,
        xml_path,
        sim_dt,
        mimic,
        velocity_command,
    )
    runtime = _make_policy_runtime(
        interface_type,
        deploy_cfg,
        policy_dir,
        env,
        mj_data,
        sdk_maps,
        velocity_command,
        getattr(args, "device", "cpu"),
    )
    cmd_ranges = (env_cfg.get("commands") or {}).get("base_velocity", {}).get("ranges")
    key_callback = _make_key_callback(cmd_ranges, velocity_command) if cmd_ranges else None
    target_sdk_q = env.target_sdk_joint_pos()
    target_dq = np.zeros(NUM_JOINTS, dtype=np.float32)

    counter = 0
    start = time.time()
    try:
        with mujoco.viewer.launch_passive(
            mj_model,
            mj_data,
            key_callback=key_callback,
            show_left_ui=False,
            show_right_ui=False,
        ) as viewer:
            while viewer.is_running() and time.time() - start < duration:
                step_start = time.time()
                sdk_q = mj_data.qpos[sdk_maps.qposadr]
                sdk_dq = mj_data.qvel[sdk_maps.qveladr]
                tau = pd_control(target_sdk_q, sdk_q, sdk_stiffness, target_dq, sdk_dq, sdk_damping)
                mj_data.ctrl[sdk_maps.actuator] = tau
                mujoco.mj_step(mj_model, mj_data)
                counter += 1
                if counter % decimation == 0:
                    env.sync_robot_from_mujoco(mj_data.qpos, mj_data.qvel, sdk_maps.qposadr, sdk_maps.qveladr)
                    if mimic is not None:
                        mimic.update_for_step(env.episode_length, env.step_dt)
                    action = runtime.act()
                    env.process_action(action)
                    runtime.after_action(action)
                    target_sdk_q = env.target_sdk_joint_pos()
                    env.episode_length += 1
                    viewer.sync()

                sleep_s = sim_dt - (time.time() - step_start)
                if sleep_s > 0:
                    time.sleep(sleep_s)
    finally:
        if interface_type in {"depth_recurrent", "depth_encoder_actor"}:
            _destroy_depth_preview_windows()

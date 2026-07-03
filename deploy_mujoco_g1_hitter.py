from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import glfw
import mujoco
import mujoco.viewer
import numpy as np
import torch
import yaml
from scipy.spatial.transform import Rotation


DEPLOY_ROOT = Path(__file__).resolve().parent
RESET_BALL_REQUESTED = False


def load_config(config_path: str) -> dict:
    path = Path(config_path).expanduser()
    if not path.is_absolute():
        path = DEPLOY_ROOT / path
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def parse_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=str, default="config/g1_hitter.yaml")
    config_args, remaining_args = config_parser.parse_known_args()
    cfg = load_config(config_args.config)

    parser = argparse.ArgumentParser(description="Deploy a G1 HITTER policy in MuJoCo.")
    parser.add_argument("--config", type=str, default=config_args.config)
    args = parser.parse_args(remaining_args)
    args.config_values = cfg
    return args


def key_callback(keycode: int) -> None:
    global RESET_BALL_REQUESTED
    if keycode == glfw.KEY_R:
        RESET_BALL_REQUESTED = True


def resolve_path(path: str) -> Path:
    p = Path(path).expanduser()
    candidates = [p, DEPLOY_ROOT / p, DEPLOY_ROOT.parent / p]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Path not found: {path}")


def projected_gravity(root_quat_wxyz: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = root_quat_wxyz
    return np.array(
        [
            2.0 * (-qz * qx + qw * qy),
            -2.0 * (qz * qy + qw * qx),
            1.0 - 2.0 * (qw * qw + qz * qz),
        ],
        dtype=np.float32,
    )


def yaw_from_quat(root_quat_wxyz: np.ndarray) -> float:
    return float(Rotation.from_quat(np.roll(root_quat_wxyz, -1)).as_euler("xyz")[2])


def body_ang_vel(root_quat_wxyz: np.ndarray, ang_vel_world: np.ndarray) -> np.ndarray:
    rot = Rotation.from_quat(np.roll(root_quat_wxyz, -1))
    return rot.inv().apply(ang_vel_world).astype(np.float32)


def pd_control(target_q: np.ndarray, q: np.ndarray, kp: np.ndarray, dq: np.ndarray, kd: np.ndarray) -> np.ndarray:
    return (target_q - q) * kp - dq * kd


def sample_ball_state(args: argparse.Namespace, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    cfg = args.config_values
    pos = np.asarray(cfg["ball_base_pos"], dtype=np.float32)
    ball_pos_offset = cfg["ball_pos_offset"]
    if ball_pos_offset is None:
        pos_ranges = cfg["ball_pos_offset_ranges"]
        pos += np.array([rng.uniform(*axis_range) for axis_range in pos_ranges], dtype=np.float32)
    else:
        pos += np.asarray(ball_pos_offset, dtype=np.float32)

    ball_velocity = cfg["ball_velocity"]
    if ball_velocity is None:
        vel_ranges = cfg["ball_velocity_ranges"]
        vel = np.array([rng.uniform(*axis_range) for axis_range in vel_ranges], dtype=np.float32)
    else:
        vel = np.asarray(ball_velocity, dtype=np.float32)
    return pos, vel


def set_ball_state(
    mj_model: mujoco.MjModel,
    mj_data: mujoco.MjData,
    ball_joint_id: int,
    ball_pos: np.ndarray,
    ball_vel: np.ndarray,
) -> None:
    qadr = mj_model.jnt_qposadr[ball_joint_id]
    vadr = mj_model.jnt_dofadr[ball_joint_id]
    mj_data.qpos[qadr : qadr + 3] = ball_pos
    mj_data.qpos[qadr + 3 : qadr + 7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    mj_data.qvel[vadr : vadr + 3] = ball_vel
    mj_data.qvel[vadr + 3 : vadr + 6] = 0.0


def reset_ball_for_serve(
    args: argparse.Namespace,
    rng: np.random.Generator,
    mj_model: mujoco.MjModel,
    mj_data: mujoco.MjData,
    ball_joint_id: int,
) -> tuple[np.ndarray, np.ndarray]:
    ball_pos, serve_vel = sample_ball_state(args, rng)
    set_ball_state(mj_model, mj_data, ball_joint_id, ball_pos, np.zeros(3, dtype=np.float32))
    mujoco.mj_forward(mj_model, mj_data)
    return ball_pos, serve_vel


def get_ball_state(mj_model: mujoco.MjModel, mj_data: mujoco.MjData, ball_joint_id: int) -> tuple[np.ndarray, np.ndarray]:
    qadr = mj_model.jnt_qposadr[ball_joint_id]
    vadr = mj_model.jnt_dofadr[ball_joint_id]
    return mj_data.qpos[qadr : qadr + 3].astype(np.float32), mj_data.qvel[vadr : vadr + 3].astype(np.float32)


def apply_ball_drag(
    mj_model: mujoco.MjModel,
    mj_data: mujoco.MjData,
    ball_body_id: int,
    ball_joint_id: int,
    cfg: dict,
) -> None:
    _, ball_vel = get_ball_state(mj_model, mj_data, ball_joint_id)
    speed = float(np.linalg.norm(ball_vel))
    if speed < 1e-6:
        mj_data.xfrc_applied[ball_body_id, :3] = 0.0
        return
    rho = float(cfg["air_density"])
    cd = float(cfg["ball_drag_coefficient"])
    radius = float(cfg["ball_drag_radius"])
    drag_force = -0.5 * rho * cd * np.pi * radius**2 * speed * ball_vel
    mj_data.xfrc_applied[ball_body_id, :3] = drag_force
    mj_data.xfrc_applied[ball_body_id, 3:] = 0.0


def compute_hitter_plan(
    ball_pos: np.ndarray,
    ball_vel: np.ndarray,
    robot_pos: np.ndarray,
    cfg: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    p = ball_pos.astype(np.float32).copy()
    v = ball_vel.astype(np.float32).copy()
    hit_plane_x = float(cfg["hitter_hit_plane_x"])
    table_z = float(cfg["hitter_table_z"])
    dt = float(cfg["hitter_plan_dt"])
    steps = int(cfg["hitter_plan_steps"])
    found = False
    strike_pos = np.zeros(3, dtype=np.float32)
    incoming_vel = v.copy()
    strike_time = np.zeros(1, dtype=np.float32)
    gravity = np.asarray(cfg["gravity"], dtype=np.float32)
    rho = float(cfg["air_density"])
    cd = float(cfg["ball_drag_coefficient"])
    radius = float(cfg["ball_drag_radius"])
    mass = float(cfg["ball_drag_mass"])
    drag_k = 0.5 * rho * cd * np.pi * radius**2 / mass

    for i in range(steps):
        prev_p = p.copy()
        prev_v = v.copy()
        speed = np.linalg.norm(v)
        v = v + (-drag_k * speed * v + gravity) * dt
        p = p + v * dt
        table_x_min, table_x_max = cfg["table_x_range"]
        table_y_min, table_y_max = cfg["table_y_range"]
        on_table = table_x_min <= p[0] <= table_x_max and table_y_min <= p[1] <= table_y_max
        if on_table and p[2] <= table_z and v[2] < 0.0:
            p[2] = table_z
            v[0] *= float(cfg["hitter_bounce_restitution_xy"])
            v[1] *= float(cfg["hitter_bounce_restitution_xy"])
            v[2] *= -float(cfg["hitter_bounce_restitution_z"])
        if prev_p[0] > hit_plane_x >= p[0] and not found:
            denom = max(prev_p[0] - p[0], 1e-6)
            alpha = np.clip((prev_p[0] - hit_plane_x) / denom, 0.0, 1.0)
            strike_pos = prev_p + alpha * (p - prev_p)
            incoming_vel = prev_v + alpha * (v - prev_v)
            strike_time[0] = (float(i) + alpha) * dt
            found = True

    landing = np.asarray(cfg["hitter_landing_pos"], dtype=np.float32)
    post_hit_t = float(cfg["hitter_post_hit_flight_time"])
    outgoing_vel = (landing - strike_pos) / post_hit_t - 0.5 * gravity * post_hit_t
    normal = outgoing_vel - incoming_vel
    normal_norm = np.linalg.norm(normal)
    if normal_norm > 1e-6:
        normal = normal / normal_norm
    restitution = float(cfg["hitter_racket_restitution"])
    racket_vel = ((np.dot(outgoing_vel, normal) + restitution * np.dot(incoming_vel, normal)) / (1.0 + restitution)) * normal

    base_offset = np.asarray(cfg["hitter_base_target_offset"], dtype=np.float32)
    base_offset[1] = abs(base_offset[1]) if strike_pos[1] < robot_pos[1] else -abs(base_offset[1])
    base_target = strike_pos + base_offset
    base_target[2] = float(cfg["future_body_height"])

    valid_min_z, valid_max_z = cfg["hitter_valid_strike_z_range"]
    valid = found and float(valid_min_z) < strike_pos[2] < float(valid_max_z) and ball_vel[0] < 0.0
    if not valid:
        fallback_racket = robot_pos.copy()
        fallback_racket[1] += float(cfg["future_paddle_y_offset"])
        fallback_racket[2] = float(cfg["future_body_height"]) + float(cfg["fallback_racket_height_offset"])
        return (
            fallback_racket.astype(np.float32),
            np.zeros(3, dtype=np.float32),
            np.asarray(cfg["default_robot_future_pos"], dtype=np.float32),
            np.zeros(1, dtype=np.float32),
        )
    return strike_pos.astype(np.float32), racket_vel.astype(np.float32), base_target.astype(np.float32), strike_time


def main() -> None:
    global RESET_BALL_REQUESTED
    args = parse_args()
    cfg = args.config_values
    rng = np.random.default_rng(42)
    xml_path = resolve_path(str(cfg["xml_path"]))
    policy_path = resolve_path(str(os.environ.get("G1_HITTER_POLICY_PATH", cfg["policy_path"])))
    joint_names = list(cfg["joint_names"])
    default_joint_pos = np.asarray(cfg["default_joint_pos"], dtype=np.float32)
    kp = np.asarray(cfg["kps"], dtype=np.float32)
    kd = np.asarray(cfg["kds"], dtype=np.float32)
    simulation_dt = float(cfg["simulation_dt"])
    control_decimation = int(cfg["control_decimation"])
    history_length = int(cfg["history_length"])
    action_scale = float(cfg["action_scale"])
    obs_clip = float(cfg["obs_clip"])
    action_clip = float(cfg["action_clip"])
    duration = float(cfg["simulation_duration"])
    robot_pos = np.asarray(cfg["robot_pos"], dtype=np.float32)
    no_realtime = bool(cfg["no_realtime"])
    serve_delay = float(cfg.get("serve_delay", 0.5))

    policy = torch.jit.load(str(policy_path), map_location="cpu")
    policy.eval()
    print(f"[INFO] Loaded policy: {policy_path}")

    mj_model = mujoco.MjModel.from_xml_path(str(xml_path))
    mj_model.opt.timestep = simulation_dt
    mj_data = mujoco.MjData(mj_model)

    joint_ids = [mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in joint_names]
    actuator_ids = [mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in joint_names]
    qpos_ids = np.array([mj_model.jnt_qposadr[jid] for jid in joint_ids], dtype=np.int32)
    qvel_ids = np.array([mj_model.jnt_dofadr[jid] for jid in joint_ids], dtype=np.int32)
    effort_limits = np.array([mj_model.jnt_actfrcrange[jid, 1] for jid in joint_ids], dtype=np.float32)
    ball_joint_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, "hitter_ball_free_joint")
    ball_body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "hitter_ball")

    mj_data.qpos[:3] = robot_pos
    mj_data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    mj_data.qpos[qpos_ids] = default_joint_pos
    mujoco.mj_forward(mj_model, mj_data)

    ball_pos, pending_ball_vel = reset_ball_for_serve(args, rng, mj_model, mj_data, ball_joint_id)
    ball_vel = np.zeros(3, dtype=np.float32)
    serve_delay_remaining = serve_delay

    num_actions = len(joint_names)
    frame_dim = 17 + 3 * num_actions
    obs_history = np.zeros((history_length, frame_dim), dtype=np.float32)
    first_obs = True
    action = np.zeros(num_actions, dtype=np.float32)
    target_q = default_joint_pos.copy()
    racket_target_pos = np.zeros(3, dtype=np.float32)
    racket_target_vel = np.zeros(3, dtype=np.float32)
    base_target_pos = np.zeros(3, dtype=np.float32)
    strike_time = np.zeros(1, dtype=np.float32)
    ball_elapsed = 0.0

    print("[INFO] Controls: R reset ball, Esc close viewer.")
    with mujoco.viewer.launch_passive(
        mj_model, mj_data, key_callback=key_callback, show_left_ui=False, show_right_ui=False
    ) as viewer:
        viewer.opt.geomgroup[:] = 0
        viewer.opt.geomgroup[0] = 1
        viewer.opt.geomgroup[1] = 1
        start = time.time()
        counter = 0
        while viewer.is_running() and time.time() - start < duration:
            step_start = time.time()
            if RESET_BALL_REQUESTED:
                ball_pos, pending_ball_vel = reset_ball_for_serve(args, rng, mj_model, mj_data, ball_joint_id)
                ball_vel = np.zeros(3, dtype=np.float32)
                serve_delay_remaining = serve_delay
                ball_elapsed = 0.0
                racket_target_pos[:] = 0.0
                racket_target_vel[:] = 0.0
                base_target_pos[:] = 0.0
                strike_time[:] = 0.0
                RESET_BALL_REQUESTED = False

            q = mj_data.qpos[qpos_ids]
            dq = mj_data.qvel[qvel_ids]
            tau = pd_control(target_q, q, kp, dq, kd)
            tau = np.clip(tau, -effort_limits, effort_limits)
            mj_data.ctrl[actuator_ids] = tau
            if serve_delay_remaining > 0.0:
                set_ball_state(mj_model, mj_data, ball_joint_id, ball_pos, np.zeros(3, dtype=np.float32))
                serve_delay_remaining = max(0.0, serve_delay_remaining - simulation_dt)
                if serve_delay_remaining == 0.0:
                    set_ball_state(mj_model, mj_data, ball_joint_id, ball_pos, pending_ball_vel)
                mujoco.mj_forward(mj_model, mj_data)
            else:
                apply_ball_drag(mj_model, mj_data, ball_body_id, ball_joint_id, cfg)

            mujoco.mj_step(mj_model, mj_data)
            if serve_delay_remaining > 0.0:
                set_ball_state(mj_model, mj_data, ball_joint_id, ball_pos, np.zeros(3, dtype=np.float32))
                mujoco.mj_forward(mj_model, mj_data)
                ball_vel = np.zeros(3, dtype=np.float32)
            else:
                ball_pos, ball_vel = get_ball_state(mj_model, mj_data, ball_joint_id)
                ball_elapsed += simulation_dt
            if serve_delay_remaining <= 0.0 and (ball_pos[2] < 0.1 or ball_elapsed > float(cfg["ball_max_episode_length"])):
                ball_pos, pending_ball_vel = reset_ball_for_serve(args, rng, mj_model, mj_data, ball_joint_id)
                ball_vel = np.zeros(3, dtype=np.float32)
                serve_delay_remaining = serve_delay
                ball_elapsed = 0.0
                racket_target_pos[:] = 0.0
                racket_target_vel[:] = 0.0
                base_target_pos[:] = 0.0
                strike_time[:] = 0.0
            counter += 1
            if counter % control_decimation == 0:
                root_pos = mj_data.qpos[:3].astype(np.float32)
                root_quat = mj_data.qpos[3:7].astype(np.float32)
                robot_pos = root_pos.copy()
                yaw = yaw_from_quat(root_quat)
                base_forward_xy = np.array([np.cos(yaw), np.sin(yaw)], dtype=np.float32)

                frame_obs = np.concatenate(
                    [
                        body_ang_vel(root_quat, mj_data.qvel[3:6]),
                        projected_gravity(root_quat),
                        base_forward_xy,
                        (base_target_pos[:2] - robot_pos[:2]).astype(np.float32),
                        (racket_target_pos - robot_pos).astype(np.float32),
                        racket_target_vel.astype(np.float32),
                        strike_time.astype(np.float32),
                        (mj_data.qpos[qpos_ids] - default_joint_pos).astype(np.float32),
                        mj_data.qvel[qvel_ids].astype(np.float32),
                        action.astype(np.float32),
                    ]
                )
                if first_obs:
                    obs_history[:] = frame_obs
                    first_obs = False
                else:
                    obs_history[:-1] = obs_history[1:]
                    obs_history[-1] = frame_obs
                obs = np.clip(obs_history.reshape(-1), -obs_clip, obs_clip)

                with torch.inference_mode():
                    action[:] = (
                        policy(torch.from_numpy(obs).unsqueeze(0)).detach().cpu().numpy().squeeze().astype(np.float32)
                    )
                action[:] = np.clip(action, -action_clip, action_clip)
                target_q[:] = action * action_scale + default_joint_pos

                racket_target_pos, racket_target_vel, base_target_pos, strike_time = compute_hitter_plan(
                    ball_pos, ball_vel, robot_pos, cfg
                )
                viewer.sync()

            if not no_realtime:
                wait = simulation_dt - (time.time() - step_start)
                if wait > 0.0:
                    time.sleep(wait)


if __name__ == "__main__":
    main()

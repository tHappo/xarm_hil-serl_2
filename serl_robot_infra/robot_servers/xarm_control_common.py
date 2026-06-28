"""
xArm compatibility helpers for HIL-SERL.

Design goal:
    Keep the original HIL-SERL FrankaEnv unchanged.
    HIL-SERL sends/receives TCP pose as:
        [x, y, z, qx, qy, qz, qw]
        meters + scipy quaternion order.

    xArm Python SDK commonly uses:
        [x, y, z, roll, pitch, yaw]
        millimeters + degrees when default_is_radian=False.

Put this file under:
    serl_robot_infra/robot_servers/xarm_control_common.py
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R


def _as_float_array(values: Any, shape: Optional[Tuple[int, ...]] = None) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if shape is not None and arr.shape != shape:
        raise ValueError(f"Expected shape {shape}, got {arr.shape}: {arr}")
    return arr


def unwrap_xarm_response(resp: Any, default: Any = None) -> Any:
    """Normalize xArm SDK return values.

    Many xArm SDK methods return either:
        (code, data)
    or directly:
        data

    This helper returns data if code == 0.  If an error appears, it raises so the
    caller can fall back to cached/default values.
    """
    if isinstance(resp, tuple):
        if len(resp) == 2:
            code, data = resp
            if int(code) != 0:
                raise RuntimeError(f"xArm SDK returned error code {code}")
            return data
        if len(resp) > 2:
            code, *data = resp
            if int(code) != 0:
                raise RuntimeError(f"xArm SDK returned error code {code}")
            return data
    if resp is None:
        if default is None:
            raise RuntimeError("xArm SDK returned None")
        return default
    return resp


def hilserl_quat_pose_to_xarm_pose(
    pose: Sequence[float], *, degrees: bool = True
) -> np.ndarray:
    """[m, quat xyzw] -> [mm, rpy]."""
    arr = _as_float_array(pose, (7,))
    xyz_mm = arr[:3] * 1000.0
    rpy = R.from_quat(arr[3:]).as_euler("xyz", degrees=degrees)
    return np.concatenate([xyz_mm, rpy])


def xarm_pose_to_hilserl_quat_pose(
    xarm_pose: Sequence[float], *, degrees: bool = True
) -> np.ndarray:
    """[mm, rpy] -> [m, quat xyzw]."""
    arr = _as_float_array(xarm_pose, (6,))
    xyz_m = arr[:3] / 1000.0
    quat = R.from_euler("xyz", arr[3:], degrees=degrees).as_quat()
    return np.concatenate([xyz_m, quat])


def xarm_pose_to_hilserl_euler_pose(
    xarm_pose: Sequence[float], *, degrees: bool = True
) -> np.ndarray:
    """[mm, rpy] -> [m, euler-rad].

    HIL-SERL's /getpos_euler endpoint returns xyz in meters and rpy in radians.
    """
    arr = _as_float_array(xarm_pose, (6,))
    xyz_m = arr[:3] / 1000.0
    if degrees:
        euler_rad = np.deg2rad(arr[3:])
    else:
        euler_rad = arr[3:]
    return np.concatenate([xyz_m, euler_rad])


def xarm_pose_to_tcp_speed(
    prev_pose: Sequence[float],
    curr_pose: Sequence[float],
    dt: float,
    *,
    degrees: bool = True,
) -> np.ndarray:
    """Finite-difference TCP speed in HIL-SERL convention.

    Output: [vx, vy, vz, wx, wy, wz], m/s + rad/s.
    """
    if dt <= 1e-6:
        return np.zeros(6, dtype=float)
    p0 = _as_float_array(prev_pose, (6,))
    p1 = _as_float_array(curr_pose, (6,))
    v = (p1[:3] - p0[:3]) / 1000.0 / dt
    r0 = R.from_euler("xyz", p0[3:], degrees=degrees)
    r1 = R.from_euler("xyz", p1[3:], degrees=degrees)
    rotvec = (r1 * r0.inv()).as_rotvec() / dt
    return np.concatenate([v, rotvec])


def pad_to_length(values: Sequence[float], length: int) -> list[float]:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size >= length:
        return arr[:length].tolist()
    out = np.zeros(length, dtype=float)
    out[: arr.size] = arr
    return out.tolist()


def zero_jacobian(dof_env: int = 7) -> list[list[float]]:
    return np.zeros((6, int(dof_env)), dtype=float).tolist()


def check_pose_jump(
    current_xarm_pose: Sequence[float],
    target_xarm_pose: Sequence[float],
    *,
    max_pos_delta_m: float,
    max_rot_delta_rad: float,
    degrees: bool = True,
) -> tuple[bool, str]:
    """Reject dangerous single-step jumps before sending to the robot."""
    current = _as_float_array(current_xarm_pose, (6,))
    target = _as_float_array(target_xarm_pose, (6,))
    pos_delta_m = float(np.linalg.norm((target[:3] - current[:3]) / 1000.0))
    r_cur = R.from_euler("xyz", current[3:], degrees=degrees)
    r_tar = R.from_euler("xyz", target[3:], degrees=degrees)
    rot_delta_rad = float(np.linalg.norm((r_tar * r_cur.inv()).as_rotvec()))
    if pos_delta_m > max_pos_delta_m:
        return False, f"translation jump {pos_delta_m:.4f} m exceeds {max_pos_delta_m:.4f} m"
    if rot_delta_rad > max_rot_delta_rad:
        return False, f"rotation jump {rot_delta_rad:.4f} rad exceeds {max_rot_delta_rad:.4f} rad"
    return True, ""


def parse_csv_floats(value: Optional[str], expected_len: Optional[int] = None) -> Optional[list[float]]:
    if value is None or value == "":
        return None
    out = [float(x.strip()) for x in value.split(",") if x.strip() != ""]
    if expected_len is not None and len(out) != expected_len:
        raise ValueError(f"Expected {expected_len} comma-separated values, got {len(out)}: {value}")
    return out


def normalize_gripper_position(raw_pos: float, open_value: float, closed_value: float) -> float:
    """Map gripper raw pulse/mm value to HIL-SERL convention: open ~= 1, closed ~= 0."""
    raw = float(raw_pos)
    lo, hi = min(open_value, closed_value), max(open_value, closed_value)
    if abs(hi - lo) < 1e-6:
        return 0.0
    if open_value > closed_value:
        return float(np.clip((raw - closed_value) / (open_value - closed_value), 0.0, 1.0))
    return float(np.clip((closed_value - raw) / (closed_value - open_value), 0.0, 1.0))


@dataclass
class XArmServerConfig:
    robot_ip: str
    flask_url: str = "127.0.0.1"
    flask_port: int = 5000
    dof_robot: int = 6
    dof_env: int = 7
    speed: float = 50.0          # mm/s for set_position
    mvacc: float = 500.0         # mm/s^2 for set_position
    max_pos_delta: float = 0.03  # m, server-side per-command safety limit
    max_rot_delta: float = 0.35  # rad, server-side per-command safety limit
    gripper_open_value: float = 850.0
    gripper_closed_value: float = 0.0
    gripper_speed: float = 5000.0
    use_gripper: bool = True
    use_ft_sensor: bool = False
    ft_force_limit: float = -1.0  # N; <=0 disables server-side force stop
    reset_joint_target: Optional[list[float]] = None  # degrees by default_is_radian=False
    workspace_low: Optional[list[float]] = None       # [x,y,z] in meters
    workspace_high: Optional[list[float]] = None      # [x,y,z] in meters
    degrees: bool = True


class XArmStateCache:
    """Small cache for finite-difference TCP velocity."""

    def __init__(self, init_pose: Sequence[float], *, degrees: bool = True):
        self.prev_pose = np.asarray(init_pose, dtype=float)
        self.curr_pose = np.asarray(init_pose, dtype=float)
        self.prev_time = time.time()
        self.curr_time = self.prev_time
        self.degrees = degrees
        self.curr_vel = np.zeros(6, dtype=float)

    def update(self, new_pose: Sequence[float]) -> np.ndarray:
        now = time.time()
        self.prev_pose = self.curr_pose.copy()
        self.prev_time = self.curr_time
        self.curr_pose = np.asarray(new_pose, dtype=float)
        self.curr_time = now
        self.curr_vel = xarm_pose_to_tcp_speed(
            self.prev_pose, self.curr_pose, self.curr_time - self.prev_time, degrees=self.degrees
        )
        return self.curr_vel

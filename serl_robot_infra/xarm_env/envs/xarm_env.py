from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from scipy.spatial.transform import Rotation


@dataclass
class XArmEnvConfig:
    robot_ip: str = "192.168.1.219"
    is_radian: bool = False

    # xArm joint HOME. Unit follows is_radian.
    # If is_radian=False, joint angles are degrees.
    home_joint_angles: list[float] = field(
        default_factory=lambda: [0, -30, 0, 60, 0, 90, 0]
    )
    home_speed: float = 30.0
    home_acc: float = 180.0

    # Control loop.
    control_hz: float = 20.0

    # Normalized action [-1, 1] to physical motion.
    max_translation_step_mm: float = 2.0
    max_rotation_step_deg: float = 2.0

    # Translation frame:
    #   "base": action xyz is interpreted in robot base frame.
    #   "tcp":  action xyz is interpreted in current TCP frame.
    translation_frame: str = "base"

    # xArm servo arguments. In mode=1, actual speed mainly depends on
    # target update frequency and adjacent target distance.
    servo_speed: float = 100.0
    servo_acc: float = 1000.0

    # Workspace safety, unit mm, base frame.
    x_min: float = 180.0
    x_max: float = 650.0
    y_min: float = -350.0
    y_max: float = 350.0
    z_min: float = 80.0
    z_max: float = 520.0

    # xArm Gripper G2, unit mm.
    gripper_enabled: bool = True
    gripper_open_mm: float = 84.0
    gripper_close_mm: float = 0.0
    gripper_speed: float = 100.0
    gripper_force: float = 50.0

    # Optional six-axis force torque.
    force_torque_enabled: bool = False

    # Startup/reset.
    startup_go_home: bool = True
    mode_settle_s: float = 0.2


class XArmSDKError(RuntimeError):
    pass


class XArmSafetyError(RuntimeError):
    pass


class XArmHILSerlEnv(gym.Env):
    """
    xArm7 Gym env for HIL-SERL.

    Action convention:
        action = [dx, dy, dz, droll, dpitch, dyaw, gripper]

    action[:6]:
        normalized in [-1, 1]

    action[6]:
        > 0: open gripper
        < 0: close gripper
        = 0: hold

    Internal xArm pose:
        [x_mm, y_mm, z_mm, roll, pitch, yaw]
        RPY unit follows config.is_radian.
    """

    metadata = {"render_modes": []}

    def __init__(self, config: XArmEnvConfig | None = None):
        super().__init__()
        self.config = config or XArmEnvConfig()

        self.arm: Any | None = None
        self._servo_started = False
        self._target_pose: np.ndarray | None = None
        self._last_step_t: float | None = None
        self._last_gripper_sign: int = 0

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(7,),
            dtype=np.float32,
        )

        self.observation_space = spaces.Dict(
            {
                "state": spaces.Dict(
                    {
                        "tcp_pose": spaces.Box(
                            low=-np.inf,
                            high=np.inf,
                            shape=(6,),
                            dtype=np.float32,
                        ),
                        "q": spaces.Box(
                            low=-np.inf,
                            high=np.inf,
                            shape=(7,),
                            dtype=np.float32,
                        ),
                        "gripper_pose": spaces.Box(
                            low=0.0,
                            high=1.0,
                            shape=(1,),
                            dtype=np.float32,
                        ),
                        "force_torque": spaces.Box(
                            low=-np.inf,
                            high=np.inf,
                            shape=(6,),
                            dtype=np.float32,
                        ),
                    }
                ),
                "images": spaces.Dict({}),
            }
        )

        self.connect()
        self.enable()

        if self.config.startup_go_home:
            self.go_home()

        self.start_servo()
        self._target_pose = self._read_pose().copy()
        self._last_step_t = time.monotonic()

    # ---------------------------------------------------------------------
    # xArm SDK helpers
    # ---------------------------------------------------------------------
    def _require_arm(self):
        if self.arm is None:
            raise XArmSDKError("xArm is not connected")
        return self.arm

    def _check_code(self, code, name: str):
        if code not in (0, None):
            raise XArmSDKError(f"{name} failed with code={code}")

    def connect(self):
        if self.arm is not None:
            return

        from xarm.wrapper import XArmAPI

        self.arm = XArmAPI(self.config.robot_ip, is_radian=self.config.is_radian)

        if hasattr(self.arm, "connect") and not getattr(self.arm, "connected", True):
            self._check_code(self.arm.connect(), "connect")

    def enable(self):
        arm = self._require_arm()

        if hasattr(arm, "clean_warn"):
            self._check_code(arm.clean_warn(), "clean_warn")
        if hasattr(arm, "clean_error"):
            self._check_code(arm.clean_error(), "clean_error")

        self._check_code(arm.motion_enable(enable=True), "motion_enable")
        self._check_code(arm.set_mode(0), "set_mode(0)")
        self._check_code(arm.set_state(0), "set_state(0)")
        time.sleep(self.config.mode_settle_s)

        if self.config.gripper_enabled and hasattr(arm, "set_gripper_enable"):
            self._check_code(arm.set_gripper_enable(True), "set_gripper_enable")

        if self.config.force_torque_enabled:
            self.enable_force_torque_sensor()

    def enable_force_torque_sensor(self):
        arm = self._require_arm()
        if not hasattr(arm, "set_ft_sensor_enable"):
            print("[WARN] xArm F/T sensor API unavailable: set_ft_sensor_enable")
            return
        self._check_code(arm.set_ft_sensor_enable(True), "set_ft_sensor_enable")

    def start_servo(self):
        arm = self._require_arm()

        if hasattr(arm, "clean_warn"):
            self._check_code(arm.clean_warn(), "clean_warn")
        if hasattr(arm, "clean_error"):
            self._check_code(arm.clean_error(), "clean_error")

        self._check_code(arm.motion_enable(enable=True), "motion_enable")
        self._check_code(arm.set_mode(1), "set_mode(1)")
        self._check_code(arm.set_state(0), "set_state(0)")
        time.sleep(self.config.mode_settle_s)

        self._servo_started = True

    def stop_servo(self, keep_enabled: bool = True):
        if self.arm is None:
            return

        arm = self.arm

        if hasattr(arm, "set_state"):
            self._check_code(arm.set_state(4), "set_state(4)")

        if keep_enabled:
            self._check_code(arm.set_mode(0), "set_mode(0)")
            self._check_code(arm.set_state(0), "set_state(0)")

        self._servo_started = False

    def go_home(self):
        arm = self._require_arm()

        self.stop_servo(keep_enabled=True)

        self._check_code(arm.set_mode(0), "set_mode(0)")
        self._check_code(arm.set_state(0), "set_state(0)")
        time.sleep(self.config.mode_settle_s)

        self._check_code(
            arm.set_servo_angle(
                angle=[float(x) for x in self.config.home_joint_angles],
                speed=float(self.config.home_speed),
                mvacc=float(self.config.home_acc),
                wait=True,
                is_radian=self.config.is_radian,
            ),
            "set_servo_angle(home)",
        )

    def _read_pose(self) -> np.ndarray:
        arm = self._require_arm()

        pose = np.asarray(getattr(arm, "position", []), dtype=np.float64)
        if pose.shape != (6,):
            code, pose = arm.get_position(is_radian=self.config.is_radian)
            self._check_code(code, "get_position")
            pose = np.asarray(pose, dtype=np.float64)

        if pose.shape != (6,):
            raise XArmSDKError(f"xArm pose must be shape (6,), got {pose.shape}")

        return pose

    def _read_joints_rad(self) -> np.ndarray:
        arm = self._require_arm()

        q = np.asarray(getattr(arm, "angles", []), dtype=np.float64)

        if q.shape[0] != 7:
            if hasattr(arm, "get_servo_angle"):
                code, q2 = arm.get_servo_angle(is_radian=True)
                if code in (0, None):
                    q = np.asarray(q2, dtype=np.float64)

        if q.shape[0] != 7:
            out = np.zeros(7, dtype=np.float64)
            out[: min(7, q.shape[0])] = q[: min(7, q.shape[0])]
            q = out

        if not self.config.is_radian:
            q = np.deg2rad(q)

        return q.astype(np.float64)

    def _read_gripper_norm(self) -> float:
        if not self.config.gripper_enabled:
            return 0.0

        arm = self._require_arm()
        pos = self.config.gripper_open_mm

        if hasattr(arm, "get_gripper_g2_position"):
            resp = arm.get_gripper_g2_position()
            if isinstance(resp, tuple) and len(resp) == 2:
                code, value = resp
                if code in (0, None):
                    pos = float(value)

        denom = max(1e-6, self.config.gripper_open_mm - self.config.gripper_close_mm)
        gripper_norm = (float(pos) - self.config.gripper_close_mm) / denom
        return float(np.clip(gripper_norm, 0.0, 1.0))

    def _read_force_torque(self) -> np.ndarray:
        if not self.config.force_torque_enabled:
            return np.zeros(6, dtype=np.float32)

        arm = self._require_arm()

        if not hasattr(arm, "get_ft_sensor_data"):
            return np.zeros(6, dtype=np.float32)

        resp = arm.get_ft_sensor_data(is_raw=False)
        if not isinstance(resp, tuple) or len(resp) != 2:
            return np.zeros(6, dtype=np.float32)

        code, data = resp
        if code not in (0, None):
            return np.zeros(6, dtype=np.float32)

        ft = np.asarray(data, dtype=np.float32)
        if ft.shape != (6,) or not np.all(np.isfinite(ft)):
            return np.zeros(6, dtype=np.float32)

        return ft

    # ---------------------------------------------------------------------
    # Safety and action conversion
    # ---------------------------------------------------------------------
    def _clip_pose_workspace(self, pose: np.ndarray) -> np.ndarray:
        c = self.config
        out = pose.copy()
        out[0] = np.clip(out[0], c.x_min, c.x_max)
        out[1] = np.clip(out[1], c.y_min, c.y_max)
        out[2] = np.clip(out[2], c.z_min, c.z_max)
        return out

    def _check_pose_safe(self, pose: np.ndarray):
        x, y, z = pose[:3]
        c = self.config

        if not (c.x_min <= x <= c.x_max):
            raise XArmSafetyError(f"x={x:.2f} outside [{c.x_min}, {c.x_max}]")
        if not (c.y_min <= y <= c.y_max):
            raise XArmSafetyError(f"y={y:.2f} outside [{c.y_min}, {c.y_max}]")
        if not (c.z_min <= z <= c.z_max):
            raise XArmSafetyError(f"z={z:.2f} outside [{c.z_min}, {c.z_max}]")

    def _compose_pose_delta(self, current_target: np.ndarray, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float64)
        action = np.clip(action, -1.0, 1.0)

        delta_xyz = action[:3] * self.config.max_translation_step_mm
        delta_rpy_deg = action[3:6] * self.config.max_rotation_step_deg

        target = current_target.copy()

        current_rot = Rotation.from_euler(
            "xyz",
            target[3:6],
            degrees=not self.config.is_radian,
        )

        if self.config.translation_frame == "tcp":
            delta_xyz = current_rot.apply(delta_xyz)
        elif self.config.translation_frame != "base":
            raise ValueError(
                f"translation_frame must be 'base' or 'tcp', got {self.config.translation_frame}"
            )

        target[:3] += delta_xyz

        delta_rot = Rotation.from_euler("xyz", delta_rpy_deg, degrees=True)
        new_rot = current_rot * delta_rot
        target[3:6] = new_rot.as_euler(
            "xyz",
            degrees=not self.config.is_radian,
        )

        target = self._clip_pose_workspace(target)
        return target

    def _send_gripper_action(self, g: float):
        if not self.config.gripper_enabled:
            return

        g = float(g)

        if abs(g) < 1e-6:
            self._last_gripper_sign = 0
            return

        sign = 1 if g > 0 else -1
        if sign == self._last_gripper_sign:
            return

        arm = self._require_arm()
        target = self.config.gripper_open_mm if g > 0 else self.config.gripper_close_mm

        if hasattr(arm, "set_gripper_g2_position"):
            self._check_code(
                arm.set_gripper_g2_position(
                    float(target),
                    speed=float(self.config.gripper_speed),
                    force=float(self.config.gripper_force),
                    wait=False,
                ),
                "set_gripper_g2_position",
            )
        elif hasattr(arm, "set_gripper_position"):
            self._check_code(
                arm.set_gripper_position(float(target), wait=False),
                "set_gripper_position",
            )
        else:
            print("[WARN] No xArm gripper API found.")

        self._last_gripper_sign = sign

    def _servo_pose(self, pose: np.ndarray):
        if not self._servo_started:
            raise XArmSDKError("servo_pose called before start_servo()")

        arm = self._require_arm()
        self._check_pose_safe(pose)

        self._check_code(
            arm.set_servo_cartesian(
                pose.astype(float).tolist(),
                speed=float(self.config.servo_speed),
                mvacc=float(self.config.servo_acc),
                is_radian=self.config.is_radian,
            ),
            "set_servo_cartesian",
        )

    # ---------------------------------------------------------------------
    # Gym API
    # ---------------------------------------------------------------------
    def get_obs(self):
        pose = self._read_pose()
        q = self._read_joints_rad()
        gripper = self._read_gripper_norm()
        ft = self._read_force_torque()

        tcp_pose = pose.astype(np.float64).copy()
        tcp_pose[:3] /= 1000.0
        if not self.config.is_radian:
            tcp_pose[3:6] = np.deg2rad(tcp_pose[3:6])

        return {
            "state": {
                "tcp_pose": tcp_pose.astype(np.float32),
                "q": q.astype(np.float32),
                "gripper_pose": np.asarray([gripper], dtype=np.float32),
                "force_torque": ft.astype(np.float32),
            },
            "images": {},
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.go_home()
        self.start_servo()

        self._target_pose = self._read_pose().copy()
        self._last_step_t = time.monotonic()
        self._last_gripper_sign = 0

        return self.get_obs(), {}

    def step(self, action):
        if self._target_pose is None:
            self._target_pose = self._read_pose().copy()

        action = np.asarray(action, dtype=np.float32)
        if action.shape != (7,):
            raise ValueError(f"xArm action must be shape (7,), got {action.shape}")

        error = None

        try:
            self._send_gripper_action(float(action[6]))
            self._target_pose = self._compose_pose_delta(self._target_pose, action)
            self._servo_pose(self._target_pose)
        except Exception as exc:
            error = repr(exc)
            print(f"[XArmHILSerlEnv ERROR] {error}")
            self.stop_servo(keep_enabled=True)

        now = time.monotonic()
        if self._last_step_t is not None:
            period = 1.0 / float(self.config.control_hz)
            sleep_s = self._last_step_t + period - now
            if sleep_s > 0:
                time.sleep(sleep_s)
        self._last_step_t = time.monotonic()

        obs = self.get_obs()
        reward = 0.0
        done = error is not None
        truncated = False

        info = {
            "target_pose": None if self._target_pose is None else self._target_pose.copy(),
            "xarm_error": error,
        }

        return obs, reward, done, truncated, info

    def close(self):
        try:
            self.stop_servo(keep_enabled=True)
        except Exception:
            pass

        if self.arm is not None and hasattr(self.arm, "disconnect"):
            try:
                self.arm.disconnect()
            except Exception:
                pass

        self.arm = None
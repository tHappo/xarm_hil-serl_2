"""
Minimal xArm server for HIL-SERL.

Put this file under:
    serl_robot_infra/robot_servers/xarm_server.py

It emulates the original HIL-SERL Franka HTTP API so that the original
franka_env/envs/franka_env.py can remain unchanged.

Original HIL-SERL expects:
    POST /pose          json={"arr": [x,y,z,qx,qy,qz,qw]}  # meters + quat xyzw
    POST /getstate      -> pose, vel, force, torque, q, dq, jacobian, gripper_pos
    POST /open_gripper
    POST /close_gripper
    POST /move_gripper
    POST /jointreset
    POST /clearerr
    POST /update_param

Requirements:
    pip install xarm-python-sdk flask scipy numpy

Important safety notes:
    1. Start with very small ACTION_SCALE in HIL-SERL config.py.
    2. Set a conservative workspace in HIL-SERL config.py and, optionally,
       also pass --workspace_low / --workspace_high here.
    3. Test with /getstate and a 1~2 mm /pose command before running RL.
"""

from __future__ import annotations

import argparse
import traceback
from typing import Any

import numpy as np
from flask import Flask, jsonify, request

from robot_servers.xarm_control_common import (
    XArmServerConfig,
    XArmStateCache,
    check_pose_jump,
    hilserl_quat_pose_to_xarm_pose,
    normalize_gripper_position,
    pad_to_length,
    parse_csv_floats,
    unwrap_xarm_response,
    xarm_pose_to_hilserl_euler_pose,
    xarm_pose_to_hilserl_quat_pose,
    zero_jacobian,
)


class XArmHardware:
    """Thin wrapper around xarm.wrapper.XArmAPI.

    The wrapper keeps all unit conversions and error handling out of Flask routes.
    """

    def __init__(self, config: XArmServerConfig):
        self.config = config
        from xarm.wrapper import XArmAPI

        # default_is_radian=False => xArm SDK returns/accepts degrees for roll/pitch/yaw and joints.
        self.arm = XArmAPI(config.robot_ip, is_radian=not config.degrees)
        self.last_gripper_raw = float(config.gripper_open_value)
        self._connect_and_initialize()

        init_pose = self.get_xarm_pose()
        self.pose_cache = XArmStateCache(init_pose, degrees=config.degrees)

    def _connect_and_initialize(self) -> None:
        self.arm.clean_warn()
        self.arm.clean_error()
        self.arm.motion_enable(enable=True)
        self.arm.set_mode(0)   # 0: position control mode. Safer for first HIL-SERL bring-up.
        self.arm.set_state(0)  # 0: ready
        if self.config.use_gripper:
            try:
                self.arm.set_gripper_enable(True)
                self.arm.set_gripper_mode(0)
                self.arm.set_gripper_speed(self.config.gripper_speed)
            except Exception:
                print("[WARN] Failed to initialize xArm gripper. Continue without hard failure.")
                traceback.print_exc()

    def clear_error(self) -> None:
        self.arm.clean_warn()
        self.arm.clean_error()
        self.arm.motion_enable(enable=True)
        self.arm.set_state(0)

    def get_xarm_pose(self) -> np.ndarray:
        """Return [x,y,z,roll,pitch,yaw], mm + deg/rad according to config.degrees."""
        try:
            pose = unwrap_xarm_response(self.arm.get_position())
        except Exception:
            # Some SDK versions expose a cached property.
            pose = getattr(self.arm, "position", None)
            if pose is None:
                raise
        pose = np.asarray(pose, dtype=float).reshape(-1)
        if pose.size < 6:
            raise RuntimeError(f"xArm pose has wrong shape: {pose}")
        return pose[:6]

    def get_joint_angles(self) -> np.ndarray:
        try:
            q = unwrap_xarm_response(self.arm.get_servo_angle())
        except Exception:
            q = getattr(self.arm, "angles", np.zeros(self.config.dof_robot))
        q = np.asarray(q, dtype=float).reshape(-1)
        return q[: self.config.dof_robot]

    def get_joint_velocities(self) -> np.ndarray:
        # xArm Python SDK does not always expose stable joint velocity through get_joint_states.
        # Return zeros if unavailable; HIL-SERL's default proprio does not require dq directly.
        try:
            js = unwrap_xarm_response(self.arm.get_joint_states())
            # SDK variants differ; search for an array with dof_robot length after q.
            if isinstance(js, (list, tuple)):
                arrays = [np.asarray(x, dtype=float).reshape(-1) for x in js if np.asarray(x).size >= self.config.dof_robot]
                if len(arrays) >= 2:
                    return arrays[1][: self.config.dof_robot]
        except Exception:
            pass
        return np.zeros(self.config.dof_robot, dtype=float)

    def get_force_torque(self) -> np.ndarray:
        """Return [Fx,Fy,Fz,Tx,Ty,Tz]. If no FT sensor is available, return zeros."""
        if not self.config.use_ft_sensor:
            return np.zeros(6, dtype=float)
        try:
            ft = unwrap_xarm_response(self.arm.get_ft_sensor_data())
            ft = np.asarray(ft, dtype=float).reshape(-1)
            out = np.zeros(6, dtype=float)
            out[: min(6, ft.size)] = ft[: min(6, ft.size)]
            return out
        except Exception:
            return np.zeros(6, dtype=float)

    def get_gripper_raw(self) -> float:
        if not self.config.use_gripper:
            return self.last_gripper_raw
        try:
            pos = unwrap_xarm_response(self.arm.get_gripper_position())
            self.last_gripper_raw = float(np.asarray(pos).reshape(-1)[0])
        except Exception:
            pass
        return float(self.last_gripper_raw)

    def _clip_workspace(self, xarm_pose: np.ndarray) -> np.ndarray:
        if self.config.workspace_low is None or self.config.workspace_high is None:
            return xarm_pose
        low_m = np.asarray(self.config.workspace_low, dtype=float)
        high_m = np.asarray(self.config.workspace_high, dtype=float)
        xyz_m = xarm_pose[:3] / 1000.0
        xyz_m = np.clip(xyz_m, low_m, high_m)
        out = xarm_pose.copy()
        out[:3] = xyz_m * 1000.0
        return out

    def _force_stop_check(self) -> tuple[bool, str]:
        if self.config.ft_force_limit <= 0:
            return True, ""
        ft = self.get_force_torque()
        norm_f = float(np.linalg.norm(ft[:3]))
        if norm_f > self.config.ft_force_limit:
            return False, f"force norm {norm_f:.2f} N exceeds {self.config.ft_force_limit:.2f} N"
        return True, ""

    def move_hilserl_pose(self, hilserl_pose: list[float]) -> tuple[bool, str]:
        """Move to [x,y,z,qx,qy,qz,qw] in meters + quat xyzw."""
        target = hilserl_quat_pose_to_xarm_pose(hilserl_pose, degrees=self.config.degrees)
        target = self._clip_workspace(target)
        current = self.get_xarm_pose()

        ok, reason = check_pose_jump(
            current,
            target,
            max_pos_delta_m=self.config.max_pos_delta,
            max_rot_delta_rad=self.config.max_rot_delta,
            degrees=self.config.degrees,
        )
        if not ok:
            return False, reason

        ok, reason = self._force_stop_check()
        if not ok:
            return False, reason

        # xArm set_position expects x/y/z in mm and roll/pitch/yaw in deg if is_radian=False.
        code = self.arm.set_position(
            x=float(target[0]),
            y=float(target[1]),
            z=float(target[2]),
            roll=float(target[3]),
            pitch=float(target[4]),
            yaw=float(target[5]),
            speed=float(self.config.speed),
            mvacc=float(self.config.mvacc),
            wait=False,
        )
        if isinstance(code, tuple):
            code = code[0]
        if int(code) != 0:
            return False, f"xArm set_position returned code {code}"
        return True, ""

    def open_gripper(self) -> None:
        if not self.config.use_gripper:
            return
        self.arm.set_gripper_position(
            int(self.config.gripper_open_value),
            wait=False,
            speed=int(self.config.gripper_speed),
        )
        self.last_gripper_raw = float(self.config.gripper_open_value)

    def close_gripper(self) -> None:
        if not self.config.use_gripper:
            return
        self.arm.set_gripper_position(
            int(self.config.gripper_closed_value),
            wait=False,
            speed=int(self.config.gripper_speed),
        )
        self.last_gripper_raw = float(self.config.gripper_closed_value)

    def move_gripper(self, raw_position: float) -> None:
        if not self.config.use_gripper:
            self.last_gripper_raw = float(raw_position)
            return
        lo = min(self.config.gripper_open_value, self.config.gripper_closed_value)
        hi = max(self.config.gripper_open_value, self.config.gripper_closed_value)
        pos = float(np.clip(raw_position, lo, hi))
        self.arm.set_gripper_position(int(pos), wait=False, speed=int(self.config.gripper_speed))
        self.last_gripper_raw = pos

    def joint_reset(self) -> tuple[bool, str]:
        if self.config.reset_joint_target is None:
            return False, "joint reset disabled: --reset_joint_target not provided"
        q = np.asarray(self.config.reset_joint_target, dtype=float).reshape(-1)
        if q.size != self.config.dof_robot:
            return False, f"reset_joint_target length {q.size} != dof_robot {self.config.dof_robot}"
        code = self.arm.set_servo_angle(
            angle=q.tolist(),
            speed=20,
            mvacc=200,
            wait=True,
        )
        if isinstance(code, tuple):
            code = code[0]
        if int(code) != 0:
            return False, f"xArm set_servo_angle returned code {code}"
        return True, ""

    def get_state_payload(self) -> dict[str, Any]:
        pose_xarm = self.get_xarm_pose()
        vel = self.pose_cache.update(pose_xarm)
        ft = self.get_force_torque()
        q = self.get_joint_angles()
        dq = self.get_joint_velocities()
        gripper_raw = self.get_gripper_raw()
        gripper_norm = normalize_gripper_position(
            gripper_raw,
            self.config.gripper_open_value,
            self.config.gripper_closed_value,
        )
        return {
            "pose": xarm_pose_to_hilserl_quat_pose(pose_xarm, degrees=self.config.degrees).tolist(),
            "vel": vel.tolist(),
            "force": ft[:3].tolist(),
            "torque": ft[3:6].tolist(),
            # Keep original FrankaEnv unchanged: pad xArm6 q/dq to 7 and return 6x7 Jacobian.
            "q": pad_to_length(q, self.config.dof_env),
            "dq": pad_to_length(dq, self.config.dof_env),
            "jacobian": zero_jacobian(self.config.dof_env),
            "gripper_pos": float(gripper_norm),
        }


def create_app(robot: XArmHardware) -> Flask:
    app = Flask(__name__)

    def safe_json_error(exc: Exception, status: int = 500):
        traceback.print_exc()
        return jsonify({"error": str(exc)}), status

    @app.route("/getpos", methods=["POST"])
    def get_pos():
        try:
            pose = robot.get_xarm_pose()
            return jsonify({"pose": xarm_pose_to_hilserl_quat_pose(pose, degrees=robot.config.degrees).tolist()})
        except Exception as exc:
            return safe_json_error(exc)

    @app.route("/getpos_euler", methods=["POST"])
    def get_pos_euler():
        try:
            pose = robot.get_xarm_pose()
            return jsonify({"pose": xarm_pose_to_hilserl_euler_pose(pose, degrees=robot.config.degrees).tolist()})
        except Exception as exc:
            return safe_json_error(exc)

    @app.route("/getvel", methods=["POST"])
    def get_vel():
        try:
            return jsonify({"vel": robot.get_state_payload()["vel"]})
        except Exception as exc:
            return safe_json_error(exc)

    @app.route("/getforce", methods=["POST"])
    def get_force():
        try:
            return jsonify({"force": robot.get_state_payload()["force"]})
        except Exception as exc:
            return safe_json_error(exc)

    @app.route("/gettorque", methods=["POST"])
    def get_torque():
        try:
            return jsonify({"torque": robot.get_state_payload()["torque"]})
        except Exception as exc:
            return safe_json_error(exc)

    @app.route("/getq", methods=["POST"])
    def get_q():
        try:
            return jsonify({"q": robot.get_state_payload()["q"]})
        except Exception as exc:
            return safe_json_error(exc)

    @app.route("/getdq", methods=["POST"])
    def get_dq():
        try:
            return jsonify({"dq": robot.get_state_payload()["dq"]})
        except Exception as exc:
            return safe_json_error(exc)

    @app.route("/getjacobian", methods=["POST"])
    def get_jacobian():
        return jsonify({"jacobian": zero_jacobian(robot.config.dof_env)})

    @app.route("/get_gripper", methods=["POST"])
    def get_gripper():
        try:
            gripper_raw = robot.get_gripper_raw()
            gripper_norm = normalize_gripper_position(
                gripper_raw,
                robot.config.gripper_open_value,
                robot.config.gripper_closed_value,
            )
            return jsonify({"gripper": float(gripper_norm)})
        except Exception as exc:
            return safe_json_error(exc)

    @app.route("/getstate", methods=["POST"])
    def get_state():
        try:
            return jsonify(robot.get_state_payload())
        except Exception as exc:
            return safe_json_error(exc)

    @app.route("/pose", methods=["POST"])
    def pose():
        try:
            data = request.get_json(force=True)
            if "arr" not in data:
                return jsonify({"error": "missing key 'arr'"}), 400
            ok, reason = robot.move_hilserl_pose(data["arr"])
            if not ok:
                return jsonify({"error": reason}), 400
            return "Moved"
        except Exception as exc:
            return safe_json_error(exc)

    @app.route("/open_gripper", methods=["POST"])
    def open_gripper():
        try:
            robot.open_gripper()
            return "Opened"
        except Exception as exc:
            return safe_json_error(exc)

    @app.route("/close_gripper", methods=["POST"])
    def close_gripper():
        try:
            robot.close_gripper()
            return "Closed"
        except Exception as exc:
            return safe_json_error(exc)

    @app.route("/close_gripper_slow", methods=["POST"])
    def close_gripper_slow():
        try:
            robot.close_gripper()
            return "Closed"
        except Exception as exc:
            return safe_json_error(exc)

    @app.route("/move_gripper", methods=["POST"])
    def move_gripper():
        try:
            data = request.get_json(force=True)
            # HIL-SERL Robotiq path sends gripper_pos.  Keep same key.
            raw_pos = float(data.get("gripper_pos", robot.config.gripper_open_value))
            robot.move_gripper(raw_pos)
            return "Moved Gripper"
        except Exception as exc:
            return safe_json_error(exc)

    @app.route("/activate_gripper", methods=["POST"])
    def activate_gripper():
        # Compatibility endpoint. xArm gripper is initialized at server start.
        return "Activated"

    @app.route("/reset_gripper", methods=["POST"])
    def reset_gripper():
        try:
            robot.open_gripper()
            return "Reset"
        except Exception as exc:
            return safe_json_error(exc)

    @app.route("/jointreset", methods=["POST"])
    def joint_reset():
        try:
            ok, reason = robot.joint_reset()
            if not ok:
                return reason
            return "Reset Joint"
        except Exception as exc:
            return safe_json_error(exc)

    @app.route("/clearerr", methods=["POST"])
    def clearerr():
        try:
            robot.clear_error()
            return "Clear"
        except Exception as exc:
            return safe_json_error(exc)

    # Compatibility no-op endpoints.  Franka uses these for impedance / payload.
    # xArm version keeps them so the original HIL-SERL env does not crash.
    @app.route("/startimp", methods=["POST"])
    def start_impedance_compat():
        return "xArm server has no Franka impedance controller"

    @app.route("/stopimp", methods=["POST"])
    def stop_impedance_compat():
        return "xArm server has no Franka impedance controller"

    @app.route("/update_param", methods=["POST"])
    def update_param_compat():
        return "xArm server ignores Franka compliance params"

    @app.route("/set_load", methods=["POST"])
    def set_load_compat():
        return "xArm server does not set payload through this endpoint"

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot_ip", required=True)
    parser.add_argument("--flask_url", default="127.0.0.1")
    parser.add_argument("--flask_port", type=int, default=5000)
    parser.add_argument("--dof_robot", type=int, default=6)
    parser.add_argument("--dof_env", type=int, default=7, help="Keep 7 to avoid changing FrankaEnv.")
    parser.add_argument("--speed", type=float, default=50.0, help="TCP speed in mm/s.")
    parser.add_argument("--mvacc", type=float, default=500.0, help="TCP acceleration in mm/s^2.")
    parser.add_argument("--max_pos_delta", type=float, default=0.03, help="Max per-command translation jump in meters.")
    parser.add_argument("--max_rot_delta", type=float, default=0.35, help="Max per-command rotation jump in radians.")
    parser.add_argument("--gripper_open_value", type=float, default=850.0)
    parser.add_argument("--gripper_closed_value", type=float, default=0.0)
    parser.add_argument("--gripper_speed", type=float, default=5000.0)
    parser.add_argument("--no_gripper", action="store_true")
    parser.add_argument("--use_ft_sensor", action="store_true")
    parser.add_argument("--ft_force_limit", type=float, default=-1.0, help="N; <=0 disables server-side force limit.")
    parser.add_argument(
        "--reset_joint_target",
        default=None,
        help="Comma-separated joint reset target in degrees, length=dof_robot. Example: '0,-30,0,60,0,30,0'",
    )
    parser.add_argument(
        "--workspace_low",
        default=None,
        help="Optional comma-separated xyz lower bound in meters, e.g. '0.20,-0.25,0.05'",
    )
    parser.add_argument(
        "--workspace_high",
        default=None,
        help="Optional comma-separated xyz upper bound in meters, e.g. '0.55,0.25,0.45'",
    )
    args = parser.parse_args()

    reset_joint_target = parse_csv_floats(args.reset_joint_target, expected_len=args.dof_robot)
    workspace_low = parse_csv_floats(args.workspace_low, expected_len=3)
    workspace_high = parse_csv_floats(args.workspace_high, expected_len=3)

    config = XArmServerConfig(
        robot_ip=args.robot_ip,
        flask_url=args.flask_url,
        flask_port=args.flask_port,
        dof_robot=args.dof_robot,
        dof_env=args.dof_env,
        speed=args.speed,
        mvacc=args.mvacc,
        max_pos_delta=args.max_pos_delta,
        max_rot_delta=args.max_rot_delta,
        gripper_open_value=args.gripper_open_value,
        gripper_closed_value=args.gripper_closed_value,
        gripper_speed=args.gripper_speed,
        use_gripper=not args.no_gripper,
        use_ft_sensor=args.use_ft_sensor,
        ft_force_limit=args.ft_force_limit,
        reset_joint_target=reset_joint_target,
        workspace_low=workspace_low,
        workspace_high=workspace_high,
        degrees=True,
    )

    robot = XArmHardware(config)
    app = create_app(robot)
    app.run(host=config.flask_url, port=config.flask_port)


if __name__ == "__main__":
    main()

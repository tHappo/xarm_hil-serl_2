from __future__ import annotations

from dataclasses import dataclass

from xarm_env.envs.xarm_env import XArmEnvConfig, XArmHILSerlEnv
from xarm_env.envs.wrappers import StableSpacemouseIntervention


@dataclass
class XArmUSBInsertionTrainConfig:
    exp_name: str = "xarm_usb_insertion"

    def get_environment(
        self,
        fake_env: bool = False,
        save_video: bool = False,
        classifier: bool = False,
    ):
        if fake_env:
            raise NotImplementedError("fake_env is not implemented for xArm env yet.")

        xarm_cfg = XArmEnvConfig(
            robot_ip="192.168.1.219",
            is_radian=False,

            # 改成你自己保存的安全 HOME。
            home_joint_angles=[
                -0.332144,
                -2.590686,
                -0.109779,
                74.35216,
                -0.000688,
                73.985467,
                -0.001547,
            ],
            home_speed=30.0,
            home_acc=180.0,

            control_hz=20.0,

            # 每一帧最大位移。20Hz 下，2mm/step 大约 40mm/s。
            max_translation_step_mm=2.0,
            max_rotation_step_deg=2.0,

            # "base" 更直观；如果希望沿 TCP 局部方向移动，改成 "tcp"。
            translation_frame="base",

            # xArm servo 参数。
            servo_speed=100.0,
            servo_acc=1000.0,

            # 根据你的工装实际调整。
            x_min=180.0,
            x_max=650.0,
            y_min=-350.0,
            y_max=350.0,
            z_min=80.0,
            z_max=520.0,

            gripper_enabled=True,
            gripper_open_mm=84.0,
            gripper_close_mm=0.0,
            gripper_speed=100.0,
            gripper_force=50.0,

            force_torque_enabled=False,

            startup_go_home=True,
            mode_settle_s=0.2,
        )

        env = XArmHILSerlEnv(config=xarm_cfg)

        env = StableSpacemouseIntervention(
            env,

            # 方向修正只改这两项。
            axis_order=(0, 1, 2, 3, 4, 5),
            axis_signs=(1, 1, 1, 1, 1, 1),

            # 遥操手感。
            translation_scale=0.85,
            rotation_scale=0.45,
            deadzone=0.05,
            precision_gamma=1.6,
            smoothing_tau=0.045,
            max_action_rate=12.0,
            intervene_threshold=0.01,

            # 按键。
            gripper_enabled=True,
            left_close=True,
            open_gripper_cmd=1.0,
            close_gripper_cmd=-1.0,
            both_buttons_reset=True,
        )

        return env

from __future__ import annotations

import os
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from franka_env.envs.wrappers import (
    Quat2EulerWrapper,
    SpacemouseIntervention,
    MultiCameraBinaryRewardClassifierWrapper,
)
from franka_env.envs.relative_env import RelativeFrame
from franka_env.envs.franka_env import DefaultEnvConfig

from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from serl_launcher.wrappers.chunking import ChunkingWrapper
from serl_launcher.networks.reward_classifier import load_classifier_func

from experiments.config import DefaultTrainingConfig
from experiments.usb_pickup_insertion.wrapper import USBEnv, GripperPenaltyWrapper

from xarm_env.envs.xarm_env import XArmEnvConfig, XArmHILSerlEnv
from xarm_env.envs.wrappers import StableSpacemouseIntervention

class EnvConfig(DefaultEnvConfig):
    SERVER_URL: str = "http://127.0.0.2:5000/"
    REALSENSE_CAMERAS = {
        "wrist_1": {
            "serial_number": "127122270350",
            "dim": (1280, 720),
            "exposure": 10500,
        },
        "wrist_2": {
            "serial_number": "127122270146",
            "dim": (1280, 720),
            "exposure": 10500,
        },
        "side_policy": {
            "serial_number": "130322274175",
            "dim": (1280, 720),
            "exposure": 13000,
        },
        "side_classifier": {
            "serial_number": "130322274175",
            "dim": (1280, 720),
            "exposure": 13000,
        },
    }
    IMAGE_CROP = {"wrist_1": lambda img: img[50:-200, 200:-200],
                  "wrist_2": lambda img: img[:-200, 200:-200],
                  "side_policy": lambda img: img[250:500, 350:650],
                  "side_classifier": lambda img: img[270:398, 500:628]}
    TARGET_POSE = np.array([0.553,0.1769683108549487,0.25097833796596336, np.pi, 0, -np.pi/2])
    RESET_POSE = TARGET_POSE + np.array([0, 0.03, 0.05, 0, 0, 0])
    ACTION_SCALE = np.array([0.015, 0.1, 1])
    RANDOM_RESET = True
    DISPLAY_IMAGE = True
    RANDOM_XY_RANGE = 0.01
    RANDOM_RZ_RANGE = 0.1
    ABS_POSE_LIMIT_HIGH = TARGET_POSE + np.array([0.03, 0.06, 0.05, 0.1, 0.1, 0.3])
    ABS_POSE_LIMIT_LOW = TARGET_POSE - np.array([0.03, 0.01, 0.03, 0.1, 0.1, 0.3])
    COMPLIANCE_PARAM = {
        "translational_stiffness": 2000,
        "translational_damping": 89,
        "rotational_stiffness": 150,
        "rotational_damping": 7,
        "translational_Ki": 0,
        "translational_clip_x": 0.006,
        "translational_clip_y": 0.0059,
        "translational_clip_z": 0.0035,
        "translational_clip_neg_x": 0.005,
        "translational_clip_neg_y": 0.005,
        "translational_clip_neg_z": 0.0035,
        "rotational_clip_x": 0.02,
        "rotational_clip_y": 0.02,
        "rotational_clip_z": 0.015,
        "rotational_clip_neg_x": 0.02,
        "rotational_clip_neg_y": 0.02,
        "rotational_clip_neg_z": 0.015,
        "rotational_Ki": 0,
    }
    PRECISION_PARAM = {
        "translational_stiffness": 2000,
        "translational_damping": 89,
        "rotational_stiffness": 150,
        "rotational_damping": 7,
        "translational_Ki": 0.0,
        "translational_clip_x": 0.01,
        "translational_clip_y": 0.01,
        "translational_clip_z": 0.01,
        "translational_clip_neg_x": 0.01,
        "translational_clip_neg_y": 0.01,
        "translational_clip_neg_z": 0.01,
        "rotational_clip_x": 0.03,
        "rotational_clip_y": 0.03,
        "rotational_clip_z": 0.03,
        "rotational_clip_neg_x": 0.03,
        "rotational_clip_neg_y": 0.03,
        "rotational_clip_neg_z": 0.03,
        "rotational_Ki": 0.0,
    }
    MAX_EPISODE_LENGTH = 120


class TrainConfig(DefaultTrainingConfig):
    image_keys = ["side_policy", "wrist_1", "wrist_2"]
    classifier_keys = ["side_classifier"]
    proprio_keys = ["tcp_pose", "tcp_vel", "tcp_force", "tcp_torque", "gripper_pose"]
    checkpoint_period = 2000
    cta_ratio = 2
    random_steps = 0
    discount = 0.98
    buffer_period = 1000
    encoder_type = "resnet-pretrained"
    setup_mode = "single-arm-learned-gripper"

    def get_environment(self, fake_env=False, save_video=False, classifier=False):
        env = USBEnv(
            fake_env=fake_env, save_video=save_video, config=EnvConfig()
        )
        if not fake_env:
            env = SpacemouseIntervention(env)
        env = RelativeFrame(env)
        env = Quat2EulerWrapper(env)
        env = SERLObsWrapper(env, proprio_keys=self.proprio_keys)
        env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)
        if classifier:
            classifier = load_classifier_func(
                key=jax.random.PRNGKey(0),
                sample=env.observation_space.sample(),
                image_keys=self.classifier_keys,
                checkpoint_path=os.path.abspath("classifier_ckpt/"),
            )

            def reward_func(obs):
                sigmoid = lambda x: 1 / (1 + jnp.exp(-x))
                return int(sigmoid(classifier(obs)) > 0.7 and obs["state"][0, 0] > 0.4)

            env = MultiCameraBinaryRewardClassifierWrapper(env, reward_func)
        env = GripperPenaltyWrapper(env, penalty=-0.02)
        return env


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

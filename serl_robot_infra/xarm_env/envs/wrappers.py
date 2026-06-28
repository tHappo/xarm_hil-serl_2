from __future__ import annotations

import time
from typing import Sequence

import gymnasium as gym
import numpy as np

from franka_env.spacemouse.spacemouse_expert import SpaceMouseExpert


class StableSpacemouseIntervention(gym.ActionWrapper):
    """
    Stable SpaceMouse intervention wrapper for xArm HIL-SERL.

    It receives policy action. If SpaceMouse has human input, it replaces the
    policy action and writes:

        info["intervene_action"] = new_action

    It is compatible with the original examples/record_success_fail.py.
    """

    def __init__(
        self,
        env,
        *,
        axis_order: Sequence[int] = (0, 1, 2, 3, 4, 5),
        axis_signs: Sequence[float] = (1, 1, 1, 1, 1, 1),
        translation_scale: float = 0.85,
        rotation_scale: float = 0.45,
        deadzone: float = 0.05,
        precision_gamma: float = 1.6,
        smoothing_tau: float = 0.045,
        max_action_rate: float = 12.0,
        intervene_threshold: float = 0.01,
        gripper_enabled: bool = True,
        left_close: bool = True,
        open_gripper_cmd: float = 1.0,
        close_gripper_cmd: float = -1.0,
        both_buttons_reset: bool = True,
    ):
        super().__init__(env)

        self.expert = SpaceMouseExpert()

        self.axis_order = np.asarray(axis_order, dtype=np.int64)
        self.axis_signs = np.asarray(axis_signs, dtype=np.float32)

        if self.axis_order.shape != (6,):
            raise ValueError("axis_order must have shape (6,)")
        if self.axis_signs.shape != (6,):
            raise ValueError("axis_signs must have shape (6,)")

        self.translation_scale = float(translation_scale)
        self.rotation_scale = float(rotation_scale)
        self.deadzone = float(deadzone)
        self.precision_gamma = float(precision_gamma)
        self.smoothing_tau = float(smoothing_tau)
        self.max_action_rate = float(max_action_rate)
        self.intervene_threshold = float(intervene_threshold)

        self.gripper_enabled = bool(gripper_enabled)
        self.left_close = bool(left_close)
        self.open_gripper_cmd = float(open_gripper_cmd)
        self.close_gripper_cmd = float(close_gripper_cmd)
        self.both_buttons_reset = bool(both_buttons_reset)

        self.left = False
        self.right = False

        self._filtered = np.zeros(6, dtype=np.float32)
        self._last_t: float | None = None

    def reset_filter(self):
        self._filtered[:] = 0.0
        self._last_t = None

    def _deadzone_precision(self, a: np.ndarray) -> np.ndarray:
        a = np.clip(np.asarray(a, dtype=np.float32), -1.0, 1.0)

        out = np.zeros_like(a)
        abs_a = np.abs(a)
        mask = abs_a > self.deadzone

        if np.any(mask):
            normalized = (abs_a[mask] - self.deadzone) / max(1e-6, 1.0 - self.deadzone)
            out[mask] = np.sign(a[mask]) * (normalized ** self.precision_gamma)

        return out

    def _smooth_rate_limit(self, a: np.ndarray) -> np.ndarray:
        now = time.monotonic()

        if self._last_t is None:
            self._last_t = now
            self._filtered[:] = a
            return self._filtered.copy()

        dt = max(1e-4, now - self._last_t)
        self._last_t = now

        if self.smoothing_tau > 0:
            alpha = dt / (self.smoothing_tau + dt)
            desired = self._filtered + alpha * (a - self._filtered)
        else:
            desired = a

        max_step = self.max_action_rate * dt
        desired = self._filtered + np.clip(
            desired - self._filtered,
            -max_step,
            max_step,
        )

        self._filtered[:] = np.clip(desired, -1.0, 1.0)

        # If raw processed input is zero, stop immediately.
        # This avoids low-pass tail causing robot drift after release.
        if np.all(a == 0.0):
            self._filtered[:] = 0.0

        return self._filtered.copy()

    def _process_motion(self, raw: np.ndarray) -> np.ndarray:
        raw = np.asarray(raw, dtype=np.float32).reshape(-1)

        if raw.shape[0] < 6:
            tmp = np.zeros(6, dtype=np.float32)
            tmp[: raw.shape[0]] = raw
            raw = tmp
        else:
            raw = raw[:6].copy()

        # Axis remap and sign flip.
        a = raw[self.axis_order] * self.axis_signs

        # Deadzone + precision curve.
        a = self._deadzone_precision(a)

        # Smooth + rate limit.
        a = self._smooth_rate_limit(a)

        # Translation and rotation scaling.
        a[:3] *= self.translation_scale
        a[3:6] *= self.rotation_scale

        return np.clip(a, -1.0, 1.0)

    def action(self, action):
        policy_action = np.asarray(action, dtype=np.float32)

        if policy_action.shape != self.action_space.shape:
            policy_action = np.zeros(self.action_space.shape, dtype=np.float32)

        expert_a, buttons = self.expert.get_action()

        buttons = list(buttons)
        while len(buttons) < 2:
            buttons.append(False)

        self.left = bool(buttons[0])
        self.right = bool(buttons[1])

        both = self.left and self.right
        reset_requested = bool(self.both_buttons_reset and both)

        motion6 = self._process_motion(np.asarray(expert_a, dtype=np.float32))

        new_action = np.zeros_like(policy_action, dtype=np.float32)
        new_action[: min(6, new_action.shape[0])] = motion6[: min(6, new_action.shape[0])]

        intervened = bool(np.linalg.norm(motion6) > self.intervene_threshold)

        if self.gripper_enabled and new_action.shape[0] >= 7:
            if both:
                new_action[6] = 0.0
                intervened = True
            elif self.left:
                new_action[6] = self.close_gripper_cmd if self.left_close else self.open_gripper_cmd
                intervened = True
            elif self.right:
                new_action[6] = self.open_gripper_cmd if self.left_close else self.close_gripper_cmd
                intervened = True
            else:
                new_action[6] = 0.0

        if reset_requested:
            new_action[:] = 0.0
            self.reset_filter()
            intervened = True

        if intervened:
            return new_action, True, reset_requested

        return policy_action, False, False

    def step(self, action):
        new_action, replaced, reset_requested = self.action(action)

        obs, rew, done, truncated, info = self.env.step(new_action)

        if replaced:
            info["intervene_action"] = new_action

        info["left"] = self.left
        info["right"] = self.right
        info["reset_requested"] = reset_requested

        # Original record_success_fail.py resets env when done or truncated.
        # Therefore, with no modification to record_success_fail.py, both buttons
        # can request HOME by setting truncated=True.
        if reset_requested:
            truncated = True

        return obs, rew, done, truncated, info
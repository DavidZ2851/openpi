"""Transforms for MolmoSpaces multi-view LeRobot datasets.

Produced by molmospaces/scripts/data/format_conversion/mlspaces_multiview_to_lerobot.py, which
stores 17-dim state/action vectors: eef_9d(9) + gripper(1) + joint_position(7). The DROID policy
wants separate joint/gripper entries and 8-dim actions (7 joints + gripper), so unpack them here.
"""

import dataclasses

import numpy as np

from openpi import transforms
from openpi.policies import droid_policy

JOINTS = slice(10, 17)
GRIPPER = slice(9, 10)


@dataclasses.dataclass(frozen=True)
class MolmoSpacesToDroid(transforms.DataTransformFn):
    """MolmoSpaces LeRobot keys -> the keys DroidInputs expects."""

    def __call__(self, data: dict) -> dict:
        state = np.asarray(data["observation.state"])
        out = {
            "observation/exterior_image_1_left": data["observation.images.exterior_1_left"],
            "observation/wrist_image_left": data["observation.images.wrist_left"],
            "observation/joint_position": state[..., JOINTS],
            "observation/gripper_position": state[..., GRIPPER],
        }
        if "action" in data:
            action = np.asarray(data["action"])
            # 7 absolute joint targets + gripper; DeltaActions later makes the joints relative,
            # matching what the pi05_droid_jointpos checkpoint predicts.
            out["actions"] = np.concatenate([action[..., JOINTS], action[..., GRIPPER]], axis=-1)
        if "prompt" in data:
            out["prompt"] = data["prompt"]
        return out


# ---------------------------------------------------------------------------
# End-effector action space (robot base frame, chunk-relative deltas).
#
# action = eef_9d (xyz + first two rotation-matrix columns, robot base frame) + gripper. For
# training, every action in a chunk is made relative to the EEF pose of the observation the chunk
# starts from: position as a difference, rotation as R_state^T @ R_action (in the same 6D form).
# The gripper stays absolute. The policy server returns these deltas; the MolmoSpaces eval client
# composes them with the EEF pose it observed (p = p_s + dp, R = R_s @ dR) and runs IK.
# ---------------------------------------------------------------------------
EEF = slice(0, 9)


def rot6d_to_matrix(r6: np.ndarray) -> np.ndarray:
    """(..., 6) first two rotation columns -> (..., 3, 3), re-orthonormalised (Gram-Schmidt)."""
    a1, a2 = r6[..., 0:3], r6[..., 3:6]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    a2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = a2 / np.linalg.norm(a2, axis=-1, keepdims=True)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1)


def matrix_to_rot6d(rot: np.ndarray) -> np.ndarray:
    return np.concatenate([rot[..., :, 0], rot[..., :, 1]], axis=-1)


def eef_delta(eef_action: np.ndarray, eef_state: np.ndarray) -> np.ndarray:
    """Absolute eef_9d actions (..., T, 9) -> deltas relative to the eef_9d state (..., 9)."""
    rot_s = rot6d_to_matrix(eef_state[..., 3:9])[..., None, :, :]
    rot_a = rot6d_to_matrix(eef_action[..., 3:9])
    d_pos = eef_action[..., 0:3] - eef_state[..., None, 0:3]
    d_rot = np.swapaxes(rot_s, -1, -2) @ rot_a
    return np.concatenate([d_pos, matrix_to_rot6d(d_rot)], axis=-1)


def eef_compose(eef_delta_9d: np.ndarray, eef_state: np.ndarray) -> np.ndarray:
    """Inverse of eef_delta: deltas (..., T, 9) + state (..., 9) -> absolute eef_9d (..., T, 9)."""
    rot_s = rot6d_to_matrix(eef_state[..., 3:9])[..., None, :, :]
    pos = eef_state[..., None, 0:3] + eef_delta_9d[..., 0:3]
    rot = rot_s @ rot6d_to_matrix(eef_delta_9d[..., 3:9])
    return np.concatenate([pos, matrix_to_rot6d(rot)], axis=-1)


@dataclasses.dataclass(frozen=True)
class MolmoSpacesToDroidEEF(transforms.DataTransformFn):
    """Like MolmoSpacesToDroid, but actions are chunk-relative EEF deltas + absolute gripper (10-dim).

    The model's state input stays the DROID one (joints + gripper), as in pi05_droid.
    """

    def __call__(self, data: dict) -> dict:
        out = MolmoSpacesToDroid()({k: v for k, v in data.items() if k != "action"})
        if "action" in data:
            state = np.asarray(data["observation.state"])
            action = np.asarray(data["action"])
            delta = eef_delta(action[..., EEF], state[..., EEF])
            out["actions"] = np.concatenate([delta, action[..., GRIPPER]], axis=-1)
        return out


@dataclasses.dataclass(frozen=True)
class DroidEEFOutputs(transforms.DataTransformFn):
    """Keep the 10 EEF action dims (9 EEF delta + gripper) of the padded model output."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][..., :10])}


# ---------------------------------------------------------------------------
# 4D RoPE inputs (PyTorch pi0 with rope_4d). The converter stores, per frame and camera, the 3D
# point (robot base frame, metres) behind each 14x14 patch of the 224x224 model image
# ("observation.patch_xyz.<camera>", (256, 3)); the eval client sends the same grids live.
# ---------------------------------------------------------------------------
EEF_XYZ = slice(0, 3)


@dataclasses.dataclass(frozen=True)
class MolmoSpacesToDroidRope4D(transforms.DataTransformFn):
    """MolmoSpacesToDroid plus the patch point grids and the gripper position."""

    def __call__(self, data: dict) -> dict:
        out = MolmoSpacesToDroid()(data)
        out["observation/patch_xyz_exterior"] = data["observation.patch_xyz.exterior_1_left"]
        out["observation/patch_xyz_wrist"] = data["observation.patch_xyz.wrist_left"]
        out["observation/eef_xyz"] = np.asarray(data["observation.state"])[..., EEF_XYZ]
        return out


@dataclasses.dataclass(frozen=True)
class DroidRope4DInputs(droid_policy.DroidInputs):
    """DroidInputs plus Observation.patch_xyz (base, left wrist, masked right wrist) and eef_xyz."""

    def __call__(self, data: dict) -> dict:
        inputs = super().__call__(data)
        exterior = np.asarray(data["observation/patch_xyz_exterior"], dtype=np.float32).reshape(-1, 3)
        wrist = np.asarray(data["observation/patch_xyz_wrist"], dtype=np.float32).reshape(-1, 3)
        inputs["patch_xyz"] = np.stack([exterior, wrist, np.zeros_like(exterior)])
        inputs["eef_xyz"] = np.asarray(data["observation/eef_xyz"], dtype=np.float32).reshape(3)
        return inputs

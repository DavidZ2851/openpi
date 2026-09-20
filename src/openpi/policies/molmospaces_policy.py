"""Transforms for MolmoSpaces multi-view LeRobot datasets.

Produced by molmospaces/scripts/data/format_conversion/mlspaces_multiview_to_lerobot.py, which
stores 17-dim state/action vectors: eef_9d(9) + gripper(1) + joint_position(7). The DROID policy
wants separate joint/gripper entries and 8-dim actions (7 joints + gripper), so unpack them here.
"""

import dataclasses

import numpy as np

from openpi import transforms

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

"""Torch forward kinematics of the DROID Franka's grasp site (TCP), for 4D RoPE action positions.

Same chain as track_residual/gripper_fk.py (exported from the MolmoSpaces franka_droid MJCF): fr3_link0 ..
fr3_link7, gripper/base, and the grasp site [0.0, 0.0, 0.155] in gripper/base. Output frame: the
MolmoSpaces robot base frame of observation.state's eef xyz and of patch_xyz (fr3_link0 + 0.58 m z);
matches the recorded eef to ~0.1 mm.
"""

import torch

# (pos, quat wxyz, joint index or -1) of each body relative to its parent, base to gripper/base.
_CHAIN = (
    ([0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], -1),  # fr3_link0
    ([0.0, 0.0, 0.333], [1.0, 0.0, 0.0, 0.0], 0),  # fr3_link1
    ([0.0, 0.0, 0.0], [0.7071067811865475, -0.7071067811865475, 0.0, 0.0], 1),  # fr3_link2
    ([0.0, -0.316, 0.0], [0.7071067811865475, 0.7071067811865475, 0.0, 0.0], 2),  # fr3_link3
    ([0.0825, 0.0, 0.0], [0.7071067811865475, 0.7071067811865475, 0.0, 0.0], 3),  # fr3_link4
    ([-0.0825, 0.384, 0.0], [0.7071067811865475, -0.7071067811865475, 0.0, 0.0], 4),  # fr3_link5
    ([0.0, 0.0, 0.0], [0.7071067811865475, 0.7071067811865475, 0.0, 0.0], 5),  # fr3_link6
    ([0.088, 0.0, 0.0], [0.7071067811865475, 0.7071067811865475, 0.0, 0.0], 6),  # fr3_link7
    ([0.0, 0.0, 0.107], [0.7071067811865476, 0.0, 0.0, 0.7071067811865476], -1),  # gripper/base
)
_GRASP_SITE = (0.0, 0.0, 0.155)
BASE_Z_OFFSET = 0.58


def _quat_to_mat(w, x, y, z):
    return torch.tensor(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=torch.float64,
    )


def tcp_position(joints: torch.Tensor) -> torch.Tensor:
    """joints (..., 7) arm angles (rad) -> grasp-site position (..., 3) in the robot base frame (metres)."""
    q = joints.to(torch.float64)
    rot = torch.eye(3, dtype=torch.float64, device=q.device).expand(*q.shape[:-1], 3, 3)
    pos = q.new_zeros(*q.shape[:-1], 3)
    for p, quat, idx in _CHAIN:
        local = _quat_to_mat(*quat).to(q.device)
        pos = pos + rot @ torch.tensor(p, dtype=torch.float64, device=q.device)
        rot = rot @ local
        if idx >= 0:
            c, s = torch.cos(q[..., idx]), torch.sin(q[..., idx])
            zero, one = torch.zeros_like(c), torch.ones_like(c)
            rz = torch.stack([c, -s, zero, s, c, zero, zero, zero, one], dim=-1).reshape(*c.shape, 3, 3)
            rot = rot @ rz
    pos = pos + rot @ torch.tensor(_GRASP_SITE, dtype=torch.float64, device=q.device)
    pos = pos + torch.tensor((0.0, 0.0, BASE_Z_OFFSET), dtype=torch.float64, device=q.device)
    return pos.to(joints.dtype)

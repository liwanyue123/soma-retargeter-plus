# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Reusable calibration of per-joint ``joint_offsets`` for a robot scaler config.

The IK target for each mapped SOMA joint is computed (see
``HumanToRobotScaler.wp_compute_scaled_effectors``) as::

    target.q = soma_global.q  *  offset.q
    target.p = scaled_root + scaled_geocentric + R(target.q) * offset.p

If we put SOMA in a known reference pose (e.g. the BVH zero frame, after
facing-direction conversion) and put the robot in a *physically equivalent*
pose, then we want ``target.q == robot_link_global.q`` and
``target.p == robot_link_global.p``. Inverting the relations above:

    offset.q = inverse(soma_global.q)  *  robot_global.q
    offset.p = inverse(robot_global.q).rotate(robot_global.p - soma_global.p)

This module exposes the math as pure functions so both the CLI tool
(``tools/calibrate_robot_offsets.py``) and the in-app calibration panel can
share it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import warp as wp


def _quat_inverse(q: wp.quat) -> wp.quat:
    return wp.quat(-q[0], -q[1], -q[2], q[3])


def _round(x: float, n: int = 6) -> float:
    return round(float(x), n)


def compute_offsets(
    soma_globals,                       # np.ndarray (num_soma_joints, 7)
    soma_joint_names: List[str],
    robot_link_globals_by_name: Dict[str, "Tuple[wp.vec3, wp.quat]"],
    ik_map: dict,
    compute_position: bool = False,
) -> Dict[str, list]:
    """Compute per-joint offsets in the format used by the scaler config.

    Args:
        soma_globals: SOMA global transforms at the reference pose
            (one row per joint, 7 floats: tx, ty, tz, qx, qy, qz, qw).
        soma_joint_names: Joint names in the same order as ``soma_globals``.
        robot_link_globals_by_name: Mapping from robot link name to its
            global ``(p, q)`` at the matching reference pose.
        ik_map: The retargeter ``ik_map`` block. Keys are SOMA joint names,
            values must contain ``t_body`` (robot link name).
        compute_position: If True, also compute ``offset.p`` from the geometric
            difference. If False, returns ``[0, 0, 0]`` for every offset.p
            (caller can merge with hand-tuned values).

    Returns:
        Dict in scaler-config format::

            { soma_joint: [[px, py, pz], [qx, qy, qz, qw]], ... }
    """
    name_to_index = {n: i for i, n in enumerate(soma_joint_names)}
    new_offsets: Dict[str, list] = {}

    for soma_joint, mapping in ik_map.items():
        link_name = mapping["t_body"]
        if soma_joint not in name_to_index:
            print(f"[WARN]: SOMA joint [{soma_joint}] not in skeleton. Skipped.")
            continue
        if link_name not in robot_link_globals_by_name:
            print(f"[WARN]: Robot link [{link_name}] not in MJCF. Skipped.")
            continue

        s_idx = name_to_index[soma_joint]
        soma_p = wp.vec3(*soma_globals[s_idx][0:3])
        soma_q = wp.quat(*soma_globals[s_idx][3:7])

        robot_p, robot_q = robot_link_globals_by_name[link_name]

        off_q = wp.mul(_quat_inverse(soma_q), robot_q)

        if compute_position:
            delta_p = wp.vec3(
                robot_p[0] - soma_p[0],
                robot_p[1] - soma_p[1],
                robot_p[2] - soma_p[2])
            off_p = wp.quat_rotate(_quat_inverse(robot_q), delta_p)
            off_p_list = [_round(off_p[0]), _round(off_p[1]), _round(off_p[2])]
        else:
            off_p_list = [0.0, 0.0, 0.0]

        new_offsets[soma_joint] = [
            off_p_list,
            [_round(off_q[0]), _round(off_q[1]), _round(off_q[2]), _round(off_q[3])],
        ]

    return new_offsets


def merge_offsets_into_config(
    scaler_cfg: dict,
    new_offsets: Dict[str, list],
    keep_existing_position: bool = True,
) -> dict:
    """Return ``scaler_cfg`` with ``joint_offsets`` updated.

    The merge is non-destructive: any existing entry that ``new_offsets`` does
    not cover (e.g. ``LeftToe`` / ``RightToe`` aliased by the scaler) is
    preserved as-is.

    Args:
        scaler_cfg: The full scaler config dict (will be mutated and returned).
        new_offsets: New offsets computed by :func:`compute_offsets`.
        keep_existing_position: If True, preserve any existing ``offset.p``
            value for each joint and only overwrite ``offset.q``.

    Returns:
        The same dict as ``scaler_cfg``, with ``joint_offsets`` updated.
    """
    existing = dict(scaler_cfg.get("joint_offsets", {}))

    for joint, vals in new_offsets.items():
        merged_pos = vals[0]
        merged_quat = vals[1]
        if keep_existing_position and joint in existing:
            merged_pos = existing[joint][0]
        existing[joint] = [merged_pos, merged_quat]

    scaler_cfg["joint_offsets"] = existing
    return scaler_cfg


def write_scaler_config(scaler_cfg: dict, path: Path) -> None:
    """Write the scaler config to disk with stable indentation."""
    Path(path).write_text(json.dumps(scaler_cfg, indent=4))


def collect_robot_link_globals(builder, body_q_array) -> Dict[str, "Tuple[wp.vec3, wp.quat]"]:
    """Build a ``{link_name: (p, q)}`` dict from a Newton model state.

    Args:
        builder: Newton ``ModelBuilder`` (used for ``body_label``).
        body_q_array: Numpy array of body transforms, shape (num_bodies, 7).
    """
    import soma_retargeter.utils.newton_utils as newton_utils

    out: Dict[str, "Tuple[wp.vec3, wp.quat]"] = {}
    for i, label in enumerate(builder.body_label):
        name = newton_utils.get_name_from_label(label)
        out[name] = (
            wp.vec3(*body_q_array[i][0:3]),
            wp.quat(*body_q_array[i][3:7]),
        )
    return out

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Reusable calibration of ``joint_offsets`` and ``joint_scales`` for a robot
scaler config.

The IK target for each mapped SOMA joint is computed (see
``HumanToRobotScaler.wp_compute_scaled_effectors``) as::

    target.q = soma_global.q  *  offset.q
    target.p = scaled_root + scaled_geocentric + R(target.q) * offset.p

where ``scaled_root = soma_hip * s_hip``, ``scaled_geocentric = (soma_link -
soma_hip) * s_link`` and ``s_X = config_scale[X] * (model_height /
human_height_assumption)`` is the *effective* per-joint scale.

If we put SOMA in a known reference pose and the robot in a *physically
equivalent* pose, three quantities can be inverted:

  * ``offset.q``   = inverse(soma.q) * robot.q
                     (always meaningful - corrects coordinate-frame mismatch)

  * ``s_link``     = |robot.p - scaled_root|  /  |soma_link - soma_hip|
                     (per-joint magnitude scale; for the hip itself we use
                      |robot_hip| / |soma_hip|)

  * ``offset.p``   = inverse(robot.q).rotate(
                       robot.p - scaled_root - (soma_link - soma_hip) * s_link)
                     (residual cm-level adjustment after scaling - if the
                      scales are calibrated, this should be small)

This module exposes the math as pure functions so the CLI tool and the
in-app calibration panel can share it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import warp as wp


def _quat_inverse(q: wp.quat) -> wp.quat:
    return wp.quat(-q[0], -q[1], -q[2], q[3])


def _round(x: float, n: int = 6) -> float:
    return round(float(x), n)


def _hip_soma_joint_name(ik_map: dict) -> str:
    """Return the SOMA joint that the ik_map treats as the root.

    Convention: the first key of ``ik_map`` is the root joint (``Hips``), per
    ``HumanToRobotScaler``.  We do *not* hardcode ``"Hips"`` here so the
    helpers stay agnostic to the source skeleton's naming.
    """
    return next(iter(ik_map.keys()))


def compute_scales(
    soma_globals,                       # np.ndarray (num_soma_joints, 7)
    soma_joint_names: List[str],
    robot_link_globals_by_name: Dict[str, "Tuple[wp.vec3, wp.quat]"],
    ik_map: dict,
    height_ratio: float,
    mode: str = 'geocentric',
    joint_parents: Dict[str, str] | None = None,
) -> Dict[str, float]:
    """Compute per-joint ``joint_scales`` from matching reference poses.

    For the root joint (first ik_map entry) the scale is::

        s_root = |robot_root.p|  /  |soma_root.p|        (Z-component-only would
                                                          be equivalent here)

    For all other joints the scale matches the bone vector magnitudes::

        s_link = |robot_link.p - robot_root.p|  /  |soma_link.p - soma_root.p|

    The returned values are *config* scales (``= effective / height_ratio``)
    so they can be dropped straight into ``soma_to_<robot>_scaler_config.json``
    without further conversion.

    Args:
        soma_globals: SOMA global transforms at the reference pose.
        soma_joint_names: Joint names in the same order as ``soma_globals``.
        robot_link_globals_by_name: ``{link_name: (p, q)}`` at the matching
            robot reference pose.
        ik_map: Retargeter ``ik_map`` (first key is the root joint).
        height_ratio: ``model_height / human_height_assumption`` (the factor
            that ``HumanToRobotScaler.__init__`` multiplies into every scale
            at runtime).

    Returns:
        ``{soma_joint: config_scale_float}``. Joints whose source vector is
        too short to scale (e.g. duplicates of the root) are skipped.

    Segmental mode (``mode='segmental'``, requires ``joint_parents``): non-root
    scales are per-SEGMENT length ratios instead of root-geocentric ratios::

        s_link = |robot_link.p - robot_parent_link.p| / |soma_link.p - soma_parent.p|

    matching ``HumanToRobotScaler``'s segmental effector accumulation
    (child = parent_effector + segment * scale).
    """
    if height_ratio <= 0:
        raise ValueError(f"height_ratio must be > 0 (got {height_ratio})")
    if mode not in ('geocentric', 'segmental'):
        raise ValueError(f"Unknown calibration mode [{mode}]")
    if mode == 'segmental' and not joint_parents:
        raise ValueError("segmental mode requires joint_parents")

    name_to_index = {n: i for i, n in enumerate(soma_joint_names)}
    root_joint = _hip_soma_joint_name(ik_map)
    root_link = ik_map[root_joint]["t_body"]

    if root_joint not in name_to_index:
        raise KeyError(f"Root joint [{root_joint}] not in SOMA skeleton")
    if root_link not in robot_link_globals_by_name:
        raise KeyError(f"Root link [{root_link}] not in robot MJCF")

    soma_root_p = np.asarray(soma_globals[name_to_index[root_joint]][0:3], dtype=np.float64)
    rp, _ = robot_link_globals_by_name[root_link]
    robot_root_p = np.array([rp[0], rp[1], rp[2]], dtype=np.float64)

    out: Dict[str, float] = {}

    soma_root_norm = float(np.linalg.norm(soma_root_p))
    if soma_root_norm > 1e-4:
        eff_root = float(np.linalg.norm(robot_root_p)) / soma_root_norm
        out[root_joint] = _round(eff_root / height_ratio, 4)

    for soma_joint, mapping in ik_map.items():
        if soma_joint == root_joint:
            continue
        link_name = mapping["t_body"]
        if soma_joint not in name_to_index or link_name not in robot_link_globals_by_name:
            continue

        if mode == 'segmental':
            parent_joint = joint_parents.get(soma_joint, "")
            parent_link = ik_map.get(parent_joint, {}).get("t_body")
            if (not parent_joint or parent_joint not in name_to_index
                    or parent_link not in robot_link_globals_by_name):
                continue
            ref_soma_p = np.asarray(soma_globals[name_to_index[parent_joint]][0:3], dtype=np.float64)
            rp_parent, _ = robot_link_globals_by_name[parent_link]
            ref_robot_p = np.array([rp_parent[0], rp_parent[1], rp_parent[2]], dtype=np.float64)
        else:
            ref_soma_p = soma_root_p
            ref_robot_p = robot_root_p

        soma_v = np.asarray(soma_globals[name_to_index[soma_joint]][0:3], dtype=np.float64) - ref_soma_p
        rp_link, _ = robot_link_globals_by_name[link_name]
        robot_v = np.array([rp_link[0], rp_link[1], rp_link[2]], dtype=np.float64) - ref_robot_p

        s_norm = float(np.linalg.norm(soma_v))
        if s_norm < 1e-4:
            continue
        eff = float(np.linalg.norm(robot_v)) / s_norm
        out[soma_joint] = _round(eff / height_ratio, 4)

    return out


def compute_offsets(
    soma_globals,                       # np.ndarray (num_soma_joints, 7)
    soma_joint_names: List[str],
    robot_link_globals_by_name: Dict[str, "Tuple[wp.vec3, wp.quat]"],
    ik_map: dict,
    compute_position: bool = False,
    joint_scales: Dict[str, float] = None,
    height_ratio: float = 1.0,
    mode: str = 'geocentric',
    joint_parents: Dict[str, str] | None = None,
) -> Dict[str, list]:
    """Compute per-joint ``joint_offsets`` for the scaler config.

    Args:
        soma_globals: SOMA global transforms at the reference pose
            (one row per joint, 7 floats: tx, ty, tz, qx, qy, qz, qw).
        soma_joint_names: Joint names in the same order as ``soma_globals``.
        robot_link_globals_by_name: ``{link_name: (p, q)}`` at the matching
            robot reference pose.
        ik_map: Retargeter ``ik_map`` (first key is the root joint).
        compute_position: If True, also solve for ``offset.p`` as the
            *residual* between the robot link and the scaled-SOMA target.
            If False, returns ``[0, 0, 0]`` for every ``offset.p`` (caller
            can merge with hand-tuned values).
        joint_scales: Optional ``{joint: config_scale}`` used only when
            ``compute_position`` is True. If omitted, scales of 1 are
            assumed (which reproduces the legacy - and broken - behaviour
            where the unscaled SOMA-vs-robot delta is written into
            ``offset.p``).
        height_ratio: ``model_height / human_height_assumption``. Combined
            with ``joint_scales`` to get the effective scale used in the
            runtime IK target formula.

    Returns:
        ``{soma_joint: [[px, py, pz], [qx, qy, qz, qw]]}``.
    """
    if mode not in ('geocentric', 'segmental'):
        raise ValueError(f"Unknown calibration mode [{mode}]")
    if mode == 'segmental' and not joint_parents:
        raise ValueError("segmental mode requires joint_parents")

    name_to_index = {n: i for i, n in enumerate(soma_joint_names)}
    root_joint = _hip_soma_joint_name(ik_map)
    soma_root_p = wp.vec3(*soma_globals[name_to_index[root_joint]][0:3])

    eff_scale = {}
    if compute_position and joint_scales:
        eff_scale = {k: float(v) * float(height_ratio) for k, v in joint_scales.items()}
    s_root = eff_scale.get(root_joint, 1.0)

    # Segmental mode: mapped base positions accumulate along the joint_parents
    # chain (child = parent_base + segment * scale), mirroring the runtime
    # kernel's pass-1 accumulation (offsets themselves are not chained).
    mapped_base: Dict[str, np.ndarray] = {}

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
            s_link = eff_scale.get(soma_joint, 1.0)
            np_soma_p = np.array([soma_p[0], soma_p[1], soma_p[2]], dtype=np.float64)
            np_root_p = np.array([soma_root_p[0], soma_root_p[1], soma_root_p[2]], dtype=np.float64)

            parent_joint = joint_parents.get(soma_joint, "") if mode == 'segmental' else ""
            if soma_joint == root_joint or (mode == 'segmental' and not parent_joint):
                base = np_root_p * s_root
            elif mode == 'segmental':
                if parent_joint not in mapped_base or parent_joint not in name_to_index:
                    print(f"[WARN]: segmental offsets: parent [{parent_joint}] of "
                          f"[{soma_joint}] not mapped yet. Skipped.")
                    continue
                np_parent_p = np.asarray(
                    soma_globals[name_to_index[parent_joint]][0:3], dtype=np.float64)
                base = mapped_base[parent_joint] + (np_soma_p - np_parent_p) * s_link
            else:
                base = np_root_p * s_root + (np_soma_p - np_root_p) * s_link
            mapped_base[soma_joint] = base

            residual = wp.vec3(
                robot_p[0] - base[0],
                robot_p[1] - base[1],
                robot_p[2] - base[2])
            off_p = wp.quat_rotate(_quat_inverse(robot_q), residual)
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
        scaler_cfg: The full scaler config dict (mutated and returned).
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


def merge_scales_into_config(
    scaler_cfg: dict,
    new_scales: Dict[str, float],
) -> dict:
    """Update ``joint_scales`` in-place, preserving any joint not in ``new_scales``.

    Mirrored joints (e.g. ``LeftToe`` / ``LeftToeBase`` which are aliased by
    ``HumanToRobotScaler``) are left as-is unless explicitly listed.
    """
    existing = dict(scaler_cfg.get("joint_scales", {}))
    existing.update({k: float(v) for k, v in new_scales.items()})
    scaler_cfg["joint_scales"] = existing
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

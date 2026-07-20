# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load CIO history-run folders into playback-ready numpy arrays.

File layout expected inside a run folder (e.g.
``.../history/t800_history/t800 dragon punch/``)::

    pipeline_paths.jsonc
    <robot>_result_pinocchio.txt          # preferred (rpy)
    <robot>_result_pinocchio_quat.txt     # optional
    <robot>_reference_clip.txt            # origin ghost (TO column layout)

Column layout matches ``VizConfigParser.get_to_col_config`` in CIO::

    rpy  : [pos(3) | yaw,pitch,roll(3) | joints(nj) | u(nj) | forces(3*nc)]
    quat : [pos(3) | qx,qy,qz,qw(4)    | joints(nj) | u(nj) | forces(3*nc)]

Base RPY in the file is stored as (yaw, pitch, roll); we convert to xyzw quat
via ``Rotation.from_euler('xyz', [roll, pitch, yaw])`` — same as
``UniversalDataLoader``.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

from soma_retargeter.to_playback.jsonc import load_jsonc
from soma_retargeter.to_playback.robots import (
    ToRobotSpec,
    get_to_robot_spec,
    infer_robot_key_from_folder,
)


@dataclass
class ToPlaybackData:
    """One loaded history run ready for dual-robot Newton playback."""

    run_dir: str
    robot_key: str
    soma_robot_type: str
    joint_names: List[str]
    contact_frame_names: List[str]

    origin_q_quat: np.ndarray  # (To, 7+nj)  pos + xyzw + joints(rad)
    origin_dt: float

    to_q_quat: np.ndarray      # (Tt, 7+nj)
    to_forces: np.ndarray      # (Tt, nc, 3) world-frame
    to_dt: float

    to_type: str = "rpy"
    status: str = ""

    @property
    def num_joints(self) -> int:
        return len(self.joint_names)

    @property
    def num_contacts(self) -> int:
        return int(self.to_forces.shape[1]) if self.to_forces.ndim == 3 else 0

    @property
    def origin_duration(self) -> float:
        return float(max(0, self.origin_q_quat.shape[0] - 1) * self.origin_dt)

    @property
    def to_duration(self) -> float:
        return float(max(0, self.to_q_quat.shape[0] - 1) * self.to_dt)

    @property
    def duration(self) -> float:
        return max(self.origin_duration, self.to_duration)

    def sample_origin(self, time_s: float) -> np.ndarray:
        return _sample_traj(self.origin_q_quat, self.origin_dt, time_s)

    def sample_to(self, time_s: float) -> np.ndarray:
        return _sample_traj(self.to_q_quat, self.to_dt, time_s)

    def sample_forces(self, time_s: float) -> np.ndarray:
        return _sample_traj(self.to_forces, self.to_dt, time_s)


def _sample_traj(data: np.ndarray, dt: float, time_s: float) -> np.ndarray:
    """Linear sample along axis 0 by time (clamped)."""
    n = int(data.shape[0])
    if n <= 0:
        raise ValueError("empty trajectory")
    if n == 1 or dt <= 0.0:
        return data[0].copy()
    t = float(np.clip(time_s, 0.0, (n - 1) * dt))
    idx = t / dt
    i0 = int(np.floor(idx))
    i1 = min(i0 + 1, n - 1)
    blend = float(idx - i0)
    if blend < 1e-5 or i0 == i1:
        return data[i0].copy()
    return (1.0 - blend) * data[i0] + blend * data[i1]


def _flatten_origin_node(pipe_origin) -> dict:
    if not isinstance(pipe_origin, dict):
        return {}
    od = pipe_origin.get("origin_data")
    if isinstance(od, dict):
        return dict(od)
    return dict(pipe_origin)


def _resolve_to_entry(pipe_to, prefer_type: Optional[str] = "rpy") -> dict:
    want = str(prefer_type).lower() if prefer_type else None
    if isinstance(pipe_to, dict):
        entries = [pipe_to]
    elif isinstance(pipe_to, list):
        entries = [e for e in pipe_to if isinstance(e, dict)]
    else:
        raise TypeError("pipeline.to must be object or list")
    if not entries:
        raise ValueError("pipeline.to is empty")
    if want:
        for e in entries:
            if str(e.get("data_type", "")).lower() == want:
                return e
    for e in entries:
        if str(e.get("data_type", "")).lower() == "rpy":
            return e
    return entries[0]


def _find_file(run_dir: Path, patterns: Sequence[str]) -> Optional[Path]:
    for pat in patterns:
        hits = sorted(run_dir.glob(pat))
        if hits:
            return hits[0]
    return None


def _load_numeric(path: Path) -> np.ndarray:
    try:
        raw = np.loadtxt(path, dtype=float, delimiter=None)
    except ValueError:
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.split("#")[0].strip()
                if not line:
                    continue
                line = line.replace(",", " ")
                try:
                    rows.append([float(x) for x in line.split()])
                except ValueError:
                    continue
        raw = np.asarray(rows, dtype=float)
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)
    return raw


def _raw_to_q_quat_and_forces(
    raw: np.ndarray,
    *,
    num_joints: int,
    num_contacts: int,
    rot_type: str,
    include_forces: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """Parse save_results layout into q_quat (T,7+nj) and forces (T,nc,3)."""
    nj = int(num_joints)
    nc = int(num_contacts)
    rot_type = rot_type.lower()
    if rot_type == "quat":
        q_dim = 7 + nj
        pos = raw[:, 0:3]
        quat = raw[:, 3:7]
        joints = raw[:, 7:q_dim]
    elif rot_type == "rpy":
        q_dim = 6 + nj
        pos = raw[:, 0:3]
        # file stores (yaw, pitch, roll) at cols 3,4,5
        yaw, pitch, roll = raw[:, 3], raw[:, 4], raw[:, 5]
        quat = Rotation.from_euler(
            "xyz", np.column_stack([roll, pitch, yaw])
        ).as_quat()  # xyzw
        joints = raw[:, 6:q_dim]
    else:
        raise ValueError(f"Unknown rot_type '{rot_type}'")

    q_quat = np.zeros((raw.shape[0], 7 + nj), dtype=np.float64)
    q_quat[:, 0:3] = pos
    q_quat[:, 3:7] = quat
    q_quat[:, 7:] = joints

    forces = np.zeros((raw.shape[0], nc, 3), dtype=np.float64)
    if include_forces and nc > 0:
        force_start = q_dim + nj  # skip u(nj)
        force_end = force_start + nc * 3
        if raw.shape[1] >= force_end:
            forces = raw[:, force_start:force_end].reshape(-1, nc, 3)
        elif raw.shape[1] > force_start:
            # Partial force block — pad
            flat = np.zeros((raw.shape[0], nc * 3), dtype=np.float64)
            n = min(nc * 3, raw.shape[1] - force_start)
            flat[:, :n] = raw[:, force_start:force_start + n]
            forces = flat.reshape(-1, nc, 3)

    return q_quat, forces


def load_history_run(
    run_dir: str | Path,
    robot_key: Optional[str] = None,
    prefer_to_type: str = "rpy",
) -> ToPlaybackData:
    """Load a CIO history run folder for TO playback.

    Args:
        run_dir: Path to one run directory containing ``pipeline_paths.jsonc``
            and result/reference txt files.
        robot_key: CIO robot key (``t800``, …). Inferred from folder name / files
            when omitted.
        prefer_to_type: ``rpy`` (default) or ``quat``.
    """
    run_path = Path(run_dir).expanduser().resolve()
    if not run_path.is_dir():
        raise FileNotFoundError(f"TO run folder not found: {run_path}")

    pipe_path = run_path / "pipeline_paths.jsonc"
    if not pipe_path.is_file():
        raise FileNotFoundError(
            f"Missing pipeline_paths.jsonc in {run_path}. "
            "Select a history run folder (the leaf that contains result txts)."
        )
    pipe_cfg = load_jsonc(pipe_path)

    joint_names = list(pipe_cfg.get("joint_names") or [])
    if not joint_names:
        raise ValueError(f"pipeline_paths.jsonc has no joint_names: {pipe_path}")

    if robot_key is None:
        robot_key = infer_robot_key_from_folder(run_path.name)
        if robot_key is None:
            robot_key = infer_robot_key_from_folder(run_path.parent.name)
    if robot_key is None:
        # Fall back to file prefix heuristics
        for cand in ("t800", "g1", "pm01", "pi_plus_s", "pi_plus", "pnd"):
            if list(run_path.glob(f"{cand}_result_pinocchio*.txt")):
                robot_key = cand
                break
    if robot_key is None:
        raise ValueError(
            f"Could not infer robot key for {run_path}. Pass robot_key explicitly."
        )

    spec: ToRobotSpec = get_to_robot_spec(robot_key)
    if not spec.enabled:
        raise ValueError(
            f"TO playback for robot '{robot_key}' is not enabled yet "
            f"(soma type would be {spec.soma_robot_type})."
        )

    prefix = spec.resolved_prefix()
    contact_names = [c.name for c in spec.contact_sites]
    nc = len(contact_names)
    nj = len(joint_names)

    origin_node = _flatten_origin_node(pipe_cfg.get("pipeline", {}).get("origin", {}))
    origin_dt = float(origin_node.get("origin_dt", 0.01))
    output_dt = float(origin_node.get("output_dt", origin_dt))

    to_entry = _resolve_to_entry(
        pipe_cfg.get("pipeline", {}).get("to", {}), prefer_type=prefer_to_type
    )
    to_type = str(to_entry.get("data_type", prefer_to_type)).lower()

    # Prefer files that live in the run folder (history archives copy them in).
    if to_type == "quat":
        to_path = _find_file(
            run_path,
            [
                f"{prefix}_result_pinocchio_quat.txt",
                "*_result_pinocchio_quat.txt",
            ],
        )
    else:
        to_path = _find_file(
            run_path,
            [
                f"{prefix}_result_pinocchio.txt",
                "*_result_pinocchio.txt",
            ],
        )
        # If only quat exists, fall back.
        if to_path is None:
            to_path = _find_file(
                run_path,
                [
                    f"{prefix}_result_pinocchio_quat.txt",
                    "*_result_pinocchio_quat.txt",
                ],
            )
            if to_path is not None:
                to_type = "quat"

    if to_path is None:
        raise FileNotFoundError(
            f"No TO result txt found in {run_path} "
            f"(looked for {prefix}_result_pinocchio*.txt)"
        )

    origin_path = _find_file(
        run_path,
        [
            f"{prefix}_reference_clip.txt",
            "*_reference_clip.txt",
        ],
    )
    if origin_path is None:
        raise FileNotFoundError(
            f"No reference_clip txt found in {run_path} "
            f"(looked for {prefix}_reference_clip.txt)"
        )

    to_raw = _load_numeric(to_path)
    origin_raw = _load_numeric(origin_path)

    # Reference clip is always save_results rpy layout (CIO docs).
    origin_q, _ = _raw_to_q_quat_and_forces(
        origin_raw, num_joints=nj, num_contacts=nc, rot_type="rpy", include_forces=False
    )
    to_q, to_forces = _raw_to_q_quat_and_forces(
        to_raw, num_joints=nj, num_contacts=nc, rot_type=to_type, include_forces=True
    )

    status = (
        f"Loaded {run_path.name}: origin {origin_q.shape[0]}f @ {origin_dt:.4f}s, "
        f"to {to_q.shape[0]}f @ {output_dt:.4f}s ({to_type}), "
        f"robot={robot_key}"
    )

    return ToPlaybackData(
        run_dir=str(run_path),
        robot_key=robot_key,
        soma_robot_type=spec.soma_robot_type,
        joint_names=joint_names,
        contact_frame_names=contact_names,
        origin_q_quat=origin_q.astype(np.float32),
        origin_dt=origin_dt,
        to_q_quat=to_q.astype(np.float32),
        to_forces=to_forces.astype(np.float32),
        to_dt=output_dt,
        to_type=to_type,
        status=status,
    )

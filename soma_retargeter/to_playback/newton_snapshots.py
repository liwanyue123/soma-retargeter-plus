# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Newton-iteration trajectory snapshots for TO convergence visualization.

Expects a folder of paired files::

    newton_iter_<K>.txt      # CIO save_results layout (rpy)
    newton_iter_<K>_dt.txt   # [index, dt] or bare dt per control segment

Load via ``load_newton_snapshots``. Align modes:

- ``absolute``: sample all iters on the same clock ``t`` (seconds). Iters whose
  duration is shorter than ``t`` are omitted.
- ``normalized``: sample at progress ``τ = t / T_ref`` on each iter
  (``T_ref = max duration``), so start/end line up across iters.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from soma_retargeter.to_playback.jsonc import load_jsonc
from soma_retargeter.to_playback.loader import _load_numeric, _raw_to_q_quat_and_forces
from soma_retargeter.to_playback.robots import (
    ToRobotSpec,
    get_to_robot_spec,
    infer_robot_key_from_folder,
)

_ITER_RE = re.compile(r"^newton_iter_(\d+)\.txt$")


@dataclass
class NewtonIterTraj:
    """One Newton-iteration trajectory with variable segment dt."""

    iter_id: int
    dts: np.ndarray          # (S,) seconds
    times: np.ndarray        # (T,) cumulative time, times[0]=0
    q_quat: np.ndarray       # (T, 7+nj) pos + xyzw + joints
    duration: float = 0.0

    def __post_init__(self):
        self.duration = float(self.times[-1]) if len(self.times) else 0.0


@dataclass
class NewtonSnapshotsData:
    """All loaded Newton iteration snapshots for one run."""

    folder: str
    robot_key: str
    soma_robot_type: str
    joint_names: List[str]
    iters: List[NewtonIterTraj] = field(default_factory=list)
    status: str = ""

    @property
    def iter_ids(self) -> List[int]:
        return [it.iter_id for it in self.iters]

    @property
    def max_duration(self) -> float:
        if not self.iters:
            return 0.0
        return float(max(it.duration for it in self.iters))

    @property
    def min_duration(self) -> float:
        if not self.iters:
            return 0.0
        return float(min(it.duration for it in self.iters))

    def timeline_duration(self, align: str = "absolute") -> float:
        """Scrubber length in seconds (both modes use max duration)."""
        _ = align
        return self.max_duration

    def iter_index(self, iter_id: int) -> int:
        for i, it in enumerate(self.iters):
            if it.iter_id == iter_id:
                return i
        raise KeyError(f"iter {iter_id} not loaded")

    def color_for_iter(self, iter_id: int) -> Tuple[float, float, float]:
        ids = self.iter_ids
        if not ids:
            return (0.7, 0.7, 0.7)
        lo, hi = ids[0], ids[-1]
        u = 0.0 if hi <= lo else (iter_id - lo) / float(hi - lo)
        # blue (early) → cyan → yellow → orange (late)
        r = float(np.clip(1.4 * u - 0.15, 0.0, 1.0))
        g = float(np.clip(1.0 - abs(u - 0.45) * 1.8, 0.15, 0.95))
        b = float(np.clip(1.15 * (1.0 - u), 0.05, 1.0))
        return (r, g, b)

    def sample_q(
        self,
        iter_id: int,
        time_s: float,
        *,
        align: str = "absolute",
        t_ref: Optional[float] = None,
    ) -> Optional[np.ndarray]:
        """Sample ``q_quat`` at scrub time. ``None`` if absolute and past end."""
        it = self.iters[self.iter_index(iter_id)]
        if it.duration <= 1e-12 or it.q_quat.shape[0] == 0:
            return None
        t_ref = float(t_ref if t_ref is not None else self.max_duration)
        if align == "normalized":
            tau = 0.0 if t_ref <= 1e-12 else float(np.clip(time_s / t_ref, 0.0, 1.0))
            t_query = tau * it.duration
        else:
            if time_s > it.duration + 1e-6:
                return None
            t_query = float(np.clip(time_s, 0.0, it.duration))
        return _interp_along_time(it.q_quat, it.times, t_query)

    def base_trail(
        self,
        iter_id: int,
        *,
        resample_dt: float = 0.1,
        align: str = "absolute",
        t_ref: Optional[float] = None,
        t_max: Optional[float] = None,
    ) -> np.ndarray:
        """Resampled base xyz along the scrub timeline. Shape ``(N, 3)``."""
        it = self.iters[self.iter_index(iter_id)]
        t_ref = float(t_ref if t_ref is not None else self.max_duration)
        if t_max is None:
            t_max = t_ref
        t_max = float(max(0.0, t_max))
        dt = max(1e-4, float(resample_dt))
        if t_max < 1e-12:
            q0 = it.q_quat[0]
            return np.asarray(q0[:3], dtype=np.float32).reshape(1, 3)

        times = np.arange(0.0, t_max + 0.5 * dt, dt, dtype=np.float64)
        pts = []
        for t in times:
            q = self.sample_q(iter_id, float(t), align=align, t_ref=t_ref)
            if q is None:
                break
            pts.append(q[:3])
        if not pts:
            return np.zeros((0, 3), dtype=np.float32)
        return np.asarray(pts, dtype=np.float32)


def _build_times(dts: np.ndarray, num_frames: int) -> np.ndarray:
    """Cumulative time vector length ``num_frames`` from segment dts."""
    dts = np.asarray(dts, dtype=np.float64).reshape(-1)
    if dts.size == 0:
        return np.linspace(0.0, max(0, num_frames - 1) * 0.01, num_frames)
    times = np.concatenate([[0.0], np.cumsum(dts)])
    if times.shape[0] == num_frames:
        return times
    if times.shape[0] > num_frames:
        return times[:num_frames]
    # Fewer intervals than poses: pad with last dt.
    last = float(dts[-1]) if dts.size else 0.01
    extra = num_frames - times.shape[0]
    pad = times[-1] + last * np.arange(1, extra + 1, dtype=np.float64)
    return np.concatenate([times, pad])


def _load_dts(path: Path) -> np.ndarray:
    raw = _load_numeric(path)
    if raw.ndim == 1:
        return raw.astype(np.float64)
    if raw.shape[1] >= 2:
        return raw[:, 1].astype(np.float64)
    return raw[:, 0].astype(np.float64)


def _interp_along_time(
    data: np.ndarray, times: np.ndarray, t_query: float
) -> np.ndarray:
    times = np.asarray(times, dtype=np.float64)
    n = int(data.shape[0])
    if n <= 0:
        raise ValueError("empty trajectory")
    if n == 1:
        return data[0].copy()
    t = float(np.clip(t_query, times[0], times[-1]))
    i1 = int(np.searchsorted(times, t, side="right"))
    i0 = max(0, i1 - 1)
    i1 = min(i1, n - 1)
    if i0 == i1 or abs(times[i1] - times[i0]) < 1e-12:
        return data[i0].copy()
    blend = (t - times[i0]) / (times[i1] - times[i0])
    return (1.0 - blend) * data[i0] + blend * data[i1]


def _discover_pipeline(folder: Path) -> Optional[Path]:
    """Find ``pipeline_paths.jsonc`` in folder or nearby parents / siblings."""
    for cand in (folder / "pipeline_paths.jsonc", folder.parent / "pipeline_paths.jsonc"):
        if cand.is_file():
            return cand
    for parent in folder.parents:
        cand = parent / "pipeline_paths.jsonc"
        if cand.is_file():
            return cand
        # stop before climbing too far
        if parent.name in ("result", "data", "resource"):
            break
    return None


def _resolve_robot_and_meta(
    folder: Path,
    robot_key: Optional[str],
) -> Tuple[str, ToRobotSpec, List[str], int]:
    pipe_path = _discover_pipeline(folder)
    joint_names: List[str] = []
    if pipe_path is not None:
        try:
            pipe_cfg = load_jsonc(pipe_path)
            joint_names = list(pipe_cfg.get("joint_names") or [])
        except Exception:
            joint_names = []

    if robot_key is None:
        robot_key = infer_robot_key_from_folder(folder.name)
        if robot_key is None:
            robot_key = infer_robot_key_from_folder(folder.parent.name)
        if robot_key is None and pipe_path is not None:
            robot_key = infer_robot_key_from_folder(pipe_path.parent.name)

    if robot_key is None:
        raise ValueError(
            f"Could not infer robot key for {folder}. Pass robot_key explicitly."
        )

    spec = get_to_robot_spec(robot_key)
    if not spec.enabled:
        raise ValueError(f"Robot '{robot_key}' is not enabled for TO playback.")

    nc = len(spec.contact_sites)
    if not joint_names:
        # Infer nj from first traj width: 6 + 2*nj + 3*nc (rpy layout).
        sample = next(folder.glob("newton_iter_*.txt"), None)
        if sample is None or "_dt" in sample.name:
            sample = None
            for p in sorted(folder.glob("newton_iter_*.txt")):
                if "_dt" not in p.name:
                    sample = p
                    break
        if sample is None:
            raise FileNotFoundError(f"No newton_iter_*.txt in {folder}")
        raw = _load_numeric(sample)
        # 6 + 2 nj + 3 nc = cols
        cols = int(raw.shape[1])
        rem = cols - 6 - 3 * nc
        if rem < 0 or rem % 2 != 0:
            raise ValueError(
                f"Cannot infer joint count from {sample.name} "
                f"(cols={cols}, nc={nc}). Provide pipeline_paths.jsonc."
            )
        nj = rem // 2
        joint_names = [f"J{i:02d}" for i in range(nj)]

    return robot_key, spec, joint_names, nc


def list_newton_iter_files(folder: str | Path) -> List[Tuple[int, Path, Path]]:
    """Return ``(iter_id, traj_path, dt_path)`` sorted by iter id."""
    root = Path(folder).expanduser().resolve()
    out: List[Tuple[int, Path, Path]] = []
    for path in sorted(root.glob("newton_iter_*.txt")):
        m = _ITER_RE.match(path.name)
        if not m:
            continue
        iter_id = int(m.group(1))
        dt_path = root / f"newton_iter_{iter_id}_dt.txt"
        if not dt_path.is_file():
            continue
        out.append((iter_id, path, dt_path))
    out.sort(key=lambda x: x[0])
    return out


def load_newton_snapshots(
    folder: str | Path,
    robot_key: Optional[str] = None,
    rot_type: str = "rpy",
) -> NewtonSnapshotsData:
    """Load all ``newton_iter_K`` + ``_dt`` pairs from a snapshots folder."""
    root = Path(folder).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Newton snapshots folder not found: {root}")

    pairs = list_newton_iter_files(root)
    if not pairs:
        raise FileNotFoundError(
            f"No newton_iter_<K>.txt + newton_iter_<K>_dt.txt pairs in {root}"
        )

    robot_key, spec, joint_names, nc = _resolve_robot_and_meta(root, robot_key)
    nj = len(joint_names)

    iters: List[NewtonIterTraj] = []
    for iter_id, traj_path, dt_path in pairs:
        raw = _load_numeric(traj_path)
        dts = _load_dts(dt_path)
        q_quat, _ = _raw_to_q_quat_and_forces(
            raw,
            num_joints=nj,
            num_contacts=nc,
            rot_type=rot_type,
            include_forces=False,
        )
        times = _build_times(dts, q_quat.shape[0])
        iters.append(
            NewtonIterTraj(
                iter_id=iter_id,
                dts=dts.astype(np.float64),
                times=times.astype(np.float64),
                q_quat=q_quat.astype(np.float32),
            )
        )

    status = (
        f"Loaded {len(iters)} Newton iters from {root.name}: "
        f"K={iters[0].iter_id}..{iters[-1].iter_id}, "
        f"T={iters[0].duration:.3f}s→{iters[-1].duration:.3f}s, "
        f"robot={robot_key}"
    )
    return NewtonSnapshotsData(
        folder=str(root),
        robot_key=robot_key,
        soma_robot_type=spec.soma_robot_type,
        joint_names=joint_names,
        iters=iters,
        status=status,
    )


def draw_snapshot_overlays(
    viewer,
    data: NewtonSnapshotsData,
    *,
    time_s: float,
    align: str = "absolute",
    resample_dt: float = 0.1,
    max_iter_id: Optional[int] = None,
    show_trails: bool = True,
    show_poses: bool = True,
    trail_width: float = 0.012,
) -> None:
    """Draw base trails + pose markers for iters up to ``max_iter_id``."""
    import warp as wp

    if not hasattr(viewer, "log_lines"):
        return

    empty_v3 = wp.array(np.zeros((0, 3), dtype=np.float32), dtype=wp.vec3)
    if not data.iters:
        viewer.log_lines(
            "/newton_snap/trails", empty_v3, empty_v3, empty_v3, width=trail_width)
        if hasattr(viewer, "log_points"):
            empty_f = wp.array(np.zeros((0,), dtype=np.float32), dtype=wp.float32)
            viewer.log_points(
                "/newton_snap/poses", empty_v3, radii=empty_f, colors=empty_v3)
        return

    if max_iter_id is None:
        max_iter_id = data.iters[-1].iter_id
    visible = [it for it in data.iters if it.iter_id <= max_iter_id]
    t_ref = data.max_duration

    starts: List[np.ndarray] = []
    ends: List[np.ndarray] = []
    colors: List[np.ndarray] = []
    pose_pts: List[np.ndarray] = []
    pose_cols: List[np.ndarray] = []

    if show_trails:
        for it in visible:
            # Grow trail only up to current scrub time (absolute seconds or τ*T_ref).
            trail = data.base_trail(
                it.iter_id,
                resample_dt=resample_dt,
                align=align,
                t_ref=t_ref,
                t_max=float(time_s),
            )
            if trail.shape[0] < 2:
                continue
            rgb = data.color_for_iter(it.iter_id)
            col = np.array(rgb, dtype=np.float32)
            for i in range(trail.shape[0] - 1):
                starts.append(trail[i])
                ends.append(trail[i + 1])
                colors.append(col)

    if show_poses:
        for it in visible:
            q = data.sample_q(it.iter_id, time_s, align=align, t_ref=t_ref)
            if q is None:
                continue
            pose_pts.append(np.asarray(q[:3], dtype=np.float32))
            pose_cols.append(np.asarray(data.color_for_iter(it.iter_id), dtype=np.float32))

    if starts:
        s = wp.array(np.stack(starts), dtype=wp.vec3)
        e = wp.array(np.stack(ends), dtype=wp.vec3)
        c = wp.array(np.stack(colors), dtype=wp.vec3)
        viewer.log_lines("/newton_snap/trails", s, e, c, width=trail_width)
    else:
        viewer.log_lines(
            "/newton_snap/trails", empty_v3, empty_v3, empty_v3, width=trail_width)

    if hasattr(viewer, "log_points"):
        empty_f = wp.array(np.zeros((0,), dtype=np.float32), dtype=wp.float32)
        if pose_pts:
            pts = wp.array(np.stack(pose_pts), dtype=wp.vec3)
            cols = wp.array(np.stack(pose_cols), dtype=wp.vec3)
            radii = wp.array(
                np.full(len(pose_pts), 0.04, dtype=np.float32), dtype=wp.float32)
            viewer.log_points("/newton_snap/poses", pts, radii=radii, colors=cols)
        else:
            viewer.log_points(
                "/newton_snap/poses", empty_v3, radii=empty_f, colors=empty_v3)


def clear_snapshot_overlays(viewer) -> None:
    """Remove Newton snapshot line/point overlays."""
    import warp as wp

    empty_v3 = wp.array(np.zeros((0, 3), dtype=np.float32), dtype=wp.vec3)
    empty_f = wp.array(np.zeros((0,), dtype=np.float32), dtype=wp.float32)
    if hasattr(viewer, "log_lines"):
        viewer.log_lines(
            "/newton_snap/trails", empty_v3, empty_v3, empty_v3, width=0.01)
    if hasattr(viewer, "log_points"):
        viewer.log_points(
            "/newton_snap/poses", empty_v3, radii=empty_f, colors=empty_v3)


def draw_dt_plot(ui, data: NewtonSnapshotsData, *, focus_iter_id: int, time_s: float,
                 align: str = "absolute", height: float = 120.0) -> None:
    """Side-panel dt curve for the focus iteration (imgui plot_lines if available)."""
    try:
        it = data.iters[data.iter_index(focus_iter_id)]
    except KeyError:
        ui.text_disabled("(focus iter not loaded)")
        return

    dts = np.asarray(it.dts, dtype=np.float32)
    if dts.size == 0:
        ui.text_disabled("(no dt samples)")
        return

    # Highlight approximate segment at current scrub time.
    t_ref = data.max_duration
    if align == "normalized":
        tau = 0.0 if t_ref <= 1e-12 else float(np.clip(time_s / t_ref, 0.0, 1.0))
        t_query = tau * it.duration
    else:
        t_query = float(np.clip(time_s, 0.0, it.duration))
    seg = int(np.searchsorted(it.times, t_query, side="right") - 1)
    seg = int(np.clip(seg, 0, max(0, dts.size - 1)))

    ui.text(f"iter {focus_iter_id}  dt[{seg}]={dts[seg]:.4f}s  "
            f"mean={dts.mean():.4f}  T={it.duration:.3f}s")

    values = dts.tolist()
    if hasattr(ui, "plot_lines"):
        ui.plot_lines(
            "##ns_dt_plot",
            values,
            graph_size=ui.ImVec2(-1, height),
        )
    elif hasattr(ui, "plot_histogram"):
        ui.plot_histogram(
            "##ns_dt_plot",
            values,
            graph_size=ui.ImVec2(-1, height),
        )
    else:
        # Fallback: mini text sparkline of min/max.
        ui.text(f"dt range [{dts.min():.4f}, {dts.max():.4f}] s  "
                f"(plot_lines unavailable)")

    # Vertical marker via text (segment index vs scrub).
    if it.duration > 1e-12:
        frac = t_query / it.duration
        ui.progress_bar(frac, ui.ImVec2(-1, 0), f"τ={frac:.3f}  t={t_query:.3f}s")

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Newton-iteration trajectory snapshots for TO convergence visualization.

Expects a folder of paired files::

    newton_iter_<K>.txt      # CIO save_results layout (rpy)
    newton_iter_<K>_dt.txt   # [index, dt] or bare dt per control segment

Scrubber steps through **iteration index** (0 → 1 → 2 → … last). At each step
up to N recent iters draw lightweight **stick skeletons** (bone lines, no skin)
sampled along each full trajectory.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from soma_retargeter.to_playback.jsonc import load_jsonc
from soma_retargeter.to_playback.loader import _load_numeric, _raw_to_q_quat_and_forces
from soma_retargeter.to_playback.robots import (
    ToRobotSpec,
    get_to_robot_spec,
    infer_robot_key_from_folder,
)
from soma_retargeter.utils.newton_utils import get_name_from_label

_ITER_RE = re.compile(r"^newton_iter_(\d+)\.txt$")
_SNAP_MAX_POSES_PER_ITER = 24
_SNAP_DEFAULT_VISIBLE_ITERS = 5
_SNAP_SKELETON_WIDTH = 0.010
_SNAP_ITER_ALPHA_MIN = 0.14
_SNAP_ITER_ALPHA_MAX = 1.0
_SNAP_GHOST_ALPHA_CUTOFF = 0.98
# Motion-direction hue sweeps (HSV). Blue→red via purple (not via green).
# hue goes UP: ~0.62 (blue) → ~1.00 (red). Soft sat/val by default.
_SNAP_MOTION_PALETTES = {
    "soft": {"hue_start": 0.60, "hue_end": 0.98, "sat": 0.36, "val": 0.70},
    "muted": {"hue_start": 0.58, "hue_end": 0.96, "sat": 0.24, "val": 0.66},
    "vivid": {"hue_start": 0.62, "hue_end": 0.99, "sat": 0.62, "val": 0.82},
    "warm": {"hue_start": 0.65, "hue_end": 0.02, "sat": 0.40, "val": 0.72},
}
_SNAP_MOTION_PALETTE_DEFAULT = "soft"
_SNAP_MOTION_PALETTE_OPTIONS = list(_SNAP_MOTION_PALETTES.keys())


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
        """Deprecated: snapshots use iteration scrub, not seconds."""
        _ = align
        return float(max(0, len(self.iters) - 1))

    def scrub_count(self) -> int:
        return len(self.iters)

    def scrub_index_clamped(self, scrub_index: float) -> int:
        n = len(self.iters)
        if n <= 0:
            return 0
        return int(np.clip(int(round(float(scrub_index))), 0, n - 1))

    def iter_id_at_scrub(self, scrub_index: float) -> int:
        return self.iters[self.scrub_index_clamped(scrub_index)].iter_id

    def iters_up_to_scrub(self, scrub_index: float) -> List[NewtonIterTraj]:
        if not self.iters:
            return []
        return self.iters[: self.scrub_index_clamped(scrub_index) + 1]

    def iters_visible_window(
        self,
        scrub_index: float,
        *,
        max_iters: int = _SNAP_DEFAULT_VISIBLE_ITERS,
    ) -> List[NewtonIterTraj]:
        """Return up to ``max_iters`` most recent iters ending at scrub index."""
        if not self.iters:
            return []
        end = self.scrub_index_clamped(scrub_index)
        max_iters = max(1, int(max_iters))
        start = max(0, end + 1 - max_iters)
        return self.iters[start: end + 1]

    def dt_axis_limits(self, *, pad: float = 0.08) -> Tuple[float, float]:
        """Global dt y-limits across all iters (stable while scrubbing)."""
        vals = []
        for it in self.iters:
            d = np.asarray(it.dts, dtype=np.float64).reshape(-1)
            if d.size:
                vals.append(d)
        if not vals:
            return 0.0, 0.02
        stacked = np.concatenate(vals)
        lo = float(np.min(stacked))
        hi = float(np.max(stacked))
        if hi <= lo:
            hi = lo + 1e-4
        span = hi - lo
        return lo - pad * span, hi + pad * span

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

    def full_base_trail(
        self,
        iter_id: int,
        *,
        resample_dt: float = 0.1,
        align: str = "absolute",
        t_ref: Optional[float] = None,
    ) -> np.ndarray:
        """Complete base xyz trail for one Newton iteration."""
        it = self.iters[self.iter_index(iter_id)]
        return self.base_trail(
            iter_id,
            resample_dt=resample_dt,
            align=align,
            t_ref=t_ref,
            t_max=float(it.duration),
        )


@dataclass
class SnapVisualShape:
    """One visual link mesh on the FK robot (no collision duplicates)."""

    label: str
    body_index: int
    local_tf: np.ndarray  # (7,) pos + quat


@dataclass
class SnapMeshBatch:
    """Pre-baked mesh instances for one viewer mesh asset."""

    mesh_path: str
    xforms: np.ndarray   # (N, 7) float32
    colors: np.ndarray   # (N, 3) float32
    alphas: np.ndarray   # (N,) float32 — iter-depth opacity (newest = 1)


@dataclass
class SnapRobotCache:
    """Single-articulation robot used to FK snapshot overlays."""

    model: object
    state: object
    skeleton_edges: List[Tuple[int, int]]
    visual_shapes: List[SnapVisualShape]
    mesh_by_label: Dict[str, str]
    device: str


@dataclass
class SnapOverlayCache:
    """Cached overlay for one scrub frame."""

    skeleton_starts: np.ndarray
    skeleton_ends: np.ndarray
    skeleton_colors: np.ndarray
    trail_starts: np.ndarray
    trail_ends: np.ndarray
    trail_colors: np.ndarray
    mesh_batches: List[SnapMeshBatch]


@dataclass
class SnapOverlayBank:
    """Pre-baked overlay for every scrub index under one view configuration."""

    key: tuple
    frames: List[SnapOverlayCache]


_overlay_bank: Optional[SnapOverlayBank] = None
_last_drawn_scrub: int = -1


def _collect_skeleton_edges(model) -> List[Tuple[int, int]]:
    joint_parents = model.joint_parent.numpy()
    joint_child = model.joint_child.numpy()
    edges: List[Tuple[int, int]] = []
    for ji in range(int(model.joint_count)):
        parent = int(joint_parents[ji])
        child = int(joint_child[ji])
        if parent >= 0 and child >= 0:
            edges.append((parent, child))
    return edges


def _collect_visual_shapes(model) -> List[SnapVisualShape]:
    import newton as nt

    visible = int(nt.ShapeFlags.VISIBLE)
    mesh_types = {int(nt.GeoType.MESH)}
    if hasattr(nt.GeoType, "CONVEX_MESH"):
        mesh_types.add(int(nt.GeoType.CONVEX_MESH))

    shape_body = model.shape_body.numpy()
    shape_tf = model.shape_transform.numpy()
    shape_flags = model.shape_flags.numpy()
    shape_types = model.shape_type.numpy()
    shape_labels = model.shape_label
    out: List[SnapVisualShape] = []
    for s in range(int(model.shape_count)):
        if not (int(shape_flags[s]) & visible):
            continue
        if int(shape_types[s]) not in mesh_types:
            continue
        label = get_name_from_label(shape_labels[s])
        if label.endswith("_col") or "_col/" in label:
            continue
        out.append(
            SnapVisualShape(
                label=label,
                body_index=int(shape_body[s]),
                local_tf=np.asarray(shape_tf[s], dtype=np.float32).reshape(7),
            )
        )
    return out


def build_viewer_mesh_map(viewer, model, *, articulation_index: int = 0) -> Dict[str, str]:
    """Map link labels → viewer mesh paths for one articulation."""
    import newton as nt

    mesh_types = {int(nt.GeoType.MESH)}
    if hasattr(nt.GeoType, "CONVEX_MESH"):
        mesh_types.add(int(nt.GeoType.CONVEX_MESH))

    bodies_per = max(1, int(model.body_count) // max(1, int(model.articulation_count)))
    lo = int(articulation_index) * bodies_per
    hi = lo + bodies_per
    shape_body = model.shape_body.numpy()
    out: Dict[str, str] = {}
    batches = getattr(viewer, "_shape_instances", None) or {}
    for shapes in batches.values():
        if int(shapes.geo_type) not in mesh_types:
            continue
        mesh_path = str(shapes.mesh)
        for s_idx in shapes.model_shapes:
            s_idx = int(s_idx)
            bi = int(shape_body[s_idx])
            if bi < lo or bi >= hi:
                continue
            label = get_name_from_label(model.shape_label[s_idx])
            if label.endswith("_col"):
                continue
            out.setdefault(label, mesh_path)
    return out


def prepare_snap_robot(viewer, main_model, robot_builder) -> SnapRobotCache:
    """Build FK model + viewer mesh map for coarse opaque snapshot ghosts."""
    import newton
    import warp as wp

    builder = newton.ModelBuilder()
    builder.add_builder(robot_builder, wp.transform_identity())
    snap_model = builder.finalize()
    return SnapRobotCache(
        model=snap_model,
        state=snap_model.state(),
        skeleton_edges=_collect_skeleton_edges(snap_model),
        visual_shapes=_collect_visual_shapes(snap_model),
        mesh_by_label=build_viewer_mesh_map(viewer, main_model, articulation_index=0),
        device=str(getattr(viewer, "device", "cpu")),
    )


def set_viewer_robots_hidden(viewer, model, hidden: bool) -> None:
    """Hide/show all body-attached shape batches (robots), keep ground visible."""
    shape_body = model.shape_body.numpy()
    batches = getattr(viewer, "_shape_instances", None) or {}
    objects = getattr(viewer, "objects", None) or {}
    for shapes in batches.values():
        hide_batch = False
        for s_idx in shapes.model_shapes:
            s_idx = int(s_idx)
            if 0 <= s_idx < len(shape_body) and int(shape_body[s_idx]) >= 0:
                hide_batch = True
                break
        if not hide_batch:
            continue
        obj = objects.get(shapes.name)
        if obj is not None and hasattr(obj, "hidden"):
            obj.hidden = bool(hidden)


def _apply_q_to_snap(cache: SnapRobotCache, q: np.ndarray) -> None:
    import newton
    import warp as wp

    q = np.asarray(q, dtype=np.float32).reshape(-1)
    wp.copy(cache.model.joint_q, wp.array(q, dtype=wp.float32), 0, 0, q.shape[0])
    newton.eval_fk(
        cache.model, cache.model.joint_q, cache.model.joint_qd, cache.state, None)


def _iter_depth_alpha(
    vis_index: int,
    vis_count: int,
    *,
    alpha_min: float = _SNAP_ITER_ALPHA_MIN,
    alpha_max: float = _SNAP_ITER_ALPHA_MAX,
) -> float:
    """Opacity by position in visible window: oldest faint, newest solid."""
    alpha_min = float(np.clip(alpha_min, 0.0, alpha_max))
    if vis_count <= 1:
        return alpha_max
    u = float(vis_index) / float(vis_count - 1)
    return float(alpha_min + (alpha_max - alpha_min) * u)


def _motion_time_tau(
    sample_index: int,
    sample_count: int,
    time_s: float,
    traj_duration: float,
    *,
    align: str,
) -> float:
    if sample_count <= 1:
        return 0.0
    if align == "normalized":
        return float(sample_index) / float(sample_count - 1)
    if traj_duration <= 1e-12:
        return 0.0
    return float(np.clip(time_s / traj_duration, 0.0, 1.0))


def _hsv_to_rgb(h: float, s: float, v: float) -> np.ndarray:
    """Convert HSV (h∈[0,1]) to RGB float32 in [0,1]."""
    h = float(h) % 1.0
    s = float(np.clip(s, 0.0, 1.0))
    v = float(np.clip(v, 0.0, 1.0))
    i = int(h * 6.0)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i = i % 6
    if i == 0:
        r, g, b = v, t, p
    elif i == 1:
        r, g, b = q, v, p
    elif i == 2:
        r, g, b = p, v, t
    elif i == 3:
        r, g, b = p, q, v
    elif i == 4:
        r, g, b = t, p, v
    else:
        r, g, b = v, p, q
    return np.array([r, g, b], dtype=np.float32)


def _color_for_motion_time(
    tau: float,
    *,
    palette: str = _SNAP_MOTION_PALETTE_DEFAULT,
) -> np.ndarray:
    """Hue sweep along traj time; default soft blue→purple→red."""
    tau = float(np.clip(tau, 0.0, 1.0))
    cfg = _SNAP_MOTION_PALETTES.get(palette) or _SNAP_MOTION_PALETTES[_SNAP_MOTION_PALETTE_DEFAULT]
    h0 = float(cfg["hue_start"])
    h1 = float(cfg["hue_end"])
    # If end < start, wrap past 1.0 so blue→red goes via purple (not green).
    if h1 < h0:
        h1 += 1.0
    h = h0 + (h1 - h0) * tau
    return _hsv_to_rgb(h, cfg["sat"], cfg["val"])


def _shape_world_xform_np7(shape: SnapVisualShape, body_q: np.ndarray) -> np.ndarray:
    import warp as wp

    local = wp.transform(
        float(shape.local_tf[0]), float(shape.local_tf[1]), float(shape.local_tf[2]),
        float(shape.local_tf[3]), float(shape.local_tf[4]),
        float(shape.local_tf[5]), float(shape.local_tf[6]),
    )
    bi = int(shape.body_index)
    if bi >= 0:
        b = body_q[bi]
        parent = wp.transform(
            float(b[0]), float(b[1]), float(b[2]),
            float(b[3]), float(b[4]), float(b[5]), float(b[6]),
        )
        world = wp.mul(parent, local)
        return np.array(
            [world.p[0], world.p[1], world.p[2], world.q[0], world.q[1], world.q[2], world.q[3]],
            dtype=np.float32,
        )
    return shape.local_tf.copy()


def _mesh_instancer_name(mesh_path: str) -> str:
    safe = mesh_path.strip("/").replace("/", "_")
    return f"/newton_snap/mesh/{safe}"


def iter_traj_sample_times(
    it: NewtonIterTraj,
    resample_dt: float,
    *,
    align: str = "absolute",
    t_ref: float,
    max_samples: int = _SNAP_MAX_POSES_PER_ITER,
) -> List[float]:
    """Uniform sample times along one iter's full duration (capped)."""
    max_samples = max(2, int(max_samples))
    if align == "normalized":
        n = min(max_samples, max(2, int(np.ceil(1.0 / max(1e-4, resample_dt))) + 1))
        taus = np.linspace(0.0, 1.0, n, dtype=np.float64)
        return [float(tau * it.duration) for tau in taus]

    dt = max(1e-4, float(resample_dt))
    if it.duration <= 1e-12:
        return [0.0]
    times = np.arange(0.0, float(it.duration) + 0.5 * dt, dt, dtype=np.float64)
    if times.size > max_samples:
        pick = np.linspace(0, times.size - 1, max_samples, dtype=int)
        times = times[pick]
    return [float(t) for t in times]


def clear_snapshot_mesh_ghosts(viewer) -> None:
    """Hide snapshot mesh instancers (/newton_snap/mesh/* and legacy /pose/*)."""
    objects = getattr(viewer, "objects", None) or {}
    for name in list(objects.keys()):
        if not isinstance(name, str):
            continue
        if name.startswith("/newton_snap/mesh/") or name.startswith("/newton_snap/pose/"):
            obj = objects[name]
            if hasattr(obj, "hidden"):
                obj.hidden = True
            if hasattr(obj, "active_instances"):
                obj.active_instances = 0


def invalidate_snapshot_overlay_cache() -> None:
    global _overlay_bank, _last_drawn_scrub
    _overlay_bank = None
    _last_drawn_scrub = -1


def _overlay_bank_key(
    data: NewtonSnapshotsData,
    *,
    align: str,
    resample_dt: float,
    max_visible_iters: int,
    show_meshes: bool,
    show_skeleton: bool,
    show_trails: bool,
    iter_alpha_min: float,
    palette: str = _SNAP_MOTION_PALETTE_DEFAULT,
) -> tuple:
    return (
        data.folder,
        align,
        round(float(resample_dt), 4),
        int(max_visible_iters),
        bool(show_meshes),
        bool(show_skeleton),
        bool(show_trails),
        round(float(iter_alpha_min), 4),
        str(palette),
    )


def _empty_overlay_cache() -> SnapOverlayCache:
    empty = np.zeros((0, 3), dtype=np.float32)
    return SnapOverlayCache(
        skeleton_starts=empty,
        skeleton_ends=empty.copy(),
        skeleton_colors=empty.copy(),
        trail_starts=empty.copy(),
        trail_ends=empty.copy(),
        trail_colors=empty.copy(),
        mesh_batches=[],
    )


def _build_overlay_cache(
    data: NewtonSnapshotsData,
    cache: SnapRobotCache,
    *,
    scrub_index: float,
    align: str,
    resample_dt: float,
    max_visible_iters: int,
    show_meshes: bool,
    show_skeleton: bool,
    show_trails: bool,
    iter_alpha_min: float = _SNAP_ITER_ALPHA_MIN,
    palette: str = _SNAP_MOTION_PALETTE_DEFAULT,
) -> SnapOverlayCache:
    visible = data.iters_visible_window(scrub_index, max_iters=max_visible_iters)
    t_ref = data.max_duration

    sk_starts: List[np.ndarray] = []
    sk_ends: List[np.ndarray] = []
    sk_colors: List[np.ndarray] = []
    per_mesh_xforms: Dict[str, List[np.ndarray]] = {}
    per_mesh_colors: Dict[str, List[np.ndarray]] = {}
    per_mesh_alphas: Dict[str, List[float]] = {}

    need_fk = (
        (show_skeleton and cache.skeleton_edges)
        or (show_meshes and cache.visual_shapes and cache.mesh_by_label)
    )
    if need_fk:
        vis_count = len(visible)
        for vis_i, it in enumerate(visible):
            iter_alpha = _iter_depth_alpha(vis_i, vis_count, alpha_min=iter_alpha_min)
            sample_times = iter_traj_sample_times(
                it, resample_dt, align=align, t_ref=t_ref)
            sample_count = len(sample_times)
            for si, t in enumerate(sample_times):
                q = data.sample_q(it.iter_id, t, align=align, t_ref=t_ref)
                if q is None:
                    continue
                tau = _motion_time_tau(
                    si, sample_count, t, it.duration, align=align)
                pose_rgb = _color_for_motion_time(tau, palette=palette)
                _apply_q_to_snap(cache, q)
                body_q = cache.state.body_q.numpy()
                if show_skeleton:
                    sk_rgb = pose_rgb * (0.35 + 0.65 * iter_alpha)
                    pos = body_q[:, :3].astype(np.float32, copy=False)
                    for parent_i, child_i in cache.skeleton_edges:
                        sk_starts.append(pos[parent_i])
                        sk_ends.append(pos[child_i])
                        sk_colors.append(sk_rgb)
                if show_meshes:
                    for shape in cache.visual_shapes:
                        mesh_path = cache.mesh_by_label.get(shape.label)
                        if mesh_path is None:
                            continue
                        per_mesh_xforms.setdefault(mesh_path, []).append(
                            _shape_world_xform_np7(shape, body_q))
                        per_mesh_colors.setdefault(mesh_path, []).append(pose_rgb)
                        per_mesh_alphas.setdefault(mesh_path, []).append(iter_alpha)

    tr_starts: List[np.ndarray] = []
    tr_ends: List[np.ndarray] = []
    tr_colors: List[np.ndarray] = []

    if show_trails:
        vis_count = len(visible)
        for vis_i, it in enumerate(visible):
            iter_alpha = _iter_depth_alpha(vis_i, vis_count, alpha_min=iter_alpha_min)
            trail = data.full_base_trail(
                it.iter_id,
                resample_dt=resample_dt,
                align=align,
                t_ref=t_ref,
            )
            if trail.shape[0] < 2:
                continue
            nseg = trail.shape[0] - 1
            for i in range(nseg):
                tau = (i + 0.5) / float(nseg)
                seg_rgb = _color_for_motion_time(tau, palette=palette) * (
                    0.35 + 0.65 * iter_alpha)
                tr_starts.append(trail[i])
                tr_ends.append(trail[i + 1])
                tr_colors.append(seg_rgb)

    def _stack_or_empty(rows: List[np.ndarray]) -> np.ndarray:
        if not rows:
            return np.zeros((0, 3), dtype=np.float32)
        return np.stack(rows).astype(np.float32, copy=False)

    mesh_batches: List[SnapMeshBatch] = []
    for mesh_path, xforms in per_mesh_xforms.items():
        if not xforms:
            continue
        mesh_batches.append(
            SnapMeshBatch(
                mesh_path=mesh_path,
                xforms=np.stack(xforms).astype(np.float32, copy=False),
                colors=np.stack(per_mesh_colors[mesh_path]).astype(np.float32, copy=False),
                alphas=np.asarray(per_mesh_alphas[mesh_path], dtype=np.float32),
            )
        )

    return SnapOverlayCache(
        skeleton_starts=_stack_or_empty(sk_starts),
        skeleton_ends=_stack_or_empty(sk_ends),
        skeleton_colors=_stack_or_empty(sk_colors),
        trail_starts=_stack_or_empty(tr_starts),
        trail_ends=_stack_or_empty(tr_ends),
        trail_colors=_stack_or_empty(tr_colors),
        mesh_batches=mesh_batches,
    )


def build_snapshot_overlay_bank(
    data: NewtonSnapshotsData,
    cache: SnapRobotCache,
    *,
    align: str,
    resample_dt: float,
    max_visible_iters: int,
    show_meshes: bool,
    show_skeleton: bool,
    show_trails: bool,
    iter_alpha_min: float = _SNAP_ITER_ALPHA_MIN,
    palette: str = _SNAP_MOTION_PALETTE_DEFAULT,
) -> SnapOverlayBank:
    """Pre-bake every scrub frame once; playback only swaps static overlays."""
    key = _overlay_bank_key(
        data,
        align=align,
        resample_dt=resample_dt,
        max_visible_iters=max_visible_iters,
        show_meshes=show_meshes,
        show_skeleton=show_skeleton,
        show_trails=show_trails,
        iter_alpha_min=iter_alpha_min,
        palette=palette,
    )
    if not data.iters or not (show_meshes or show_skeleton or show_trails):
        return SnapOverlayBank(key=key, frames=[])

    frames: List[SnapOverlayCache] = []
    n = len(data.iters)
    for scrub_i in range(n):
        frames.append(
            _build_overlay_cache(
                data,
                cache,
                scrub_index=float(scrub_i),
                align=align,
                resample_dt=resample_dt,
                max_visible_iters=max_visible_iters,
                show_meshes=show_meshes,
                show_skeleton=show_skeleton,
                show_trails=show_trails,
                iter_alpha_min=iter_alpha_min,
                palette=palette,
            )
        )
    return SnapOverlayBank(key=key, frames=frames)


def warm_snapshot_overlay_bank(
    data: NewtonSnapshotsData,
    cache: SnapRobotCache,
    *,
    align: str,
    resample_dt: float,
    max_visible_iters: int,
    show_meshes: bool,
    show_skeleton: bool,
    show_trails: bool,
    iter_alpha_min: float = _SNAP_ITER_ALPHA_MIN,
    palette: str = _SNAP_MOTION_PALETTE_DEFAULT,
) -> SnapOverlayBank:
    """Build and store the overlay bank for the current snapshot view settings."""
    global _overlay_bank, _last_drawn_scrub

    print(
        f"[INFO]: Pre-baking {len(data.iters)} Newton snapshot scrub frames "
        f"(Δt={resample_dt:.2f}s, max={max_visible_iters}, "
        f"mesh={'on' if show_meshes else 'off'}, palette={palette})…")
    _overlay_bank = build_snapshot_overlay_bank(
        data,
        cache,
        align=align,
        resample_dt=resample_dt,
        max_visible_iters=max_visible_iters,
        show_meshes=show_meshes,
        show_skeleton=show_skeleton,
        show_trails=show_trails,
        iter_alpha_min=iter_alpha_min,
        palette=palette,
    )
    _last_drawn_scrub = -1
    print(f"[INFO]: Newton snapshot bake done ({len(_overlay_bank.frames)} frames).")
    return _overlay_bank


def _upload_mesh_batches(viewer, batches: List[SnapMeshBatch], *, device: str) -> None:
    import warp as wp
    from soma_retargeter.to_playback import newton_alpha

    if not hasattr(viewer, "log_instances"):
        return

    alpha_ok = newton_alpha.is_mesh_alpha_enabled()
    objects = getattr(viewer, "objects", None) or {}
    active: set[str] = set()

    def _upload_split(
        inst_name: str,
        mesh_path: str,
        xforms: np.ndarray,
        colors: np.ndarray,
        alphas: np.ndarray | None,
        *,
        ghost_pass: bool,
    ) -> None:
        n = int(xforms.shape[0])
        if n <= 0 or mesh_path not in objects:
            return
        active.add(inst_name)
        materials = None
        out_colors = colors
        if ghost_pass and alpha_ok and alphas is not None and alphas.shape[0] == n:
            # Metallic must NOT be in (0.04, 0.06): that band is reserved by
            # enable_studio_reflect_fade() and multiplies soma_alpha by ~0.16–0.58,
            # which made even opacity=0.9 look nearly invisible.
            m_np = np.zeros((n, 4), dtype=np.float32)
            for j in range(n):
                m_np[j] = newton_alpha.encode_material_alpha(
                    (0.55, 0.0, 0.0, 0.0), float(alphas[j]))
            materials = wp.array(m_np, dtype=wp.vec4, device=device)
        else:
            # Always rewrite materials: None leaves stale -alpha in the VBO.
            m_np = np.tile(
                np.array([0.55, 0.0, 0.0, 0.0], dtype=np.float32), (n, 1))
            materials = wp.array(m_np, dtype=wp.vec4, device=device)
            if alphas is not None and alphas.shape[0] == n:
                out_colors = colors * np.clip(alphas, 0.0, 1.0).reshape(-1, 1)

        viewer.log_instances(
            inst_name,
            mesh_path,
            wp.array(xforms, dtype=wp.transform, device=device),
            wp.ones(n, dtype=wp.vec3, device=device),
            wp.array(out_colors, dtype=wp.vec3, device=device),
            materials,
            hidden=False,
        )
        inst = objects.get(inst_name)
        if inst is not None and ghost_pass:
            inst._soma_pass = "ghost"
        elif inst is not None and hasattr(inst, "_soma_pass"):
            inst._soma_pass = None

    for batch in batches:
        mesh_path = batch.mesh_path
        n = int(batch.xforms.shape[0])
        if n <= 0:
            continue
        base_name = _mesh_instancer_name(mesh_path)
        alphas = batch.alphas
        if alphas is None or alphas.shape[0] != n:
            alphas = np.ones(n, dtype=np.float32)

        opaque_mask = alphas >= _SNAP_GHOST_ALPHA_CUTOFF
        ghost_mask = ~opaque_mask
        if np.any(opaque_mask):
            _upload_split(
                base_name,
                mesh_path,
                batch.xforms[opaque_mask],
                batch.colors[opaque_mask],
                alphas[opaque_mask],
                ghost_pass=False,
            )
        else:
            # Hide stale opaque batch when everything is ghost this frame.
            legacy = objects.get(base_name)
            if legacy is not None:
                if hasattr(legacy, "hidden"):
                    legacy.hidden = True
                if hasattr(legacy, "active_instances"):
                    legacy.active_instances = 0
        if np.any(ghost_mask):
            _upload_split(
                f"{base_name}/ghost",
                mesh_path,
                batch.xforms[ghost_mask],
                batch.colors[ghost_mask],
                alphas[ghost_mask],
                ghost_pass=True,
            )
        else:
            ghost = objects.get(f"{base_name}/ghost")
            if ghost is not None:
                if hasattr(ghost, "hidden"):
                    ghost.hidden = True
                if hasattr(ghost, "active_instances"):
                    ghost.active_instances = 0

    for obj_name in list(objects.keys()):
        if isinstance(obj_name, str) and obj_name.startswith("/newton_snap/mesh/"):
            if obj_name not in active:
                obj = objects[obj_name]
                if hasattr(obj, "hidden"):
                    obj.hidden = True
                if hasattr(obj, "active_instances"):
                    obj.active_instances = 0


def _upload_overlay_cache(
    viewer,
    overlay: SnapOverlayCache,
    *,
    trail_width: float,
    device: str,
) -> None:
    import warp as wp

    empty_v3 = wp.array(np.zeros((0, 3), dtype=np.float32), dtype=wp.vec3)

    if overlay.skeleton_starts.shape[0] > 0:
        viewer.log_lines(
            "/newton_snap/skeleton",
            wp.array(overlay.skeleton_starts, dtype=wp.vec3),
            wp.array(overlay.skeleton_ends, dtype=wp.vec3),
            wp.array(overlay.skeleton_colors, dtype=wp.vec3),
            width=_SNAP_SKELETON_WIDTH,
        )
    else:
        viewer.log_lines(
            "/newton_snap/skeleton", empty_v3, empty_v3, empty_v3, width=_SNAP_SKELETON_WIDTH)

    if overlay.trail_starts.shape[0] > 0:
        viewer.log_lines(
            "/newton_snap/trails",
            wp.array(overlay.trail_starts, dtype=wp.vec3),
            wp.array(overlay.trail_ends, dtype=wp.vec3),
            wp.array(overlay.trail_colors, dtype=wp.vec3),
            width=trail_width,
        )
    else:
        viewer.log_lines(
            "/newton_snap/trails", empty_v3, empty_v3, empty_v3, width=trail_width)

    _upload_mesh_batches(viewer, overlay.mesh_batches, device=device)


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
    scrub_index: float,
    align: str = "absolute",
    resample_dt: float = 0.5,
    max_visible_iters: int = _SNAP_DEFAULT_VISIBLE_ITERS,
    show_trails: bool = False,
    show_meshes: bool = True,
    show_skeleton: bool = False,
    iter_alpha_min: float = _SNAP_ITER_ALPHA_MIN,
    palette: str = _SNAP_MOTION_PALETTE_DEFAULT,
    snap_cache: Optional[SnapRobotCache] = None,
    trail_width: float = 0.012,
) -> None:
    """Draw pre-baked coarse mesh + optional skeleton; upload only on scrub change."""
    import warp as wp

    global _overlay_bank, _last_drawn_scrub

    if not hasattr(viewer, "log_lines"):
        return

    if not data.iters:
        invalidate_snapshot_overlay_cache()
        empty_v3 = wp.array(np.zeros((0, 3), dtype=np.float32), dtype=wp.vec3)
        viewer.log_lines(
            "/newton_snap/skeleton", empty_v3, empty_v3, empty_v3, width=_SNAP_SKELETON_WIDTH)
        viewer.log_lines(
            "/newton_snap/trails", empty_v3, empty_v3, empty_v3, width=trail_width)
        clear_snapshot_mesh_ghosts(viewer)
        return

    show_meshes = bool(show_meshes and snap_cache is not None and snap_cache.mesh_by_label)
    show_skeleton = bool(show_skeleton and snap_cache is not None)
    bank_key = _overlay_bank_key(
        data,
        align=align,
        resample_dt=resample_dt,
        max_visible_iters=max_visible_iters,
        show_meshes=show_meshes,
        show_skeleton=show_skeleton,
        show_trails=bool(show_trails),
        iter_alpha_min=float(iter_alpha_min),
        palette=str(palette),
    )
    if _overlay_bank is None or _overlay_bank.key != bank_key:
        if snap_cache is None or not (show_meshes or show_skeleton or show_trails):
            _overlay_bank = SnapOverlayBank(
                key=bank_key,
                frames=[_empty_overlay_cache() for _ in data.iters],
            )
        else:
            print(
                f"[INFO]: Rebuilding {len(data.iters)} Newton snapshot frames "
                f"(Δt={resample_dt:.2f}s, max={max_visible_iters}, "
                f"mesh={'on' if show_meshes else 'off'}, palette={palette})…")
            _overlay_bank = build_snapshot_overlay_bank(
                data,
                snap_cache,
                align=align,
                resample_dt=resample_dt,
                max_visible_iters=max_visible_iters,
                show_meshes=show_meshes,
                show_skeleton=show_skeleton,
                show_trails=bool(show_trails),
                iter_alpha_min=float(iter_alpha_min),
                palette=str(palette),
            )
        _last_drawn_scrub = -1

    scrub_i = data.scrub_index_clamped(scrub_index)
    if scrub_i == _last_drawn_scrub:
        return

    if scrub_i < 0 or scrub_i >= len(_overlay_bank.frames):
        return

    device = str(getattr(viewer, "device", "cpu"))
    if snap_cache is not None:
        device = snap_cache.device
    _upload_overlay_cache(
        viewer, _overlay_bank.frames[scrub_i], trail_width=trail_width, device=device)
    _last_drawn_scrub = scrub_i


def clear_snapshot_overlays(viewer) -> None:
    """Remove Newton snapshot line overlays."""
    import warp as wp

    invalidate_snapshot_overlay_cache()
    clear_snapshot_mesh_ghosts(viewer)
    empty_v3 = wp.array(np.zeros((0, 3), dtype=np.float32), dtype=wp.vec3)
    if hasattr(viewer, "log_lines"):
        viewer.log_lines(
            "/newton_snap/skeleton", empty_v3, empty_v3, empty_v3, width=_SNAP_SKELETON_WIDTH)
        viewer.log_lines(
            "/newton_snap/trails", empty_v3, empty_v3, empty_v3, width=0.01)


def draw_dt_plot(
    ui,
    data: NewtonSnapshotsData,
    *,
    focus_iter_id: int,
    height: float = 160.0,
) -> None:
    """dt curve with fixed axes, first-iter baseline, and tick labels.

    Y-limits are locked to the global min/max across **all** loaded Newton
    iters so scrubbing does not re-scale the plot. The first iter (K=min) is
    drawn as a blue baseline (``dt_init``); the scrub iter is orange (``dt_opt``).
    """
    try:
        it = data.iters[data.iter_index(focus_iter_id)]
    except KeyError:
        ui.text_disabled("(scrub iter not loaded)")
        return

    dts = np.asarray(it.dts, dtype=np.float64)
    if dts.size == 0:
        ui.text_disabled("(no dt samples)")
        return

    base_it = data.iters[0]
    dts0 = np.asarray(base_it.dts, dtype=np.float64)
    y_lo, y_hi = data.dt_axis_limits()
    x_max = max(1, max(int(x.dts.size) for x in data.iters) - 1)

    ui.text(
        f"iter {focus_iter_id}  mean dt={dts.mean():.4f}s  "
        f"T={it.duration:.3f}s  segments={dts.size}  "
        f"|  baseline K={base_it.iter_id} mean={dts0.mean():.4f}s"
    )
    ui.text_disabled(
        f"y-axis locked [{y_lo:.4f}, {y_hi:.4f}] s   "
        f"x = segment index 0..{x_max}"
    )

    _draw_dt_plot_canvas(
        ui,
        dts_opt=dts,
        dts_init=dts0,
        x_max=x_max,
        y_lo=y_lo,
        y_hi=y_hi,
        height=height,
        init_id=base_it.iter_id,
        opt_id=focus_iter_id,
    )


def _nice_ticks(lo: float, hi: float, target: int = 5) -> List[float]:
    """Return ~target nicely rounded tick values in [lo, hi]."""
    lo = float(lo)
    hi = float(hi)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return [lo, hi]
    span = hi - lo
    raw = span / max(1, target)
    exp = np.floor(np.log10(max(raw, 1e-12)))
    base = 10.0 ** exp
    step = base
    for mult in (1.0, 2.0, 2.5, 5.0, 10.0):
        if raw <= mult * base:
            step = mult * base
            break
    start = np.ceil(lo / step) * step
    ticks = []
    t = start
    for _ in range(64):
        if t > hi + 0.5 * step:
            break
        if t >= lo - 1e-12:
            ticks.append(float(t))
        t += step
    if not ticks:
        ticks = [lo, hi]
    return ticks


def _draw_dt_plot_canvas(
    ui,
    *,
    dts_opt: np.ndarray,
    dts_init: np.ndarray,
    x_max: int,
    y_lo: float,
    y_hi: float,
    height: float,
    init_id: int,
    opt_id: int,
) -> None:
    """Custom ImGui canvas: axes + baseline + current dt curve."""
    pad_l, pad_r, pad_t, pad_b = 52.0, 10.0, 8.0, 28.0
    width = max(120.0, float(ui.get_content_region_avail().x))
    plot_h = float(height)
    ui.invisible_button("##ns_dt_canvas", ui.ImVec2(width, plot_h))
    p0 = ui.get_item_rect_min()
    p1 = ui.get_item_rect_max()
    dl = ui.get_window_draw_list()

    x0 = float(p0.x) + pad_l
    y0 = float(p0.y) + pad_t
    x1 = float(p1.x) - pad_r
    y1 = float(p1.y) - pad_b
    if x1 <= x0 + 4 or y1 <= y0 + 4:
        return

    def col(r, g, b, a=1.0):
        return ui.color_convert_float4_to_u32(ui.ImVec4(r, g, b, a))

    col_bg = col(0.10, 0.12, 0.15, 0.85)
    col_grid = col(0.35, 0.38, 0.42, 0.35)
    col_axis = col(0.75, 0.78, 0.82, 0.90)
    col_init = col(0.35, 0.55, 0.95, 1.0)
    col_opt = col(0.95, 0.55, 0.18, 1.0)
    col_text = col(0.85, 0.88, 0.92, 1.0)

    dl.add_rect_filled(ui.ImVec2(x0, y0), ui.ImVec2(x1, y1), col_bg, 2.0)
    dl.add_rect(ui.ImVec2(x0, y0), ui.ImVec2(x1, y1), col_axis, 0.0, 0, 1.0)

    y_span = max(1e-9, y_hi - y_lo)

    def to_x(i: float) -> float:
        return x0 + (float(i) / float(max(1, x_max))) * (x1 - x0)

    def to_y(v: float) -> float:
        u = (float(v) - y_lo) / y_span
        return y1 - u * (y1 - y0)

    # Horizontal grid + y tick labels.
    for tick in _nice_ticks(y_lo, y_hi, target=6):
        yy = to_y(tick)
        dl.add_line(ui.ImVec2(x0, yy), ui.ImVec2(x1, yy), col_grid, 1.0)
        label = f"{tick:.4f}"
        dl.add_text(ui.ImVec2(float(p0.x) + 2.0, yy - 7.0), col_text, label)

    # Vertical grid + x tick labels.
    x_ticks = _nice_ticks(0.0, float(x_max), target=6)
    for tick in x_ticks:
        xx = to_x(tick)
        dl.add_line(ui.ImVec2(xx, y0), ui.ImVec2(xx, y1), col_grid, 1.0)
        dl.add_text(ui.ImVec2(xx - 8.0, y1 + 4.0), col_text, f"{int(round(tick))}")

    def _polyline(values: np.ndarray, color: int, thickness: float = 1.5):
        if values.size < 2:
            return
        pts = []
        n = int(values.size)
        for i in range(n):
            pts.append(ui.ImVec2(to_x(i), to_y(float(values[i]))))
        dl.add_polyline(pts, color, 0, thickness)

    # Baseline = first Newton iter (dt_init).
    _polyline(dts_init, col_init, 1.6)
    # Current scrub iter (dt_opt).
    _polyline(dts_opt, col_opt, 1.8)

    # Axis titles (compact).
    dl.add_text(ui.ImVec2(x0, float(p0.y) - 1.0), col_text, "dt (s)")
    dl.add_text(ui.ImVec2((x0 + x1) * 0.5 - 40.0, y1 + 14.0), col_text, "segment index")

    # Legend.
    lx = x0 + 6.0
    ly = y0 + 6.0
    dl.add_line(ui.ImVec2(lx, ly + 5), ui.ImVec2(lx + 18, ly + 5), col_init, 2.0)
    dl.add_text(ui.ImVec2(lx + 22, ly), col_text, f"dt_init (K={init_id})")
    dl.add_line(ui.ImVec2(lx + 130, ly + 5), ui.ImVec2(lx + 148, ly + 5), col_opt, 2.0)
    dl.add_text(ui.ImVec2(lx + 152, ly), col_text, f"dt_opt (K={opt_id})")

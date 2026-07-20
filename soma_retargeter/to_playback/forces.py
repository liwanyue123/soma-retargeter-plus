# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contact-force arrow helpers for TO playback."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
import warp as wp

from soma_retargeter.to_playback.robots import ContactSite
from soma_retargeter.utils.newton_utils import get_name_from_label


def build_contact_body_indices(
    model,
    robot_index: int,
    sites: Sequence[ContactSite],
) -> List[int]:
    """Resolve parent body indices for contact sites on one articulation."""
    labels = [get_name_from_label(l) for l in model.body_label]
    bodies_per_robot = model.body_count // max(1, model.articulation_count)
    base = int(robot_index) * int(bodies_per_robot)
    # Body labels are duplicated per articulation with the same names.
    name_to_local = {}
    for i in range(bodies_per_robot):
        name_to_local[labels[base + i]] = base + i
    indices = []
    for site in sites:
        if site.parent_body not in name_to_local:
            raise KeyError(
                f"Contact parent body '{site.parent_body}' not found on robot {robot_index}"
            )
        indices.append(name_to_local[site.parent_body])
    return indices


def contact_world_positions(
    body_q: np.ndarray,
    body_indices: Sequence[int],
    sites: Sequence[ContactSite],
    root_offset: np.ndarray | None = None,
) -> np.ndarray:
    """Transform contact locals into world using Newton ``body_q`` (N,7).

    ``body_q`` rows are ``[x,y,z, qx,qy,qz,qw]``. Optional ``root_offset`` is
    already baked into FK when robot offsets are applied via joint_q, so leave
    it None in the normal TO playback path.
    """
    out = np.zeros((len(sites), 3), dtype=np.float64)
    for i, (bi, site) in enumerate(zip(body_indices, sites)):
        tx = body_q[bi]
        p = tx[0:3]
        q = tx[3:7]  # xyzw
        # rotate local point
        # q * v * q^{-1}
        local = np.asarray(site.local_xyz, dtype=np.float64)
        # scipy-free: use warp via numpy quat rotate
        x, y, z = local
        qx, qy, qz, qw = q
        # standard quat rotate
        ix = qw * x + qy * z - qz * y
        iy = qw * y + qz * x - qx * z
        iz = qw * z + qx * y - qy * x
        iw = -qx * x - qy * y - qz * z
        rx = ix * qw + iw * -qx + iy * -qz - iz * -qy
        ry = iy * qw + iw * -qy + iz * -qx - ix * -qz
        rz = iz * qw + iw * -qz + ix * -qy - iy * -qx
        out[i] = p + np.array([rx, ry, rz])
        if root_offset is not None:
            out[i] = out[i] + root_offset
    return out


def _quat_align_z_to(direction: np.ndarray) -> np.ndarray:
    """Return xyzw quaternion rotating +Z onto ``direction``."""
    d = np.asarray(direction, dtype=np.float64)
    n = float(np.linalg.norm(d))
    if n < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    d = d / n
    z = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    c = float(np.dot(z, d))
    if c > 0.999999:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    if c < -0.999999:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    axis = np.cross(z, d)
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    ang = float(np.arccos(np.clip(c, -1.0, 1.0)))
    s = np.sin(ang * 0.5)
    return np.array(
        [axis[0] * s, axis[1] * s, axis[2] * s, np.cos(ang * 0.5)],
        dtype=np.float64)


def _ensure_force_arrow_meshes(viewer) -> tuple[str, str]:
    """Cache unit cylinder + cone meshes used as arrow shaft / head."""
    import newton

    cyl = "/to_playback/_arrow_cylinder"
    cone = "/to_playback/_arrow_cone"
    if not getattr(viewer, "_soma_force_arrow_meshes", None):
        # Unit meshes along +Z: radius=1, half_height=1 (total height 2 before instance scale).
        viewer.log_geo(cyl, newton.GeoType.CYLINDER, (1.0, 1.0), 0.0, True, hidden=True)
        viewer.log_geo(cone, newton.GeoType.CONE, (1.0, 1.0), 0.0, True, hidden=True)
        viewer._soma_force_arrow_meshes = (cyl, cone)
    return viewer._soma_force_arrow_meshes


def clear_contact_force_overlays(viewer) -> None:
    """Hide force arrows and clear legacy line/point overlays."""
    empty_v3 = wp.array(np.zeros((0, 3), dtype=np.float32), dtype=wp.vec3)
    empty_f = wp.array(np.zeros((0,), dtype=np.float32), dtype=wp.float32)
    if hasattr(viewer, "log_lines"):
        viewer.log_lines(
            "/to_playback/contact_forces", empty_v3, empty_v3, empty_v3, width=0.01)
    if hasattr(viewer, "log_points"):
        viewer.log_points(
            "/to_playback/contact_points", empty_v3, radii=empty_f, colors=empty_v3)
    meshes = getattr(viewer, "_soma_force_arrow_meshes", None)
    if meshes and hasattr(viewer, "log_instances"):
        cyl, cone = meshes
        viewer.log_instances(
            "/to_playback/contact_force_shafts", cyl, None, None, None, None, hidden=True)
        viewer.log_instances(
            "/to_playback/contact_force_heads", cone, None, None, None, None, hidden=True)


def draw_contact_forces(
    viewer,
    positions: np.ndarray,
    forces: np.ndarray,
    *,
    force_scale: float = 0.025,
    force_threshold: float = 0.01,
    color: Tuple[float, float, float] = (0.95, 0.12, 0.10),
    shaft_radius: float = 0.018,
    head_radius: float = 0.042,
    head_length_frac: float = 0.30,
    min_head_length: float = 0.045,
) -> None:
    """Draw red 3D force arrows (cylinder shaft + cone head).

    Arrow length = ``|F| * force_scale`` (metres).
    """
    positions = np.asarray(positions, dtype=np.float64)
    forces = np.asarray(forces, dtype=np.float64)
    if positions.ndim != 2 or forces.ndim != 2:
        return

    n = min(positions.shape[0], forces.shape[0])
    shaft_xforms, shaft_scales = [], []
    head_xforms, head_scales = [], []
    colors = []

    for i in range(n):
        f = forces[i]
        norm = float(np.linalg.norm(f))
        if norm < force_threshold:
            continue
        direction = f / norm
        length = max(norm * float(force_scale), 1e-4)
        head_len = max(min_head_length, length * float(head_length_frac))
        head_len = min(head_len, length * 0.55)
        shaft_len = max(length - head_len, 1e-4)

        quat = _quat_align_z_to(direction)
        shaft_mid = positions[i] + direction * (0.5 * shaft_len)
        tip = positions[i] + direction * length
        # Cone spans [-h/2, +h/2] with apex at +Z → center so apex = tip.
        head_mid = tip - direction * (0.5 * head_len)

        shaft_xforms.append([*shaft_mid, *quat])
        # Unit mesh half_height=1 → instance Z scale is half the desired length.
        shaft_scales.append([shaft_radius, shaft_radius, 0.5 * shaft_len])
        head_xforms.append([*head_mid, *quat])
        head_scales.append([head_radius, head_radius, 0.5 * head_len])
        colors.append(color)

    cyl_mesh, cone_mesh = _ensure_force_arrow_meshes(viewer)
    mat = (0.35, 0.0, 0.0, 0.0)

    if shaft_xforms:
        n_arr = len(shaft_xforms)
        cols = wp.array(np.asarray(colors, dtype=np.float32), dtype=wp.vec3)
        mats = wp.array(
            np.tile(np.asarray(mat, dtype=np.float32), (n_arr, 1)), dtype=wp.vec4)
        viewer.log_instances(
            "/to_playback/contact_force_shafts",
            cyl_mesh,
            wp.array(np.asarray(shaft_xforms, dtype=np.float32), dtype=wp.transform),
            wp.array(np.asarray(shaft_scales, dtype=np.float32), dtype=wp.vec3),
            cols,
            mats,
            hidden=False,
        )
        viewer.log_instances(
            "/to_playback/contact_force_heads",
            cone_mesh,
            wp.array(np.asarray(head_xforms, dtype=np.float32), dtype=wp.transform),
            wp.array(np.asarray(head_scales, dtype=np.float32), dtype=wp.vec3),
            cols,
            mats,
            hidden=False,
        )
    else:
        clear_contact_force_overlays(viewer)


# Soft daylight studio look (sky gradient + ambient). Newton has no mesh alpha;
# origin "transparency" is approximated by blending mesh colors toward a pale tint.
_STUDIO_SKY_UPPER = (0.72, 0.82, 0.92)
_STUDIO_SKY_LOWER = (0.48, 0.52, 0.56)
_STUDIO_LIGHT = (1.00, 0.98, 0.95)
_STUDIO_GROUND = (0.28, 0.29, 0.31)
_GHOST_TINT = (0.88, 0.90, 0.93)


def _lerp_rgb(
    a: Tuple[float, float, float],
    b: Tuple[float, float, float],
    t: float,
) -> Tuple[float, float, float]:
    t = float(np.clip(t, 0.0, 1.0))
    return (a[0] + (b[0] - a[0]) * t,
            a[1] + (b[1] - a[1]) * t,
            a[2] + (b[2] - a[2]) * t)


def snapshot_shape_colors(viewer) -> dict:
    """Capture current per-shape RGB colors from the Newton viewer."""
    slot_map = getattr(viewer, "_shape_to_slot", None)
    color_arr = getattr(viewer, "model_shape_color", None)
    if slot_map is None or color_arr is None:
        return {}
    arr = color_arr.numpy()
    out = {}
    for s_idx, slot in enumerate(slot_map):
        if int(slot) < 0:
            continue
        c = arr[int(slot)]
        out[int(s_idx)] = (float(c[0]), float(c[1]), float(c[2]))
    return out


def apply_studio_environment(viewer) -> None:
    """Softer sky / ground / light for TO playback."""
    renderer = getattr(viewer, "renderer", None)
    if renderer is None:
        return
    renderer.draw_sky = True
    if hasattr(renderer, "draw_shadows"):
        renderer.draw_shadows = True
    renderer.sky_upper = _STUDIO_SKY_UPPER
    renderer.sky_lower = _STUDIO_SKY_LOWER
    renderer.background_color = _STUDIO_SKY_UPPER
    if hasattr(renderer, "_light_color"):
        renderer._light_color = _STUDIO_LIGHT


def snapshot_shape_materials(viewer) -> dict:
    """Capture base material vec4s keyed by model shape index."""
    batches = getattr(viewer, "_shape_instances", None)
    if not batches:
        return {}
    out = {}
    for shapes in batches.values():
        mats = shapes.materials.numpy()
        for local_i, s_idx in enumerate(shapes.model_shapes):
            out[int(s_idx)] = tuple(float(x) for x in mats[local_i])
    return out


def apply_to_playback_appearance(
    viewer,
    model,
    *,
    origin_opacity: float = 0.38,
    original_colors: dict | None = None,
    original_materials: dict | None = None,
    tint_ground: bool = True,
    use_mesh_alpha: bool = True,
) -> tuple[dict, dict]:
    """Style dual robots: keep mesh hues; fade origin with real alpha if patched.

    Returns ``(original_colors, original_materials)`` for later restore.

    Args:
        origin_opacity: 1 = solid origin, 0 = invisible. Typical ghost ~0.35–0.45.
    """
    from soma_retargeter.to_playback import newton_alpha

    if original_colors is None:
        original_colors = snapshot_shape_colors(viewer)
    if original_materials is None:
        original_materials = snapshot_shape_materials(viewer)

    shape_body = model.shape_body.numpy()
    articulations = max(1, int(model.articulation_count))
    bodies_per = model.body_count // articulations

    # Ground tint (RGB only).
    if tint_ground and original_colors and hasattr(viewer, "update_shape_colors"):
        ground_colors = {}
        for si, bi in enumerate(shape_body):
            if bi < 0 and si in original_colors:
                ground_colors[si] = _STUDIO_GROUND
            elif si in original_colors:
                ground_colors[si] = original_colors[si]
        if ground_colors:
            viewer.update_shape_colors(ground_colors)

    alpha_ok = use_mesh_alpha and newton_alpha.is_mesh_alpha_enabled()
    if alpha_ok and articulations >= 2:
        alphas = {}
        for si, bi in enumerate(shape_body):
            if bi < 0:
                continue  # keep ground materials (checkerboard) untouched
            if bi < bodies_per:
                alphas[si] = float(np.clip(origin_opacity, 0.0, 1.0))
            # TO robot (bi >= bodies_per): leave materials alone (fully opaque)
        newton_alpha.set_shape_alphas(
            viewer, alphas, base_materials=original_materials)
    elif articulations >= 2 and original_colors and hasattr(viewer, "update_shape_colors"):
        # Fallback: pale ghost tint when alpha patch is unavailable.
        ghost = 1.0 - float(np.clip(origin_opacity, 0.0, 1.0))
        colors = {}
        for si, bi in enumerate(shape_body):
            if si not in original_colors:
                continue
            base = original_colors[si]
            if bi < 0:
                colors[si] = _STUDIO_GROUND if tint_ground else base
            elif bi < bodies_per:
                colors[si] = _lerp_rgb(base, _GHOST_TINT, ghost)
            else:
                colors[si] = base
        viewer.update_shape_colors(colors)

    return original_colors, original_materials


def restore_shape_appearance(
    viewer,
    original_colors: dict | None,
    original_materials: dict | None = None,
) -> None:
    restore_shape_colors(viewer, original_colors)
    if original_materials:
        from soma_retargeter.to_playback import newton_alpha
        if newton_alpha.is_mesh_alpha_enabled():
            # Restore opaque materials (positive z).
            import warp as wp
            batches = getattr(viewer, "_shape_instances", None) or {}
            for shapes in batches.values():
                mats = shapes.materials.numpy().copy()
                changed = False
                for local_i, s_idx in enumerate(shapes.model_shapes):
                    s_idx = int(s_idx)
                    if s_idx in original_materials:
                        mats[local_i] = original_materials[s_idx]
                        changed = True
                if changed:
                    shapes.materials = wp.array(
                        mats, dtype=wp.vec4, device=shapes.device)
                    shapes.materials_changed = True


def restore_shape_colors(viewer, original_colors: dict | None) -> None:
    if not original_colors or not hasattr(viewer, "update_shape_colors"):
        return
    viewer.update_shape_colors(original_colors)


# Back-compat alias used by earlier code paths.
def apply_dual_robot_shape_colors(viewer, model, **_kwargs):
    apply_studio_environment(viewer)
    return apply_to_playback_appearance(viewer, model)


def default_layout_offsets(
    mode: str,
    separation: float = 1.2,
) -> List[wp.transform]:
    """Return two root offsets for origin/to robots.

    Default ``overlay`` keeps both robots at the same origin (ghost + TO
    coincident); ``side_by_side`` separates them along world +Y.
    """
    mode = (mode or "overlay").lower()
    if mode == "overlay":
        return [wp.transform_identity(), wp.transform_identity()]
    # Side-by-side along world +Y (viewer ground plane is XY-ish with Z up).
    half = 0.5 * float(separation)
    return [
        wp.transform(wp.vec3(0.0, -half, 0.0), wp.quat_identity()),
        wp.transform(wp.vec3(0.0, half, 0.0), wp.quat_identity()),
    ]

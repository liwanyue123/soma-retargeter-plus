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
        _hide_reflected_instancer(viewer, "/to_playback/floor_reflect/contact_force_shafts")
        _hide_reflected_instancer(viewer, "/to_playback/floor_reflect/contact_force_heads")


def _quat_to_mat33(q: np.ndarray) -> np.ndarray:
    """``q`` is (..., 4) xyzw → (..., 3, 3) rotation matrix."""
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    r00 = 1.0 - 2.0 * (yy + zz)
    r01 = 2.0 * (xy - wz)
    r02 = 2.0 * (xz + wy)
    r10 = 2.0 * (xy + wz)
    r11 = 1.0 - 2.0 * (xx + zz)
    r12 = 2.0 * (yz - wx)
    r20 = 2.0 * (xz - wy)
    r21 = 2.0 * (yz + wx)
    r22 = 1.0 - 2.0 * (xx + yy)
    return np.stack(
        [
            np.stack([r00, r01, r02], axis=-1),
            np.stack([r10, r11, r12], axis=-1),
            np.stack([r20, r21, r22], axis=-1),
        ],
        axis=-2,
    )


# Plane z=0 reflection matrix (row-major 4×4, matches Newton VBO layout).
_REFLECT_PLANE = np.diag([1.0, 1.0, -1.0, 1.0]).astype(np.float32)


def _mat44_batch_from_xform_scale(xforms: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Build (N,4,4) instance matrices matching ``_compute_shape_vbo_xforms``."""
    xforms = np.asarray(xforms, dtype=np.float32)
    scales = np.asarray(scales, dtype=np.float32)
    r = _quat_to_mat33(xforms[:, 3:7])
    n = xforms.shape[0]
    m = np.zeros((n, 4, 4), dtype=np.float32)
    m[:, 3, 3] = 1.0
    m[:, 0, 0] = r[:, 0, 0] * scales[:, 0]
    m[:, 1, 0] = r[:, 1, 0] * scales[:, 0]
    m[:, 2, 0] = r[:, 2, 0] * scales[:, 0]
    m[:, 0, 1] = r[:, 0, 1] * scales[:, 1]
    m[:, 1, 1] = r[:, 1, 1] * scales[:, 1]
    m[:, 2, 1] = r[:, 2, 1] * scales[:, 1]
    m[:, 0, 2] = r[:, 0, 2] * scales[:, 2]
    m[:, 1, 2] = r[:, 1, 2] * scales[:, 2]
    m[:, 2, 2] = r[:, 2, 2] * scales[:, 2]
    m[:, 3, 0] = xforms[:, 0]
    m[:, 3, 1] = xforms[:, 1]
    m[:, 3, 2] = xforms[:, 2]
    return m


def _reflect_mat44_batch(m: np.ndarray) -> np.ndarray:
    """Apply planar reflection to Newton VBO ``(N,4,4)`` matrices.

    Newton stores ``M^T`` (translation in row 3). OpenGL reads the same bytes as
    column-major ``M``. Reflection is ``M' = Reflect @ M``, so the stored array
    must be updated as ``M^T @ Reflect`` (right multiply), not left.
    """
    return np.matmul(m, _REFLECT_PLANE)


_REFLECT_ALPHA_MAX = 0.44
_REFLECT_ALPHA_MIN = 0.08


def _reflect_alpha_for_height(z: float) -> tuple[float, float]:
    """Return (alpha, roughness) for a mirror instance at world height ``z``.

    Near the floor (feet) → sharper/brighter; head height → softer/faded.
    Even at contact, alpha stays below 1 so reflections never read as solid white.
    """
    h = float(max(z, 0.0))
    t = float(np.clip((h - 0.06) / 1.05, 0.0, 1.0))
    # smoothstep-ish without importing extra
    t = t * t * (3.0 - 2.0 * t)
    alpha = (_REFLECT_ALPHA_MAX - _REFLECT_ALPHA_MIN) * (1.0 - t) + _REFLECT_ALPHA_MIN
    rough = 0.22 + 0.62 * t
    return alpha, rough


def _iter_shape_vbo_batches(viewer):
    """Yield ``(shapes, mat44_np, xforms_np)`` for live instances.

    Prefer the packed VBO mat44 buffer (CUDA path). ``xforms_np`` is world
    transform (N,7) for height-based reflection fade.
    """
    packed_groups = getattr(viewer, "_packed_groups", None)
    host_m = getattr(viewer, "_packed_vbo_xforms_host", None)
    packed_x = getattr(viewer, "_packed_world_xforms", None)
    if packed_groups and host_m is not None:
        all_m = host_m.numpy()
        all_x = packed_x.numpy() if packed_x is not None else None
        for _key, shapes, offset, count in packed_groups:
            x_np = all_x[offset : offset + count] if all_x is not None else None
            yield shapes, all_m[offset : offset + count], x_np
        return

    batches = getattr(viewer, "_shape_instances", None) or {}
    for shapes in batches.values():
        xforms = getattr(shapes, "world_xforms", None)
        scales = getattr(shapes, "scales", None)
        if xforms is None or scales is None:
            continue
        x_np = xforms.numpy()
        s_np = scales.numpy()
        n = min(len(x_np), len(s_np))
        if n <= 0:
            continue
        yield shapes, _mat44_batch_from_xform_scale(x_np[:n], s_np[:n]), x_np[:n]


def _iter_shape_world_batches(viewer):
    """Yield ``(shapes, xforms_np)`` with live world transforms.

    ViewerGL's CUDA packed path updates ``_packed_world_xforms`` but does
    **not** refresh ``shapes.world_xforms`` for meshes — reading the latter
    yields zeros (no reflections / origin ghosts). Prefer the packed buffer.
    """
    packed_groups = getattr(viewer, "_packed_groups", None)
    packed_xforms = getattr(viewer, "_packed_world_xforms", None)
    if packed_groups and packed_xforms is not None:
        all_x = packed_xforms.numpy()
        for _key, shapes, offset, count in packed_groups:
            yield shapes, all_x[offset : offset + count]
        return

    batches = getattr(viewer, "_shape_instances", None) or {}
    for shapes in batches.values():
        xforms = getattr(shapes, "world_xforms", None)
        if xforms is None:
            continue
        yield shapes, xforms.numpy()


def clear_floor_reflections(viewer) -> None:
    """Hide planar floor-mirror / catcher instances."""
    objects = getattr(viewer, "objects", None) or {}
    prefixes = ("/to_playback/floor_reflect/", "/to_playback/floor_catcher")
    for name in list(objects.keys()):
        if isinstance(name, str) and name.startswith(prefixes):
            obj = objects[name]
            if hasattr(obj, "hidden"):
                obj.hidden = True
            if hasattr(obj, "active_instances"):
                obj.active_instances = 0


def draw_floor_reflections(viewer, model) -> None:
    """Draw floor reflections via exact ``M^T @ Reflect`` on packed VBO mat44.

    Quaternion + scale tricks cannot represent a true mirror (det=-1); they
    flip feet/backward and shatter the body. We reuse the same mat44 Newton
    uploads for the robot, right-multiply by ``diag(1,1,-1,1)`` (Newton stores
    ``M^T``), and push with ``update_from_pinned``.
    """
    import newton as nt

    from soma_retargeter.to_playback import newton_alpha

    if not hasattr(viewer, "log_instances"):
        return
    if not newton_alpha.is_mesh_alpha_enabled():
        return

    device = getattr(viewer, "device", "cpu")
    shape_body = model.shape_body.numpy()

    mesh_types = {int(nt.GeoType.MESH)}
    if hasattr(nt.GeoType, "CONVEX_MESH"):
        mesh_types.add(int(nt.GeoType.CONVEX_MESH))

    batches = list(_iter_shape_vbo_batches(viewer))
    if not batches:
        return

    horiz = []
    for shapes, _m, x_np in batches:
        if x_np is None or int(shapes.geo_type) not in mesh_types:
            continue
        model_shapes = np.asarray(shapes.model_shapes, dtype=np.int32)
        for i in range(len(x_np)):
            s_idx = int(model_shapes[i]) if i < len(model_shapes) else -1
            if s_idx < 0 or s_idx >= len(shape_body) or int(shape_body[s_idx]) < 0:
                continue
            if float(x_np[i, 2]) > 0.25:
                horiz.append(float(np.linalg.norm(x_np[i, 0:2])))
    robots_away = bool(horiz) and float(np.median(horiz)) > 0.45

    mirrored = 0
    for shapes, m_np, x_np in batches:
        if int(shapes.geo_type) == int(nt.GeoType.PLANE):
            obj = getattr(viewer, "objects", {}).get(shapes.name)
            if obj is not None:
                obj._soma_pass = "floor"
            continue

        if int(shapes.geo_type) not in mesh_types:
            continue

        main_obj = getattr(viewer, "objects", {}).get(shapes.name)
        if main_obj is not None and getattr(main_obj, "hidden", False):
            continue

        colors = getattr(shapes, "colors", None)
        n = int(m_np.shape[0])
        if n <= 0 or x_np is None or len(x_np) < n:
            continue

        model_shapes = np.asarray(shapes.model_shapes, dtype=np.int32)
        keep = []
        for i in range(n):
            s_idx = int(model_shapes[i]) if i < len(model_shapes) else -1
            if s_idx < 0 or s_idx >= len(shape_body):
                continue
            if int(shape_body[s_idx]) < 0:
                continue
            if float(x_np[i, 2]) < 0.05:
                continue
            if robots_away and float(np.linalg.norm(x_np[i, 0:2])) < 0.25:
                continue
            keep.append(i)

        mirror_name = f"/to_playback/floor_reflect/{shapes.name}"
        if not keep:
            obj = getattr(viewer, "objects", {}).get(mirror_name)
            if obj is not None:
                obj.hidden = True
                obj.active_instances = 0
            continue

        idx = np.asarray(keep, dtype=np.int32)
        m_ref = _reflect_mat44_batch(m_np[idx]).astype(np.float32, copy=False)

        c_m = None
        if colors is not None:
            c_m = colors.numpy()[idx].copy()
            for j, ki in enumerate(idx):
                z_orig = float(x_np[ki, 2])
                a_h, _ = _reflect_alpha_for_height(z_orig)
                c_m[j] *= 0.40 + 0.35 * (a_h / _REFLECT_ALPHA_MAX)

        m_m = np.zeros((len(keep), 4), dtype=np.float32)
        for j, ki in enumerate(idx):
            z_orig = float(x_np[ki, 2])
            a_h, rough = _reflect_alpha_for_height(z_orig)
            m_m[j] = newton_alpha.encode_material_alpha((rough, 0.05, 0.0, 0.0), a_h)

        instancer = getattr(viewer, "objects", {}).get(mirror_name)
        if instancer is None or len(keep) > instancer.num_instances:
            cap = max(len(keep), 1)
            dummy_x = wp.zeros(cap, dtype=wp.transform, device=device)
            dummy_s = wp.ones(cap, dtype=wp.vec3, device=device)
            viewer.log_instances(
                mirror_name,
                shapes.mesh,
                dummy_x,
                dummy_s,
                None,
                None,
                hidden=False,
            )
            instancer = viewer.objects[mirror_name]

        instancer.hidden = False
        instancer._soma_pass = "reflect"
        instancer.update_from_pinned(
            np.ascontiguousarray(m_ref),
            len(keep),
            wp.array(c_m, dtype=wp.vec3, device=device) if c_m is not None else None,
            wp.array(m_m, dtype=wp.vec4, device=device),
        )
        mirrored += len(keep)

    if not getattr(viewer, "_soma_reflect_logged", False):
        viewer._soma_reflect_logged = True
        src = "mat44" if getattr(viewer, "_packed_vbo_xforms_host", None) is not None else "legacy"
        print(f"[INFO]: Floor reflections: {mirrored} mesh instances ({src} VBO)")


def _hide_reflected_instancer(viewer, mirror_name: str) -> None:
    obj = getattr(viewer, "objects", {}).get(mirror_name)
    if obj is not None:
        obj.hidden = True
        obj.active_instances = 0


def _upload_reflected_mat44_batch(
    viewer,
    mirror_name: str,
    mesh_name: str,
    m_ref: np.ndarray,
    *,
    colors: np.ndarray | None,
    alphas: np.ndarray | None,
) -> None:
    """Upload a reflected mat44 batch tagged for the floor mirror draw pass."""
    from soma_retargeter.to_playback import newton_alpha

    n = int(m_ref.shape[0])
    if n <= 0:
        _hide_reflected_instancer(viewer, mirror_name)
        return

    device = getattr(viewer, "device", "cpu")
    if alphas is None:
        alphas = np.full(n, _REFLECT_ALPHA_MAX, dtype=np.float32)

    m_m = np.zeros((n, 4), dtype=np.float32)
    for j in range(n):
        a_h, rough = _reflect_alpha_for_height(float(alphas[j]))
        m_m[j] = newton_alpha.encode_material_alpha((rough, 0.05, 0.0, 0.0), a_h)

    instancer = getattr(viewer, "objects", {}).get(mirror_name)
    if instancer is None or n > instancer.num_instances:
        cap = max(n, 1)
        dummy_x = wp.zeros(cap, dtype=wp.transform, device=device)
        dummy_s = wp.ones(cap, dtype=wp.vec3, device=device)
        viewer.log_instances(
            mirror_name,
            mesh_name,
            dummy_x,
            dummy_s,
            None,
            None,
            hidden=False,
        )
        instancer = viewer.objects[mirror_name]

    instancer.hidden = False
    instancer._soma_pass = "reflect"
    instancer.update_from_pinned(
        np.ascontiguousarray(m_ref.astype(np.float32, copy=False)),
        n,
        wp.array(colors, dtype=wp.vec3, device=device) if colors is not None else None,
        wp.array(m_m, dtype=wp.vec4, device=device),
    )


def draw_contact_force_reflections(
    viewer,
    *,
    color: Tuple[float, float, float] = (0.95, 0.12, 0.10),
) -> None:
    """Mirror contact-force arrows across the floor (call after :func:`draw_contact_forces`)."""
    from soma_retargeter.to_playback import newton_alpha

    if not newton_alpha.is_mesh_alpha_enabled():
        return

    cyl_mesh, cone_mesh = _ensure_force_arrow_meshes(viewer)
    pairs = (
        ("/to_playback/contact_force_shafts", "/to_playback/floor_reflect/contact_force_shafts", cyl_mesh),
        ("/to_playback/contact_force_heads", "/to_playback/floor_reflect/contact_force_heads", cone_mesh),
    )
    for src_name, mirror_name, mesh_name in pairs:
        src = getattr(viewer, "objects", {}).get(src_name)
        if src is None or getattr(src, "hidden", True) or int(src.active_instances) <= 0:
            _hide_reflected_instancer(viewer, mirror_name)
            continue

        n = int(src.active_instances)
        m_np = src.world_xforms.numpy()[:n]
        m_ref = _reflect_mat44_batch(m_np)

        cols = np.tile(np.asarray(color, dtype=np.float32), (n, 1))
        cols *= 0.38 + 0.28 * (_REFLECT_ALPHA_MAX / max(_REFLECT_ALPHA_MAX, 1e-6))

        z_heights = np.maximum(m_np[:, 3, 2], 0.0).astype(np.float32)
        _upload_reflected_mat44_batch(
            viewer,
            mirror_name,
            mesh_name,
            m_ref,
            colors=cols,
            alphas=z_heights,
        )


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
    ``shaft_radius`` / ``head_radius`` control arrow thickness (metres).
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
    mat = (_STUDIO_ROUGHNESS, 0.0, 0.0, 0.0)

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


# User palette (RGB 0–255 → 0–1): light / sky / ground.
def _rgb255(r: int, g: int, b: int) -> Tuple[float, float, float]:
    return (r / 255.0, g / 255.0, b / 255.0)


_STUDIO_LIGHT = _rgb255(226, 197, 141)
# Newton UI "Sky Color" → sky_upper (zenith).  "Ground Color" → sky_lower (fog/horizon).
_STUDIO_SKY_UPPER = _rgb255(52, 42, 42)
_STUDIO_GROUND = _rgb255(63, 73, 81)
_STUDIO_SKY_LOWER = _STUDIO_GROUND
_GHOST_TINT = (0.78, 0.80, 0.82)
_STUDIO_ROUGHNESS = 0.78
_STUDIO_METALLIC = 0.0
_STUDIO_FLOOR_ROUGHNESS = 0.28
_STUDIO_FLOOR_METALLIC = 0.22
_STUDIO_REFLECT_ALPHA = _REFLECT_ALPHA_MAX
_STUDIO_SPECULAR = 0.40
_STUDIO_DIFFUSE = 1.0
_STUDIO_SHADOW_RADIUS = 16.0
_STUDIO_SHADOW_EXTENTS = 8.0
_STUDIO_GROUND_GRID = 2.0
_STUDIO_SUN_Z_UP = (0.12, -0.18, 0.98)


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


def robot_root_xy(state, model, robot_index: int = 1) -> Tuple[float, float]:
    """World horizontal coords of an articulation root after FK."""
    articulations = max(1, int(model.articulation_count))
    bodies_per = model.body_count // articulations
    root_bi = int(robot_index) * int(bodies_per)
    tx = state.body_q.numpy()[root_bi]
    return horizontal_plane_xy(tx[0:3], model.up_axis)


def horizontal_plane_xy(
    pos3: Sequence[float],
    up_axis: str | int,
) -> Tuple[float, float]:
    """Map a world position to 2D floor coordinates for the model up-axis."""
    axis = str(up_axis).upper()
    if axis in ("Y", "1"):
        return float(pos3[0]), float(pos3[2])
    if axis in ("X", "0"):
        return float(pos3[1]), float(pos3[2])
    return float(pos3[0]), float(pos3[1])


def set_studio_grid_origin(viewer, origin_xy: Tuple[float, float]) -> None:
    """Center the floor grid vignette on ``origin_xy`` (robot first-frame root)."""
    from soma_retargeter.to_playback import newton_alpha

    xy = (float(origin_xy[0]), float(origin_xy[1]))
    renderer = getattr(viewer, "renderer", None)
    if renderer is not None:
        renderer._soma_grid_origin = xy
    viewer._soma_studio_grid_origin = xy
    newton_alpha.set_grid_origin(xy)
    if not getattr(viewer, "_soma_grid_origin_logged", False):
        viewer._soma_grid_origin_logged = True
        print(f"[INFO]: Studio grid origin (TO frame 0): ({xy[0]:.3f}, {xy[1]:.3f})")


def set_studio_grid_origin_from_joint_q(
    viewer,
    model,
    q: Sequence[float],
    *,
    robot_index: int = 1,
    robot_offset=None,
) -> None:
    """Set grid origin from a root joint_q sample (before/after FK)."""
    import warp as wp

    q = np.asarray(q, dtype=np.float64)
    if robot_offset is not None:
        root_tx = wp.mul(robot_offset, wp.transform(*q[:7].astype(np.float32)))
        pos = np.asarray(root_tx[0:3], dtype=np.float64)
    else:
        pos = q[0:3]
    set_studio_grid_origin(viewer, horizontal_plane_xy(pos, model.up_axis))


def update_studio_grid_origin_from_state(
    viewer,
    state,
    model,
    *,
    robot_index: int = 1,
) -> None:
    set_studio_grid_origin(viewer, robot_root_xy(state, model, robot_index))


def apply_studio_environment(viewer) -> None:
    """Push Newton GL toward a dark matte studio look (soft key, low specular)."""
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
    if hasattr(renderer, "specular_scale"):
        renderer.specular_scale = _STUDIO_SPECULAR
    if hasattr(renderer, "diffuse_scale"):
        renderer.diffuse_scale = _STUDIO_DIFFUSE
    if hasattr(renderer, "shadow_radius"):
        renderer.shadow_radius = _STUDIO_SHADOW_RADIUS
    if hasattr(renderer, "shadow_extents"):
        renderer.shadow_extents = _STUDIO_SHADOW_EXTENTS
    if hasattr(renderer, "spotlight_enabled"):
        # Uniform fill instead of camera-cone spotlight falloff.
        renderer.spotlight_enabled = False
    if hasattr(renderer, "_env_intensity"):
        renderer._env_intensity = 0.18
    # Prefer a soft overhead key (Z-up). Setting early skips Newton's lazy default.
    if hasattr(renderer, "_sun_direction"):
        d = np.asarray(_STUDIO_SUN_Z_UP, dtype=np.float64)
        d = d / (np.linalg.norm(d) + 1e-12)
        renderer._sun_direction = d

    # Soften self-shadow mottling on dense CAD shells (looks like surface damage).
    # Prefer the early argv patch; this is a no-op if already applied.
    try:
        from soma_retargeter.to_playback import newton_alpha
        newton_alpha.enable_soft_shadow_bias()
    except Exception:
        pass


def _studio_material_from_base(
    base_xyzw: Sequence[float],
    *,
    checker: float = 0.0,
) -> Tuple[float, float, float, float]:
    """Keep texture flag; force matte roughness / no metal / no checker."""
    w = float(base_xyzw[3]) if len(base_xyzw) > 3 else 0.0
    return (_STUDIO_ROUGHNESS, _STUDIO_METALLIC, float(checker), w)


def apply_studio_shape_materials(
    viewer,
    model,
    *,
    original_materials: dict,
    origin_opacity: float = 0.38,
    use_mesh_alpha: bool = True,
) -> None:
    """Matte materials on robots; dark Isaac-style grid on the ground plane."""
    import warp as wp

    from soma_retargeter.to_playback import newton_alpha

    batches = getattr(viewer, "_shape_instances", None)
    if not batches or not original_materials:
        return

    shape_body = model.shape_body.numpy()
    articulations = max(1, int(model.articulation_count))
    bodies_per = model.body_count // articulations
    alpha_ok = use_mesh_alpha and newton_alpha.is_mesh_alpha_enabled()
    origin_a = float(np.clip(origin_opacity, 0.0, 1.0))

    for shapes in batches.values():
        mats = shapes.materials.numpy().copy()
        changed = False
        for local_i, s_idx in enumerate(shapes.model_shapes):
            s_idx = int(s_idx)
            if s_idx not in original_materials:
                continue
            base = original_materials[s_idx]
            bi = int(shape_body[s_idx]) if s_idx < len(shape_body) else -1
            if bi < 0:
                # Hard metal floor + grid (Material.z = density >= 2).
                mats[local_i] = (
                    _STUDIO_FLOOR_ROUGHNESS,
                    _STUDIO_FLOOR_METALLIC,
                    float(_STUDIO_GROUND_GRID),
                    0.0,
                )
            elif bi < bodies_per and articulations >= 2 and alpha_ok:
                m = _studio_material_from_base(base, checker=0.0)
                mats[local_i] = newton_alpha.encode_material_alpha(m, origin_a)
            else:
                mats[local_i] = _studio_material_from_base(base, checker=0.0)
            changed = True
        if changed:
            shapes.materials = wp.array(mats, dtype=wp.vec4, device=shapes.device)
            shapes.materials_changed = True
        # Tag ground instancer for translucent draw-order pass.
        try:
            import newton as nt
            if int(shapes.geo_type) == int(nt.GeoType.PLANE):
                obj = getattr(viewer, "objects", {}).get(shapes.name)
                if obj is not None:
                    obj._soma_pass = "floor"
        except Exception:
            pass


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

    # Dark floor base color (grid lines drawn by isaac-grid shader patch).
    if tint_ground and original_colors and hasattr(viewer, "update_shape_colors"):
        ground_colors = {}
        for si, bi in enumerate(shape_body):
            if bi < 0 and si in original_colors:
                ground_colors[si] = _STUDIO_GROUND
            elif si in original_colors:
                ground_colors[si] = original_colors[si]
        if ground_colors:
            viewer.update_shape_colors(ground_colors)
    elif original_colors and hasattr(viewer, "update_shape_colors"):
        ground_colors = {}
        for si, bi in enumerate(shape_body):
            if bi < 0 and si in original_colors:
                ground_colors[si] = original_colors[si]
        if ground_colors:
            viewer.update_shape_colors(ground_colors)

    # Matte materials (high roughness / no metal / no checker). Origin alpha
    # is encoded into Material.z when the GL patch is active.
    apply_studio_shape_materials(
        viewer,
        model,
        original_materials=original_materials,
        origin_opacity=origin_opacity,
        use_mesh_alpha=use_mesh_alpha,
    )

    alpha_ok = use_mesh_alpha and newton_alpha.is_mesh_alpha_enabled()
    if (not alpha_ok) and articulations >= 2 and original_colors and hasattr(
            viewer, "update_shape_colors"):
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

    Overlay applies a tiny ±Y epsilon so coincident translucent meshes do not
    z-fight (looks like mottled / torn surface damage in the GL viewer).
    """
    mode = (mode or "overlay").lower()
    if mode == "overlay":
        # ~2 mm total — visually “same origin”, enough to stop depth fighting.
        eps = 0.001
        return [
            wp.transform(wp.vec3(0.0, -eps, 0.0), wp.quat_identity()),
            wp.transform(wp.vec3(0.0, eps, 0.0), wp.quat_identity()),
        ]
    # Side-by-side along world +Y (viewer ground plane is XY-ish with Z up).
    half = 0.5 * float(separation)
    return [
        wp.transform(wp.vec3(0.0, -half, 0.0), wp.quat_identity()),
        wp.transform(wp.vec3(0.0, half, 0.0), wp.quat_identity()),
    ]

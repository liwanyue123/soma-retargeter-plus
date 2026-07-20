# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime patch: add per-instance mesh alpha to Newton ViewerGL.

Stock Newton GL hardcodes ``FragColor = vec4(color, 1.0)`` and stores shape
colors as RGB only. This module:

1. Patches the shape fragment shader so ``Material.z < 0`` means
   ``alpha = -Material.z`` (checker disabled in that mode).
2. Enables ``GL_BLEND`` during the shape pass.
3. Makes ``log_state`` re-upload materials when ``materials_changed`` is set.

Call :func:`enable_mesh_alpha` **before** ``newton.examples.init()`` / ViewerGL
construction so the patched shader source is compiled.
"""

from __future__ import annotations

_ENABLED = False
_MARKER = "SOMA_MESH_ALPHA"


def enable_mesh_alpha() -> bool:
    """Install Newton GL mesh-alpha support. Safe to call multiple times."""
    global _ENABLED
    if _ENABLED:
        return True

    import newton._src.viewer.gl.shaders as shaders
    from newton._src.viewer.gl.opengl import RendererGL
    from newton._src.viewer.viewer import ViewerBase

    frag = shaders.shape_fragment_shader
    if _MARKER not in frag:
        old_mat = (
            "    float roughness = clamp(Material.x, 0.0, 1.0);\n"
            "    float metallic = clamp(Material.y, 0.0, 1.0);\n"
            "    float checker_enable = Material.z;\n"
            "    float texture_enable = Material.w;"
        )
        new_mat = (
            f"    // {_MARKER}\n"
            "    float roughness = clamp(Material.x, 0.0, 1.0);\n"
            "    float metallic = clamp(Material.y, 0.0, 1.0);\n"
            "    float checker_enable = (Material.z < 0.0) ? 0.0 : Material.z;\n"
            "    float texture_enable = Material.w;\n"
            "    float soma_alpha = (Material.z < 0.0) "
            "? clamp(-Material.z, 0.0, 1.0) : 1.0;"
        )
        if old_mat not in frag:
            raise RuntimeError(
                "Newton shape_fragment_shader material block not found; "
                "cannot patch mesh alpha."
            )
        frag = frag.replace(old_mat, new_mat)
        if "FragColor = vec4(color, 1.0);" not in frag:
            raise RuntimeError(
                "Newton shape_fragment_shader FragColor line not found."
            )
        frag = frag.replace(
            "FragColor = vec4(color, 1.0);",
            f"FragColor = vec4(color, soma_alpha); // {_MARKER}",
        )
        shaders.shape_fragment_shader = frag

    if not getattr(RendererGL._render_scene, "_soma_alpha_patched", False):
        _orig_render_scene = RendererGL._render_scene

        def _render_scene_alpha(self, objects):
            gl = RendererGL.gl
            gl.glEnable(gl.GL_BLEND)
            gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
            try:
                return _orig_render_scene(self, objects)
            finally:
                gl.glDisable(gl.GL_BLEND)

        _render_scene_alpha._soma_alpha_patched = True
        RendererGL._render_scene = _render_scene_alpha

    if not getattr(ViewerBase.log_state, "_soma_alpha_patched", False):
        _orig_log_state = ViewerBase.log_state

        def _log_state_alpha(self, state):
            if self.model is None:
                return _orig_log_state(self, state)
            need_mats = any(
                getattr(shapes, "materials_changed", False)
                for shapes in getattr(self, "_shape_instances", {}).values()
            )
            if need_mats:
                self.model_changed = True
            try:
                return _orig_log_state(self, state)
            finally:
                for shapes in getattr(self, "_shape_instances", {}).values():
                    shapes.materials_changed = False

        _log_state_alpha._soma_alpha_patched = True
        ViewerBase.log_state = _log_state_alpha

    _ENABLED = True
    return True


def is_mesh_alpha_enabled() -> bool:
    return _ENABLED


def encode_material_alpha(material_xyzw, alpha: float):
    """Return material vec4 with alpha encoded in negative ``z``."""
    import numpy as np

    m = np.asarray(material_xyzw, dtype=np.float32).reshape(4).copy()
    a = float(np.clip(alpha, 0.0, 1.0))
    # Keep w (texture_enable). Encode opacity as -alpha in z.
    m[2] = -a
    return m


def set_shape_alphas(viewer, alpha_by_shape: dict[int, float], *, base_materials: dict | None = None) -> None:
    """Set per-shape opacity via encoded materials. Triggers GPU re-upload."""
    import numpy as np
    import warp as wp

    if not alpha_by_shape:
        return
    batches = getattr(viewer, "_shape_instances", None)
    if not batches:
        return

    for shapes in batches.values():
        mats = shapes.materials.numpy().copy()
        changed = False
        for local_i, s_idx in enumerate(shapes.model_shapes):
            s_idx = int(s_idx)
            if s_idx not in alpha_by_shape:
                continue
            if base_materials is not None and s_idx in base_materials:
                base = base_materials[s_idx]
            else:
                base = mats[local_i]
            mats[local_i] = encode_material_alpha(base, alpha_by_shape[s_idx])
            changed = True
        if changed:
            shapes.materials = wp.array(mats, dtype=wp.vec4, device=shapes.device)
            shapes.materials_changed = True

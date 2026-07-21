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
_GRID_ORIGIN = (0.0, 0.0)


def set_grid_origin(origin_xy: tuple[float, float]) -> None:
    """Update the floor-grid center (world horizontal plane, Z-up default)."""
    global _GRID_ORIGIN
    _GRID_ORIGIN = (float(origin_xy[0]), float(origin_xy[1]))


def get_grid_origin() -> tuple[float, float]:
    return _GRID_ORIGIN


def _inject_grid_origin_uniform(shaders) -> bool:
    marker = "SOMA_GRID_ORIGIN_UNIFORM"
    frag = shaders.shape_fragment_shader
    if marker in frag:
        return True
    anchor = "uniform vec3 fogColor;"
    if anchor not in frag:
        return False
    shaders.shape_fragment_shader = frag.replace(
        anchor,
        f"{anchor}\nuniform vec2 soma_grid_origin; // {marker}",
    )
    return True


def _patch_shape_shader_grid_origin() -> bool:
    from newton._src.viewer.gl.shaders import ShaderShape, str_buffer

    if getattr(ShaderShape, "_soma_grid_origin_patched", False):
        return True

    _orig_update = ShaderShape.update

    def _update(self, *args, **kwargs):
        _orig_update(self, *args, **kwargs)
        with self:
            loc = getattr(self, "_soma_grid_origin_loc", None)
            if loc is None:
                loc = self._gl.glGetUniformLocation(
                    self.shader_program.id, str_buffer("soma_grid_origin"))
                self._soma_grid_origin_loc = loc
            if loc >= 0:
                self._gl.glUniform2f(loc, _GRID_ORIGIN[0], _GRID_ORIGIN[1])

    ShaderShape.update = _update
    ShaderShape._soma_grid_origin_patched = True
    return True


def enable_studio_grid_origin() -> bool:
    """Pass robot-root grid center into the floor shader each frame."""
    import newton._src.viewer.gl.shaders as shaders

    ok_uniform = _inject_grid_origin_uniform(shaders)
    ok_patch = _patch_shape_shader_grid_origin()
    return ok_uniform and ok_patch


def enable_isaac_grid_floor() -> bool:
    """Studio cyclorama floor: grid + near vignette + far dissolve into void.

    Encoding: ``scale = Material.z - 1`` (cells / metre). Example: ``z=2`` → 1 m
    cells. Grid ring is centered on ``soma_grid_origin`` (robot first frame).
    """
    import re
    import newton._src.viewer.gl.shaders as shaders

    enable_studio_grid_origin()

    marker = "SOMA_ISAAC_GRID_V9"
    frag = shaders.shape_fragment_shader
    if marker in frag:
        return True

    stock = (
        "    // Optional checker pattern in object-space so it follows instance transforms\n"
        "    if (checker_enable > 0.0)\n"
        "    {\n"
        "        vec2 uv = LocalPos.xy * checker_scale;\n"
        "        float cb = checker(uv);\n"
        "        vec3 albedo2 = albedo*0.7;\n"
        "        // pick between the two colors\n"
        "        albedo = mix(albedo, albedo2, cb);\n"
        "    }"
    )
    new = (
        f"    // {marker}: grid centered on robot first-frame root (soma_grid_origin)\n"
        "    if (checker_enable >= 2.0)\n"
        "    {\n"
        "        float scale = max(checker_enable - 1.0, 0.25);\n"
        "        vec2 floor_pos = (up_axis == 2) ? FragPos.xy\n"
        "                       : ((up_axis == 1) ? FragPos.xz : FragPos.yz);\n"
        "        vec2 floor_xy = floor_pos - soma_grid_origin;\n"
        "        vec2 uv = floor_xy * scale;\n"
        "        vec2 fw = max(fwidth(uv), vec2(1e-5));\n"
        "        vec2 d = min(fract(uv), 1.0 - fract(uv));\n"
        "        vec2 d_px = d / fw;\n"
        "        float min_px = min(d_px.x, d_px.y);\n"
        "        float line = 1.0 - smoothstep(0.0, 1.15, min_px);\n"
        "        float dots = 1.0 - smoothstep(0.0, 1.8, length(d_px));\n"
        "        float grid = max(line * 0.38, dots * 0.22);\n"
        "        float dist_h = length(floor_xy);\n"
        "        // Wider studio disk: gentle center, slow fade to void (~38 m sharp, ~80 m gone).\n"
        "        float near_dark = 1.0 - smoothstep(4.0, 38.0, dist_h);\n"
        "        float far_dark = smoothstep(30.0, 80.0, dist_h);\n"
        "        grid *= (1.0 - far_dark) * (1.0 - 0.18 * near_dark);\n"
        "        albedo *= mix(1.0, 0.62, near_dark);\n"
        "        vec3 line_col = albedo * 1.55 + vec3(0.02);\n"
        "        albedo = mix(albedo, line_col, clamp(grid, 0.0, 1.0));\n"
        "        vec3 far_col = pow(fogColor, vec3(2.2)) * 0.42;\n"
        "        albedo = mix(albedo, far_col, far_dark * 0.65);\n"
        "        albedo *= mix(1.0, 0.74, far_dark);\n"
        "    }\n"
        "    else if (checker_enable > 0.0)\n"
        "    {\n"
        "        vec2 uv = LocalPos.xy * checker_scale;\n"
        "        float cb = checker(uv);\n"
        "        vec3 albedo2 = albedo*0.7;\n"
        "        albedo = mix(albedo, albedo2, cb);\n"
        "    }"
    )

    if stock in frag:
        shaders.shape_fragment_shader = frag.replace(stock, new)
        return True

    if "SOMA_ISAAC_GRID" in frag:
        frag2, n = re.subn(
            r"    // SOMA_ISAAC_GRID(?:_V[0-9]+)?:.*?else if \(checker_enable > 0\.0\)\n"
            r"    \{\n"
            r"        vec2 uv = LocalPos\.xy \* checker_scale;\n"
            r"        float cb = checker\(uv\);\n"
            r"        vec3 albedo2 = albedo\*0\.7;\n"
            r"        albedo = mix\(albedo, albedo2, cb\);\n"
            r"    \}",
            new,
            frag,
            count=1,
            flags=re.S,
        )
        if n:
            shaders.shape_fragment_shader = frag2
            return True
    return False


def enable_studio_floor_matte() -> bool:
    """Hard metallic studio floor: strong specular + env reflection (robot mirrors separate)."""
    import re
    import newton._src.viewer.gl.shaders as shaders

    marker = "SOMA_FLOOR_METAL"
    frag = shaders.shape_fragment_shader
    if marker in frag:
        return True

    # Replace older matte kill-spec patch if present.
    if "SOMA_FLOOR_MATTE" in frag:
        frag2, n = re.subn(
            r"    // SOMA_FLOOR_MATTE:.*?diffuse \*= mix\(1\.0, 0\.25, far_merge\);\n"
            r"    \}",
            "",
            frag,
            count=1,
            flags=re.S,
        )
        if n:
            shaders.shape_fragment_shader = frag2
            frag = frag2

    old_light = (
        "    // Metals should contribute little diffuse light.\n"
        "    diffuse *= 1.0 - metallic;\n"
        "    vec3 color = ambient + (1.0 - shadow) * spotlightAttenuation * (diffuse + spec);"
    )
    new_light = (
        "    // Metals should contribute little diffuse light.\n"
        "    diffuse *= 1.0 - metallic;\n"
        f"    // {marker}: hard floor specular before lighting composite\n"
        "    if (checker_enable >= 2.0) {\n"
        "        spec *= 1.6;\n"
        "        float dist_h = length(FragPos.xy - view_pos.xy);\n"
        "        dist_h = min(dist_h, length(vec2(FragPos.x, FragPos.z) - vec2(view_pos.x, view_pos.z)));\n"
        "        float far_merge = smoothstep(5.0, 20.0, dist_h);\n"
        "        diffuse *= mix(1.0, 0.30, far_merge);\n"
        "        spec *= (1.0 - 0.85 * far_merge);\n"
        "    }\n"
        "    vec3 color = ambient + (1.0 - shadow) * spotlightAttenuation * (diffuse + spec);"
    )
    if old_light not in frag:
        return False
    frag = frag.replace(old_light, new_light)

    old_env = (
        "    float reflection_strength = clamp(metallic * pow(1.0 - roughness, 2.0), 0.0, 1.0);\n"
        "    vec3 env_tint = mix(vec3(1.0), albedo, metallic);\n"
        "    vec3 env_reflection = env_color * env_tint * env_intensity;\n"
        "    color = mix(color, env_reflection, reflection_strength);"
    )
    new_env = (
        "    float reflection_strength = clamp(metallic * pow(1.0 - roughness, 2.0), 0.0, 1.0);\n"
        f"    // {marker}: fresnel gloss on floor\n"
        "    if (checker_enable >= 2.0) {\n"
        "        float fres = pow(clamp(1.0 - NdotV, 0.0, 1.0), 2.5);\n"
        "        reflection_strength = clamp(max(reflection_strength, 0.12) + 0.28 * fres, 0.0, 0.55);\n"
        "        float dist_h = length(FragPos.xy - view_pos.xy);\n"
        "        dist_h = min(dist_h, length(vec2(FragPos.x, FragPos.z) - vec2(view_pos.x, view_pos.z)));\n"
        "        reflection_strength *= (1.0 - 0.85 * smoothstep(7.0, 26.0, dist_h));\n"
        "    }\n"
        "    vec3 env_tint = mix(vec3(1.0), albedo, metallic);\n"
        "    vec3 env_reflection = env_color * env_tint * env_intensity;\n"
        "    color = mix(color, env_reflection, reflection_strength);"
    )
    if old_env not in frag:
        shaders.shape_fragment_shader = frag
        return False
    shaders.shape_fragment_shader = frag.replace(old_env, new_env)
    return True


def enable_studio_floor_alpha() -> bool:
    """Keep studio floor opaque.

    Earlier translucency made the sky bleed through (fog band) and hid mirrors
    at normal camera angles. Reflections are composited in the draw-order pass
    with depth testing disabled instead.
    """
    import re
    import newton._src.viewer.gl.shaders as shaders

    marker = "SOMA_FLOOR_OPAQUE"
    frag = shaders.shape_fragment_shader
    if marker in frag:
        return True

    # Strip fresnel / constant floor alpha patches; restore opaque soma_alpha.
    if "SOMA_FLOOR_ALPHA" in frag:
        frag2, n = re.subn(
            r"    // SOMA_FLOOR_ALPHA(?:_V2)?:.*?\n"
            r"(?:    if \(checker_enable >= 2\.0\) \{\n"
            r"        float ndv = .*?\n"
            r"        soma_alpha = .*?\n"
            r"    \}\n)?"
            r"    FragColor = vec4\(color, soma_alpha\); // SOMA_MESH_ALPHA",
            f"    // {marker}: reflections use depth-off pass, floor stays solid\n"
            f"    FragColor = vec4(color, soma_alpha); // {_MARKER}",
            frag,
            count=1,
            flags=re.S,
        )
        if n:
            shaders.shape_fragment_shader = frag2
            frag = frag2

    # Also undo early constant-alpha assignment if still present.
    if "Material.z >= 2.0) ? 0.78" in frag or "Material.z >= 2.0) ? 0.78" in frag:
        frag = frag.replace(
            "    float soma_alpha = (Material.z < 0.0)\n"
            "        ? clamp(-Material.z, 0.0, 1.0)\n"
            "        : ((Material.z >= 2.0) ? 0.78 : 1.0);",
            "    float soma_alpha = (Material.z < 0.0) "
            "? clamp(-Material.z, 0.0, 1.0) : 1.0;",
        )
        shaders.shape_fragment_shader = frag

    if marker not in shaders.shape_fragment_shader:
        # Ensure marker exists even when there was nothing to strip.
        old = f"FragColor = vec4(color, soma_alpha); // {_MARKER}"
        if old in shaders.shape_fragment_shader:
            shaders.shape_fragment_shader = shaders.shape_fragment_shader.replace(
                old,
                f"    // {marker}\n    {old}",
            )
    return marker in shaders.shape_fragment_shader or True


def enable_studio_draw_order() -> bool:
    """Opaque floor + robots, then frosted ghosts, then frosted floor mirrors.

    Translucent origin / snapshot ghosts / floor reflections use a depth
    prepass + alpha color pass so only the outer shell is visible (internal
    motors/ribs stay hidden). Opaque TO draws first so it is not darkened by
    multi-layer origin blend.
    """
    from newton._src.viewer.gl.opengl import RendererGL

    if getattr(RendererGL._draw_objects, "_soma_draw_order_v14", False):
        return True

    _ensure_alpha_filter_shader()

    def _set_alpha_filter(renderer, mode: int) -> None:
        loc = getattr(getattr(renderer, "_shape_shader", None), "loc_soma_alpha_filter", None)
        if loc is None or int(loc) < 0:
            return
        gl = RendererGL.gl
        shader = renderer._shape_shader
        shader.use()
        gl.glUniform1i(loc, int(mode))

    def _force_front_face_cull(objects):
        saved = []
        for o in objects:
            mesh = getattr(o, "mesh", None)
            if mesh is not None and hasattr(mesh, "backface_culling"):
                saved.append((mesh, bool(mesh.backface_culling)))
                mesh.backface_culling = True
        return saved

    def _restore_front_face_cull(saved):
        for mesh, prev in saved:
            mesh.backface_culling = prev

    def _draw_frosted_shell(objects, *, front_face=None):
        """Outer-surface transparency: hide internals, keep see-through shell.

        ``front_face`` defaults to CCW; floor mirrors pass ``GL_CW`` because the
        reflection matrix flips winding.
        """
        if not objects:
            return
        gl = RendererGL.gl
        if front_face is None:
            front_face = gl.GL_CCW
        saved_cull = _force_front_face_cull(objects)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_CULL_FACE)
        gl.glCullFace(gl.GL_BACK)
        gl.glFrontFace(front_face)

        gl.glColorMask(False, False, False, False)
        gl.glDepthMask(True)
        gl.glDepthFunc(gl.GL_LESS)
        gl.glDisable(gl.GL_BLEND)
        for o in objects:
            o.render()

        gl.glColorMask(True, True, True, True)
        gl.glDepthMask(False)
        gl.glDepthFunc(gl.GL_LEQUAL)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        for o in objects:
            o.render()

        _restore_front_face_cull(saved_cull)
        gl.glFrontFace(gl.GL_CCW)
        gl.glCullFace(gl.GL_BACK)
        gl.glDepthFunc(gl.GL_LESS)
        gl.glDepthMask(True)
        gl.glDisable(gl.GL_BLEND)

    def _draw_reflect_pass(reflect):
        """Floor mirrors: frosted outer shell (no internal motors in the puddle).

        Reflections sit under the floor, so scene depth would hide them. Clear
        depth for this pass only (reflect is drawn last), then reuse the same
        shell prepass as origin/snapshot ghosts. Winding is CW after mirror.
        """
        if not reflect:
            return
        gl = RendererGL.gl
        gl.glClear(gl.GL_DEPTH_BUFFER_BIT)
        _draw_frosted_shell(reflect, front_face=gl.GL_CW)

    def _draw_ghost_pass(ghost):
        """Snapshot traj ghosts: frosted outer shell only."""
        _draw_frosted_shell(ghost)

    def _draw_rest_alpha_split(self, rest):
        """Opaque TO first, then frosted translucent origin (no inner structure)."""
        if not rest:
            return
        gl = RendererGL.gl
        _set_alpha_filter(self, 1)
        gl.glColorMask(True, True, True, True)
        gl.glDepthMask(True)
        gl.glDepthFunc(gl.GL_LESS)
        gl.glDisable(gl.GL_BLEND)
        for o in rest:
            o.render()
        _set_alpha_filter(self, 2)
        _draw_frosted_shell(rest)
        _set_alpha_filter(self, 0)

    def _draw_objects_ordered(self, objects):
        gl = RendererGL.gl

        is_shadow = False
        try:
            bound = gl.GLint()
            gl.glGetIntegerv(gl.GL_DRAW_FRAMEBUFFER_BINDING, bound)
            shadow_fbo = getattr(self, "_shadow_fbo", None)
            if shadow_fbo is not None and int(bound.value) == int(shadow_fbo.value):
                is_shadow = True
        except Exception:
            is_shadow = False

        reflect, floor, ghost, rest = [], [], [], []
        for name, o in objects.items():
            if not hasattr(o, "render"):
                continue
            pass_id = getattr(o, "_soma_pass", None)
            if pass_id is None and isinstance(name, str):
                if name.startswith("/to_playback/floor_reflect"):
                    pass_id = "reflect"
                elif name.startswith("/to_playback/floor_catcher"):
                    continue
                elif name.startswith("/newton_snap/mesh/") and name.endswith("/ghost"):
                    pass_id = "ghost"
            if pass_id == "reflect":
                if not is_shadow:
                    reflect.append(o)
            elif pass_id == "ghost":
                if not is_shadow:
                    ghost.append(o)
            elif pass_id == "floor":
                floor.append(o)
            else:
                rest.append(o)

        if is_shadow:
            _set_alpha_filter(self, 1)
            for o in floor:
                o.render()
            for o in rest:
                o.render()
            _set_alpha_filter(self, 0)
            return

        for o in floor:
            o.render()
        _draw_rest_alpha_split(self, rest)
        _draw_ghost_pass(ghost)
        _draw_reflect_pass(reflect)

    _draw_objects_ordered._soma_draw_order_v14 = True
    _draw_objects_ordered._soma_draw_order_v13 = True
    _draw_objects_ordered._soma_draw_order_v12 = True
    _draw_objects_ordered._soma_draw_order_v11 = True
    _draw_objects_ordered._soma_draw_order_v10 = True
    _draw_objects_ordered._soma_draw_order_v9 = True
    _draw_objects_ordered._soma_draw_order_v8 = True
    _draw_objects_ordered._soma_draw_order_v7 = True
    _draw_objects_ordered._soma_draw_order_v6 = True
    _draw_objects_ordered._soma_draw_order_v5 = True
    _draw_objects_ordered._soma_draw_order_v4 = True
    _draw_objects_ordered._soma_draw_order = True
    RendererGL._draw_objects = _draw_objects_ordered
    return True


def _ensure_alpha_filter_shader() -> bool:
    """Add soma_alpha_filter uniform + discard so opaque/transparent can be split."""
    import newton._src.viewer.gl.shaders as shaders
    from newton._src.viewer.gl.shaders import ShaderShape

    marker = "SOMA_ALPHA_FILTER_V1"
    frag = shaders.shape_fragment_shader
    if marker not in frag:
        # Require mesh-alpha soma_alpha variable.
        needle = "float soma_alpha = (Material.z < 0.0) ? clamp(-Material.z, 0.0, 1.0) : 1.0;"
        if needle not in frag:
            # Try multiline form from enable_mesh_alpha.
            needle = (
                "    float soma_alpha = (Material.z < 0.0) "
                "? clamp(-Material.z, 0.0, 1.0) : 1.0;"
            )
        if needle not in frag:
            return False
        insert = (
            f"\n    // {marker}: 0=all, 1=opaque-only, 2=transparent-only\n"
            "    if (soma_alpha_filter == 1 && soma_alpha < 0.995) discard;\n"
            "    if (soma_alpha_filter == 2 && soma_alpha >= 0.995) discard;"
        )
        # Declare uniform near other uniforms if missing.
        if "uniform int soma_alpha_filter;" not in frag:
            frag = frag.replace(
                "uniform float shadow_extents;",
                "uniform float shadow_extents;\n"
                f"uniform int soma_alpha_filter; // {marker}",
            )
            if "uniform int soma_alpha_filter;" not in frag:
                # Fallback: inject after first uniform block line in fragment.
                frag = frag.replace(
                    "#version 330 core\n",
                    "#version 330 core\n"
                    f"uniform int soma_alpha_filter; // {marker}\n",
                    1,
                )
        frag = frag.replace(needle, needle + insert)
        shaders.shape_fragment_shader = frag

    # Ensure ShaderShape caches the uniform location.
    if not getattr(ShaderShape.__init__, "_soma_alpha_filter_patched", False):
        _orig_init = ShaderShape.__init__

        def _init_with_filter(self, gl):
            _orig_init(self, gl)
            try:
                with self:
                    self.loc_soma_alpha_filter = self._get_uniform_location("soma_alpha_filter")
                    if self.loc_soma_alpha_filter is not None and int(self.loc_soma_alpha_filter) >= 0:
                        gl.glUniform1i(self.loc_soma_alpha_filter, 0)
            except Exception:
                self.loc_soma_alpha_filter = -1

        _init_with_filter._soma_alpha_filter_patched = True
        ShaderShape.__init__ = _init_with_filter

    # Refresh any live RendererGL shape shaders so the new uniform exists.
    try:
        from newton._src.viewer.gl.opengl import RendererGL
        import gc
        for obj in gc.get_objects():
            if isinstance(obj, RendererGL) and getattr(obj, "_shape_shader", None) is not None:
                try:
                    obj._shape_shader = ShaderShape(RendererGL.gl)
                except Exception:
                    pass
    except Exception:
        pass
    return marker in shaders.shape_fragment_shader or True


def enable_studio_sky_gradient() -> bool:
    """Studio cyclorama: soft lit mid-wall, dark vignette edges (match ref)."""
    import re
    import newton._src.viewer.gl.shaders as shaders

    marker = "SOMA_SKY_GRAD_V3"
    sky = shaders.sky_fragment_shader
    if marker in sky:
        return True

    stock_body = (
        "    float h = up_axis == 0 ? FragPos.x : (up_axis == 1 ? FragPos.y : FragPos.z);\n"
        "    float height = max(0.0, h / far_plane);\n"
        "    vec3 sky = mix(sky_lower, sky_upper, height);\n"
        "\n"
        "    float diff = max(dot(sun_direction, normalize(FragPos)), 0.0);\n"
        "    vec3 sun = pow(diff, 32) * vec3(1.0, 0.8, 0.6) * 0.5;\n"
        "\n"
        "    FragColor = vec4(sky + sun, 1.0);"
    )
    new_body = (
        f"    // {marker}: soft key on cyclorama wall + dark vignette\n"
        "    float h = up_axis == 0 ? FragPos.x : (up_axis == 1 ? FragPos.y : FragPos.z);\n"
        "    float elev = clamp(h / far_plane, 0.0, 1.0);\n"
        "    float t = smoothstep(0.08, 0.88, elev);\n"
        "    vec3 sky = mix(sky_lower, sky_upper, t);\n"
        "    vec3 dir = normalize(FragPos);\n"
        "    float key = pow(max(dot(dir, normalize(sun_direction)), 0.0), 1.6);\n"
        "    sky += sky_lower * key * 0.55;\n"
        "    float rim = pow(1.0 - abs(dot(dir, vec3(\n"
        "        up_axis == 0 ? 1.0 : 0.0,\n"
        "        up_axis == 1 ? 1.0 : 0.0,\n"
        "        up_axis == 2 ? 1.0 : 0.0))), 1.8);\n"
        "    sky *= mix(1.0, 0.72, rim * 0.65);\n"
        "    FragColor = vec4(sky, 1.0);"
    )

    if stock_body in sky:
        shaders.sky_fragment_shader = sky.replace(stock_body, new_body)
        return True

    if "SOMA_SKY_GRAD" in sky:
        sky2, n = re.subn(
            r"    // SOMA_SKY_GRAD(?:_V[0-9]+)?:.*?FragColor = vec4\(sky(?: \+ sun)?, 1\.0\);",
            new_body,
            sky,
            count=1,
            flags=re.S,
        )
        if n:
            shaders.sky_fragment_shader = sky2
            return True
    return False


def enable_studio_horizon_fog() -> bool:
    """Dissolve distant floor into sky_lower — matched colors, no bright mist band."""
    import re
    import newton._src.viewer.gl.shaders as shaders

    marker = "SOMA_STUDIO_FOG_V6"
    frag = shaders.shape_fragment_shader
    if marker in frag:
        return True

    new = (
        f"    // fog  {marker}: soft dark dissolve (no bright horizon band)\n"
        "    float dist = length(FragPos - view_pos);\n"
        "    float fog_start = 30.0;\n"
        "    float fog_end   = 88.0;\n"
        "    float fog_factor = clamp((dist - fog_start) / (fog_end - fog_start), 0.0, 1.0);\n"
        "    fog_factor = fog_factor * fog_factor;\n"
        "    // Fog toward a darkened sky_lower so the seam stays soft and dark.\n"
        "    vec3 fog_col = pow(fogColor, vec3(2.2)) * 0.70;\n"
        "    color = mix(color, fog_col, fog_factor);"
    )

    stock = (
        "    // fog\n"
        "    float dist = length(FragPos - view_pos);\n"
        "    float fog_start = 20.0;\n"
        "    float fog_end   = 200.0;\n"
        "    float fog_factor = clamp((dist - fog_start) / (fog_end - fog_start), 0.0, 1.0);\n"
        "    color = mix(color, pow(fogColor, vec3(2.2)), fog_factor);"
    )
    if stock in frag:
        shaders.shape_fragment_shader = frag.replace(stock, new)
        return True

    if "SOMA_STUDIO_FOG" in frag:
        frag2, n = re.subn(
            r"    // fog  SOMA_STUDIO_FOG(?:_V[0-9]+)?:.*?color = mix\(color, fog_col, fog_factor\);",
            new,
            frag,
            count=1,
            flags=re.S,
        )
        if n:
            shaders.shape_fragment_shader = frag2
            return True
    return False


def enable_soft_shadow_bias() -> bool:
    """Softer, lighter contact shadows that stay stuck to the feet.

    Earlier large depth bias caused peter-panning (shadow detached under the
    sole). Keep a mild bias for CAD acne and cut shadow darkness.
    """
    import re
    import newton._src.viewer.gl.shaders as shaders

    marker = "SOMA_SHADOW_SOFT_V2"
    frag = shaders.shape_fragment_shader
    if marker in frag:
        return True

    if "SOMA_SHADOW_BIAS" in frag:
        frag2, n = re.subn(
            r"    // SOMA_SHADOW_BIAS:.*?float biased_depth = frag_depth - depthBias;",
            "    float NdotL_bias = max(dot(normal, lightDir), 0.0);\n"
            "    float depthBias = mix(0.0003, 0.00002, NdotL_bias);\n"
            "    float biased_depth = frag_depth - depthBias;",
            frag,
            count=1,
            flags=re.S,
        )
        if n:
            shaders.shape_fragment_shader = frag2
            frag = frag2

    old_bias = (
        "    float NdotL_bias = max(dot(normal, lightDir), 0.0);\n"
        "    float depthBias = mix(0.0003, 0.00002, NdotL_bias);\n"
        "    float biased_depth = frag_depth - depthBias;"
    )
    new_bias = (
        f"    // {marker}: mild bias — stick to soles, avoid acne\n"
        "    float NdotL_bias = max(dot(normal, lightDir), 0.0);\n"
        "    float depthBias = mix(0.00055, 0.00004, NdotL_bias);\n"
        "    float biased_depth = frag_depth - depthBias;"
    )
    if old_bias in frag:
        frag = frag.replace(old_bias, new_bias)

    old_ret = "    shadow /= 16.0;\n    return shadow * fade;"
    new_ret = (
        "    shadow /= 16.0;\n"
        f"    // {marker}: lighter contact shadow (was pitch-black)\n"
        "    return shadow * fade * 0.28;"
    )
    if old_ret in frag:
        frag = frag.replace(old_ret, new_ret)
    elif "return shadow * fade;" in frag:
        frag = frag.replace(
            "    return shadow * fade;",
            f"    // {marker}\n    return shadow * fade * 0.28;",
        )

    shaders.shape_fragment_shader = frag

    try:
        from newton._src.viewer.gl.opengl import RendererGL
        from newton._src.viewer.gl.shaders import ShapeShader
        import gc
        for obj in gc.get_objects():
            if isinstance(obj, RendererGL) and getattr(obj, "_shape_shader", None) is not None:
                try:
                    obj._shape_shader = ShapeShader(RendererGL.gl)
                except Exception:
                    pass
    except Exception:
        pass
    return marker in shaders.shape_fragment_shader


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
            origin = getattr(self, "_soma_grid_origin", None)
            if origin is not None:
                set_grid_origin(origin)
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


def enable_studio_reflect_fade() -> bool:
    """Floor mirrors: fade/blur parts farther from the symmetry plane (z=0).

    Reflect instances are tagged with ``metallic ≈ 0.05`` and negative
    ``Material.z`` (encoded alpha). Feet stay sharp; head softens out.
    """
    import re
    import newton._src.viewer.gl.shaders as shaders

    marker = "SOMA_REFLECT_FADE_V3"
    frag = shaders.shape_fragment_shader
    if marker in frag:
        return True

    block = (
        f"    // {marker}: contact-sharp reflection, never fully opaque\n"
        "    if (metallic > 0.04 && metallic < 0.06 && Material.z < 0.0) {\n"
        "        float dist_plane = abs(FragPos.z);\n"
        "        float contact = 1.0 - smoothstep(0.04, 1.35, dist_plane);\n"
        "        soma_alpha *= mix(0.16, 0.58, contact);\n"
        "        color *= mix(0.68, 0.92, contact);\n"
        "        color = mix(color, pow(fogColor, vec3(2.2)), (1.0 - contact) * 0.45);\n"
        "    }\n"
        "\n"
        "    // gamma correction (sRGB)\n"
        "    color = pow(color, vec3(1.0 / 2.2));"
    )

    if "SOMA_REFLECT_FADE" in frag:
        frag2, n = re.subn(
            r"    // SOMA_REFLECT_FADE(?:_V[0-9]+)?:.*?// gamma correction \(sRGB\)\n"
            r"    color = pow\(color, vec3\(1\.0 / 2\.2\)\);",
            block,
            frag,
            count=1,
            flags=re.S,
        )
        if n:
            shaders.shape_fragment_shader = frag2
            return True

    anchor = "    // gamma correction (sRGB)\n    color = pow(color, vec3(1.0 / 2.2));"
    if anchor not in frag:
        return False
    shaders.shape_fragment_shader = frag.replace(anchor, block)
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

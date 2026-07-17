# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import newton

import pathlib
import time
import warp as wp

import json
import numpy as np

import soma_retargeter.utils.math_utils as math_utils
import soma_retargeter.utils.newton_utils as newton_utils
import soma_retargeter.assets.bvh as bvh_utils
import soma_retargeter.assets.csv as csv_utils
import soma_retargeter.utils.io_utils as io_utils
import soma_retargeter.pipelines.utils as pipeline_utils
import soma_retargeter.robotics.calibration as calibration_utils

from soma_retargeter.renderers.skeleton_renderer import SkeletonRenderer
from soma_retargeter.renderers.mesh_renderer import SkeletalMeshRenderer
from soma_retargeter.renderers.coordinate_renderer import CoordinateRenderer
from soma_retargeter.animation.skeleton import SkeletonInstance
from soma_retargeter.utils.space_conversion_utils import SpaceConverter, get_facing_direction_type_from_str

from tqdm import trange

_UI_NEWTON_PANEL_WIDTH  = 320
_UI_NEWTON_PANEL_MARGIN = 10
_UI_NEWTON_PANEL_ALPHA  = 0.9
_DEFAULT_COLOR = (235.0 / 255.0, 245.0 / 255.0, 112.0 / 255.0)

class Viewer:
    def __init__(self, viewer, config):
        self.viewer = viewer
        self.viewer.vsync = True
        self.config = config
        self.converter = SpaceConverter(
            get_facing_direction_type_from_str(self.config['retarget_source_facing_direction']),
            yaw_offset_deg=float(self.config.get('retarget_source_yaw_offset_deg', 0.0)))

        # Source skeleton type (selectable via the `--data` flag, which overrides
        # 'retarget_source' in the config). Defaults to SOMA for back-compat.
        # Computed here (before the headless early-return below) since
        # batched_retargeting() also relies on these.
        self.source_str = self.config.get('retarget_source', 'soma')
        self.source_type = pipeline_utils.get_source_type_from_str(self.source_str)
        self.source_position_scale = float(self.config.get(
            'retarget_source_position_scale',
            pipeline_utils.get_source_position_scale(self.source_type)))
        # Optional per-group multipliers on top of the uniform scale. Useful when
        # the actor's proportions don't match the robot (e.g. longer arms but
        # shorter legs): tune upper/lower independently instead of compromising
        # on a single uniform value. Defaults are 1.0 (no effect); calibration
        # scales/offsets stay unaffected until recomputed.
        self.source_position_scale_upper = float(self.config.get(
            'retarget_source_position_scale_upper', 1.0))
        self.source_position_scale_lower = float(self.config.get(
            'retarget_source_position_scale_lower', 1.0))
        self.source_recenter_xy = bool(self.config.get(
            'retarget_source_recenter_xy',
            pipeline_utils.get_source_recenter_xy(self.source_type)))

        # Per-axis motion-amplitude scales (x, y, z), applied together:
        #   full_amplitude_scale: scales every effector (hands/feet too; may
        #     slip/drift) relative to its own standing reference.
        #   root_amplitude_scale: scales only the root trajectory relative to
        #     its standing reference, leaving body pose/proportions untouched.
        # (1, 1, 1) == identity for both.
        self.full_amplitude_scale = [1.0, 1.0, 1.0]
        self.root_amplitude_scale = [1.0, 1.0, 1.0]

        if isinstance(self.viewer, newton.viewer.ViewerNull):
            # Headless mode for batch processing
            return
        
        self.fps      = 60
        self.frame_dt = 1.0 / self.fps
        self.time     = 0.0

        self.is_playing          = True
        self.playback_time       = 0.0
        self.playback_speed      = 1.0
        self.playback_loop       = True
        self.playback_total_time = 0.0

        self.retarget_source_options = ['soma', 'mydata', 'mydata2', 'mydata3', 'mydata4', 'lafan1']
        if self.source_str not in self.retarget_source_options:
            self.retarget_source_options.append(self.source_str)
        self.retarget_target_options = [
            'unitree_g1',
            'engineai_pm01',
            'engineai_t800',
            'hightorque_pi_plus',
            'hightorque_pi_plus_s',
            'pndbotics_adam_lite',
            'pndbotics_adam_sp',
        ]
        self.retarget_solver_options = ['Newton']
        self.retarget_solver_idx     = 0
        self.retarget_source_idx     = self.retarget_source_options.index(self.source_str)

        # Resolve currently selected robot from config (falls back to G1).
        self.robot_type = self.config.get('retarget_target', 'unitree_g1')
        if self.robot_type not in self.retarget_target_options:
            print(f"[WARN]: retarget_target [{self.robot_type}] not in known list, defaulting to unitree_g1")
            self.robot_type = 'unitree_g1'
        self.retarget_target_idx = self.retarget_target_options.index(self.robot_type)
        self.csv_config = csv_utils.get_csv_config_for_robot(self.robot_type)

        # SOMA has a skin mesh; native skeletons don't, so default to drawing bones.
        self._has_source_mesh = (self.source_type == pipeline_utils.SourceType.SOMA)
        self.show_skeleton_mesh = self._has_source_mesh
        self.show_skeleton = not self._has_source_mesh
        self.show_skeleton_joint_axes = False
        self.show_gizmos = True

        self.viewer.renderer.set_title("BVH to CSV Converter")
        self.viewer.register_ui_callback(lambda ui: self.gui(ui), position="free")

        self.robot_builder = newton.ModelBuilder()
        pipeline_utils.add_robot_model(self.robot_builder, self.robot_type)

        self.num_robots = 1
        self.robot_offsets = [wp.transform(wp.vec3(0.0, i - (self.num_robots - 1) / 2.0, 0.0), wp.quat_identity()) for i in range(self.num_robots)]
        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        for _ in range(self.num_robots):
            builder.add_builder(self.robot_builder, wp.transform_identity())
        self.model = builder.finalize()

        self.viewer.set_model(self.model)
        self.viewer.set_world_offsets([0, 0, 0])
        self.state = self.model.state()

        self.robot_num_joint_q = self.model.joint_coord_count // self.model.articulation_count
        self.robot_joint_q_offsets = [int(i * self.robot_num_joint_q) for i in range(self.model.articulation_count)]
        self.robot_default_joint_q_values = self.model.joint_q.numpy()

        self.coordinate_renderer = CoordinateRenderer()
        self.show_ik_map_axes = False
        self.ik_map_coordinate_renderer = CoordinateRenderer()
        # Overlay of the IK effector *targets* (where the retargeter asks each
        # mapped robot body to go). Computed on demand from the live scaler +
        # offsets config, so it mirrors exactly what retargeting solves for.
        self.show_effector_targets = False
        self.effector_target_coordinate_renderer = CoordinateRenderer()
        self._effector_scaler_cache = {}
        self._effector_target_status = ""
        self.skeleton = None
        self.skeleton_renderer = None
        self.skeletal_mesh_renderer = None

        self.animation_offsets = []
        self.animation_buffers = []
        self.skeleton_instances = []
        self.robot_csv_animation_buffers = [None for _ in range(self.num_robots)]

        default_bvh = self.config.get('default_bvh_file', None)
        if default_bvh:
            bvh_path = pathlib.Path(default_bvh)
            if not bvh_path.is_absolute():
                bvh_path = io_utils.get_project_root() / bvh_path
            if bvh_path.exists():
                print(f"[INFO]: Pre-loading default BVH: {bvh_path}")
                self.load_bvh_file(str(bvh_path))
            else:
                print(f"[WARN]: default_bvh_file not found: {bvh_path}")

        self._init_calibration()
        self._ik_map_body_indices = self._build_ik_map_body_indices()
        self._ik_map_label_data = []  # [(name, color_u32, (x,y,z)), ...] populated each frame
        self._init_scene_objects()

    def _init_scene_objects(self):
        """State for user-placed scene primitives (boxes, etc.)."""
        self.scene_objects = []
        self._scene_object_next_id = 0
        self._selected_scene_object_idx = -1
        self.show_scene_object_gizmos = True
        self._scene_objects_status = ""
        self._scene_object_logged_ids = set()
        self._scene_unit_box_mesh_path = None

    def _init_calibration(self):
        """Set up state for the in-app bias-calibration panel.

        Loads the SOMA zero-pose skeleton (used as the visual + numerical
        reference) and discovers the robot's revolute joints so we can
        expose them as sliders in the GUI.
        """
        self.calibration_mode = False
        self.show_zero_pose_reference = False

        retargeter_cfg = pipeline_utils.get_retargeter_config(
            self.source_type,
            pipeline_utils.get_target_type_from_str(self.robot_type))
        self._calibration_retargeter_cfg = retargeter_cfg
        # Path to the retargeter config on disk (so the scale slider can persist).
        rt_filename = pipeline_utils._RETARGETER_CONFIG_FILENAME.get(
            (self.source_type, self.robot_type))
        self._calibration_retargeter_cfg_path = (
            io_utils.get_config_file(self.robot_type, rt_filename) if rt_filename else None)

        scaler_cfg_path = io_utils.get_config_file(retargeter_cfg['human_robot_scaler_config'])
        self._calibration_scaler_cfg_path = scaler_cfg_path
        # joint_offsets (calibration results) may live in a separate file from
        # joint_scales (tracking params); fall back to the scaler config if not.
        offsets_cfg_rel = retargeter_cfg.get('joint_offsets_config')
        self._calibration_offsets_cfg_path = (
            io_utils.get_config_file(offsets_cfg_rel) if offsets_cfg_rel else scaler_cfg_path)

        init_bvh = io_utils.get_config_file(retargeter_cfg['initialization_pose'])
        # By default the init pose loads at the SAME scale/convention as the live
        # motion (so the zero pose and data are the same size). Optional config
        # keys still let a retargeter config override the loading parameters and
        # world-space orientation if an init pose is authored differently.
        init_pos_scale = retargeter_cfg.get('init_pose_position_scale', self.source_position_scale)
        init_recenter  = retargeter_cfg.get('init_pose_recenter_xy',    self.source_recenter_xy)
        ref_skel, ref_anim = bvh_utils.load_bvh(
            init_bvh, position_scale=init_pos_scale, recenter_xy=init_recenter,
            position_scale_upper=self.source_position_scale_upper,
            position_scale_lower=self.source_position_scale_lower)
        self.zero_pose_reference_skeleton = ref_skel
        self.zero_pose_reference_local_zero = ref_anim.get_local_transforms(0).copy()
        # Optional converter override: if the init-pose BVH has a different
        # coordinate convention than the live source, use a dedicated converter
        # so the skeleton overlay renders upright in the viewer.
        init_facing = retargeter_cfg.get('init_pose_facing_direction', None)
        if init_facing is not None:
            init_yaw = float(retargeter_cfg.get('init_pose_yaw_offset_deg', 0.0))
            init_converter = SpaceConverter(
                get_facing_direction_type_from_str(init_facing), yaw_offset_deg=init_yaw)
            init_xform = init_converter.transform(wp.transform_identity())
        else:
            init_xform = self.converter.transform(wp.transform_identity())
        self.zero_pose_reference_instance = SkeletonInstance(
            ref_skel, [0.6, 0.7, 1.0], init_xform)
        self.zero_pose_reference_instance.set_local_transforms(self.zero_pose_reference_local_zero)

        self.zero_pose_reference_mesh = pipeline_utils.get_source_model_mesh(
            self.source_type, ref_skel)
        self.zero_pose_reference_mesh_renderer = (
            SkeletalMeshRenderer(self.zero_pose_reference_mesh)
            if self.zero_pose_reference_mesh is not None else None)
        self.zero_pose_reference_skeleton_renderer = SkeletonRenderer(ref_skel, [0])

        # Discover revolute joints (skip free + fixed) so we can build sliders.
        # joint_limit_lower/upper are indexed per-DOF (length = joint_dof_count),
        # while joint_q_start is per-joint coord index. For a revolute joint we
        # look up its single limit at joint_qd_start[ji].
        self.calibration_revolute_joints = []
        for ji in range(self.robot_builder.joint_count):
            if self.robot_builder.joint_type[ji] != newton.JointType.REVOLUTE:
                continue
            label = self.robot_builder.joint_label[ji]
            name = newton_utils.get_name_from_label(label)
            q_idx = int(self.robot_builder.joint_q_start[ji])
            qd_idx = int(self.robot_builder.joint_qd_start[ji])
            lo = float(self.robot_builder.joint_limit_lower[qd_idx])
            hi = float(self.robot_builder.joint_limit_upper[qd_idx])
            self.calibration_revolute_joints.append({
                "name": name, "q_idx": q_idx, "lo": lo, "hi": hi})

        self.calibration_joint_q = self.robot_default_joint_q_values.astype(np.float32).copy()

        # Default reference pose for known robots: pose that physically
        # matches SOMA zero pose ("holding-tray" stance, elbows bent ~75 deg)
        # PLUS a yaw rotation on the base so the robot faces SOMA's forward
        # direction (-Y in world). See _build_reference_pose for details.
        self._calibration_reference = self._build_reference_pose()
        self.reset_calibration_pose_to_reference()

        self._calibration_status = ""
        self._calibration_last_offsets = None
        self._calibration_last_scales = None

        # A previously-saved slider value (persisted in the retargeter config)
        # takes effect here: rescale the already-loaded motion + zero pose so the
        # source starts at the size the user last dialed in.
        persisted_scale = retargeter_cfg.get('retarget_source_position_scale')
        if (persisted_scale is not None
                and abs(float(persisted_scale) - self.source_position_scale) > 1e-9):
            self._apply_source_scale(float(persisted_scale))
        for group in ('upper', 'lower'):
            persisted_group = retargeter_cfg.get(f'retarget_source_position_scale_{group}')
            current = getattr(self, f'source_position_scale_{group}')
            if (persisted_group is not None
                    and abs(float(persisted_group) - current) > 1e-9):
                self._apply_source_scale_group(group, float(persisted_group))
        # At startup the in-memory scale matches the config on disk (nothing to save).
        self._source_scale_dirty = False

    def _apply_source_scale(self, new_scale):
        """Rescale the loaded source data (motion + zero-pose reference) in place.

        This drives the calibration 'Source Position Scale' slider: it keeps the
        zero pose and the motion at the SAME size, updates the viewer live, and
        invalidates any previously-computed joint_scales/joint_offsets (which are
        size-dependent). The chosen scale is also what feeds bias computation and
        the Newton retargeting init pose.
        """
        new_scale = float(new_scale)
        old = float(getattr(self, "source_position_scale", 1.0))
        if old <= 1e-9 or abs(new_scale - old) < 1e-9:
            self.source_position_scale = new_scale
            return
        ratio = new_scale / old

        # Live motion skeleton + animation buffers.
        if getattr(self, "skeleton", None) is not None:
            self.skeleton._reference_local_transforms[..., 0:3] *= ratio
        for buf in (getattr(self, "animation_buffers", None) or []):
            if buf is not None:
                buf.local_transforms[..., 0:3] *= ratio

        # Zero-pose reference (calibration overlay + numeric reference).
        if getattr(self, "zero_pose_reference_skeleton", None) is not None:
            self.zero_pose_reference_skeleton._reference_local_transforms[..., 0:3] *= ratio
        if getattr(self, "zero_pose_reference_local_zero", None) is not None:
            self.zero_pose_reference_local_zero[..., 0:3] *= ratio
            if getattr(self, "zero_pose_reference_instance", None) is not None:
                self.zero_pose_reference_instance.set_local_transforms(
                    self.zero_pose_reference_local_zero)

        self.source_position_scale = new_scale
        # Live scale differs from what's persisted in the config until saved.
        self._source_scale_dirty = True
        # Size-dependent calibration results are now stale.
        self._calibration_last_scales = None
        self._calibration_last_offsets = None
        self._invalidate_effector_scaler()
        self._calibration_status = (
            f"Source position scale = {new_scale:.3f}. "
            "Motion + zero pose rescaled; recompute scales/offsets.")

    def _apply_source_scale_group(self, group, new_scale):
        """Rescale only the upper- or lower-body joints of the loaded source data.

        Mirrors :meth:`_apply_source_scale` but multiplies the local translations
        of a single joint group (see ``bvh_utils.lower_body_joint_mask``): the
        root/hips + leg chains form the lower group, everything else the upper
        group. Lets arm reach and leg length be tuned independently when the
        actor's proportions don't match the robot's.

        Args:
            group: Either ``'upper'`` or ``'lower'``.
            new_scale: New multiplier for that group (relative to the uniform scale).
        """
        assert group in ('upper', 'lower')
        new_scale = float(new_scale)
        attr = f'source_position_scale_{group}'
        old = float(getattr(self, attr, 1.0))
        if old <= 1e-9 or abs(new_scale - old) < 1e-9:
            setattr(self, attr, new_scale)
            return
        ratio = new_scale / old

        def rescale(skeleton_like, transforms):
            lower_mask = bvh_utils.lower_body_joint_mask(skeleton_like.joint_names)
            mask = lower_mask if group == 'lower' else ~lower_mask
            transforms[..., mask, 0:3] *= ratio

        # Live motion skeleton + animation buffers.
        if getattr(self, "skeleton", None) is not None:
            rescale(self.skeleton, self.skeleton._reference_local_transforms)
            for buf in (getattr(self, "animation_buffers", None) or []):
                if buf is not None:
                    rescale(self.skeleton, buf.local_transforms)

        # Zero-pose reference (calibration overlay + numeric reference).
        if getattr(self, "zero_pose_reference_skeleton", None) is not None:
            rescale(self.zero_pose_reference_skeleton,
                    self.zero_pose_reference_skeleton._reference_local_transforms)
            if getattr(self, "zero_pose_reference_local_zero", None) is not None:
                rescale(self.zero_pose_reference_skeleton, self.zero_pose_reference_local_zero)
                if getattr(self, "zero_pose_reference_instance", None) is not None:
                    self.zero_pose_reference_instance.set_local_transforms(
                        self.zero_pose_reference_local_zero)

        setattr(self, attr, new_scale)
        # Live scale differs from what's persisted in the config until saved.
        self._source_scale_dirty = True
        # Size-dependent calibration results are now stale.
        self._calibration_last_scales = None
        self._calibration_last_offsets = None
        self._invalidate_effector_scaler()
        self._calibration_status = (
            f"Source {group}-body scale = {new_scale:.3f}. "
            "Motion + zero pose rescaled; recompute scales/offsets.")

    def _do_write_source_scale(self):
        """Persist the current source position scale into the retargeter config."""
        path = getattr(self, "_calibration_retargeter_cfg_path", None)
        if path is None:
            self._calibration_status = "No retargeter config path; cannot save scale."
            return
        cfg = io_utils.load_json(path)
        cfg['retarget_source_position_scale'] = round(float(self.source_position_scale), 6)
        cfg['retarget_source_position_scale_upper'] = round(float(self.source_position_scale_upper), 6)
        cfg['retarget_source_position_scale_lower'] = round(float(self.source_position_scale_lower), 6)
        with open(path, 'w') as f:
            json.dump(cfg, f, indent=4)
        self._calibration_retargeter_cfg = cfg
        self._source_scale_dirty = False
        self._calibration_status = (
            f"Saved source position scale (uniform={self.source_position_scale:.3f}, "
            f"upper={self.source_position_scale_upper:.3f}, "
            f"lower={self.source_position_scale_lower:.3f}) to {os.path.basename(str(path))}")
        print(f"[INFO]: {self._calibration_status}")

    def _build_reference_pose(self):
        """Per-robot calibration reference pose.

        Loaded from ``tools/reference_poses/<robot_type>_reference_pose.json`` if present, so
        adding a new robot does not require touching this file - just drop a
        reference-pose JSON in the right place.

        The JSON schema (same one used by the CLI calibrator) has these keys:
          * ``base_pos``         (3 floats, world-space base position)
          * ``base_quat_xyzw``   (4 floats, base orientation; encode any yaw
                                  alignment with SOMA here, e.g. -90 deg
                                  around Z to face world -Y)
          * ``joint_order``      (list of joint names)
          * ``joint_angles_rad`` (list, same length as ``joint_order``)

        Returns a dict::

            { "base_pos": [...], "base_quat_xyzw": [...], "angles": {name: rad} }

        Falls back to identity / zero angles if the file does not exist.
        """
        repo_root = pathlib.Path(__file__).parent.parent
        ref_path = repo_root / "tools" / "reference_poses" / f"{self.robot_type}_reference_pose.json"
        if not ref_path.exists():
            print(f"[INFO]: No reference pose JSON for [{self.robot_type}] at {ref_path}. "
                  "Calibration will use identity base + zero joint angles.")
            return {"base_pos": None, "base_quat_xyzw": None, "angles": {}}

        ref = json.loads(ref_path.read_text())
        order = ref.get("joint_order", [])
        rads = ref.get("joint_angles_rad", [])
        angles = {}
        if len(order) == len(rads):
            angles = {nm: float(a) for nm, a in zip(order, rads)}
        else:
            print(f"[WARN]: Reference pose [{ref_path.name}] has joint_order "
                  f"({len(order)}) and joint_angles_rad ({len(rads)}) mismatch.")

        return {
            "base_pos": ref.get("base_pos"),
            "base_quat_xyzw": ref.get("base_quat_xyzw"),
            "angles": angles,
        }

    def _build_ik_map_body_indices(self):
        """Return an ordered list of (soma_joint_name, robot_body_name, robot_local_body_idx)
        for all unique robot bodies referenced in the retargeter's ik_map (t_body / r_body).

        Indices are local to ``self.robot_builder`` (0-based within the robot,
        not accounting for any ground-plane bodies in the full model).

        Also stores ``self._ik_map_source_joint_names`` — the ordered list of
        source (SOMA/human) joint names from the ik_map keys, used to overlay
        coordinate axes on the human skeleton.
        """
        ik_map = self._calibration_retargeter_cfg.get('ik_map', {})
        robot_body_names = [
            newton_utils.get_name_from_label(lbl)
            for lbl in self.robot_builder.body_label]

        # Source joint names are the ik_map keys (one per mapped human joint).
        self._ik_map_source_joint_names = list(ik_map.keys())
        # Skeleton indices resolved lazily in load_bvh_file once a skeleton is available.
        self._ik_map_source_joint_indices = []

        seen = set()
        result = []
        for soma_joint, mapping in ik_map.items():
            # t_body and r_body are often the same body; deduplicate by body name
            for key in ('t_body', 'r_body'):
                body_name = mapping.get(key)
                if body_name and body_name not in seen:
                    seen.add(body_name)
                    if body_name in robot_body_names:
                        result.append((soma_joint, body_name, robot_body_names.index(body_name)))
        return result  # list of (soma_joint, robot_body_name, robot_local_body_idx)

    # ImGui U32 colors for IK-map overlay labels (IM_COL32 = (a<<24)|(b<<16)|(g<<8)|r)
    _IK_LABEL_COLOR_ROBOT  = (220 << 24) | (255 << 16) | (220 << 8) | 120   # light-blue,  robot bodies
    _IK_LABEL_COLOR_HUMAN  = (220 << 24) | (120 << 16) | (255 << 8) | 255   # light-yellow, human joints
    _IK_LABEL_COLOR_SHADOW = (160 << 24) | (  0 << 16) | (  0 << 8) |   0   # semi-transparent black

    def _render_ik_map_axes(self):
        """Draw RGB coordinate frames for every body referenced in the IK map.

        Renders two sets of axes:
        - Robot bodies: read from ``self.state.body_q`` after FK evaluation.
        - Human skeleton joints: read from each skeleton instance's global transforms,
          filtered to the source joints listed in the ik_map.

        Also caches ``self._ik_map_label_data`` with world positions so that
        ``_draw_ik_map_labels()`` (called from the ImGui callback) can overlay
        text without re-computing transforms.
        """
        label_data = []

        # --- Robot body axes ---
        robot_transforms = []
        if self._ik_map_body_indices:
            body_q = self.state.body_q.numpy()   # shape: (total_model_bodies, 7)
            robot_body_count = self.robot_builder.body_count
            # Ground-plane bodies come first in self.model (added before robots in __init__).
            # Offset = total bodies - robot bodies across all articulations.
            body_offset = self.model.body_count - self.num_robots * robot_body_count

            for robot_idx in range(self.num_robots):
                base = body_offset + robot_idx * robot_body_count
                for _, robot_body_name, local_idx in self._ik_map_body_indices:
                    t = body_q[base + local_idx]
                    robot_transforms.append(wp.transform(wp.vec3(*t[0:3]), wp.quat(*t[3:7])))
                    label_data.append((robot_body_name, self._IK_LABEL_COLOR_ROBOT, (t[0], t[1], t[2])))

        if robot_transforms:
            self.ik_map_coordinate_renderer.draw(self.viewer, robot_transforms, 0.15, 998, width=0.008)

        # --- Human skeleton axes ---
        if (not self.calibration_mode
                and self._ik_map_source_joint_indices
                and len(self.skeleton_instances) > 0):
            for i, skel_inst in enumerate(self.skeleton_instances):
                # Temporarily apply animation offset (mirrors what render() does).
                prev_xform = wp.transform(skel_inst.xform)
                skel_inst.xform = wp.mul(self.animation_offsets[i], skel_inst.xform)
                all_tx = skel_inst.compute_global_transforms()
                skel_inst.xform = prev_xform

                selected = [all_tx[idx] for idx in self._ik_map_source_joint_indices]
                if selected:
                    self.ik_map_coordinate_renderer.draw(self.viewer, selected, 0.1, 9000 + i, width=0.006)

                # Cache human label positions (use the ik_map key names in order)
                for joint_name, idx in zip(self._ik_map_source_joint_names, self._ik_map_source_joint_indices):
                    t = all_tx[idx]
                    label_data.append((joint_name, self._IK_LABEL_COLOR_HUMAN, (t[0], t[1], t[2])))

        self._ik_map_label_data = label_data

    def _draw_ik_map_labels(self, ui):
        """Overlay text labels for IK-map coordinate frames using ImGui's foreground draw list.

        Projects each cached world position (from ``_ik_map_label_data``) to screen
        space using the viewer camera's view/projection matrices, then draws the body
        or joint name with a drop-shadow for readability.
        """
        camera = getattr(self.viewer, 'camera', None)
        if camera is None or not self._ik_map_label_data:
            return

        # Pyglet's Mat4 is stored column-major; np.reshape(4,4) gives M_math^T.
        # To project: clip_row = p_row @ V_np @ P_np  (equivalent to P_math @ V_math @ p_col).
        V_np = np.array(camera.get_view_matrix(), dtype=np.float64).reshape(4, 4)
        P_np = np.array(camera.get_projection_matrix(), dtype=np.float64).reshape(4, 4)
        W = float(camera.width)
        H = float(camera.height)

        draw_list = ui.get_foreground_draw_list()

        for name, color, pos in self._ik_map_label_data:
            p = np.array([pos[0], pos[1], pos[2], 1.0], dtype=np.float64)
            clip = p @ V_np @ P_np
            if clip[3] <= 1e-6:
                continue  # behind or at camera
            ndc = clip[:3] / clip[3]
            if abs(ndc[0]) > 1.1 or abs(ndc[1]) > 1.1:
                continue  # off-screen
            sx = float((ndc[0] + 1.0) * 0.5 * W)
            sy = float((1.0 - ndc[1]) * 0.5 * H)

            # Drop shadow (1px offset) then main label
            draw_list.add_text(ui.ImVec2(sx + 1, sy + 1), self._IK_LABEL_COLOR_SHADOW, name)
            draw_list.add_text(ui.ImVec2(sx, sy), color, name)

    def _invalidate_effector_scaler(self):
        """Drop the cached effector-target scaler so it is rebuilt from the
        latest scale + on-disk scaler/offsets config on next use."""
        self._effector_scaler_cache = {}

    def _get_effector_scaler(self, skeleton):
        """Build/cache a ``HumanToRobotScaler`` for the current retargeter config.

        Keyed by the skeleton object so both the live-motion skeleton and the
        calibration zero-pose skeleton can be visualized. The cache is cleared
        whenever the source scale changes or calibration values are written, so
        the overlay always reflects what retargeting would solve for.
        """
        key = id(skeleton)
        scaler = self._effector_scaler_cache.get(key)
        if scaler is not None and scaler.skeleton is skeleton:
            return scaler
        try:
            from soma_retargeter.robotics.human_to_robot_scaler import HumanToRobotScaler
            cfg = self._calibration_retargeter_cfg
            scaler = HumanToRobotScaler(
                skeleton,
                float(cfg['model_height']),
                self._calibration_scaler_cfg_path,
                offsets_file=self._calibration_offsets_cfg_path)
        except Exception as exc:  # noqa: BLE001 - surface build errors in the UI
            self._effector_target_status = f"effector-target scaler failed: {exc}"
            print(f"[WARN]: {self._effector_target_status}")
            scaler = None
        self._effector_scaler_cache[key] = scaler
        return scaler

    def _render_effector_targets(self):
        """Draw RGB coordinate frames at each IK effector *target*.

        Each frame shows where the retargeter wants the mapped robot body to be:
        orientation = ``source_q * offset.q`` and position = scaled root +
        scaled geocentric + ``R(q) * offset.p`` (the exact target the IK solver
        chases). Compare against ``Show IK Map Axes`` (the robot bodies' actual
        frames) to see the residual the IK has to absorb.
        """
        def draw_from_instance(inst, base_id):
            scaler = self._get_effector_scaler(inst.skeleton)
            if scaler is None:
                return
            eff = scaler.compute_effectors_from_skeleton(inst, True)
            transforms = [wp.transform(wp.vec3(*t[0:3]), wp.quat(*t[3:7])) for t in eff]
            if transforms:
                self.effector_target_coordinate_renderer.draw(
                    self.viewer, transforms, 0.12, base_id, width=0.01)

        if self.calibration_mode:
            draw_from_instance(self.zero_pose_reference_instance, 8500)
        elif len(self.skeleton_instances) > 0:
            for i, inst in enumerate(self.skeleton_instances):
                prev_xform = wp.transform(inst.xform)
                inst.xform = wp.mul(self.animation_offsets[i], inst.xform)
                draw_from_instance(inst, 8500 + i)
                inst.xform = prev_xform

    def reset_calibration_pose_to_reference(self):
        """Reset ``self.calibration_joint_q`` to the per-robot reference pose."""
        self.calibration_joint_q = self.robot_default_joint_q_values.astype(np.float32).copy()
        ref = self._calibration_reference
        if ref.get("base_pos"):
            self.calibration_joint_q[0:3] = ref["base_pos"]
        if ref.get("base_quat_xyzw"):
            self.calibration_joint_q[3:7] = ref["base_quat_xyzw"]
        ref_angles = ref.get("angles", {})
        for j in self.calibration_revolute_joints:
            self.calibration_joint_q[j["q_idx"]] = ref_angles.get(j["name"], 0.0)

    def reset_calibration_pose_to_zero(self):
        """Reset ``self.calibration_joint_q`` to all-zero joints (free base kept identity)."""
        self.calibration_joint_q = self.robot_default_joint_q_values.astype(np.float32).copy()
        self.calibration_joint_q[3:7] = [0.0, 0.0, 0.0, 1.0]
        for j in self.calibration_revolute_joints:
            self.calibration_joint_q[j["q_idx"]] = 0.0

    def gui(self, ui):
        self.ui_playback_controls(ui)
        self.ui_scene_options(ui)
        self.ui_right_sidebar(ui)
        if self.show_ik_map_axes:
            self._draw_ik_map_labels(ui)

    def load_csv_file(self, path):
        self.robot_csv_animation_buffers[0] = csv_utils.load_csv(path, csv_config=self.csv_config)
        self.compute_playback_total_time()

    def load_bvh_file(self, path):
        self._loaded_bvh_path = path
        self.animation_buffers = []
        self.skeleton_instances = []
        if self.skeleton_renderer is not None:
            self.skeleton_renderer.clear(self.viewer)
        if self.skeletal_mesh_renderer is not None:
            self.skeletal_mesh_renderer.clear(self.viewer)
        if self.coordinate_renderer is not None:
            self.coordinate_renderer.clear(self.viewer)
        if self.effector_target_coordinate_renderer is not None:
            self.effector_target_coordinate_renderer.clear(self.viewer)
        self._invalidate_effector_scaler()

        self.skeleton, animation = bvh_utils.load_bvh(
            path, position_scale=self.source_position_scale, recenter_xy=self.source_recenter_xy,
            position_scale_upper=self.source_position_scale_upper,
            position_scale_lower=self.source_position_scale_lower)
        self.skeleton_renderer = SkeletonRenderer(self.skeleton, [0])
        self.skeleton_instances = [SkeletonInstance(self.skeleton, _DEFAULT_COLOR, self.converter.transform(wp.transform_identity()))]

        # Resolve IK-map source joint names to skeleton joint indices.
        self._ik_map_source_joint_indices = [
            self.skeleton.joint_index(name)
            for name in self._ik_map_source_joint_names
            if self.skeleton.joint_index(name) >= 0
        ]
        self.animation_offsets = [wp.transform_identity()] * len(self.skeleton_instances)
        self.animation_buffers = [animation]

        self.skeletal_mesh = pipeline_utils.get_source_model_mesh(self.source_type, self.skeleton)
        if self.skeletal_mesh is not None:
            self.skeletal_mesh_renderer = SkeletalMeshRenderer(self.skeletal_mesh)
        else:
            # Native skeleton: no skin mesh -> show bones instead.
            self.skeletal_mesh_renderer = None
            self.show_skeleton = True
        self.compute_playback_total_time()

    def compute_playback_total_time(self):
        bvh_max_time = 0.0
        for buffer in self.animation_buffers:
            if buffer is not None:
                bvh_max_time = max(bvh_max_time, buffer.num_frames * (1 / buffer.sample_rate))
        
        csv_max_time = 0.0
        for buffer in self.robot_csv_animation_buffers:
            if buffer is not None:
                csv_max_time = max(csv_max_time, buffer.num_frames * (1 / buffer.sample_rate))

        self.playback_total_time = max(bvh_max_time, csv_max_time)
        self.playback_time = wp.clamp(self.playback_time, 0.0, self.playback_total_time)

    def update_robot_states(self):
        if self.calibration_mode:
            wp.copy(
                self.model.joint_q,
                wp.array(self.calibration_joint_q, dtype=wp.float32),
                0, 0, len(self.calibration_joint_q))
            newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state, None)
            return

        for i in range(self.num_robots):
            robot_offset = self.robot_offsets[i]

            joint_q_offset = self.robot_joint_q_offsets[i]
            if self.robot_csv_animation_buffers[i] is not None:
                buffer = self.robot_csv_animation_buffers[i]
                # Apply visual offset
                prev_xform = wp.transform(buffer.xform)
                buffer.xform = robot_offset

                data = buffer.sample(self.playback_time)
                wp.copy(self.model.joint_q, wp.array(data, dtype=wp.float32), joint_q_offset, 0, self.robot_num_joint_q)
                buffer.xform = prev_xform
            else:
                root_tx = wp.mul(
                    robot_offset,
                    wp.transform(*self.robot_default_joint_q_values[joint_q_offset:(joint_q_offset + 7)]))

                wp.copy(
                    self.model.joint_q,
                    wp.array(self.robot_default_joint_q_values[joint_q_offset:(joint_q_offset + self.robot_num_joint_q)], dtype=wp.float32),
                    joint_q_offset,
                    0, self.robot_num_joint_q)
                wp.copy(self.model.joint_q, wp.array(root_tx[0:7], dtype=wp.float32), joint_q_offset, 0, 7)

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state, None)

    def step(self):
        self.time += self.frame_dt
        if self.is_playing and not self.calibration_mode:
            self.playback_time += self.frame_dt * self.playback_speed
            if self.playback_loop and self.playback_total_time > 0.0:
                self.playback_time %= self.playback_total_time
            else:
                self.playback_time = max(0.0, min(self.playback_time, self.playback_total_time))

        if not self.calibration_mode:
            for i in range(len(self.animation_buffers)):
                self.skeleton_instances[i].set_local_transforms(self.animation_buffers[i].sample(self.playback_time))

        def clamp_gizmo_transform(tx: wp.transform):
            return wp.transform(
                wp.vec3(tx.p[0], tx.p[1], tx.p[2]),
                math_utils.quat_twist(wp.vec3(0.0, 0.0, 1.0), tx.q))

        for i in range(len(self.robot_offsets)):
            self.robot_offsets[i] = clamp_gizmo_transform(self.robot_offsets[i])
        for i in range(len(self.animation_offsets)):
            self.animation_offsets[i] = clamp_gizmo_transform(self.animation_offsets[i])
        for obj in self.scene_objects:
            self._ground_scene_box(obj)

        self.update_robot_states()

    def render(self):
        self.viewer.begin_frame(self.time)
        if not self.calibration_mode and len(self.animation_buffers) > 0:
            for i in range(len(self.skeleton_instances)):
                prev_xform = wp.transform(self.skeleton_instances[i].xform)
                self.skeleton_instances[i].xform = wp.mul(self.animation_offsets[i], self.skeleton_instances[i].xform)
                if self.show_skeleton:
                    self.skeleton_renderer.draw(self.viewer, self.skeleton_instances[i], i)
                if self.show_skeleton_joint_axes:
                    tx = self.skeleton_instances[i].compute_global_transforms()
                    self.coordinate_renderer.draw(self.viewer, tx, 0.1, i)
                if self.show_skeleton_mesh and self.skeletal_mesh_renderer is not None:
                    self.skeletal_mesh_renderer.draw(self.viewer, self.skeleton_instances[i], self.skeleton_instances[i].color, i)
                self.skeleton_instances[i].xform = prev_xform

        if self.calibration_mode and self.show_zero_pose_reference:
            if self.zero_pose_reference_mesh_renderer is not None:
                self.zero_pose_reference_mesh_renderer.draw(
                    self.viewer, self.zero_pose_reference_instance,
                    self.zero_pose_reference_instance.color, 99)
            else:
                self.zero_pose_reference_skeleton_renderer.draw(
                    self.viewer, self.zero_pose_reference_instance, 99)

        if self.show_gizmos:
            for i, offset in enumerate(self.robot_offsets):
                self.viewer.log_gizmo(f"robot_offset{i}", offset)
            for i, offset in enumerate(self.animation_offsets):
                self.viewer.log_gizmo(f"animation_offset{i}", offset)

        self._render_scene_objects()

        self.viewer.log_state(self.state)

        if self.show_ik_map_axes:
            self._render_ik_map_axes()

        if self.show_effector_targets:
            self._render_effector_targets()

        self.viewer.end_frame()

    def run(self):
        while self.viewer.is_running():
            with wp.ScopedTimer("step", active=False):
                self.step()
            with wp.ScopedTimer("render", active=False):
                self.render()

        self.viewer.close()

    def retarget_motion(self):
        retarget_source = self.retarget_source_options[self.retarget_source_idx]
        retarget_target = self.retarget_target_options[self.retarget_target_idx]
        retarget_solver = self.retarget_solver_options[self.retarget_solver_idx]
        
        if (retarget_solver == 'Newton'):
            import soma_retargeter.pipelines.newton_pipeline as newton_pipeline
            # Pass the live source scale so the pipeline's init pose is loaded at
            # the same size as the (already-scaled) motion buffers we feed it.
            pipeline = newton_pipeline.NewtonPipeline(
                self.skeleton, retarget_source, retarget_target,
                source_position_scale=self.source_position_scale,
                source_position_scale_upper=self.source_position_scale_upper,
                source_position_scale_lower=self.source_position_scale_lower)
        else:
            raise(ValueError(f"[ERROR]: Unknown retargeter solver [{retarget_solver}"))
        
        pipeline.set_root_amplitude_scale(*self.root_amplitude_scale)
        pipeline.set_full_amplitude_scale(*self.full_amplitude_scale)
        r_offsets = [wp.transform(wp.vec3(0,0,0), wp.quat(*s.xform[3:7])) for s in self.skeleton_instances]
        pipeline.add_input_motions(self.animation_buffers, r_offsets, True)
        buffers = pipeline.execute()
        
        if buffers is not None:
            t_offsets = [wp.transform(wp.vec3(*s.xform[:3]), wp.quat_identity()) for s in self.skeleton_instances]
            for i, buffer in enumerate(buffers):
                buffer.xform = t_offsets[i]

        self.robot_csv_animation_buffers[0] = buffers[0]

    def ui_scene_options(self, ui):
        import tkinter as tk
        from tkinter import filedialog as tk_filedialog
        
        viewport = ui.get_main_viewport()

        panel_size = ui.ImVec2(320, 320)
        ui.set_next_window_pos(
            ui.ImVec2(
                viewport.size.x - _UI_NEWTON_PANEL_MARGIN - panel_size.x,
                viewport.size.y - _UI_NEWTON_PANEL_MARGIN - panel_size.y))
        
        ui.set_next_window_size(panel_size)
        ui.set_next_window_bg_alpha(_UI_NEWTON_PANEL_ALPHA)

        ui.begin("Scene Options", flags=(ui.WindowFlags_.no_collapse | ui.WindowFlags_.no_resize))
        ui.separator()

        # Motion options
        if ui.collapsing_header("Motion", flags=ui.TreeNodeFlags_.default_open):
            ui.separator()
            ui.align_text_to_frame_padding()
            ui.text("BVH Motion:")
            ui.same_line()
            
            ui.push_id(100)
            if ui.button("Load"):
                root = tk.Tk()
                root.withdraw()
                bvh_path = tk_filedialog.askopenfilename(
                    title='Load BVH File',
                    defaultextension=".bvh",
                    filetypes=[('BVH files', '*.bvh')])

                if bvh_path:
                    self.load_bvh_file(bvh_path)
            ui.pop_id()

            if (len(self.animation_buffers) == 0):
                ui.begin_disabled()

            ui.same_line()
            if ui.button("Retarget"):
                self.retarget_motion()

            ui.spacing()
            ui.align_text_to_frame_padding()
            ui.text("Motion Amplitude:")
            ui.text_disabled("Adjust, then press Retarget to apply.")
            ui.text_disabled("Left: Full Body (may slip/drift)  Right: Root only")

            _AMP_SLIDER_WIDTH = 80
            amp_axis_labels = ("X", "Y", "Z")
            for axis, axis_label in enumerate(amp_axis_labels):
                ui.set_next_item_width(_AMP_SLIDER_WIDTH)
                fb_changed, fb_val = ui.slider_float(
                    f"FB {axis_label}##fbamp{axis}", float(self.full_amplitude_scale[axis]), 0.1, 2.0, "%.2f")
                if fb_changed:
                    self.full_amplitude_scale[axis] = fb_val

                ui.same_line()
                ui.set_next_item_width(_AMP_SLIDER_WIDTH)
                r_changed, r_val = ui.slider_float(
                    f"Root {axis_label}##rootamp{axis}", float(self.root_amplitude_scale[axis]), 0.1, 2.0, "%.2f")
                if r_changed:
                    self.root_amplitude_scale[axis] = r_val

            if ui.small_button("Reset Full Body##fbamp"):
                self.full_amplitude_scale = [1.0, 1.0, 1.0]
            ui.same_line()
            if ui.small_button("Reset Root##rootamp"):
                self.root_amplitude_scale = [1.0, 1.0, 1.0]

            if (len(self.animation_buffers) == 0):
                ui.end_disabled()

            ui.align_text_to_frame_padding()
            ui.text("CSV Motion:")
            ui.same_line()
            
            ui.push_id(200)
            if ui.button("Load"):
                root = tk.Tk()
                root.withdraw()
                csv_path = tk_filedialog.askopenfilename(
                    title='Load CSV File',
                    defaultextension=".csv",
                    filetypes=[('CSV files', '*.csv')])

                if csv_path:
                    self.load_csv_file(csv_path)

            if self.robot_csv_animation_buffers[0] is None:
                ui.begin_disabled()
            ui.pop_id()

            ui.same_line()
            if ui.button("Save"):
                root = tk.Tk()
                root.withdraw()

                save_path = tk_filedialog.asksaveasfilename(
                    title="Save CSV File",
                    defaultextension=".csv",
                    filetypes=[("CSV files", "*.csv")])
                if save_path:
                    # Bake in the live viewport gizmo offset (e.g. dragging the
                    # robot up/down) so the saved CSV matches what is shown on
                    # screen -- mirrors the override done in update_robot_states().
                    save_buffer = self.robot_csv_animation_buffers[0]
                    prev_xform = wp.transform(save_buffer.xform)
                    save_buffer.xform = self.robot_offsets[0]
                    try:
                        csv_utils.save_csv(save_path, save_buffer, csv_config=self.csv_config)
                    finally:
                        save_buffer.xform = prev_xform

            if self.robot_csv_animation_buffers[0] is None:
                ui.end_disabled()

        # Visibility options
        ui.spacing()
        if ui.collapsing_header("Visibility", flags=ui.TreeNodeFlags_.default_open):
            ui.separator()

            changed, self.show_skeleton_mesh = ui.checkbox("Show Mesh", self.show_skeleton_mesh)
            if changed and self.skeletal_mesh_renderer is not None:
                self.skeletal_mesh_renderer.clear(self.viewer)
            changed, self.show_skeleton = ui.checkbox("Show Skeleton", self.show_skeleton)
            if changed and self.skeleton_renderer is not None:
                self.skeleton_renderer.clear(self.viewer)
            changed, self.show_skeleton_joint_axes = ui.checkbox("Show Joint Axes", self.show_skeleton_joint_axes)
            if changed and self.coordinate_renderer is not None:
                self.coordinate_renderer.clear(self.viewer)
            changed, self.show_ik_map_axes = ui.checkbox("Show IK Map Axes", self.show_ik_map_axes)
            if changed and not self.show_ik_map_axes:
                self.ik_map_coordinate_renderer.clear(self.viewer)
            changed, self.show_effector_targets = ui.checkbox("Show Effector Targets", self.show_effector_targets)
            if changed:
                # Rebuild from the latest on-disk config each time it's toggled on.
                self._invalidate_effector_scaler()
                if not self.show_effector_targets:
                    self.effector_target_coordinate_renderer.clear(self.viewer)
            if self.show_effector_targets and self._effector_target_status:
                ui.text_colored(ui.ImVec4(1.0, 0.6, 0.4, 1.0), self._effector_target_status)
            _, self.show_gizmos = ui.checkbox("Show Gizmos", self.show_gizmos)
            ui.same_line()
            if ui.button("Reset"):
                self.robot_offsets = [wp.transform(wp.vec3(0.0, i - (self.num_robots - 1) / 2.0, 0.0), wp.quat_identity()) for i in range(self.num_robots)]
                self.animation_offsets = [wp.transform_identity()] * len(self.skeleton_instances)
        ui.end()

    def ui_right_sidebar(self, ui):
        """Single right-side window with collapsible Calibration + Scene Objects."""
        viewport = ui.get_main_viewport()
        panel_width = 360
        margin = _UI_NEWTON_PANEL_MARGIN
        max_panel_h = float(viewport.size.y) - 2.0 * margin
        ui.set_next_window_pos(
            ui.ImVec2(viewport.size.x - margin - panel_width, margin))
        ui.set_next_window_size_constraints(
            ui.ImVec2(panel_width, 80),
            ui.ImVec2(panel_width, max_panel_h))
        ui.set_next_window_bg_alpha(_UI_NEWTON_PANEL_ALPHA)

        ui.begin(
            "Right Panels",
            flags=(ui.WindowFlags_.always_auto_resize | ui.WindowFlags_.no_collapse))

        if ui.collapsing_header(
                "Calibration (Compute Bias)",
                flags=ui.TreeNodeFlags_.default_open):
            self._draw_calibration_content(ui)

        ui.separator()

        if ui.collapsing_header(
                "Scene Objects",
                flags=ui.TreeNodeFlags_.default_open):
            self._draw_scene_objects_content(ui)

        ui.end()

    def _clear_reference_overlay(self):
        """Remove the zero-pose reference overlay (mesh for SOMA, bones otherwise)."""
        if self.zero_pose_reference_mesh_renderer is not None:
            self.zero_pose_reference_mesh_renderer.clear(self.viewer)
        if getattr(self, "zero_pose_reference_skeleton_renderer", None) is not None:
            self.zero_pose_reference_skeleton_renderer.clear(self.viewer)

    def _draw_calibration_content(self, ui):
        """Calibration controls (inside collapsible section)."""
        import tkinter as tk
        from tkinter import filedialog as tk_filedialog

        changed, self.calibration_mode = ui.checkbox(
            "Enable Calibration Mode", self.calibration_mode)
        if changed and not self.calibration_mode:
            self._clear_reference_overlay()
        elif changed and self.calibration_mode:
            self.is_playing = False

        ui.text_colored(
            ui.ImVec4(0.6, 0.8, 1.0, 1.0),
            f"Robot: {self.robot_type}   "
            f"Revolute joints: {len(self.calibration_revolute_joints)}")

        ui.separator()
        ui.text("Source Position Scale (live)")
        ui.set_next_item_width(200)
        sc_changed, sc_val = ui.slider_float(
            "scale##srcscale", float(self.source_position_scale), 0.1, 5.0, "%.3f")
        if sc_changed:
            self._apply_source_scale(sc_val)
        ui.same_line()
        if ui.button("Save scale to config"):
            self._do_write_source_scale()
        ui.set_next_item_width(200)
        up_changed, up_val = ui.slider_float(
            "upper body##srcscaleup", float(self.source_position_scale_upper), 0.1, 5.0, "%.3f")
        if up_changed:
            self._apply_source_scale_group('upper', up_val)
        ui.set_next_item_width(200)
        lo_changed, lo_val = ui.slider_float(
            "lower body##srcscalelo", float(self.source_position_scale_lower), 0.1, 5.0, "%.3f")
        if lo_changed:
            self._apply_source_scale_group('lower', lo_val)
        ui.text_colored(
            ui.ImVec4(0.7, 0.7, 0.7, 1.0),
            "Rescales motion + zero pose together; used for bias + retarget.\n"
            "Upper/lower multiply on top of the uniform scale (arms vs legs).")

        if not self.calibration_mode:
            ui.text_wrapped(
                "Turn on Calibration Mode to freeze BVH playback, "
                "load the zero-pose reference, and edit the robot's "
                "joint angles to match.")
            return

        ui.separator()
        ref_changed, self.show_zero_pose_reference = ui.checkbox(
            "Show Zero Pose Overlay", self.show_zero_pose_reference)
        if ref_changed and not self.show_zero_pose_reference:
            self._clear_reference_overlay()

        ui.separator()
        # One-click "foolproof" calibration: match the robot to the zero pose
        # (sliders below), then click. Persists the live source scale (if
        # unsaved) and recomputes + writes joint_scales AND joint_offsets
        # (including the offset.p residual) from one calibration snapshot.
        if ui.button("One-Click Calibrate (scales + offsets)"):
            self._do_calibrate_all()
        ui.text_colored(
            ui.ImVec4(0.7, 0.85, 0.7, 1.0),
            "Saves the current source scale, then recomputes & writes "
            "joint_scales AND joint_offsets (incl. offset.p).")

        have_scales = bool(self._calibration_last_scales)
        have_offsets = self._calibration_last_offsets is not None
        if not have_scales and not have_offsets:
            ui.begin_disabled()
        if ui.button("Print scales##s"):
            print(json.dumps(self._calibration_last_scales, indent=4))
        ui.same_line()
        if ui.button("Print offsets##o"):
            print(json.dumps(self._calibration_last_offsets, indent=4))
        if not have_scales and not have_offsets:
            ui.end_disabled()

        if have_scales:
            ui.text("Computed joint_scales:")
            if ui.begin_child("##scales_preview", ui.ImVec2(0, 120)):
                for k, v in self._calibration_last_scales.items():
                    ui.text(f"{k:14s}  {v:.4f}")
            ui.end_child()

        if have_offsets:
            ui.text("Computed offset.q (xyzw) per joint:")
            if ui.begin_child("##offset_preview", ui.ImVec2(0, 140)):
                for soma_joint, vals in self._calibration_last_offsets.items():
                    q = vals[1]
                    ui.text(
                        f"{soma_joint:14s}  "
                        f"({q[0]:+.3f}, {q[1]:+.3f}, {q[2]:+.3f}, {q[3]:+.3f})")
            ui.end_child()

        ui.separator()
        if ui.collapsing_header("Robot Joint Sliders",
                                flags=ui.TreeNodeFlags_.default_open):
            ui.text("Adjust joint angles (rad) to match the zero pose.")
            if ui.button("Reset to Reference"):
                self.reset_calibration_pose_to_reference()
            ui.same_line()
            if ui.button("Reset to Zero"):
                self.reset_calibration_pose_to_zero()
            ui.same_line()
            if ui.button("Save Pose..."):
                root = tk.Tk()
                root.withdraw()
                save_path = tk_filedialog.asksaveasfilename(
                    title="Save Reference Pose JSON",
                    defaultextension=".json",
                    filetypes=[("JSON files", "*.json")])
                if save_path:
                    self._save_calibration_pose(save_path)

            if ui.button("Load Pose..."):
                root = tk.Tk()
                root.withdraw()
                load_path = tk_filedialog.askopenfilename(
                    title="Load Reference Pose JSON",
                    defaultextension=".json",
                    filetypes=[("JSON files", "*.json")])
                if load_path:
                    self._load_calibration_pose(load_path)

            ui.spacing()
            if ui.begin_child("##slider_scroll", ui.ImVec2(0, 280)):
                for j in self.calibration_revolute_joints:
                    cur = float(self.calibration_joint_q[j["q_idx"]])
                    lo = max(j["lo"], -6.283)
                    hi = min(j["hi"], 6.283)
                    ui.set_next_item_width(180)
                    s_changed, new_val = ui.slider_float(
                        f"{j['name']}##js{j['q_idx']}",
                        cur, lo, hi, "%.3f")
                    if s_changed:
                        self.calibration_joint_q[j["q_idx"]] = new_val
            ui.end_child()

        if self._calibration_status:
            ui.separator()
            ui.text_wrapped(self._calibration_status)

    def _ground_scene_box(self, obj):
        """Keep box bottom on z=0; center z is always half the box height."""
        if obj.get("type") != "box":
            return
        sz = obj["size"]
        tx = obj["transform"]
        z = max(0.001, float(sz[2])) * 0.5
        p = tx.p
        if abs(float(p[2]) - z) < 1e-8:
            return
        obj["transform"] = wp.transform(
            wp.vec3(float(p[0]), float(p[1]), z),
            tx.q)

    def _add_scene_box(self):
        obj_id = self._scene_object_next_id
        self._scene_object_next_id += 1
        n = len(self.scene_objects)
        size = wp.vec3(0.5, 0.5, 0.5)
        obj = {
            "id": obj_id,
            "name": f"Cube {obj_id}",
            "type": "box",
            "size": size,
            "transform": wp.transform(
                wp.vec3(1.0, float(n) * 0.5, float(size[2]) * 0.5),
                wp.quat_identity()),
            "color": wp.vec3(0.9, 0.45, 0.15),
        }
        self.scene_objects.append(obj)
        self._selected_scene_object_idx = len(self.scene_objects) - 1

    def _ensure_scene_unit_box_mesh(self):
        if self._scene_unit_box_mesh_path is not None:
            return self._scene_unit_box_mesh_path
        path = "/scene_objects/_unit_box_mesh"
        self.viewer.log_geo(
            path,
            newton.GeoType.BOX,
            (1.0, 1.0, 1.0),
            0.0,
            True,
            hidden=True,
        )
        self._scene_unit_box_mesh_path = path
        return path

    def _hide_scene_object(self, obj_id):
        path = f"/scene_objects/box_{obj_id}"
        if hasattr(self.viewer, "objects") and path in self.viewer.objects:
            obj = self.viewer.objects[path]
            destroy = getattr(obj, "destroy", None)
            if callable(destroy):
                destroy()
            del self.viewer.objects[path]
        if hasattr(self.viewer, "_gizmo_log"):
            self.viewer._gizmo_log.pop(f"scene_object_{obj_id}", None)
        self._scene_object_logged_ids.discard(obj_id)

    def _remove_selected_scene_object(self):
        if not (0 <= self._selected_scene_object_idx < len(self.scene_objects)):
            return
        removed_id = self.scene_objects[self._selected_scene_object_idx]["id"]
        self._hide_scene_object(removed_id)
        del self.scene_objects[self._selected_scene_object_idx]
        if not self.scene_objects:
            self._selected_scene_object_idx = -1
        else:
            self._selected_scene_object_idx = min(
                self._selected_scene_object_idx,
                len(self.scene_objects) - 1)

    def _save_scene_objects(self, path):
        objects = []
        for obj in self.scene_objects:
            sz = obj["size"]
            p = obj["transform"].p
            q = obj["transform"].q
            col = obj["color"]
            objects.append({
                "id": obj["id"],
                "name": obj["name"],
                "type": obj["type"],
                "size": [float(sz[0]), float(sz[1]), float(sz[2])],
                "position": [float(p[0]), float(p[1]), float(p[2])],
                "quat_xyzw": [float(q[0]), float(q[1]), float(q[2]), float(q[3])],
                "color": [float(col[0]), float(col[1]), float(col[2])],
            })
        data = {
            "_comment": "Scene objects saved from BVH to CSV Converter.",
            "next_id": self._scene_object_next_id,
            "objects": objects,
        }
        pathlib.Path(path).write_text(json.dumps(data, indent=4))
        self._scene_objects_status = f"Saved {len(objects)} object(s) to {path}"
        print(f"[INFO]: {self._scene_objects_status}")

    def _load_scene_objects(self, path):
        try:
            data = json.loads(pathlib.Path(path).read_text())
        except Exception as e:
            self._scene_objects_status = f"Failed to load: {e}"
            return
        loaded = []
        for entry in data.get("objects", []):
            if "size" in entry:
                sz = entry["size"]
            elif "half_size" in entry:
                hs = entry["half_size"]
                sz = [2.0 * float(hs[0]), 2.0 * float(hs[1]), 2.0 * float(hs[2])]
            else:
                sz = [0.5, 0.5, 0.5]
            pos = entry.get("position", [0.0, 0.0, 0.0])
            quat = entry.get("quat_xyzw", [0.0, 0.0, 0.0, 1.0])
            col = entry.get("color", [0.9, 0.45, 0.15])
            obj = {
                "id": int(entry.get("id", self._scene_object_next_id)),
                "name": entry.get("name", f"Cube {entry.get('id', 0)}"),
                "type": entry.get("type", "box"),
                "size": wp.vec3(float(sz[0]), float(sz[1]), float(sz[2])),
                "transform": wp.transform(
                    wp.vec3(float(pos[0]), float(pos[1]), float(pos[2])),
                    wp.quat(float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))),
                "color": wp.vec3(float(col[0]), float(col[1]), float(col[2])),
            }
            self._ground_scene_box(obj)
            loaded.append(obj)
        for old_id in list(self._scene_object_logged_ids):
            self._hide_scene_object(old_id)
        self.scene_objects = loaded
        self._scene_object_next_id = int(data.get("next_id", 0))
        for obj in loaded:
            self._scene_object_next_id = max(self._scene_object_next_id, obj["id"] + 1)
        self._selected_scene_object_idx = 0 if loaded else -1
        self._scene_objects_status = f"Loaded {len(loaded)} object(s) from {path}"
        print(f"[INFO]: {self._scene_objects_status}")

    def _draw_scene_objects_content(self, ui):
        """Scene-object controls (inside collapsible section)."""
        import tkinter as tk
        from tkinter import filedialog as tk_filedialog

        if ui.button("Add Cube"):
            self._add_scene_box()
        ui.same_line()
        can_remove = 0 <= self._selected_scene_object_idx < len(self.scene_objects)
        if not can_remove:
            ui.begin_disabled()
        if ui.button("Remove"):
            self._remove_selected_scene_object()
        if not can_remove:
            ui.end_disabled()
        ui.same_line()
        if ui.button("Save..."):
            root = tk.Tk()
            root.withdraw()
            save_path = tk_filedialog.asksaveasfilename(
                title="Save Scene Objects",
                defaultextension=".json",
                filetypes=[("JSON files", "*.json")])
            if save_path:
                self._save_scene_objects(save_path)
        ui.same_line()
        if ui.button("Load..."):
            root = tk.Tk()
            root.withdraw()
            load_path = tk_filedialog.askopenfilename(
                title="Load Scene Objects",
                filetypes=[("JSON files", "*.json")])
            if load_path:
                self._load_scene_objects(load_path)

        _, self.show_scene_object_gizmos = ui.checkbox(
            "Show Transform Gizmo (drag in viewport)", self.show_scene_object_gizmos)

        ui.separator()
        ui.text("Select a cube, edit XY / size, or drag the gizmo (Z locked to ground).")
        list_h = max(100.0, min(220.0, 24.0 * max(1, len(self.scene_objects)) + 8.0))
        if ui.begin_child("##scene_object_list", ui.ImVec2(0, list_h)):
            for i, obj in enumerate(self.scene_objects):
                selected = i == self._selected_scene_object_idx
                # imgui.selectable returns (clicked, selected); a non-empty tuple is
                # always truthy, so we must unpack clicked explicitly.
                clicked, _ = ui.selectable(
                    f"{obj['name']}##scene_obj_{obj['id']}", selected)
                if clicked:
                    self._selected_scene_object_idx = i
        ui.end_child()

        if 0 <= self._selected_scene_object_idx < len(self.scene_objects):
            obj = self.scene_objects[self._selected_scene_object_idx]
            ui.push_id(obj["id"])
            ui.separator()
            ui.text(f"Editing: {obj['name']}")

            sz = obj["size"]
            sz_vals = [float(sz[0]), float(sz[1]), float(sz[2])]
            ui.set_next_item_width(-1)
            sz_changed, new_sz = ui.input_float3(
                "Size (m) X / Y / Z", sz_vals, "%.4f")
            if sz_changed:
                obj["size"] = wp.vec3(
                    max(0.001, float(new_sz[0])),
                    max(0.001, float(new_sz[1])),
                    max(0.001, float(new_sz[2])))
                self._ground_scene_box(obj)

            tx = obj["transform"]
            p = tx.p
            # Z is derived from height so the box bottom stays on the ground.
            pos_xy = [float(p[0]), float(p[1])]
            ui.set_next_item_width(-1)
            pos_changed, new_xy = ui.input_float2(
                "Position XY (m)", pos_xy, "%.4f")
            if pos_changed:
                obj["transform"] = wp.transform(
                    wp.vec3(float(new_xy[0]), float(new_xy[1]), float(p[2])),
                    tx.q)
                self._ground_scene_box(obj)
            ui.text(f"Position Z (locked, bottom on ground): {float(obj['transform'].p[2]):.4f} m")
            ui.pop_id()

        status = getattr(self, "_scene_objects_status", None)
        if status:
            ui.separator()
            ui.text_wrapped(status)

    def _render_scene_objects(self):
        active_ids = set()
        unit_mesh = self._ensure_scene_unit_box_mesh()

        for i, obj in enumerate(self.scene_objects):
            if obj.get("type") != "box":
                continue
            obj_id = obj["id"]
            active_ids.add(obj_id)
            sz = obj["size"]
            path = f"/scene_objects/box_{obj_id}"
            xforms = wp.array([obj["transform"]], dtype=wp.transform)
            scales = wp.array(
                [wp.vec3(float(sz[0]) * 0.5, float(sz[1]) * 0.5, float(sz[2]) * 0.5)],
                dtype=wp.vec3)
            colors = wp.array([obj["color"]], dtype=wp.vec3)
            # Unit box (half=1 m) × scale (full_size / 2) = box with full_size edge lengths.
            self.viewer.log_instances(
                path, unit_mesh, xforms, scales, colors, None, hidden=False)

            show_gizmo = (
                self.show_scene_object_gizmos
                and i == self._selected_scene_object_idx)
            if show_gizmo:
                self.viewer.log_gizmo(f"scene_object_{obj_id}", obj["transform"])

        for stale_id in self._scene_object_logged_ids - active_ids:
            self._hide_scene_object(stale_id)
        self._scene_object_logged_ids = active_ids

    def _save_calibration_pose(self, path):
        ref_data = {
            "_comment": "Reference pose saved from the in-app calibration panel.",
            "robot_type": self.robot_type,
            "base_pos": [float(x) for x in self.calibration_joint_q[0:3]],
            "base_quat_xyzw": [float(x) for x in self.calibration_joint_q[3:7]],
            "joint_order": [j["name"] for j in self.calibration_revolute_joints],
            "joint_angles_rad": [
                float(self.calibration_joint_q[j["q_idx"]])
                for j in self.calibration_revolute_joints],
        }
        pathlib.Path(path).write_text(json.dumps(ref_data, indent=4))
        self._calibration_status = f"Saved pose to {path}"
        print(f"[INFO]: {self._calibration_status}")

    def _load_calibration_pose(self, path):
        try:
            ref = json.loads(pathlib.Path(path).read_text())
        except Exception as e:
            self._calibration_status = f"Failed to load: {e}"
            return
        if "base_pos" in ref:
            self.calibration_joint_q[0:3] = ref["base_pos"]
        if "base_quat_xyzw" in ref:
            self.calibration_joint_q[3:7] = ref["base_quat_xyzw"]
        order = ref.get("joint_order")
        angles = ref.get("joint_angles_rad", [])
        if order and len(order) == len(angles):
            name_to_idx = {j["name"]: j["q_idx"] for j in self.calibration_revolute_joints}
            for nm, ang in zip(order, angles):
                if nm in name_to_idx:
                    self.calibration_joint_q[name_to_idx[nm]] = float(ang)
        else:
            # Fallback: assume same order as our revolute joints
            for j, ang in zip(self.calibration_revolute_joints, angles):
                self.calibration_joint_q[j["q_idx"]] = float(ang)
        self._calibration_status = f"Loaded pose from {path}"
        print(f"[INFO]: {self._calibration_status}")

    def _calibration_height_ratio(self):
        cfg = self._calibration_retargeter_cfg
        scaler = io_utils.load_json(self._calibration_scaler_cfg_path)
        return float(cfg.get('model_height', 1.8)) / float(
            scaler.get('human_height_assumption', 1.8))

    def _calibration_collect(self):
        """Snapshot source zero-pose and current robot pose for calibration."""
        ref_globals = self.zero_pose_reference_skeleton.compute_global_transforms(
            self.zero_pose_reference_local_zero,
            self.converter.transform(wp.transform_identity()))
        body_q = self.state.body_q.numpy()
        link_globals = calibration_utils.collect_robot_link_globals(
            self.robot_builder, body_q)
        return ref_globals, link_globals

    def _do_compute_scales(self):
        """Compute joint_scales from the current matching reference poses."""
        ik_map = self._calibration_retargeter_cfg.get('ik_map', {})
        ref_globals, link_globals = self._calibration_collect()
        scaler_cfg = io_utils.load_json(self._calibration_scaler_cfg_path)
        new_scales = calibration_utils.compute_scales(
            ref_globals,
            self.zero_pose_reference_skeleton.joint_names,
            link_globals,
            ik_map,
            height_ratio=self._calibration_height_ratio(),
            mode=scaler_cfg.get('scaling_mode', 'geocentric'),
            joint_parents=scaler_cfg.get('joint_parents'))
        self._calibration_last_scales = new_scales
        self._calibration_status = (
            f"Computed {len(new_scales)} joint_scales. "
            "Click 'Write scales to Config' to persist.")
        print(f"[INFO]: {self._calibration_status}")

    def _do_compute_bias(self):
        """Compute joint_offsets from current source zero pose + robot pose."""
        ik_map = self._calibration_retargeter_cfg.get('ik_map', {})
        ref_globals, link_globals = self._calibration_collect()

        # Use the latest in-memory scales if the user just computed them,
        # otherwise fall back to whatever is currently saved in the config.
        scaler = io_utils.load_json(self._calibration_scaler_cfg_path)
        scales = getattr(self, "_calibration_last_scales", None) or scaler.get('joint_scales', {})

        compute_pos = bool(getattr(self, "_calc_position", False))
        new_offsets = calibration_utils.compute_offsets(
            ref_globals,
            self.zero_pose_reference_skeleton.joint_names,
            link_globals,
            ik_map,
            compute_position=compute_pos,
            joint_scales=scales,
            height_ratio=self._calibration_height_ratio(),
            mode=scaler.get('scaling_mode', 'geocentric'),
            joint_parents=scaler.get('joint_parents'))

        self._calibration_last_offsets = new_offsets
        self._calibration_status = (
            f"Computed {len(new_offsets)} offsets. "
            f"Position: {'computed (residual)' if compute_pos else 'kept (existing)'}")
        print(f"[INFO]: {self._calibration_status}")

    def _do_write_offsets(self):
        if self._calibration_last_offsets is None:
            return
        path = self._calibration_offsets_cfg_path
        offsets_cfg = io_utils.load_json(path)
        keep_pos = not bool(getattr(self, "_calc_position", False))
        calibration_utils.merge_offsets_into_config(
            offsets_cfg, self._calibration_last_offsets,
            keep_existing_position=keep_pos)
        calibration_utils.write_scaler_config(offsets_cfg, path)
        self._invalidate_effector_scaler()
        self._calibration_status = f"Wrote offsets to {path}"
        print(f"[INFO]: {self._calibration_status}")

    def _do_write_scales(self):
        if not getattr(self, "_calibration_last_scales", None):
            return
        path = self._calibration_scaler_cfg_path
        scaler_cfg = io_utils.load_json(path)
        calibration_utils.merge_scales_into_config(
            scaler_cfg, self._calibration_last_scales)
        calibration_utils.write_scaler_config(scaler_cfg, path)
        self._invalidate_effector_scaler()
        self._calibration_status = f"Wrote joint_scales to {path}"
        print(f"[INFO]: {self._calibration_status}")

    def _do_calibrate_all(self):
        """One-click calibration: persist scale + compute/write scales & offsets.

        Foolproof entry point that guarantees ``joint_scales`` and
        ``joint_offsets`` (including the size-dependent ``offset.p`` residual) are
        recomputed together from the SAME reference poses, and that the live
        source position scale is saved to the retargeter config. This avoids the
        classic footgun of changing the scale (or recomputing scales) while
        leaving a stale ``offset.p`` behind.
        """
        steps = []

        # 1. Persist the live source scale (even if the user never clicked Save).
        if getattr(self, "_source_scale_dirty", False):
            self._do_write_source_scale()
            steps.append("saved scale")

        # 2. joint_scales: compute + write.
        self._do_compute_scales()
        self._do_write_scales()
        steps.append(f"{len(self._calibration_last_scales or {})} scales")

        # 3. joint_offsets: force the offset.p residual to be recomputed so it
        #    always matches the freshly written scales, then write.
        prev_calc_position = getattr(self, "_calc_position", False)
        self._calc_position = True
        try:
            self._do_compute_bias()
            self._do_write_offsets()
        finally:
            self._calc_position = prev_calc_position
        steps.append(f"{len(self._calibration_last_offsets or {})} offsets (+offset.p)")

        # Report the largest recomputed offset.p residual. If this is big (e.g.
        # >5 cm on an arm joint), the robot calibration pose does not match the
        # source zero pose well there, and that residual rotates with the joint
        # at retarget -> the arm drifts/stretches away from the calibration pose.
        max_p, max_j = 0.0, ""
        for j, vals in (self._calibration_last_offsets or {}).items():
            p = vals[0]
            m = (p[0] ** 2 + p[1] ** 2 + p[2] ** 2) ** 0.5
            if m > max_p:
                max_p, max_j = m, j
        steps.append(f"max|offset.p|={max_p * 100:.1f}cm @ {max_j}")

        self._calibration_status = "One-click calibrate done: " + ", ".join(steps)
        print(f"[INFO]: {self._calibration_status}")

    def ui_playback_controls(self, ui):
        viewport = ui.get_main_viewport()
        
        panel_height = 105
        panel_width = viewport.size.x - 2 * (2 * _UI_NEWTON_PANEL_MARGIN + _UI_NEWTON_PANEL_WIDTH)
        
        ui.set_next_window_pos(ui.ImVec2(_UI_NEWTON_PANEL_WIDTH + _UI_NEWTON_PANEL_MARGIN, viewport.size.y - _UI_NEWTON_PANEL_MARGIN - panel_height))
        ui.set_next_window_size(ui.ImVec2(panel_width, panel_height))
        ui.set_next_window_bg_alpha(_UI_NEWTON_PANEL_ALPHA)

        ui.begin("Playback Controls", flags=(ui.WindowFlags_.no_collapse | ui.WindowFlags_.no_resize))
        # Time slider
        ui.align_text_to_frame_padding()
        ui.text("Time (s):")
        ui.same_line()
        ui.set_next_item_width(panel_width - 150)
        changed, new_time = ui.slider_float(
            "##TimeSlider",
            self.playback_time,
            0.0,
            self.playback_total_time,
            "%.2f")
        if changed:
            self.playback_time = wp.clamp(new_time, 0.0, self.playback_total_time)
        ui.same_line()
        ui.text_colored(ui.ImVec4(0.6, 0.8, 1.0, 1.0), f"{self.playback_total_time:.2f}s")
        
        self.is_playing = not ui.button("Pause") if self.is_playing else ui.button("Play ")
        ui.same_line()

        # Speed slider
        ui.align_text_to_frame_padding()
        ui.text("Speed")
        ui.same_line()
        ui.set_next_item_width(100)
        changed, new_speed = ui.slider_float(
            "##SpeedSlider",
            self.playback_speed,
            -2.0, 2.0,
            "%.2f"
        )
        if changed:
            self.playback_speed = new_speed
        ui.same_line()
        _, self.playback_loop = ui.checkbox("Loop", self.playback_loop)
        ui.end()

    def batched_retargeting(self):
        if not os.path.isdir(self.config['import_folder']):
            print(f"[ERROR]: Import folder does not exist {self.config['import_folder']}.")
            exit(-1)

        import_path = pathlib.Path(self.config['import_folder'])
        if len(self.config['export_folder']) == 0:
            print("[ERROR]: No export folder specified.")
            exit(-1)

        export_path = pathlib.Path(self.config['export_folder'])
        if not export_path.is_dir():
            print(f"[WARNING]: Export folder does not exist! Creating new folder at {str(export_path)}!")
            export_path.mkdir(parents=True, exist_ok=True)

        batch_size = self.config['batch_size']
        bvh_files = list(import_path.rglob("*.bvh"))
        if (len(bvh_files) == 0):
            print(f"[ERROR]: Import folder {str(import_path)}, does not contain any BVH files.")
            exit(-1)

        # Sort files based on size (largest first)
        bvh_files.sort(key=lambda p: p.stat().st_size, reverse=True)
        batches = [bvh_files[i:i + batch_size] for i in range(0, len(bvh_files), batch_size)]
        
        # All skeletons should be the same, load one as our reference
        bvh_skeleton, _ = bvh_utils.load_bvh(
            batches[0][0], position_scale=self.source_position_scale, recenter_xy=self.source_recenter_xy,
            position_scale_upper=self.source_position_scale_upper,
            position_scale_lower=self.source_position_scale_lower)

        bvh_tx_converter = self.converter.transform(wp.transform_identity())
        expected_num_joints = bvh_skeleton.num_joints

        retarget_source = self.config['retarget_source']
        retarget_solver = self.config['retargeter']
        retarget_target = self.config["retarget_target"]
        batch_csv_config = csv_utils.get_csv_config_for_robot(retarget_target)
        retarget_pipeline = None
        if (retarget_solver == 'Newton'):
            import soma_retargeter.pipelines.newton_pipeline as newton_pipeline
            retarget_pipeline = newton_pipeline.NewtonPipeline(
                bvh_skeleton, retarget_source, retarget_target,
                source_position_scale=self.source_position_scale,
                source_position_scale_upper=self.source_position_scale_upper,
                source_position_scale_lower=self.source_position_scale_lower)
        if retarget_pipeline is None:
            print(f"[ERROR]: Invalid retarget solver selected [{retarget_solver}]. Use 'Newton'.")
            exit(-1)
        retarget_pipeline.set_root_amplitude_scale(*self.root_amplitude_scale)
        retarget_pipeline.set_full_amplitude_scale(*self.full_amplitude_scale)

        nb_retargeted_motions = 0
        start_time = time.time()

        for i, batch in enumerate(batches):
            print(f"[INFO]: Processing batch {i+1} of {len(batches)}")
            
            print(f"[INFO]: Loading {len(batch)} animations...")
            animations = []
            for file_path in batch:
                _, animation = bvh_utils.load_bvh(
                    file_path, bvh_skeleton,
                    position_scale=self.source_position_scale, recenter_xy=self.source_recenter_xy,
                    position_scale_upper=self.source_position_scale_upper,
                    position_scale_lower=self.source_position_scale_lower)
                # All animations should be on the same skeleton
                assert expected_num_joints == animation.skeleton.num_joints, (
                    f"[ERROR]: Unexpected number of joints in input motion. Expected {expected_num_joints}, "
                    f"got {animation.skeleton.num_joints}")
                
                animations.append(animation)
            assert(len(animations) == len(batch))

            if (len(animations) > 0):
                print("[INFO]: Retargeting...")
                retarget_pipeline.clear()
                retarget_pipeline.add_input_motions(animations, [bvh_tx_converter] * len(animations), True)
                csv_buffers = retarget_pipeline.execute()

                assert(len(csv_buffers) == len(animations))
                for i in trange(len(csv_buffers), desc="[INFO]: Exporting CSV Files"):
                    csv_buffer = csv_buffers[i]
                    dst_path = export_path / pathlib.Path(batch[i]).relative_to(import_path).with_suffix(".csv")
                    dst_path.parent.mkdir(parents=True, exist_ok=True)
                    csv_utils.save_csv(dst_path, csv_buffer, csv_config=batch_csv_config)

            nb_retargeted_motions += len(batch)

        elapsed_time = time.time() - start_time
        elapsed_str = f"{int(elapsed_time // 3600):02d}:{int((elapsed_time % 3600) // 60):02d}:{int(elapsed_time % 60):02d}"
        print(
            f"[INFO]: Retargeted {nb_retargeted_motions} animations successfully "
            f"in {elapsed_str} "
            f"[{(elapsed_time/nb_retargeted_motions):.2f}s per motion]!")

def main():
    import newton.examples

    parser = newton.examples.create_parser()
    parser.set_defaults(viewer=("null"))
    parser.add_argument(
        "--config",
        type=lambda x: None if x == "None" else str(x),
        default="./assets/default_bvh_to_csv_converter_config.json",
        help="Input json config file.")
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Source skeleton type (e.g. 'soma', 'mydata'). Overrides "
             "'retarget_source' in the config so you can run your own "
             "skeleton natively without editing the config file.")

    viewer, args = newton.examples.init(parser)
    if not pathlib.Path(args.config).exists():
        print(f"[ERROR]: Main config json file not found: {args.config}")
        exit(1)

    config = io_utils.load_json(args.config)
    if args.data:
        print(f"[INFO]: --data override: retarget_source = '{args.data}'")
        config['retarget_source'] = args.data
        # Apply this source's coordinate convention + unit scale so it loads
        # upright, on the ground, and at a sensible size (these are the SOMA
        # vs native-skeleton differences). The config's facing is SOMA's, so we
        # override it for --data; an explicit position scale in the config wins.
        config['retarget_source_facing_direction'] = pipeline_utils.get_source_facing_direction(args.data)
        config.setdefault(
            'retarget_source_position_scale',
            pipeline_utils.get_source_position_scale(args.data))
        config.setdefault(
            'retarget_source_yaw_offset_deg',
            pipeline_utils.get_source_yaw_offset_deg(args.data))
        print(f"[INFO]: --data convention: facing="
              f"{config['retarget_source_facing_direction']}, "
              f"position_scale={config['retarget_source_position_scale']}, "
              f"yaw_offset_deg={config['retarget_source_yaw_offset_deg']}")
    with wp.ScopedDevice(args.device):
        app = Viewer(viewer, config)
        if not isinstance(viewer, newton.viewer.ViewerNull):
            app.run()
        else:
            app.batched_retargeting()

if __name__ == "__main__":
    main()

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
        self.converter = SpaceConverter(get_facing_direction_type_from_str(self.config['retarget_source_facing_direction']))

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

        self.retarget_source_options = ['soma']
        self.retarget_target_options = [
            'unitree_g1',
            'engineai_pm01',
            'hightorque_pi_plus',
            'pndbotics_adam_lite',
        ]
        self.retarget_solver_options = ['Newton']
        self.retarget_solver_idx     = 0
        self.retarget_source_idx     = 0

        # Resolve currently selected robot from config (falls back to G1).
        self.robot_type = self.config.get('retarget_target', 'unitree_g1')
        if self.robot_type not in self.retarget_target_options:
            print(f"[WARN]: retarget_target [{self.robot_type}] not in known list, defaulting to unitree_g1")
            self.robot_type = 'unitree_g1'
        self.retarget_target_idx = self.retarget_target_options.index(self.robot_type)
        self.csv_config = csv_utils.get_csv_config_for_robot(self.robot_type)

        self.show_skeleton_mesh = True
        self.show_skeleton = False
        self.show_skeleton_joint_axes = False
        self.show_gizmos = True

        self.viewer.renderer.set_title("BVH to CSV Converter")
        self.viewer.register_ui_callback(lambda ui: self.gui(ui), position="free")

        self.robot_builder = newton.ModelBuilder()
        self.robot_builder.add_mjcf(
            str(pipeline_utils.get_robot_mjcf_path(self.robot_type)))

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

    def _init_calibration(self):
        """Set up state for the in-app bias-calibration panel.

        Loads the SOMA zero-pose skeleton (used as the visual + numerical
        reference) and discovers the robot's revolute joints so we can
        expose them as sliders in the GUI.
        """
        self.calibration_mode = False
        self.show_soma_reference = False

        retargeter_cfg = pipeline_utils.get_retargeter_config(
            pipeline_utils.SourceType.SOMA,
            pipeline_utils.get_target_type_from_str(self.robot_type))
        self._calibration_retargeter_cfg = retargeter_cfg

        scaler_cfg_path = io_utils.get_config_file(retargeter_cfg['human_robot_scaler_config'])
        self._calibration_scaler_cfg_path = scaler_cfg_path

        init_bvh = io_utils.get_config_file(retargeter_cfg['initialization_pose'])
        soma_skel, soma_anim = bvh_utils.load_bvh(init_bvh)
        self.soma_reference_skeleton = soma_skel
        self.soma_reference_local_zero = soma_anim.get_local_transforms(0).copy()
        self.soma_reference_instance = SkeletonInstance(
            soma_skel, [0.6, 0.7, 1.0],
            self.converter.transform(wp.transform_identity()))
        self.soma_reference_instance.set_local_transforms(self.soma_reference_local_zero)

        self.soma_reference_mesh = pipeline_utils.get_source_model_mesh(
            pipeline_utils.SourceType.SOMA, soma_skel)
        self.soma_reference_mesh_renderer = SkeletalMeshRenderer(self.soma_reference_mesh)

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

    def _build_reference_pose(self):
        """Per-robot calibration reference pose.

        Loaded from ``tools/<robot_type>_reference_pose.json`` if present, so
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
        ref_path = repo_root / "tools" / f"{self.robot_type}_reference_pose.json"
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
        self.ui_calibration(ui)

    def load_csv_file(self, path):
        self.robot_csv_animation_buffers[0] = csv_utils.load_csv(path, csv_config=self.csv_config)
        self.compute_playback_total_time()

    def load_bvh_file(self, path):
        self.animation_buffers = []
        self.skeleton_instances = []
        if self.skeleton_renderer is not None:
            self.skeleton_renderer.clear(self.viewer)
        if self.skeletal_mesh_renderer is not None:
            self.skeletal_mesh_renderer.clear(self.viewer)
        if self.coordinate_renderer is not None:
            self.coordinate_renderer.clear(self.viewer)

        self.skeleton, animation = bvh_utils.load_bvh(path)
        self.skeleton_renderer = SkeletonRenderer(self.skeleton, [0])
        self.skeleton_instances = [SkeletonInstance(self.skeleton, _DEFAULT_COLOR, self.converter.transform(wp.transform_identity()))]
        self.animation_offsets = [wp.transform_identity()] * len(self.skeleton_instances)
        self.animation_buffers = [animation]

        self.skeletal_mesh = pipeline_utils.get_source_model_mesh(pipeline_utils.SourceType.SOMA, self.skeleton)
        self.skeletal_mesh_renderer = SkeletalMeshRenderer(self.skeletal_mesh)
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
                wp.vec3(tx.p[0], tx.p[1], 0.0),
                math_utils.quat_twist(wp.vec3(0.0, 0.0, 1.0), tx.q))

        for i in range(len(self.robot_offsets)):
            self.robot_offsets[i] = clamp_gizmo_transform(self.robot_offsets[i])
        for i in range(len(self.animation_offsets)):
            self.animation_offsets[i] = clamp_gizmo_transform(self.animation_offsets[i])

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
                if self.show_skeleton_mesh:
                    self.skeletal_mesh_renderer.draw(self.viewer, self.skeleton_instances[i], self.skeleton_instances[i].color, i)
                self.skeleton_instances[i].xform = prev_xform

        if self.calibration_mode and self.show_soma_reference:
            self.soma_reference_mesh_renderer.draw(
                self.viewer, self.soma_reference_instance,
                self.soma_reference_instance.color, 99)

        if self.show_gizmos:
            for i, offset in enumerate(self.robot_offsets):
                self.viewer.log_gizmo(f"robot_offset{i}", offset)
            for i, offset in enumerate(self.animation_offsets):
                self.viewer.log_gizmo(f"animation_offset{i}", offset)
        
        self.viewer.log_state(self.state)
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
            pipeline = newton_pipeline.NewtonPipeline(self.skeleton, retarget_source, retarget_target)
        else:
            raise(ValueError(f"[ERROR]: Unknown retargeter solver [{retarget_solver}"))
        
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
                    csv_utils.save_csv(save_path, self.robot_csv_animation_buffers[0], csv_config=self.csv_config)

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
            _, self.show_gizmos = ui.checkbox("Show Gizmos", self.show_gizmos)
            ui.same_line()
            if ui.button("Reset"):
                self.robot_offsets = [wp.transform(wp.vec3(0.0, i - (self.num_robots - 1) / 2.0, 0.0), wp.quat_identity()) for i in range(self.num_robots)]
                self.animation_offsets = [wp.transform_identity()] * len(self.skeleton_instances)
        ui.end()

    def ui_calibration(self, ui):
        """Right-side panel for computing per-joint bias (joint_offsets)."""
        import tkinter as tk
        from tkinter import filedialog as tk_filedialog

        viewport = ui.get_main_viewport()

        panel_size = ui.ImVec2(360, 600)
        ui.set_next_window_pos(
            ui.ImVec2(
                viewport.size.x - _UI_NEWTON_PANEL_MARGIN - panel_size.x,
                _UI_NEWTON_PANEL_MARGIN))
        ui.set_next_window_size(panel_size, ui.Cond_.first_use_ever)
        ui.set_next_window_bg_alpha(_UI_NEWTON_PANEL_ALPHA)

        ui.begin("Calibration (Compute Bias)",
                 flags=ui.WindowFlags_.no_collapse)

        changed, self.calibration_mode = ui.checkbox(
            "Enable Calibration Mode", self.calibration_mode)
        if changed:
            if self.calibration_mode:
                self.is_playing = False
            else:
                self.soma_reference_mesh_renderer.clear(self.viewer)

        ui.text_colored(
            ui.ImVec4(0.6, 0.8, 1.0, 1.0),
            f"Robot: {self.robot_type}   "
            f"Revolute joints: {len(self.calibration_revolute_joints)}")

        if not self.calibration_mode:
            ui.text_wrapped(
                "Turn on Calibration Mode to freeze BVH playback, "
                "load the SOMA zero-pose reference, and edit the robot's "
                "joint angles to match.")
            ui.end()
            return

        ui.separator()
        ref_changed, self.show_soma_reference = ui.checkbox(
            "Show SOMA Zero Pose Overlay", self.show_soma_reference)
        if ref_changed and not self.show_soma_reference:
            self.soma_reference_mesh_renderer.clear(self.viewer)

        ui.separator()
        if ui.collapsing_header("Robot Joint Sliders",
                                flags=ui.TreeNodeFlags_.default_open):
            ui.text("Adjust joint angles (rad) to match SOMA zero pose.")
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

        ui.separator()
        if ui.collapsing_header("Compute joint_scales",
                                flags=ui.TreeNodeFlags_.default_open):
            ui.text_wrapped(
                "Per-joint magnitude scales |robot_vec| / |soma_vec|, derived "
                "from the matching reference poses above. Run this whenever "
                "you change robots, the reference pose, or model_height.")
            if ui.button("Compute Scales"):
                self._do_compute_scales()
            ui.same_line()
            if not self._calibration_last_scales:
                ui.begin_disabled()
            if ui.button("Write scales to Config"):
                self._do_write_scales()
            ui.same_line()
            if ui.button("Print scales##s"):
                print(json.dumps(self._calibration_last_scales, indent=4))
            if not self._calibration_last_scales:
                ui.end_disabled()

            if self._calibration_last_scales:
                ui.spacing()
                if ui.begin_child("##scales_preview", ui.ImVec2(0, 120)):
                    for k, v in self._calibration_last_scales.items():
                        ui.text(f"{k:14s}  {v:.4f}")
                ui.end_child()

        ui.separator()
        if ui.collapsing_header("Compute joint_offsets",
                                flags=ui.TreeNodeFlags_.default_open):
            ui.text(f"ik_map entries: {len(self._calibration_retargeter_cfg.get('ik_map', {}))}")
            ui.text_wrapped(
                "offset.q always recomputes (corrects coordinate-frame "
                "mismatch). offset.p is left at its hand-tuned value unless "
                "you tick the box below; ticking it recomputes the residual "
                "AFTER scaling, so make sure joint_scales is up to date.")
            _, self._calc_position = ui.checkbox(
                "Also overwrite offset.p (residual after scaling)",
                getattr(self, "_calc_position", False))

            if ui.button("Compute Offsets"):
                self._do_compute_bias()
            ui.same_line()
            if self._calibration_last_offsets is None:
                ui.begin_disabled()
            if ui.button("Write offsets to Config"):
                self._do_write_offsets()
            ui.same_line()
            if ui.button("Print offsets##o"):
                print(json.dumps(self._calibration_last_offsets, indent=4))
            if self._calibration_last_offsets is None:
                ui.end_disabled()

            if self._calibration_last_offsets is not None:
                ui.spacing()
                ui.text("Computed offset.q (xyzw) per SOMA joint:")
                if ui.begin_child("##offset_preview", ui.ImVec2(0, 140)):
                    for soma_joint, vals in self._calibration_last_offsets.items():
                        q = vals[1]
                        ui.text(
                            f"{soma_joint:14s}  "
                            f"({q[0]:+.3f}, {q[1]:+.3f}, {q[2]:+.3f}, {q[3]:+.3f})")
                ui.end_child()

        if self._calibration_status:
            ui.separator()
            ui.text_wrapped(self._calibration_status)

        ui.end()

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
        """Snapshot SOMA zero-pose and current robot pose for calibration."""
        soma_globals = self.soma_reference_skeleton.compute_global_transforms(
            self.soma_reference_local_zero,
            self.converter.transform(wp.transform_identity()))
        body_q = self.state.body_q.numpy()
        link_globals = calibration_utils.collect_robot_link_globals(
            self.robot_builder, body_q)
        return soma_globals, link_globals

    def _do_compute_scales(self):
        """Compute joint_scales from the current matching reference poses."""
        ik_map = self._calibration_retargeter_cfg.get('ik_map', {})
        soma_globals, link_globals = self._calibration_collect()
        new_scales = calibration_utils.compute_scales(
            soma_globals,
            self.soma_reference_skeleton.joint_names,
            link_globals,
            ik_map,
            height_ratio=self._calibration_height_ratio())
        self._calibration_last_scales = new_scales
        self._calibration_status = (
            f"Computed {len(new_scales)} joint_scales. "
            "Click 'Write scales to Config' to persist.")
        print(f"[INFO]: {self._calibration_status}")

    def _do_compute_bias(self):
        """Compute joint_offsets from current SOMA zero pose + robot pose."""
        ik_map = self._calibration_retargeter_cfg.get('ik_map', {})
        soma_globals, link_globals = self._calibration_collect()

        # Use the latest in-memory scales if the user just computed them,
        # otherwise fall back to whatever is currently saved in the config.
        scaler = io_utils.load_json(self._calibration_scaler_cfg_path)
        scales = getattr(self, "_calibration_last_scales", None) or scaler.get('joint_scales', {})

        compute_pos = bool(getattr(self, "_calc_position", False))
        new_offsets = calibration_utils.compute_offsets(
            soma_globals,
            self.soma_reference_skeleton.joint_names,
            link_globals,
            ik_map,
            compute_position=compute_pos,
            joint_scales=scales,
            height_ratio=self._calibration_height_ratio())

        self._calibration_last_offsets = new_offsets
        self._calibration_status = (
            f"Computed {len(new_offsets)} offsets. "
            f"Position: {'computed (residual)' if compute_pos else 'kept (existing)'}")
        print(f"[INFO]: {self._calibration_status}")

    def _do_write_offsets(self):
        if self._calibration_last_offsets is None:
            return
        path = self._calibration_scaler_cfg_path
        scaler_cfg = io_utils.load_json(path)
        keep_pos = not bool(getattr(self, "_calc_position", False))
        calibration_utils.merge_offsets_into_config(
            scaler_cfg, self._calibration_last_offsets,
            keep_existing_position=keep_pos)
        calibration_utils.write_scaler_config(scaler_cfg, path)
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
        self._calibration_status = f"Wrote joint_scales to {path}"
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
        bvh_importer = bvh_utils.BVHImporter()
        bvh_skeleton, _ = bvh_importer.create_skeleton(batches[0][0])

        bvh_tx_converter = self.converter.transform(wp.transform_identity())
        expected_num_joints = bvh_skeleton.num_joints

        retarget_source = self.config['retarget_source']
        retarget_solver = self.config['retargeter']
        retarget_target = self.config["retarget_target"]
        batch_csv_config = csv_utils.get_csv_config_for_robot(retarget_target)
        retarget_pipeline = None
        if (retarget_solver == 'Newton'):
            import soma_retargeter.pipelines.newton_pipeline as newton_pipeline
            retarget_pipeline = newton_pipeline.NewtonPipeline(bvh_skeleton, retarget_source, retarget_target)
        if retarget_pipeline is None:
            print(f"[ERROR]: Invalid retarget solver selected [{retarget_solver}]. Use 'Newton'.")
            exit(-1)

        nb_retargeted_motions = 0
        start_time = time.time()

        for i, batch in enumerate(batches):
            print(f"[INFO]: Processing batch {i+1} of {len(batches)}")
            
            print(f"[INFO]: Loading {len(batch)} animations...")
            animations = []
            for file_path in batch:
                _, animation = bvh_utils.load_bvh(file_path, bvh_skeleton)
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

    viewer, args = newton.examples.init(parser)
    if not pathlib.Path(args.config).exists():
        print(f"[ERROR]: Main config json file not found: {args.config}")
        exit(1)

    config = io_utils.load_json(args.config)
    with wp.ScopedDevice(args.device):
        app = Viewer(viewer, config)
        if not isinstance(viewer, newton.viewer.ViewerNull):
            app.run()
        else:
            app.batched_retargeting()

if __name__ == "__main__":
    main()

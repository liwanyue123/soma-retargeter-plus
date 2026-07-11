# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import warp as wp

import soma_retargeter.utils.io_utils as io_utils
import soma_retargeter.utils.pose_utils as pose_utils

from soma_retargeter.animation.skeleton import Skeleton, SkeletonInstance
from soma_retargeter.animation.animation_buffer import AnimationBuffer


class HumanToRobotScaler:
    """
    Scale and map human motion to robot-aligned effectors.
    """
    def __init__(self, skeleton: Skeleton, human_height, config_file, offsets_file=None):
        """
        Args:
            skeleton: Common skeleton of the source clips.
            human_height: Source actor height (``model_height``) used for the
                height-ratio scaling.
            config_file: Scaler config holding ``joint_scales`` / ``joint_parents``
                (and, unless ``offsets_file`` is given, ``joint_offsets``).
            offsets_file: Optional separate config holding ``{"joint_offsets": ...}``.
                When provided, ``joint_offsets`` are read from here instead of from
                ``config_file`` (keeps tracking scales and calibration offsets in
                separate files per data source).
        """
        config = io_utils.load_json(config_file)
        self.robot_type = config['robot_type']
        self.skeleton = skeleton

        ratio = human_height / config['human_height_assumption']
        joint_scales = config['joint_scales']
        for key in joint_scales.keys():
            joint_scales[key] *= ratio

        joint_offsets = {}
        if offsets_file is not None:
            joint_offset_data = io_utils.load_json(offsets_file)['joint_offsets']
        else:
            joint_offset_data = config['joint_offsets']
        for joint_name, entry in joint_offset_data.items():
            t_offset, q_offset = entry
            joint_offsets[joint_name] = wp.transform(
                wp.vec3(*t_offset),
                wp.normalize(wp.quat(*q_offset)))

        # SOMA convention: ToeBase shares the Toe joint's offset. Native skeletons
        # may already define ToeBase directly (and have no "Toe"), so only alias
        # when a "Toe" entry exists and "ToeBase" is not already provided.
        if "LeftToe" in joint_offsets:
            joint_offsets.setdefault("LeftToeBase", joint_offsets["LeftToe"])
        if "RightToe" in joint_offsets:
            joint_offsets.setdefault("RightToeBase", joint_offsets["RightToe"])

        self.mapped_joints = [name for name in self.skeleton.joint_names if name in joint_scales.keys()]
        self.mapped_joint_indices = wp.array([self.skeleton.joint_index(name) for name in self.mapped_joints], dtype=wp.int32)
        self.mapped_joint_scales = wp.array([joint_scales[name] for name in self.mapped_joints], dtype=wp.float32)
        self.mapped_joint_offsets = wp.array([joint_offsets[name] for name in self.mapped_joints], dtype=wp.transform)

        joint_parents = config['joint_parents']
        self.mapped_joint_parents = [
            -1 if joint_parents[name] == "" else self.mapped_joints.index(joint_parents[name])
            for name in self.mapped_joints]

        # Standing references for per-axis motion-amplitude scaling, injected by
        # the pipeline once the init pose is known. Default to origin / zeros,
        # which (together with amp == (1,1,1)) reproduces the original behaviour.
        #   ref_root: raw unscaled root world position (root-only mode).
        #   ref_effectors: standing effector world positions (full-body mode).
        self.ref_root = wp.vec3(0.0, 0.0, 0.0)
        self.ref_effectors = wp.array(
            [wp.vec3(0.0, 0.0, 0.0)] * len(self.mapped_joints), dtype=wp.vec3)

    def set_amplitude_reference(self, root_world_position):
        """Set the standing root reference for motion-amplitude scaling.

        Args:
            root_world_position: Raw (unscaled) world position of the root joint
                in the initialization/standing pose. Amplitude scaling shrinks or
                grows the root trajectory's displacement from this point.
        """
        self.ref_root = wp.vec3(
            float(root_world_position[0]),
            float(root_world_position[1]),
            float(root_world_position[2]))

    def set_effector_reference(self, effector_positions):
        """Set the standing effector references used by full-body amplitude scaling.

        Args:
            effector_positions: Iterable of (x, y, z) standing world positions, one
                per mapped joint (typically the effectors of the initialization
                pose). Full-body mode scales each effector's displacement from
                here.
        """
        self.ref_effectors = wp.array(
            [wp.vec3(float(p[0]), float(p[1]), float(p[2])) for p in effector_positions],
            dtype=wp.vec3)

    def effector_names(self):
        """
        Return the list of mapped joint names used as effectors.

        Returns:
            list[str]: Names of joints for which effectors are computed.
        """
        return self.mapped_joints

    def compute_effectors_from_skeleton(self, skeleton_instance: SkeletonInstance, scale_animation: bool, root_amplitude_scale: wp.vec3 = None, full_amplitude_scale: wp.vec3 = None):
        """
        Compute scaled effectors from a single skeleton instance.

        The method computes global joint transforms from the skeleton instance,
        then applies per-joint scaling and offsets to produce effector
        transforms in world space.

        Args:
            skeleton_instance: SkeletonInstance whose skeleton must match the scaler's ``skeleton``.
            scale_animation: Whether to apply per-joint scaling when computing
                effectors. If False, only height scaling is applied.
            root_amplitude_scale: Per-axis scale applied to the root trajectory's
                displacement from its standing reference. Composes with
                ``full_amplitude_scale``. Defaults to ``(1, 1, 1)``.
            full_amplitude_scale: Per-axis scale applied to every effector's
                displacement from its own standing reference (scales hands/feet
                too; may cause foot slip/drift). Composes with
                ``root_amplitude_scale``. Defaults to ``(1, 1, 1)``.

        Returns:
            np.ndarray: Array of effector transforms (one per mapped joint) in the
            layout ``(num_mapped_joints, wp.transform)``.

        Raises:
            ValueError: If ``skeleton_instance.skeleton`` does not match the scaler's ``skeleton``.
        """
        if skeleton_instance.skeleton != self.skeleton:
            raise ValueError("[ERROR]: SkeletonInstance.skeleton is not equal to self.skeleton.")

        @wp.kernel
        def compute_global_pose_kernel(
            in_num_joints     : wp.int32,
            in_root_tx        : wp.transform,
            in_parent_indices : wp.array(dtype=wp.int32),
            in_local_pose     : wp.array(dtype=wp.transform),
            out_result        : wp.array(dtype=wp.transform)
        ):
            pose_utils.wp_compute_global_pose(in_num_joints, in_root_tx, in_parent_indices, in_local_pose, out_result)

        @wp.kernel
        def compute_scaled_effectors_kernel(
            in_num_mapped_joints    : wp.int32,
            in_global_pose          : wp.array(dtype=wp.transform),
            in_mapped_joint_indices : wp.array(dtype=wp.int32),
            in_mapped_joint_scales  : wp.array(dtype=wp.float32),
            in_mapped_joint_offsets : wp.array(dtype=wp.transform),
            in_scale_animation      : wp.bool,
            in_root_amp_scale       : wp.vec3,
            in_ref_root             : wp.vec3,
            in_ref_effectors        : wp.array(dtype=wp.vec3),
            in_full_amp_scale       : wp.vec3,
            out_result              : wp.array(dtype=wp.transform)
        ):
            HumanToRobotScaler.wp_compute_scaled_effectors(
                in_num_mapped_joints, in_global_pose, in_mapped_joint_indices,
                in_mapped_joint_scales, in_mapped_joint_offsets, in_scale_animation,
                in_root_amp_scale, in_ref_root, in_ref_effectors, in_full_amp_scale, out_result)

        wp_global_pose = wp.array([wp.transform_identity()] * skeleton_instance.num_joints, dtype=wp.transform)
        wp.launch(
            compute_global_pose_kernel,
            dim=1,
            inputs=[
                skeleton_instance.num_joints,
                skeleton_instance.xform,
                wp.array(skeleton_instance.parent_indices, dtype=wp.int32),
                wp.array(skeleton_instance.local_transforms, dtype=wp.transform)],
                outputs=[wp_global_pose])

        root_amp = root_amplitude_scale if root_amplitude_scale is not None else wp.vec3(1.0, 1.0, 1.0)
        full_amp = full_amplitude_scale if full_amplitude_scale is not None else wp.vec3(1.0, 1.0, 1.0)
        wp_effectors = wp.array([wp.transform_identity()] * len(self.mapped_joint_indices), dtype=wp.transform)
        wp.launch(
            compute_scaled_effectors_kernel,
            dim=1,
            inputs=[
                len(self.mapped_joint_indices),
                wp_global_pose,
                self.mapped_joint_indices,
                self.mapped_joint_scales,
                self.mapped_joint_offsets,
                scale_animation,
                root_amp,
                self.ref_root,
                self.ref_effectors,
                full_amp
            ],
            outputs=[wp_effectors])

        return wp_effectors.numpy()

    def compute_effectors_from_buffer(self, animation_buffer: AnimationBuffer, scale_animation: bool, xform: wp.transform = wp.transform_identity(), root_amplitude_scale: wp.vec3 = None, full_amplitude_scale: wp.vec3 = None):
        """
        Compute scaled effectors for all frames in an animation buffer.

        This is a batched variant of ``compute_effectors_from_skeleton`` that
        operates over all frames in an AnimationBuffer.

        Args:
            animation_buffer: AnimationBuffer whose skeleton must match the scaler's ``skeleton``.
            scale_animation: Whether to apply per-joint scaling when computing
                effectors. If False, only height scaling is applied.
            xform: Optional root transform applied to all frames before global
                pose computation.
            root_amplitude_scale: Per-axis scale applied to the root trajectory's
                displacement from its standing reference. Composes with
                ``full_amplitude_scale``. Defaults to ``(1, 1, 1)``.
            full_amplitude_scale: Per-axis scale applied to every effector's
                displacement from its own standing reference (scales hands/feet
                too; may cause foot slip/drift). Composes with
                ``root_amplitude_scale``. Defaults to ``(1, 1, 1)``.

        Returns:
            np.ndarray: Array of transforms of shape ``(num_frames, num_mapped_joints, wp.transform)``.

        Raises:
            ValueError: If ``animation_buffer.skeleton`` does not match the scaler's ``skeleton``.
        """
        if animation_buffer.skeleton != self.skeleton:
            raise ValueError("[ERROR]: AnimationBuffer.skeleton is not equal to self.skeleton.")

        @wp.kernel
        def batched_compute_global_pose_kernel(
            in_num_joints     : wp.int32,
            in_root_tx        : wp.transform,
            in_parent_indices : wp.array(dtype=wp.int32),
            in_local_pose     : wp.array2d(dtype=wp.transform),
            out_result        : wp.array2d(dtype=wp.transform)
        ):
            frame_idx = wp.tid()
            pose_utils.wp_compute_global_pose(
                in_num_joints, in_root_tx, in_parent_indices, in_local_pose[frame_idx], out_result[frame_idx])

        @wp.kernel
        def batched_compute_scaled_effectors_2d_kernel(
            in_num_mapped_joints    : wp.int32,
            in_global_pose          : wp.array2d(dtype=wp.transform),
            in_mapped_joint_indices : wp.array(dtype=wp.int32),
            in_mapped_joint_scales  : wp.array(dtype=wp.float32),
            in_mapped_joint_offsets : wp.array(dtype=wp.transform),
            in_scale_animation      : wp.bool,
            in_root_amp_scale       : wp.vec3,
            in_ref_root             : wp.vec3,
            in_ref_effectors        : wp.array(dtype=wp.vec3),
            in_full_amp_scale       : wp.vec3,
            out_result              : wp.array2d(dtype=wp.transform)
        ):
            frame_idx = wp.tid()
            HumanToRobotScaler.wp_compute_scaled_effectors(
               in_num_mapped_joints, in_global_pose[frame_idx], in_mapped_joint_indices,
               in_mapped_joint_scales, in_mapped_joint_offsets, in_scale_animation,
               in_root_amp_scale, in_ref_root, in_ref_effectors, in_full_amp_scale, out_result[frame_idx])

        wp_global_poses = wp.empty(shape=(animation_buffer.num_frames, self.skeleton.num_joints), dtype=wp.transform)
        wp.launch(
            batched_compute_global_pose_kernel,
            dim=animation_buffer.num_frames,
            inputs=[
                self.skeleton.num_joints,
                xform,
                wp.array(self.skeleton.parent_indices, dtype=wp.int32),
                wp.array2d(animation_buffer.local_transforms, dtype=wp.transform)],
                outputs=[wp_global_poses])

        root_amp = root_amplitude_scale if root_amplitude_scale is not None else wp.vec3(1.0, 1.0, 1.0)
        full_amp = full_amplitude_scale if full_amplitude_scale is not None else wp.vec3(1.0, 1.0, 1.0)
        wp_effectors = wp.empty(shape=(animation_buffer.num_frames, len(self.mapped_joint_indices)), dtype=wp.transform)
        wp.launch(
            batched_compute_scaled_effectors_2d_kernel,
            dim=animation_buffer.num_frames,
            inputs=[
                len(self.mapped_joint_indices),
                wp_global_poses,
                self.mapped_joint_indices,
                self.mapped_joint_scales,
                self.mapped_joint_offsets,
                scale_animation,
                root_amp,
                self.ref_root,
                self.ref_effectors,
                full_amp
            ],
            outputs=[wp_effectors])

        return wp_effectors.numpy()

    def create_scaled_skeleton(self, skeleton_instance: SkeletonInstance):
        """
        Create a scaled Skeleton from a skeleton instance.

        This method computes scaled global effectors from the input skeleton
        instance, converts them to local transforms based on the mapped joint
        hierarchy, and returns a new Skeleton containing only the mapped joints.

        Args:
            skeleton_instance: SkeletonInstance to be converted into a scaled skeleton.

        Returns:
            Skeleton: A new skeleton with joints, parents, and local transforms
            derived from the mapped joints and their scaled effectors.
        """
        global_tx = self.compute_effectors_from_skeleton(skeleton_instance, True)

        num_joints = len(self.mapped_joints)
        wp_local_tx = wp.array([wp.transform_identity()] * num_joints, dtype=wp.transform)

        wp.launch(
            pose_utils.compute_local_pose_kernel,
            dim=1,
            inputs=[
                num_joints,
                skeleton_instance.xform,
                wp.array(self.mapped_joint_parents, dtype=wp.int32),
                wp.array(global_tx, dtype=wp.transform)],
            outputs=[wp_local_tx])

        return Skeleton(
            num_joints,
            self.mapped_joints,
            self.mapped_joint_parents,
            wp_local_tx.numpy())

    @wp.func
    def wp_compute_scaled_effectors(
        in_num_mapped_joints    : wp.int32,
        in_global_pose          : wp.array(dtype=wp.transform),
        in_mapped_joint_indices : wp.array(dtype=wp.int32),
        in_mapped_joint_scales  : wp.array(dtype=wp.float32),
        in_mapped_joint_offsets : wp.array(dtype=wp.transform),
        in_scale_animation      : wp.bool,
        in_root_amp_scale       : wp.vec3,
        in_ref_root             : wp.vec3,
        in_ref_effectors        : wp.array(dtype=wp.vec3),
        in_full_amp_scale       : wp.vec3,
        out_result              : wp.array(dtype=wp.transform)
    ):
        root_t = in_global_pose[in_mapped_joint_indices[0]].p

        scale = wp.where(in_scale_animation, wp.vec3(in_mapped_joint_scales[0]), wp.vec3(1.0, 1.0, in_mapped_joint_scales[0]))
        # Per-axis MOTION-amplitude scaling. The two modes now compose instead of
        # being mutually exclusive:
        #   root: scales the root trajectory's displacement from its standing
        #     reference; body pose/proportions stay intact (no slip).
        #   full-body: on top of that, scales each whole effector's displacement
        #     from its own standing reference (scales hands/feet too -> may
        #     introduce foot slip / drift).
        # Both == (1,1,1) reproduces the original mapped effectors exactly, so
        # this is fully backward compatible.
        mapped_root = wp.cw_mul(root_t, scale)
        mapped_ref = wp.cw_mul(in_ref_root, scale)
        scaled_root_t = mapped_ref + wp.cw_mul(mapped_root - mapped_ref, in_root_amp_scale)

        for i in range(in_num_mapped_joints):
            idx = in_mapped_joint_indices[i]
            pose_tx = in_global_pose[idx]
            offset_tx = in_mapped_joint_offsets[i]

            scale = wp.where(in_scale_animation, wp.vec3(in_mapped_joint_scales[i]), wp.vec3(1.0, 1.0, in_mapped_joint_scales[i]))
            geocentric_scaled_t = wp.cw_mul((pose_tx.p - root_t), scale)

            q = wp.mul(pose_tx.q, offset_tx.q)
            base_t = geocentric_scaled_t + scaled_root_t + wp.quat_rotate(q, offset_tx.p)
            t = in_ref_effectors[i] + wp.cw_mul(base_t - in_ref_effectors[i], in_full_amp_scale)
            out_result[i] = wp.transform(t, q)

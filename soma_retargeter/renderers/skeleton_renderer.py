# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import warp as wp
import newton

import soma_retargeter.utils.pose_utils as pose_utils

from soma_retargeter.renderers.base_renderer import BaseRenderer
from soma_retargeter.animation.skeleton import Skeleton, SkeletonInstance


# Octahedral ("diamond") bone built from 6 vertices in a head-local frame whose
# +Z axis points from the bone head (parent joint) to its tail (child joint):
#   0 = head, 1 = tail, 2..5 = the four ridge vertices near the head.
_BONE_FACES = np.array([
    [0, 2, 3], [0, 3, 4], [0, 4, 5], [0, 5, 2],   # head -> ridge
    [1, 3, 2], [1, 4, 3], [1, 5, 4], [1, 2, 5],   # tail -> ridge
], dtype=np.int32)


def _make_uv_sphere(rings: int = 8, sectors: int = 12):
    """Return (vertices, faces) for a unit sphere centred at the origin."""
    verts = []
    for i in range(rings + 1):
        phi = np.pi * i / rings
        sp, cp = np.sin(phi), np.cos(phi)
        for j in range(sectors + 1):
            theta = 2.0 * np.pi * j / sectors
            verts.append((sp * np.cos(theta), sp * np.sin(theta), cp))
    faces = []
    stride = sectors + 1
    for i in range(rings):
        for j in range(sectors):
            a = i * stride + j
            b = a + stride
            faces.append((a, b, a + 1))
            faces.append((a + 1, b, b + 1))
    return np.asarray(verts, dtype=np.float32), np.asarray(faces, dtype=np.int32)


def _basis_from_z(direction: np.ndarray) -> np.ndarray:
    """3x3 rotation mapping +Z onto ``direction`` (Rodrigues' formula)."""
    d = direction / (np.linalg.norm(direction) + 1e-12)
    z = np.array([0.0, 0.0, 1.0])
    c = float(np.dot(z, d))
    if c > 0.999999:
        return np.eye(3)
    if c < -0.999999:
        return np.diag([1.0, -1.0, -1.0])  # 180 deg about X
    v = np.cross(z, d)
    s2 = float(np.dot(v, v))
    vx = np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0],
    ])
    return np.eye(3) + vx + vx @ vx * ((1.0 - c) / s2)


class SkeletonRenderer(BaseRenderer):
    """Renders a skeleton as solid, shaded octahedral bones with ball joints."""

    def __init__(self, skeleton: Skeleton, masked_indices=None):
        super().__init__()
        self.skeleton = skeleton
        self.bones = self._build_bones(masked_indices)
        # Joints to draw a ball at: every endpoint of a drawn bone.
        joint_set = set()
        for parent_idx, child_idx in self.bones:
            joint_set.update((parent_idx, child_idx))
        self.joint_indices = sorted(joint_set)
        self._sphere_verts, self._sphere_faces = _make_uv_sphere()

    def draw(self, viewer, skeleton_instance: SkeletonInstance, id: wp.int32):
        """Build and display the bone + joint meshes for the given pose."""
        if skeleton_instance.skeleton != self.skeleton:
            raise ValueError(f"[ERROR]: SkeletonInstance.skeleton [{skeleton_instance.skeleton}] is not equal to SkeletonRenderer.skeleton [{self.skeleton}]")

        global_transforms = pose_utils.compute_global_pose(
            self.skeleton, skeleton_instance.local_transforms, skeleton_instance.xform)
        positions = np.asarray(global_transforms[:, 0:3], dtype=np.float32)

        bone_pts, bone_idx, char_size = self._build_bone_mesh(positions)
        joint_pts, joint_idx = self._build_joint_mesh(positions, char_size)

        bone_color = self._as_vec3(skeleton_instance.color)
        # Joints get a lighter accent so they read as distinct ball joints.
        joint_color = wp.vec3(
            0.45 * bone_color[0] + 0.55,
            0.45 * bone_color[1] + 0.55,
            0.45 * bone_color[2] + 0.55)

        bones_name = f"/skeleton_bones_{id}"
        self._register_unique_id(bones_name)
        viewer.log_mesh(bones_name, wp.array(bone_pts, dtype=wp.vec3), wp.array(bone_idx, dtype=wp.int32))
        self._set_color(viewer, bones_name, bone_color)

        if len(joint_pts) > 0:
            joints_name = f"/skeleton_joints_{id}"
            self._register_unique_id(joints_name)
            viewer.log_mesh(joints_name, wp.array(joint_pts, dtype=wp.vec3), wp.array(joint_idx, dtype=wp.int32))
            self._set_color(viewer, joints_name, joint_color)

    def clear(self, viewer):
        """Remove all skeleton meshes from the viewer."""
        self._clear(viewer.objects)

    def _build_bone_mesh(self, positions: np.ndarray):
        """Return (points, indices, characteristic_size) for the octahedral bones."""
        lengths = [float(np.linalg.norm(positions[c] - positions[p])) for p, c in self.bones]
        char_size = float(np.median(lengths)) if lengths else 1.0
        if char_size < 1e-6:
            char_size = 1.0

        all_pts = np.empty((len(self.bones) * 6, 3), dtype=np.float32)
        all_idx = np.empty((len(self.bones) * 8, 3), dtype=np.int32)
        for b, (parent_idx, child_idx) in enumerate(self.bones):
            head = positions[parent_idx]
            tail = positions[child_idx]
            direction = tail - head
            length = max(float(np.linalg.norm(direction)), 1e-5)
            width = max(length * 0.09, char_size * 0.04)
            rot = _basis_from_z(direction)
            local = np.array([
                [0.0, 0.0, 0.0],
                [0.0, 0.0, length],
                [width, 0.0, 0.12 * length],
                [0.0, width, 0.12 * length],
                [-width, 0.0, 0.12 * length],
                [0.0, -width, 0.12 * length],
            ], dtype=np.float32)
            all_pts[b * 6:(b + 1) * 6] = head + (rot @ local.T).T
            all_idx[b * 8:(b + 1) * 8] = _BONE_FACES + b * 6
        return all_pts, all_idx.reshape(-1), char_size

    def _build_joint_mesh(self, positions: np.ndarray, char_size: float):
        """Return (points, indices) for spheres at each drawn joint."""
        if not self.joint_indices:
            return np.empty((0, 3), dtype=np.float32), np.empty((0,), dtype=np.int32)

        radius = char_size * 0.07
        nv = len(self._sphere_verts)
        all_pts = np.empty((len(self.joint_indices) * nv, 3), dtype=np.float32)
        all_idx = np.empty((len(self.joint_indices) * len(self._sphere_faces), 3), dtype=np.int32)
        for k, joint_idx in enumerate(self.joint_indices):
            all_pts[k * nv:(k + 1) * nv] = positions[joint_idx] + radius * self._sphere_verts
            all_idx[k * len(self._sphere_faces):(k + 1) * len(self._sphere_faces)] = self._sphere_faces + k * nv
        return all_pts, all_idx.reshape(-1)

    @staticmethod
    def _as_vec3(color) -> wp.vec3:
        return wp.vec3(float(color[0]), float(color[1]), float(color[2]))

    @staticmethod
    def _set_color(viewer, object_name, color: wp.vec3):
        """Set a flat per-object color on the ViewerGL backend (no-op elsewhere)."""
        if isinstance(viewer, newton.viewer.ViewerGL):
            if object_name in viewer.objects:
                from pyglet import gl
                gl.glBindVertexArray(viewer.objects[object_name].vao)
                gl.glVertexAttrib3f(7, color[0], color[1], color[2])
                gl.glBindVertexArray(0)

    def _build_bones(self, mask_indices):
        mask = set(mask_indices) if mask_indices is not None else set()
        bones = []
        for idx in range(1, self.skeleton.num_joints):
            parent_idx = self.skeleton.joint_parent(idx)
            if (idx in mask or parent_idx in mask):
                continue
            bones.append((parent_idx, idx))
        return bones

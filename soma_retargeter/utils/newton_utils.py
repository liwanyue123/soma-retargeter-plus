# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import warp as wp
import numpy as np

import soma_retargeter.utils.pose_utils as pose_utils
from soma_retargeter.animation.animation_buffer import AnimationBuffer
from soma_retargeter.animation.skeleton import SkeletonInstance


def _split_normals_by_angle(vertices, faces, angle_deg=30.0):
    """
    Compute per-corner normals with sharp-edge splitting (CAD-style auto-smooth).

    Newton loads meshes via trimesh, which welds duplicate vertices and averages
    normals across every edge — sharp CAD edges get smeared, making dense STL
    robots look "melted"/decimated in the GL viewer. This rebuilds the mesh with
    one vertex per face corner, where each corner normal averages only adjacent
    face normals within ``angle_deg`` of its own face.

    Args:
        vertices: (N, 3) float array of welded vertices.
        faces: (F, 3) int array of triangle vertex indices.
        angle_deg: Crease angle; faces meeting at a sharper angle keep distinct normals.

    Returns:
        tuple: (new_vertices (3F, 3), new_indices (3F,), new_normals (3F, 3)).
    """
    faces = faces.reshape(-1, 3)
    num_faces = faces.shape[0]
    v0, v1, v2 = (vertices[faces[:, i]] for i in range(3))
    face_normals = np.cross(v1 - v0, v2 - v0)
    lengths = np.linalg.norm(face_normals, axis=1, keepdims=True)
    lengths[lengths < 1e-20] = 1.0
    face_normals /= lengths

    corner_vertex = faces.reshape(-1)
    corner_face = np.repeat(np.arange(num_faces), 3)

    # CSR-style adjacency: for every vertex, the list of faces touching it.
    order = np.argsort(corner_vertex, kind="stable")
    adjacent_face = corner_face[order]
    counts = np.bincount(corner_vertex, minlength=len(vertices))
    starts = np.concatenate(([0], np.cumsum(counts)[:-1]))

    # Pair every corner with every face adjacent to its vertex.
    reps = counts[corner_vertex]
    cum = np.cumsum(reps)
    within = np.arange(cum[-1]) - np.repeat(cum - reps, reps)
    neighbor = adjacent_face[np.repeat(starts[corner_vertex], reps) + within]
    corner_id = np.repeat(np.arange(3 * num_faces), reps)

    cos_thresh = np.cos(np.deg2rad(angle_deg))
    same_group = (face_normals[corner_face[corner_id]] * face_normals[neighbor]).sum(axis=1) >= cos_thresh

    grouped_id = corner_id[same_group]
    grouped_normals = face_normals[neighbor[same_group]]
    corner_normals = np.stack(
        [np.bincount(grouped_id, weights=grouped_normals[:, c], minlength=3 * num_faces) for c in range(3)],
        axis=1,
    )
    lengths = np.linalg.norm(corner_normals, axis=1, keepdims=True)
    lengths[lengths < 1e-20] = 1.0
    corner_normals /= lengths

    new_vertices = vertices[corner_vertex]
    new_indices = np.arange(3 * num_faces, dtype=np.int32)
    return new_vertices.astype(np.float32), new_indices, corner_normals.astype(np.float32)


def sharpen_visual_mesh_normals(builder, angle_deg=25.0):
    """
    Rebuild all visible mesh shapes in a ``ModelBuilder`` with sharp-edge normals.

    Call after ``add_mjcf``/``add_urdf`` and before ``finalize``. Only visible
    (visual) mesh shapes are touched; hidden collision meshes are left alone.

    Args:
        builder: Newton ``ModelBuilder`` holding freshly imported robot shapes.
        angle_deg: Crease angle passed to :func:`_split_normals_by_angle`.
            Default 25° keeps panel edges, avoids crack-like hard creases on
            curved STL shells that read as surface damage.
    """
    import newton

    processed = set()
    n_meshes = 0
    for i, source in enumerate(builder.shape_source):
        if source is None or id(source) in processed:
            continue
        if not (builder.shape_flags[i] & int(newton.ShapeFlags.VISIBLE)):
            continue
        vertices = getattr(source, "vertices", None)
        indices = getattr(source, "indices", None)
        if vertices is None or indices is None or len(indices) == 0:
            continue
        processed.add(id(source))
        new_vertices, new_indices, new_normals = _split_normals_by_angle(
            np.asarray(vertices, dtype=np.float32), np.asarray(indices, dtype=np.int32), angle_deg
        )
        source._vertices = new_vertices
        source._indices = new_indices
        source._normals = new_normals
        if source._uvs is not None:
            source._uvs = source._uvs[np.asarray(indices, dtype=np.int32).reshape(-1)]
        source._cached_hash = None
        source.mesh = None
        n_meshes += 1
    print(
        f"[INFO]: HQ visuals — sharpened normals on {n_meshes} mesh(es) "
        f"(crease={angle_deg:.0f}°)."
    )


def create_child_parent_map(model):
    """
    Build a mapping between child and parent joints from Newton model.

    Args:
        model: A Newton model object containing joint_parent and joint_child attributes.

    Returns:
        dict: A dictionary where keys are child joint indices and values are their
              corresponding parent joint indices.
    """
    child_parent_map = {}
    joint_parents = model.joint_parent.numpy()
    joint_child = model.joint_child.numpy()
    for i in range(len(joint_parents)):
        parent_index = joint_parents[i]
        child_index = joint_child[i]
        child_parent_map[child_index] = parent_index
    return child_parent_map


def create_joint_coord_masks(model, active_body_masks, default_mask_fill_value):
    """
    Create a joint coord mask array for a Newton model based on specified active body masks.

    Args:
        model: A model object containing joint coordinate information, including:
            - joint_coord_count: Total number of joint coordinates
            - joint_q_start: Array of starting indices for each joint's coordinates
            - joint_dof_dim: Array of DOF dimensions for each joint
            - body_label: List of body labels in order of body indices
        active_body_masks (dict): Dictionary mapping body names to their mask values.
            Only bodies present in this dictionary will have their masks updated.
        default_mask_fill_value (float): The default value to fill the entire mask array with
            before applying active body mask values.

    Returns:
        numpy.ndarray: Array of mask values for each joint coordinate.
    """
    mask_np = np.full(model.joint_coord_count, default_mask_fill_value, dtype=np.float32)
    joint_q_start_np = model.joint_q_start.numpy()
    joint_dof_dim_np = model.joint_dof_dim.numpy()
    body_name_to_idx = {get_name_from_label(k): i for i, k in enumerate(model.body_label)}
    for (key, value) in active_body_masks.items():
        idx = body_name_to_idx[key]
        start_idx = joint_q_start_np[idx]
        dim = joint_dof_dim_np[idx][1]
        mask_np[start_idx:start_idx+dim] = value

    return mask_np


def create_buffer_with_initialization_frames(
        init_pose: SkeletonInstance,
        animation_buffer: AnimationBuffer,
        num_frames_to_insert: int,
        num_stabilization_frames: int):
    """
    Construct a new AnimationBuffer that prepends a sequence of initialization frames
    transitioning smoothly from a given initial pose into an existing animation.
    The generated sequence includes:
      1. Root blending frames (transitioning the global position & orientation)
      2. Joint blending frames (interpolating joint rotations)
      3. Stabilization frames (steady pose holding before animation playback)

    Args:
        init_pose (SkeletonInstance): Starting skeleton pose to initialize from.
        animation_buffer (AnimationBuffer): Existing animation to blend into.
        num_frames_to_insert (int): Total number of transition frames to generate.
        num_stabilization_frames (int): Additional frames to hold the first blended pose.
    Returns:
        AnimationBuffer: A new buffer containing the prepended initialization frames followed by the original animation data.
    """
    num_root_blend_frames = max(0, num_frames_to_insert // 2)
    num_joint_blend_frames = max(0, num_frames_to_insert - num_root_blend_frames)
    num_stabilization_frames = max(0, num_stabilization_frames)

    index_map = np.fromiter(
        (init_pose.skeleton.joint_index(name) for name in animation_buffer.skeleton.joint_names),
        dtype=np.int32,
        count=animation_buffer.skeleton.num_joints)

    mask = index_map != -1
    start_pose = animation_buffer.skeleton.reference_local_transforms
    start_pose[mask] = init_pose.get_local_transforms()[index_map[mask]]
    end_pose = animation_buffer.get_local_transforms(0)

    # Step 1: Root blending from init_pose to first frame of animation_buffer
    start_pose_wp = wp.transform(start_pose[0][:3], start_pose[0][3:])
    root_str_t = wp.transform_get_translation(start_pose_wp)
    root_str_q = wp.transform_get_rotation(start_pose_wp)
    end_pose_wp = wp.transform(end_pose[0][:3], end_pose[0][3:])
    root_end_t = wp.transform_get_translation(end_pose_wp)
    root_end_q = wp.transform_get_rotation(end_pose_wp)
    initialization_poses = []
    for i in range(num_root_blend_frames):
        t = i / (num_root_blend_frames - 1)
        initialization_poses.append(np.copy(start_pose))
        initialization_poses[i][0] = wp.transform(
            wp.lerp(root_str_t, root_end_t, t),
            wp.quat_slerp(root_str_q, root_end_q, t))

    # Step 2: Pose blending from last initialization_poses to first frame of animation_buffer
    start_pose = initialization_poses[-1]
    for i in range(num_joint_blend_frames):
        initialization_poses.append(
            pose_utils.blend_poses(start_pose, end_pose, (i + 1) / num_joint_blend_frames))

    for i in range(num_stabilization_frames):
        initialization_poses.append(end_pose)

    return AnimationBuffer(
        animation_buffer.skeleton,
        animation_buffer.num_frames + num_frames_to_insert + num_stabilization_frames,
        animation_buffer.sample_rate,
        np.concatenate((np.stack(initialization_poses), animation_buffer.local_transforms)))


def get_name_from_label(label: str):
    """Return the leaf component of a hierarchical label.

    Args:
        label: Slash-delimited label string (e.g. ``"robot/link1"``).

    Returns:
        The final path component of the label.
    """
    return label.split("/")[-1]

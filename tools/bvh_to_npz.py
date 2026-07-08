#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Convert BVH motion files to NPZ archives.

Each output .npz contains:
    joint_names          : (J,)       str     — joint name for each index
    parent_indices       : (J,)       int32   — parent joint index (-1 for root)
    fps                  : scalar     float32 — frames per second
    num_frames           : scalar     int32   — total number of frames
    local_transforms     : (F, J, 7)  float32 — per-frame local joint transforms
                                                 layout: [px, py, pz, qx, qy, qz, qw]
    ref_local_transforms : (J, 7)     float32 — reference (T-pose) local transforms

Usage
-----
    python tools/bvh_to_npz.py dataset/my_data2/*.bvh
    python tools/bvh_to_npz.py dataset/my_data2/foo.bvh --out dataset/my_data2/foo.npz
    python tools/bvh_to_npz.py dataset/my_data2/*.bvh --position_scale 1.0
"""

import argparse
import sys
from pathlib import Path

import numpy as np


def convert(bvh_path: Path, out_path: Path, position_scale: float, recenter_xy: bool) -> None:
    import soma_retargeter.assets.bvh as bvh_utils

    skeleton, animation = bvh_utils.load_bvh(
        str(bvh_path),
        position_scale=position_scale,
        recenter_xy=recenter_xy,
    )

    # local_transforms is (num_frames, num_joints) of wp.transform (7 float32 each)
    local_tf = animation.local_transforms.view(np.float32).reshape(
        animation.num_frames, skeleton.num_joints, 7
    )
    ref_tf = skeleton.reference_local_transforms.view(np.float32).reshape(
        skeleton.num_joints, 7
    )

    np.savez_compressed(
        out_path,
        joint_names         = np.array(skeleton.joint_names, dtype=object),
        parent_indices      = skeleton.parent_indices.astype(np.int32),
        fps                 = np.float32(animation.sample_rate),
        num_frames          = np.int32(animation.num_frames),
        local_transforms    = local_tf.astype(np.float32),
        ref_local_transforms= ref_tf.astype(np.float32),
    )
    print(f"[OK] {bvh_path.name} -> {out_path}  "
          f"({animation.num_frames} frames, {skeleton.num_joints} joints, "
          f"{animation.sample_rate:.1f} fps)")


def main():
    parser = argparse.ArgumentParser(description="Convert BVH files to NPZ.")
    parser.add_argument("bvh_files", nargs="+", help="Input BVH file(s)")
    parser.add_argument("--out", default=None,
                        help="Output .npz path (only valid for a single input file; "
                             "otherwise output is placed next to each input file)")
    parser.add_argument("--position_scale", type=float, default=1.0,
                        help="Extra position scale applied on top of the built-in "
                             "cm->m (×0.01) conversion. Default: 1.0 (standard BVH)")
    parser.add_argument("--recenter_xy", action="store_true",
                        help="Shift the clip so the root's frame-0 XY sits at the origin")
    args = parser.parse_args()

    bvh_paths = [Path(p) for p in args.bvh_files]

    if args.out and len(bvh_paths) > 1:
        print("[ERROR] --out can only be used with a single input file.", file=sys.stderr)
        sys.exit(1)

    for bvh_path in bvh_paths:
        if not bvh_path.exists():
            print(f"[ERROR] File not found: {bvh_path}", file=sys.stderr)
            continue
        out_path = Path(args.out) if args.out else bvh_path.with_suffix(".npz")
        convert(bvh_path, out_path, args.position_scale, args.recenter_xy)


if __name__ == "__main__":
    main()

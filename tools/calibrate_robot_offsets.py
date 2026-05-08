#!/usr/bin/env python3
"""
Calibrate ``joint_offsets`` and/or ``joint_scales`` of a robot scaler config.

Given:
  - SOMA's zero-pose BVH (already in repo)
  - The robot's MJCF
  - The robot's reference joint angles (a pose that physically matches the
    SOMA zero pose), stored in ``tools/<robot>_reference_pose.json``

The math lives in :mod:`soma_retargeter.robotics.calibration`. This script is
a thin CLI wrapper.

Recipes:

  # Print computed offsets (q only) - safe, no writes:
  python tools/calibrate_robot_offsets.py engineai_pm01

  # Update offset.q and joint_scales together (recommended for new robots):
  python tools/calibrate_robot_offsets.py engineai_pm01 --write --scales

  # Also recompute offset.p as the residual after scaling
  # (only meaningful when scales are also up-to-date):
  python tools/calibrate_robot_offsets.py engineai_pm01 --write --scales --calc-pos
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import warp as wp
import newton

import soma_retargeter.assets.bvh as bvh_utils
import soma_retargeter.utils.io_utils as io_utils
import soma_retargeter.pipelines.utils as pipeline_utils
import soma_retargeter.robotics.calibration as calibration
from soma_retargeter.animation.skeleton import SkeletonInstance
from soma_retargeter.utils.space_conversion_utils import (
    SpaceConverter, get_facing_direction_type_from_str)


def _load_soma_zero_globals(retargeter_cfg, facing_direction):
    init_bvh_path = io_utils.get_config_file(retargeter_cfg['initialization_pose'])
    soma_skel, soma_anim = bvh_utils.load_bvh(init_bvh_path)
    converter = SpaceConverter(get_facing_direction_type_from_str(facing_direction))
    soma_inst = SkeletonInstance(
        soma_skel, [0, 0, 0], converter.transform(wp.transform_identity()))
    soma_inst.set_local_transforms(soma_anim.get_local_transforms(0))
    return soma_skel, soma_inst.compute_global_transforms()


def _load_robot_globals(robot_type, ref_data):
    builder = newton.ModelBuilder()
    builder.add_mjcf(str(pipeline_utils.get_robot_mjcf_path(robot_type)))
    model = builder.finalize()

    base_pos = ref_data['base_pos']
    base_quat_xyzw = ref_data.get('base_quat_xyzw', [0, 0, 0, 1])
    joint_angles_rad = ref_data['joint_angles_rad']

    joint_q = np.zeros(model.joint_coord_count, dtype=np.float32)
    joint_q[0:3] = base_pos
    joint_q[3:7] = base_quat_xyzw
    if len(joint_angles_rad) + 7 != model.joint_coord_count:
        raise ValueError(
            f"Reference pose has {len(joint_angles_rad)} joint angles, but "
            f"model expects {model.joint_coord_count - 7}")
    joint_q[7:] = joint_angles_rad
    model.joint_q.assign(wp.array(joint_q, dtype=wp.float32))

    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    body_q = state.body_q.numpy()
    return calibration.collect_robot_link_globals(builder, body_q)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("robot_type", help="e.g. engineai_pm01")
    parser.add_argument(
        "--ref",
        default=None,
        help="Path to reference pose JSON. Defaults to "
             "tools/<robot_type>_reference_pose.json")
    parser.add_argument(
        "--write",
        action="store_true",
        help="If set, write the new joint_offsets / joint_scales back into "
             "the scaler config.")
    parser.add_argument(
        "--scales",
        action="store_true",
        help="If set, also (re)compute joint_scales from the geometry of the "
             "matching reference poses (|robot_vec| / |soma_vec| per joint).")
    parser.add_argument(
        "--calc-pos",
        action="store_true",
        help="If set, also compute offset.p as the residual after scaling. "
             "Use with --scales for sane values. Without --scales the "
             "residual is computed against unscaled SOMA, which can produce "
             "decimeter-level offsets and make the robot squat / drift.")
    parser.add_argument(
        "--keep-pos",
        action="store_true",
        default=True,
        help="If set (default), preserve existing offset.p values from the "
             "scaler config and only overwrite offset.q. Ignored when "
             "--calc-pos is set.")
    parser.add_argument(
        "--no-keep-pos",
        dest="keep_pos",
        action="store_false",
        help="Disable the default --keep-pos behavior.")
    args = parser.parse_args()

    ref_path = Path(args.ref) if args.ref else (
        Path(__file__).parent / f"{args.robot_type}_reference_pose.json")
    if not ref_path.exists():
        print(f"[ERROR]: Reference pose file not found: {ref_path}")
        sys.exit(1)

    retargeter_cfg = pipeline_utils.get_retargeter_config(
        pipeline_utils.SourceType.SOMA,
        pipeline_utils.get_target_type_from_str(args.robot_type))
    scaler_cfg_path = io_utils.get_config_file(retargeter_cfg['human_robot_scaler_config'])
    scaler_cfg = io_utils.load_json(scaler_cfg_path)
    ik_map = retargeter_cfg['ik_map']

    height_ratio = float(retargeter_cfg.get('model_height', 1.8)) / float(
        scaler_cfg.get('human_height_assumption', 1.8))

    soma_skel, soma_globals = _load_soma_zero_globals(
        retargeter_cfg, facing_direction="Mujoco")
    robot_link_globals = _load_robot_globals(
        args.robot_type, json.loads(ref_path.read_text()))

    new_scales = None
    if args.scales:
        new_scales = calibration.compute_scales(
            soma_globals,
            soma_skel.joint_names,
            robot_link_globals,
            ik_map,
            height_ratio=height_ratio)
        print("\n=== Computed joint_scales (config values) ===")
        print(json.dumps(new_scales, indent=4))

    # When computing offset.p we want the *new* scales (if computed in this
    # run) so the residual subtracts the right scaled SOMA position.
    scales_for_offset = new_scales if new_scales is not None else scaler_cfg.get('joint_scales', {})

    new_offsets = calibration.compute_offsets(
        soma_globals,
        soma_skel.joint_names,
        robot_link_globals,
        ik_map,
        compute_position=args.calc_pos,
        joint_scales=scales_for_offset,
        height_ratio=height_ratio)

    print("\n=== Computed joint_offsets block ===")
    print(json.dumps(new_offsets, indent=4))

    if args.write:
        if new_scales is not None:
            calibration.merge_scales_into_config(scaler_cfg, new_scales)
        calibration.merge_offsets_into_config(
            scaler_cfg, new_offsets,
            keep_existing_position=(args.keep_pos and not args.calc_pos))
        calibration.write_scaler_config(scaler_cfg, scaler_cfg_path)
        print(f"\n[INFO]: Wrote scaler config to {scaler_cfg_path}")
    else:
        print("\n[INFO]: Pass --write to update the scaler config in place.")


if __name__ == "__main__":
    main()

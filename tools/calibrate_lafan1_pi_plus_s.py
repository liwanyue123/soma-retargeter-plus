"""
calibrate_lafan1_pi_plus_s.py  --  Headless equivalent of the in-app
Calibration panel's "One-click calibrate" (compute + write joint_offsets,
using the existing joint_scales) for lafan1 -> hightorque_pi_plus_s.

Why this needs re-running: ``joint_offsets["<joint>"].q`` is calibrated as
``inverse(source_zero_pose.q) * robot_reference_pose.q`` (see
soma_retargeter/robotics/calibration.py). It was previously computed against
an EARLIER version of configs/sources/lafan1/init_pose.bvh (before the Hips
orientation fix in tools/gen_lafan1_init_pose.py) and against yaw_offset_deg=0
(before pipelines/utils.py added the -90 deg lafan1 yaw). Both of those
changed the zero pose's GLOBAL per-joint orientation, so the saved offset.q
values are now stale -- they silently rotate every retargeted frame by
whatever the zero pose's orientation shifted by, which is exactly the
"upper body off by ~90 deg" symptom. This script recomputes offset.q (and
offset.p, for consistency with a real one-click calibrate) against the
CURRENT zero pose + yaw, using the robot's already-authored reference pose
(tools/reference_poses/hightorque_pi_plus_s_reference_pose.json) as the
matching "robot in the same physical stance" pose.

joint_scales are NOT recomputed here: they come from bone-length RATIOS
(vector magnitudes), which don't depend on the zero pose's rotation
convention at all -- only compute_offsets (rotation-dependent) needs it.

Run: conda run -n soma-retargeter python tools/calibrate_lafan1_pi_plus_s.py
"""
import json
import os

import numpy as np
import warp as wp
import newton

import soma_retargeter.assets.bvh as bvh_utils
import soma_retargeter.pipelines.utils as pipeline_utils
import soma_retargeter.robotics.calibration as calibration_utils
import soma_retargeter.utils.io_utils as io_utils
from soma_retargeter.pipelines.utils import SourceType, TargetType
from soma_retargeter.utils.space_conversion_utils import SpaceConverter, get_facing_direction_type_from_str

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_HERE, "..")
_ROBOT_TYPE = "hightorque_pi_plus_s"
_REF_POSE_PATH = os.path.join(_ROOT, "tools", "reference_poses", f"{_ROBOT_TYPE}_reference_pose.json")


def main():
    wp.init()

    retargeter_cfg = pipeline_utils.get_retargeter_config(SourceType.LAFAN1, TargetType.HIGHTORQUE_PI_PLUS_S)
    ik_map = retargeter_cfg["ik_map"]
    scaler_cfg_path = io_utils.get_config_file(retargeter_cfg["human_robot_scaler_config"])
    offsets_cfg_path = io_utils.get_config_file(retargeter_cfg["joint_offsets_config"])
    scaler_cfg = io_utils.load_json(scaler_cfg_path)
    height_ratio = float(retargeter_cfg["model_height"]) / float(scaler_cfg["human_height_assumption"])

    # ---- 1) source zero-pose global transforms, at the CURRENT convention ----
    facing = pipeline_utils.get_source_facing_direction(SourceType.LAFAN1)
    yaw = pipeline_utils.get_source_yaw_offset_deg(SourceType.LAFAN1)
    scale = pipeline_utils.get_source_position_scale(SourceType.LAFAN1)
    recenter = pipeline_utils.get_source_recenter_xy(SourceType.LAFAN1)
    converter = SpaceConverter(get_facing_direction_type_from_str(facing), yaw_offset_deg=yaw)

    init_bvh = io_utils.get_config_file(retargeter_cfg["initialization_pose"])
    ref_skel, ref_anim = bvh_utils.load_bvh(init_bvh, position_scale=scale, recenter_xy=recenter)
    local_zero = ref_anim.get_local_transforms(0)
    ref_globals = ref_skel.compute_global_transforms(local_zero, converter.transform(wp.transform_identity()))

    # ---- 2) robot body globals at the matching reference stance ----
    robot_builder = newton.ModelBuilder()
    pipeline_utils.add_robot_model(robot_builder, _ROBOT_TYPE)
    builder = newton.ModelBuilder()
    builder.add_builder(robot_builder, xform=wp.transform_identity())
    model = builder.finalize()
    state = model.state()

    joint_q = model.joint_q.numpy().astype(np.float32).copy()
    ref = json.loads(open(_REF_POSE_PATH).read())
    if ref.get("base_pos"):
        joint_q[0:3] = ref["base_pos"]
    if ref.get("base_quat_xyzw"):
        joint_q[3:7] = ref["base_quat_xyzw"]
    angles = dict(zip(ref.get("joint_order", []), ref.get("joint_angles_rad", [])))
    for ji in range(robot_builder.joint_count):
        if robot_builder.joint_type[ji] != newton.JointType.REVOLUTE:
            continue
        from soma_retargeter.utils import newton_utils
        name = newton_utils.get_name_from_label(robot_builder.joint_label[ji])
        if name in angles:
            q_idx = int(robot_builder.joint_q_start[ji])
            joint_q[q_idx] = float(angles[name])

    wp.copy(model.joint_q, wp.array(joint_q, dtype=wp.float32))
    newton.eval_fk(model, model.joint_q, model.joint_qd, state, None)
    body_q = state.body_q.numpy()
    link_globals = calibration_utils.collect_robot_link_globals(robot_builder, body_q)

    # ---- 3) compute + write joint_offsets (keep existing joint_scales) ----
    joint_scales = scaler_cfg["joint_scales"]
    new_offsets = calibration_utils.compute_offsets(
        ref_globals, ref_skel.joint_names, link_globals, ik_map,
        compute_position=True, joint_scales=joint_scales, height_ratio=height_ratio)

    offsets_cfg = io_utils.load_json(offsets_cfg_path)
    calibration_utils.merge_offsets_into_config(offsets_cfg, new_offsets, keep_existing_position=False)
    calibration_utils.write_scaler_config(offsets_cfg, offsets_cfg_path)

    print(f"[OK] Recomputed {len(new_offsets)} joint_offsets -> {offsets_cfg_path}")
    for name, (p, q) in list(new_offsets.items())[:3]:
        print(f"  {name}: p={np.round(p, 4)} q={np.round(q, 4)}")


if __name__ == "__main__":
    main()

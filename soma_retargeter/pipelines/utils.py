# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from enum import IntEnum, auto
from pathlib import Path

import warp as wp

import soma_retargeter.utils.io_utils as io_utils
import soma_retargeter.assets.usd as usd_utils


class SourceType(IntEnum):
    """Enumeration of supported source model types."""
    SOMA = auto()
    # Native (non-SOMA) source skeleton. Selected via `--data mydata`; uses your
    # own joint names + a dedicated retargeter config, with no SOMA skin mesh.
    MYDATA = auto()
    # Second native skeleton variant. Same coordinate conventions as MYDATA but
    # with its own calibration configs (scaler / offsets) so the two datasets
    # can be retargeted independently. Selected via `--data mydata2`.
    MYDATA2 = auto()
    # Fingerless Mixamo-style spine, 21 joints (e.g.
    # dataset/my_data3/PM_dance_002_han.bvh). Joint names are a strict subset
    # of MYDATA2's, but the actor's proportions differ substantially
    # (shoulders/torso ~1.4x vs near-equal legs), so it carries its own init
    # pose + calibration. Selected via `--data mydata3`.
    MYDATA3 = auto()
    # Fourth native skeleton variant (e.g. dataset/my_data4/kongfang_take_002.bvh): same
    # 60-joint naming as MYDATA (Spine..Spine3, fingers, ToeBase) but a
    # different coordinate convention -- standard Y-up centimetre BVH whose
    # rest pose lays every bone along local +Z (rotations encode the whole
    # pose). Selected via `--data mydata4`.
    MYDATA4 = auto()
    # LAFAN1 dataset skeleton (Y-up, standard-centimetre BVH, Mixamo/LAFAN1
    # joint names, no fingers). Selected via `--data lafan1`.
    LAFAN1 = auto()


class TargetType(IntEnum):
    """Enumeration of supported target model types."""
    UNITREE_G1 = auto()
    ENGINEAI_PM01 = auto()
    HIGHTORQUE_PI_PLUS = auto()
    HIGHTORQUE_PI_PLUS_S = auto()
    PNDBOTICS_ADAM_LITE = auto()
    PNDBOTICS_ADAM_SP = auto()

_SOURCE_TYPE_TO_STR = {
    SourceType.SOMA    : "soma",
    SourceType.MYDATA  : "mydata",
    SourceType.MYDATA2 : "mydata2",
    SourceType.MYDATA3 : "mydata3",
    SourceType.MYDATA4 : "mydata4",
    SourceType.LAFAN1  : "lafan1",
}
_STR_TO_SOURCE_TYPE = {s : t for t, s in _SOURCE_TYPE_TO_STR.items()}

_TARGET_TYPE_TO_STR = {
    TargetType.UNITREE_G1            : "unitree_g1",
    TargetType.ENGINEAI_PM01         : "engineai_pm01",
    TargetType.HIGHTORQUE_PI_PLUS    : "hightorque_pi_plus",
    TargetType.HIGHTORQUE_PI_PLUS_S  : "hightorque_pi_plus_s",
    TargetType.PNDBOTICS_ADAM_LITE   : "pndbotics_adam_lite",
    TargetType.PNDBOTICS_ADAM_SP     : "pndbotics_adam_sp",
}
_STR_TO_TARGET_TYPE = {s : t for t, s in _TARGET_TYPE_TO_STR.items()}

# Per-robot relative MJCF path under assets/robots/<robot_type>/ (and inside
# Newton's downloadable asset bundle, which uses the same layout).
_ROBOT_MJCF_RELATIVE_PATH = {
    "unitree_g1":           "mjcf/g1_29dof_rev_1_0.xml",
    "engineai_pm01":        "pm.xml",
    "hightorque_pi_plus":   "xml/PiPlus_S_12L8A0G2H0W.xml",
    "hightorque_pi_plus_s": "xml/PiPlus_S_12L8A0G2H1W_LSE_260611.xml",
    "pndbotics_adam_lite":  "adam_lite.xml",
    "pndbotics_adam_sp":    "adam_sp.xml",
}

# Initial base Z height (metres) for URDF-based robots. MJCF files encode
# the base height directly in the worldbody (e.g. <body pos="0 0 0.93">),
# so they are not listed here. For URDF robots, add_urdf defaults to origin
# (0, 0, 0), so we supply the standing height via the xform argument to
# avoid the robot spawning inside the ground.
_ROBOT_URDF_INITIAL_BASE_Z = {
    # (no URDF-based robots currently; add entries here if needed)
}

# Per-robot retargeter config filename under
# soma_retargeter/configs/<robot_type>/.
# Filenames are relative to soma_retargeter/configs/<robot_type>/ and grouped
# into a per-source subfolder (soma/ or mydata/).
_RETARGETER_CONFIG_FILENAME = {
    (SourceType.SOMA, "unitree_g1"):          "soma/soma_to_g1_retargeter_config.json",
    (SourceType.SOMA, "engineai_pm01"):       "soma/soma_to_pm01_retargeter_config.json",
    (SourceType.SOMA, "hightorque_pi_plus"):  "soma/soma_to_pi_plus_retargeter_config.json",
    (SourceType.SOMA, "hightorque_pi_plus_s"): "soma/soma_to_pi_plus_s_retargeter_config.json",
    (SourceType.SOMA, "pndbotics_adam_lite"): "soma/soma_to_adam_lite_retargeter_config.json",
    (SourceType.SOMA, "pndbotics_adam_sp"):   "soma/soma_to_adam_sp_retargeter_config.json",
    # Native skeleton (route A): retarget straight from your own joint names.
    (SourceType.MYDATA, "unitree_g1"):          "mydata/mydata_to_g1_retargeter_config.json",
    (SourceType.MYDATA, "engineai_pm01"):       "mydata/mydata_to_pm01_retargeter_config.json",
    (SourceType.MYDATA, "hightorque_pi_plus"):  "mydata/mydata_to_pi_plus_retargeter_config.json",
    (SourceType.MYDATA, "hightorque_pi_plus_s"): "mydata/mydata_to_pi_plus_s_retargeter_config.json",
    (SourceType.MYDATA, "pndbotics_adam_lite"): "mydata/mydata_to_adam_lite_retargeter_config.json",
    (SourceType.MYDATA, "pndbotics_adam_sp"):   "mydata/mydata_to_adam_sp_retargeter_config.json",
    # Second native skeleton variant with its own calibration configs.
    (SourceType.MYDATA2, "unitree_g1"):          "mydata2/mydata2_to_g1_retargeter_config.json",
    (SourceType.MYDATA2, "engineai_pm01"):       "mydata2/mydata2_to_pm01_retargeter_config.json",
    (SourceType.MYDATA2, "hightorque_pi_plus"):  "mydata2/mydata2_to_pi_plus_retargeter_config.json",
    (SourceType.MYDATA2, "hightorque_pi_plus_s"): "mydata2/mydata2_to_pi_plus_s_retargeter_config.json",
    (SourceType.MYDATA2, "pndbotics_adam_lite"): "mydata2/mydata2_to_adam_lite_retargeter_config.json",
    (SourceType.MYDATA2, "pndbotics_adam_sp"):   "mydata2/mydata2_to_adam_sp_retargeter_config.json",
    (SourceType.MYDATA3, "unitree_g1"):          "mydata3/mydata3_to_g1_retargeter_config.json",
    (SourceType.MYDATA3, "engineai_pm01"):       "mydata3/mydata3_to_pm01_retargeter_config.json",
    (SourceType.MYDATA3, "hightorque_pi_plus"):  "mydata3/mydata3_to_pi_plus_retargeter_config.json",
    (SourceType.MYDATA3, "hightorque_pi_plus_s"): "mydata3/mydata3_to_pi_plus_s_retargeter_config.json",
    (SourceType.MYDATA3, "pndbotics_adam_lite"): "mydata3/mydata3_to_adam_lite_retargeter_config.json",
    (SourceType.MYDATA3, "pndbotics_adam_sp"):   "mydata3/mydata3_to_adam_sp_retargeter_config.json",

    (SourceType.MYDATA4, "unitree_g1"):          "mydata4/mydata4_to_g1_retargeter_config.json",
    (SourceType.MYDATA4, "engineai_pm01"):       "mydata4/mydata4_to_pm01_retargeter_config.json",
    (SourceType.MYDATA4, "hightorque_pi_plus"):  "mydata4/mydata4_to_pi_plus_retargeter_config.json",
    (SourceType.MYDATA4, "hightorque_pi_plus_s"): "mydata4/mydata4_to_pi_plus_s_retargeter_config.json",
    (SourceType.MYDATA4, "pndbotics_adam_lite"): "mydata4/mydata4_to_adam_lite_retargeter_config.json",
    (SourceType.MYDATA4, "pndbotics_adam_sp"):   "mydata4/mydata4_to_adam_sp_retargeter_config.json",
    # LAFAN1 dataset skeleton. Add more (source, robot) rows here (+ matching
    # configs) to extend it.
    (SourceType.LAFAN1, "hightorque_pi_plus_s"): "lafan1/lafan1_to_pi_plus_s_retargeter_config.json",
    (SourceType.LAFAN1, "engineai_pm01"):       "lafan1/lafan1_to_pm01_retargeter_config.json",
}


def get_robot_mjcf_path(robot_type: str) -> Path:
    """Resolve the MJCF path for a robot.

    Resolution order:
        1. Local override under ``<project_root>/assets/robots/<robot_type>/...``.
        2. Newton built-in asset via ``newton.utils.download_asset``.

    Args:
        robot_type: Robot type string (e.g. ``"unitree_g1"``).

    Returns:
        Filesystem path to the MJCF file.

    Raises:
        ValueError: If ``robot_type`` has no MJCF mapping registered.
        FileNotFoundError: If neither the local override nor the Newton built-in
            asset resolve to an existing file.
    """
    relative = _ROBOT_MJCF_RELATIVE_PATH.get(robot_type)
    if relative is None:
        allowed = ", ".join(_ROBOT_MJCF_RELATIVE_PATH.keys())
        raise ValueError(
            f"No MJCF mapping registered for robot type [{robot_type}]. "
            f"Allowed values: {allowed}"
        )

    local_path = io_utils.get_robot_asset(robot_type, *relative.split("/"))
    if local_path.exists():
        print(f"[INFO]: Using local MJCF for [{robot_type}]: {local_path}")
        return local_path

    import newton
    fallback = newton.utils.download_asset(robot_type) / relative
    if not Path(fallback).exists():
        raise FileNotFoundError(
            f"[ERROR]: MJCF for robot [{robot_type}] not found locally at "
            f"[{local_path}] nor in Newton built-in assets at [{fallback}]."
        )
    print(f"[INFO]: Using Newton built-in MJCF for [{robot_type}]: {fallback}")
    return fallback


def add_robot_model(builder, robot_type: str) -> None:
    """Add a robot's model to a Newton ``ModelBuilder``, dispatching on file type.

    Most robots ship as MJCF (``add_mjcf``). URDF robots are also supported
    via ``add_urdf``; for those the initial base Z is looked up from
    ``_ROBOT_URDF_INITIAL_BASE_Z`` and passed as ``xform`` to avoid spawning
    inside the ground plane (MJCF encodes the base height in the worldbody
    directly, so no extra handling is needed).

    Args:
        builder: Newton ``ModelBuilder`` to add the robot to.
        robot_type: Robot type string (e.g. ``"pndbotics_adam_sp"``).
    """
    path = str(get_robot_mjcf_path(robot_type))
    if path.lower().endswith(".urdf"):
        base_z = _ROBOT_URDF_INITIAL_BASE_Z.get(robot_type, 0.0)
        xform = wp.transform((0.0, 0.0, base_z), wp.quat_identity())
        builder.add_urdf(path, floating=True, xform=xform)
    else:
        builder.add_mjcf(path)


# Per-source coordinate convention + unit scale defaults. SOMA mocap is Y-up
# (Maya/"Mujoco" conversion) in centimetres (the loader's built-in *0.01 puts it
# in metres). Native ("mydata") mocap is already Z-up (Newton frame) and authored
# ~3.16x smaller than cm-scale, so we apply an extra position scale to bring it to
# real human size (~1.77 m) and keep it consistent with the robot.
_SOURCE_FACING_DIRECTION = {
    SourceType.SOMA    : "Mujoco",
    SourceType.MYDATA  : "Newton",
    # mydata2 is a standard Y-up BVH (same convention as SOMA/Mujoco).
    SourceType.MYDATA2 : "Mujoco",
    # mydata3 shares mydata2's conventions (Y-up standard BVH).
    SourceType.MYDATA3 : "Mujoco",
    # mydata4 (kongfang) is also a standard Y-up BVH; the bones-along-+Z rest
    # pose is a rig-internal detail that FK resolves, not a world convention.
    SourceType.MYDATA4 : "Mujoco",
    # LAFAN1 is also a standard Y-up BVH.
    SourceType.LAFAN1  : "Mujoco",
}
_SOURCE_POSITION_SCALE = {
    SourceType.SOMA    : 1.0,
    SourceType.MYDATA  : 3.16,
    # mydata2 is a standard centimetre-scale BVH; the built-in ×0.01 already
    # converts it to metres, so no extra scale factor is needed.
    SourceType.MYDATA2 : 1.0,
    SourceType.MYDATA3 : 1.0,
    # mydata4 is centimetre-scale too (~157 cm standing height in file units).
    SourceType.MYDATA4 : 1.0,
    # LAFAN1 is also a standard centimetre-scale BVH; no extra scale needed.
    SourceType.LAFAN1  : 1.0,
}
# SOMA clips are authored at the origin; native clips carry a world offset, so
# recenter them horizontally to behave the same.
# NOTE: recenter_xy removes local_transforms[0, 0, 0:2] = [px, py] from the
# root joint. For Z-up (mydata/Newton) this correctly zeroes the XY horizontal
# plane while preserving the Z height. For Y-up (mydata2/Mujoco) py IS the
# height axis, so recentering would snap the hip to Y=0 (ground), burying the
# lower body. Keep it False for Y-up sources and rely on the space converter to
# land the character at the right height.
_SOURCE_RECENTER_XY = {
    SourceType.SOMA    : False,
    SourceType.MYDATA  : True,
    SourceType.MYDATA2 : False,
    SourceType.MYDATA3 : False,
    SourceType.MYDATA4 : False,
    SourceType.LAFAN1  : False,
}
# Extra yaw (deg, about up axis) to align the source's forward with the robot.
# mydata is captured back-to-back with the robot, so flip it 180 deg.
# mydata2 uses the same facing direction as SOMA (no extra flip needed).
# lafan1's own rig has a different native forward axis than SOMA/mydata2 (+X
# in its own Y-up frame vs their +Z), which the Mujoco facing conversion maps
# to Newton +X instead of the -Y that SOMA/mydata2 end up facing -- this -90
# deg yaw rotates it to match that same -Y convention.
_SOURCE_YAW_OFFSET_DEG = {
    SourceType.SOMA    : 0.0,
    SourceType.MYDATA  : 180.0,
    SourceType.MYDATA2 : 0.0,
    SourceType.MYDATA3 : 0.0,
    # mydata4 faces -Z in its Y-up frame (toes/left-right placement in the
    # capture confirm it), opposite of SOMA/mydata2's +Z, so flip it 180 deg.
    SourceType.MYDATA4 : 180.0,
    SourceType.LAFAN1  : -90.0,
}


def get_source_facing_direction(source) -> str:
    """Default facing-direction string for a source type (str or SourceType)."""
    if isinstance(source, str):
        source = get_source_type_from_str(source)
    return _SOURCE_FACING_DIRECTION.get(source, "Mujoco")


def get_source_position_scale(source) -> float:
    """Extra position scale applied at load for a source type (str or SourceType)."""
    if isinstance(source, str):
        source = get_source_type_from_str(source)
    return _SOURCE_POSITION_SCALE.get(source, 1.0)


def get_source_recenter_xy(source) -> bool:
    """Whether to horizontally recenter clips at load for a source type."""
    if isinstance(source, str):
        source = get_source_type_from_str(source)
    return _SOURCE_RECENTER_XY.get(source, False)


def get_source_yaw_offset_deg(source) -> float:
    """Extra yaw (deg about up axis) to align the source's forward with the robot."""
    if isinstance(source, str):
        source = get_source_type_from_str(source)
    return _SOURCE_YAW_OFFSET_DEG.get(source, 0.0)


def get_source_str_from_type(source: SourceType) -> str:
    """
    Get the string name associated with a given source type.

    Args:
        source (SourceType): The source type enum value.

    Returns:
        str: The string representation of the source type.
    """
    return _SOURCE_TYPE_TO_STR[source]


def get_source_type_from_str(source: str) -> SourceType:
    """
    Convert a string to its corresponding SourceType enum value.

    Args:
        source (str): The string representation of a source.

    Returns:
        SourceType: The corresponding source type enum.

    Raises:
        ValueError: If the provided string does not correspond to a valid source type.
    """
    try:
        return _STR_TO_SOURCE_TYPE[source]
    except KeyError:
        allowed = ", ".join(_STR_TO_SOURCE_TYPE.keys())
        raise ValueError(f"Unknown source type: [{source}]. Allowed values: {allowed}") from None


def get_target_str_from_type(target: TargetType) -> str:
    """
    Get the string name associated with a given target type.

    Args:
        target (TargetType): The target type enum value.

    Returns:
        str: The string representation of the target type.
    """
    return _TARGET_TYPE_TO_STR[target]


def get_target_type_from_str(target: str) -> TargetType:
    """
    Convert a string to its corresponding TargetType enum value.

    Args:
        target (str): The string representation of a target.

    Returns:
        TargetType: The corresponding target type enum.

    Raises:
        ValueError: If the provided string does not correspond to a valid target type.
    """
    try:
        return _STR_TO_TARGET_TYPE[target]
    except KeyError:
        allowed = ", ".join(_STR_TO_TARGET_TYPE.keys())
        raise ValueError(f"Unknown target type: [{target}]. Allowed values: {allowed}") from None


def get_source_model_mesh(source: SourceType, skeleton) -> dict:
    """
    Retrieve model mesh for a given source type.

    Args:
        source (SourceType): The source type for which properties should be retrieved.
        skeleton: The skeleton associated with the source model, used for loading the mesh.

    Returns:
        SkeletalMesh: The skeleton mesh for the given source type.

    Raises:
        ValueError: If the source type is not recognized.
    """
    if source == SourceType.SOMA:
        return usd_utils.load_skeletal_mesh_from_usd(
            str(io_utils.get_config_file('sources', 'soma', 'soma_base_skel_minimal.usd')),
            skeleton,
            '/OUTPUT/c_geometry_grp',
            '/OUTPUT/c_skeleton_grp/Root')

    # Native / non-SOMA skeletons (mydata and any future source) have no skin
    # mesh bound to SOMA joint names; callers fall back to drawing the skeleton
    # bones directly.
    return None


def get_retargeter_config(source: SourceType, target: TargetType) -> dict:
    """
    Load the retargeter configuration between a specific source and target.

    Args:
        source (SourceType): The source type.
        target (TargetType): The target type.

    Returns:
        dict: The loaded JSON configuration for the retargeter.

    Raises:
        ValueError: If the source or target type is not supported.
    """
    config_dir = _TARGET_TYPE_TO_STR.get(target)
    if config_dir is None:
        raise ValueError(f"Unknown target type [{target}].")

    filename = _RETARGETER_CONFIG_FILENAME.get((source, config_dir))
    if filename is None:
        raise ValueError(
            f"No retargeter config registered for source=[{source}] "
            f"target=[{config_dir}].")

    return io_utils.load_json(
        io_utils.get_config_file(config_dir, filename)
    )

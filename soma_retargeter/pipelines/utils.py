# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from enum import IntEnum, auto
from pathlib import Path

import soma_retargeter.utils.io_utils as io_utils
import soma_retargeter.assets.usd as usd_utils


class SourceType(IntEnum):
    """Enumeration of supported source model types."""
    SOMA = auto()


class TargetType(IntEnum):
    """Enumeration of supported target model types."""
    UNITREE_G1 = auto()
    ENGINEAI_PM01 = auto()

_SOURCE_TYPE_TO_STR = {
    SourceType.SOMA : "soma"
}
_STR_TO_SOURCE_TYPE = {s : t for t, s in _SOURCE_TYPE_TO_STR.items()}

_TARGET_TYPE_TO_STR = {
    TargetType.UNITREE_G1 : "unitree_g1",
    TargetType.ENGINEAI_PM01 : "engineai_pm01",
}
_STR_TO_TARGET_TYPE = {s : t for t, s in _TARGET_TYPE_TO_STR.items()}

# Per-robot relative MJCF path under assets/robots/<robot_type>/ (and inside
# Newton's downloadable asset bundle, which uses the same layout).
_ROBOT_MJCF_RELATIVE_PATH = {
    "unitree_g1": "mjcf/g1_29dof_rev_1_0.xml",
    "engineai_pm01": "pm.xml",
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
            str(io_utils.get_config_file('soma', 'soma_base_skel_minimal.usd')),
            skeleton,
            '/OUTPUT/c_geometry_grp',
            '/OUTPUT/c_skeleton_grp/Root')

    raise ValueError(f"Unknown source type {source}.")


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
    if target == TargetType.UNITREE_G1:
        config_dir = 'unitree_g1'
        if source == SourceType.SOMA:
            filename = 'soma_to_g1_retargeter_config.json'
        else:
            raise ValueError(f"Unknown source type [{source}] for target [{target}].")
    elif target == TargetType.ENGINEAI_PM01:
        config_dir = 'engineai_pm01'
        if source == SourceType.SOMA:
            filename = 'soma_to_pm01_retargeter_config.json'
        else:
            raise ValueError(f"Unknown source type [{source}] for target [{target}].")
    else:
        raise ValueError(f"Unknown target type [{target}].")

    return io_utils.load_json(
        io_utils.get_config_file(config_dir, filename)
    )

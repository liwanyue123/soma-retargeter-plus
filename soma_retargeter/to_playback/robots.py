# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-robot specs for TO playback (CIO history ↔ soma assets)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class ContactSite:
    """Contact force application point in a parent body frame."""

    name: str
    parent_body: str
    local_xyz: Tuple[float, float, float]


@dataclass(frozen=True)
class ToRobotSpec:
    """Maps a CIO ``robot`` key onto soma-retargeter assets."""

    key: str
    """CIO / history folder key, e.g. ``t800``."""

    soma_robot_type: str
    """``retarget_target`` / MJCF key, e.g. ``engineai_t800``."""

    enabled: bool = True
    """False = listed in UI as coming-soon."""

    file_prefix: str = ""
    """Basename prefix for result/reference files (defaults to ``key``)."""

    converter_config: str = ""
    """Viewer launch config under ``assets/`` (hint for matching robot session)."""

    contact_sites: Tuple[ContactSite, ...] = field(default_factory=tuple)
    """Force frames in CIO ``contacts.frames`` order (parent + local xyz)."""

    def resolved_prefix(self) -> str:
        return self.file_prefix or self.key


# ---------------------------------------------------------------------------
# Contact locals from CIO URDFs (J_FIXED_* / FOOT frames). Order matches
# ``*_robot_config.json`` ``contacts.frames`` so force columns line up.
# ---------------------------------------------------------------------------

_T800_CONTACTS: Tuple[ContactSite, ...] = (
    ContactSite("LINK_FOOT_L_1", "LINK_ANKLE_ROLL_L", (0.12, 0.04, -0.065)),
    ContactSite("LINK_FOOT_L_2", "LINK_ANKLE_ROLL_L", (0.12, -0.05, -0.065)),
    ContactSite("LINK_FOOT_L_3", "LINK_ANKLE_ROLL_L", (-0.09, 0.04, -0.065)),
    ContactSite("LINK_FOOT_L_4", "LINK_ANKLE_ROLL_L", (-0.09, -0.04, -0.065)),
    ContactSite("LINK_FOOT_R_1", "LINK_ANKLE_ROLL_R", (0.12, 0.05, -0.065)),
    ContactSite("LINK_FOOT_R_2", "LINK_ANKLE_ROLL_R", (0.12, -0.04, -0.06)),
    ContactSite("LINK_FOOT_R_3", "LINK_ANKLE_ROLL_R", (-0.09, 0.04, -0.065)),
    ContactSite("LINK_FOOT_R_4", "LINK_ANKLE_ROLL_R", (-0.09, -0.04, -0.065)),
)

# G1: a few CIO parents are absent in soma MJCF — bake their fixed-joint
# offsets into the nearest soma body (wrist_yaw / pelvis / torso).
_G1_CONTACTS: Tuple[ContactSite, ...] = (
    ContactSite("LINK_FOOT_L_1", "left_ankle_roll_link", (0.12, 0.032, -0.035)),
    ContactSite("LINK_FOOT_L_2", "left_ankle_roll_link", (0.12, -0.032, -0.035)),
    ContactSite("LINK_FOOT_L_3", "left_ankle_roll_link", (-0.055, 0.023, -0.035)),
    ContactSite("LINK_FOOT_L_4", "left_ankle_roll_link", (-0.055, -0.023, -0.035)),
    ContactSite("LINK_FOOT_R_1", "right_ankle_roll_link", (0.12, 0.032, -0.035)),
    ContactSite("LINK_FOOT_R_2", "right_ankle_roll_link", (0.12, -0.032, -0.035)),
    ContactSite("LINK_FOOT_R_3", "right_ankle_roll_link", (-0.055, 0.023, -0.035)),
    ContactSite("LINK_FOOT_R_4", "right_ankle_roll_link", (-0.055, -0.023, -0.035)),
    ContactSite("LINK_HIP_PITCH_L_single", "left_hip_pitch_link", (0.015, 0.05, -0.03)),
    ContactSite("LINK_HIP_PITCH_R_single", "right_hip_pitch_link", (0.015, -0.05, -0.03)),
    ContactSite("LINK_KNEE_L_1", "left_knee_link", (0.01, -0.005, 0.0)),
    ContactSite("LINK_KNEE_R_1", "right_knee_link", (0.01, 0.005, 0.0)),
    ContactSite("LINK_SHOULDER_L_1", "left_shoulder_pitch_link", (0.0, 0.04, -0.015)),
    ContactSite("LINK_SHOULDER_R_1", "right_shoulder_pitch_link", (0.0, -0.04, -0.015)),
    ContactSite("LINK_ELBOW_L_1", "left_shoulder_yaw_link", (0.01, -0.005, -0.08)),
    ContactSite("LINK_ELBOW_R_1", "right_shoulder_yaw_link", (0.01, 0.005, -0.08)),
    # sphere_hand @ (0.029,-0.003,0) from wrist_yaw + end (0,0,0.023)
    ContactSite("LINK_ELBOW_END_L", "left_wrist_yaw_link", (0.029, -0.003, 0.023)),
    ContactSite("LINK_ELBOW_END_R", "right_wrist_yaw_link", (0.029, -0.003, 0.023)),
    ContactSite("LINK_BACK_1", "torso_link", (-0.055, 0.085, 0.25)),
    ContactSite("LINK_BACK_2", "torso_link", (-0.055, -0.085, 0.25)),
    ContactSite("LINK_BACK_3", "torso_link", (-0.055, -0.05, 0.01)),
    ContactSite("LINK_BACK_4", "torso_link", (-0.055, 0.05, 0.01)),
    ContactSite("LINK_BASE_1", "pelvis", (0.0, 0.0, -0.091)),
    # head_link @ (0.0039635,0,-0.044) from torso + tip (0,0,0.465)
    ContactSite("LINK_HEAD_1", "torso_link", (0.0039635, 0.0, 0.421)),
)

_PM01_CONTACTS: Tuple[ContactSite, ...] = (
    ContactSite("LINK_FOOT_L_1", "LINK_ANKLE_ROLL_L", (0.115, 0.04, -0.04)),
    ContactSite("LINK_FOOT_L_2", "LINK_ANKLE_ROLL_L", (0.115, -0.04, -0.04)),
    ContactSite("LINK_FOOT_L_3", "LINK_ANKLE_ROLL_L", (-0.06, 0.036, -0.04)),
    ContactSite("LINK_FOOT_L_4", "LINK_ANKLE_ROLL_L", (-0.06, -0.036, -0.04)),
    ContactSite("LINK_FOOT_L_5", "LINK_ANKLE_ROLL_L", (0.16, 0.005, -0.02)),
    ContactSite("LINK_FOOT_R_1", "LINK_ANKLE_ROLL_R", (0.115, 0.04, -0.04)),
    ContactSite("LINK_FOOT_R_2", "LINK_ANKLE_ROLL_R", (0.115, -0.04, -0.04)),
    ContactSite("LINK_FOOT_R_3", "LINK_ANKLE_ROLL_R", (-0.06, 0.036, -0.04)),
    ContactSite("LINK_FOOT_R_4", "LINK_ANKLE_ROLL_R", (-0.06, -0.036, -0.04)),
    ContactSite("LINK_FOOT_R_5", "LINK_ANKLE_ROLL_R", (0.16, 0.005, -0.02)),
    ContactSite("LINK_HIP_PITCH_L_single", "LINK_HIP_PITCH_L", (0.005, 0.049359, -0.0132)),
    ContactSite("LINK_HIP_PITCH_R_single", "LINK_HIP_PITCH_R", (0.005, -0.049359, -0.0132)),
    ContactSite("LINK_KNEE_L_1", "LINK_KNEE_PITCH_L", (0.0, 0.0, 0.0)),
    ContactSite("LINK_KNEE_R_1", "LINK_KNEE_PITCH_R", (0.0, 0.0, 0.0)),
    ContactSite("LINK_SHOULDER_L_1", "LINK_SHOULDER_PITCH_L", (0.0, 0.065, -0.02)),
    ContactSite("LINK_SHOULDER_R_1", "LINK_SHOULDER_PITCH_R", (0.0, -0.065, -0.02)),
    ContactSite("LINK_ELBOW_L_1", "LINK_ELBOW_PITCH_L", (0.0, 0.0, 0.0)),
    ContactSite("LINK_ELBOW_R_1", "LINK_ELBOW_PITCH_R", (0.0, 0.0, 0.0)),
    ContactSite("LINK_ELBOW_END_L", "LINK_ELBOW_YAW_L", (0.03, -0.02, -0.14)),
    ContactSite("LINK_ELBOW_END_R", "LINK_ELBOW_YAW_R", (0.03, 0.02, -0.14)),
    ContactSite("LINK_BACK_1", "LINK_TORSO_YAW", (-0.005, 0.0, 0.16)),
    ContactSite("LINK_BACK_2", "LINK_TORSO_YAW", (-0.11, -0.05, 0.21)),
    ContactSite("LINK_BACK_3", "LINK_TORSO_YAW", (-0.07, 0.08, 0.01)),
    ContactSite("LINK_BACK_4", "LINK_TORSO_YAW", (-0.07, -0.08, 0.01)),
    ContactSite("LINK_BASE_1", "LINK_BASE", (0.02, 0.0, -0.03)),
    ContactSite("LINK_HEAD_1", "LINK_HEAD_YAW", (0.005, 0.0, 0.115)),
)

_PI_PLUS_FOOT_CONTACTS: Tuple[ContactSite, ...] = (
    ContactSite("LINK_FOOT_L_1", "l_ankle_roll_link", (0.056, 0.033, -0.05)),
    ContactSite("LINK_FOOT_L_2", "l_ankle_roll_link", (0.056, -0.033, -0.05)),
    ContactSite("LINK_FOOT_L_3", "l_ankle_roll_link", (-0.11, 0.025, -0.05)),
    ContactSite("LINK_FOOT_L_4", "l_ankle_roll_link", (-0.11, -0.025, -0.05)),
    ContactSite("LINK_FOOT_R_1", "r_ankle_roll_link", (0.056, 0.033, -0.05)),
    ContactSite("LINK_FOOT_R_2", "r_ankle_roll_link", (0.056, -0.033, -0.05)),
    ContactSite("LINK_FOOT_R_3", "r_ankle_roll_link", (-0.11, 0.025, -0.05)),
    ContactSite("LINK_FOOT_R_4", "r_ankle_roll_link", (-0.11, -0.025, -0.05)),
)

_PI_PLUS_S_CONTACTS: Tuple[ContactSite, ...] = _PI_PLUS_FOOT_CONTACTS + (
    ContactSite("LINK_HIP_PITCH_L_single", "l_hip_pitch_link", (0.0, 0.034, -0.002)),
    ContactSite("LINK_HIP_PITCH_R_single", "r_hip_pitch_link", (0.0, -0.034, -0.002)),
    ContactSite("LINK_KNEE_L_1", "l_calf_link", (0.0, 0.0, 0.0)),
    ContactSite("LINK_KNEE_R_1", "r_calf_link", (0.0, 0.0, 0.0)),
    ContactSite("LINK_SHOULDER_L_1", "l_shoulder_pitch_link", (0.0, 0.03, 0.0)),
    ContactSite("LINK_SHOULDER_R_1", "r_shoulder_pitch_link", (0.0, -0.03, 0.0)),
    ContactSite("LINK_ELBOW_L_1", "l_elbow_link", (0.0, 0.0, 0.0)),
    ContactSite("LINK_ELBOW_R_1", "r_elbow_link", (0.0, 0.0, 0.0)),
    ContactSite("LINK_ELBOW_END_L", "l_wrist_link", (0.0, 0.0, -0.1)),
    ContactSite("LINK_ELBOW_END_R", "r_wrist_link", (0.0, 0.0, -0.1)),
    ContactSite("LINK_BACK_1", "base_link", (0.0, 0.0, 0.15)),
    ContactSite("LINK_BASE_1", "base_link", (0.0, 0.0, -0.045)),
    ContactSite("LINK_HEAD_1", "head_pitch_link", (0.0, 0.0, 0.06)),
)

_PND_CONTACTS: Tuple[ContactSite, ...] = (
    ContactSite("LINK_FOOT_L_1", "toeLeft", (0.155, 0.035, -0.063)),
    ContactSite("LINK_FOOT_L_2", "toeLeft", (0.155, -0.03, -0.063)),
    ContactSite("LINK_FOOT_L_3", "toeLeft", (-0.06, 0.035, -0.063)),
    ContactSite("LINK_FOOT_L_4", "toeLeft", (-0.06, -0.025, -0.063)),
    ContactSite("LINK_FOOT_R_1", "toeRight", (0.155, 0.035, -0.063)),
    ContactSite("LINK_FOOT_R_2", "toeRight", (0.155, -0.03, -0.063)),
    ContactSite("LINK_FOOT_R_3", "toeRight", (-0.06, 0.035, -0.063)),
    ContactSite("LINK_FOOT_R_4", "toeRight", (-0.06, -0.025, -0.063)),
    ContactSite("LINK_HIP_PITCH_L_single", "hipPitchLeft", (-0.065, 0.07, 0.0)),
    ContactSite("LINK_HIP_PITCH_R_single", "hipPitchRight", (-0.065, -0.07, 0.0)),
    ContactSite("LINK_THIGH_L_1", "thighLeft_geom_2", (0.0, -0.01, -0.1)),
    ContactSite("LINK_THIGH_R_1", "thighRight_geom_2", (0.0, 0.01, -0.1)),
    ContactSite("LINK_THIGH_L_2", "thighLeft_geom_2", (0.01, -0.025, -0.28)),
    ContactSite("LINK_THIGH_R_2", "thighRight_geom_2", (0.01, 0.025, -0.28)),
    ContactSite("LINK_KNEE_L_1", "shinLeft", (0.0, 0.002, 0.0)),
    ContactSite("LINK_KNEE_R_1", "shinRight", (0.0, -0.002, 0.0)),
    ContactSite("LINK_CALF_L_1", "shinLeft_geom_3", (0.0, -0.005, -0.095)),
    ContactSite("LINK_CALF_R_1", "shinRight_geom_3", (0.0, 0.005, -0.095)),
    ContactSite("LINK_CALF_L_2", "shinLeft_geom_3", (0.0, 0.005, -0.168)),
    ContactSite("LINK_CALF_R_2", "shinRight_geom_3", (0.0, -0.005, -0.168)),
    ContactSite("LINK_SHOULDER_L_1", "shoulderPitchLeft", (0.0, 0.04, -0.005)),
    ContactSite("LINK_SHOULDER_R_1", "shoulderPitchRight", (0.0, -0.04, -0.005)),
    ContactSite("LINK_ELBOW_L_1", "shoulderYawLeft", (0.0, 0.005, -0.2)),
    ContactSite("LINK_ELBOW_R_1", "shoulderYawRight", (0.0, -0.005, -0.2)),
    ContactSite("LINK_ELBOW_END_L", "wristYawLeft", (0.0, 0.0, -0.275)),
    ContactSite("LINK_ELBOW_END_R", "wristYawRight", (0.0, 0.0, -0.275)),
    ContactSite("LINK_BACK_1", "torso", (0.0, 0.0, 0.15)),
    ContactSite("LINK_BASE_1", "pelvis", (-0.005, 0.0, 0.01)),
    ContactSite("LINK_HEAD_1", "torso", (0.04, 0.0, 0.48)),
)


TO_PLAYBACK_ROBOTS: Dict[str, ToRobotSpec] = {
    "t800": ToRobotSpec(
        key="t800",
        soma_robot_type="engineai_t800",
        enabled=True,
        converter_config="t800_bvh_to_csv_converter_config.json",
        contact_sites=_T800_CONTACTS,
    ),
    "g1": ToRobotSpec(
        key="g1",
        soma_robot_type="unitree_g1",
        enabled=True,
        converter_config="g1_bvh_to_csv_converter_config.json",
        contact_sites=_G1_CONTACTS,
    ),
    "pm01": ToRobotSpec(
        key="pm01",
        soma_robot_type="engineai_pm01",
        enabled=True,
        converter_config="pm01_bvh_to_csv_converter_config.json",
        contact_sites=_PM01_CONTACTS,
    ),
    "pi_plus": ToRobotSpec(
        key="pi_plus",
        soma_robot_type="hightorque_pi_plus",
        enabled=True,
        converter_config="pi_plus_bvh_to_csv_converter_config.json",
        contact_sites=_PI_PLUS_FOOT_CONTACTS,
    ),
    "pi_plus_s": ToRobotSpec(
        key="pi_plus_s",
        soma_robot_type="hightorque_pi_plus_s",
        enabled=True,
        converter_config="pi_plus_s_bvh_to_csv_converter_config.json",
        contact_sites=_PI_PLUS_S_CONTACTS,
    ),
    "pnd": ToRobotSpec(
        key="pnd",
        soma_robot_type="pndbotics_adam_lite",
        enabled=True,
        converter_config="adam_lite_bvh_to_csv_converter_config.json",
        contact_sites=_PND_CONTACTS,
    ),
}


def list_to_robot_keys(enabled_only: bool = False) -> List[str]:
    keys = list(TO_PLAYBACK_ROBOTS.keys())
    if enabled_only:
        keys = [k for k in keys if TO_PLAYBACK_ROBOTS[k].enabled]
    return keys


def get_to_robot_spec(key: str) -> ToRobotSpec:
    k = key.lower().strip()
    if k not in TO_PLAYBACK_ROBOTS:
        allowed = ", ".join(TO_PLAYBACK_ROBOTS)
        raise KeyError(f"Unknown TO playback robot '{key}'. Known: {allowed}")
    return TO_PLAYBACK_ROBOTS[k]


def infer_robot_key_from_folder(folder_name: str) -> str | None:
    """Guess CIO robot key from a history run folder or parent name."""
    name = folder_name.lower().replace(" ", "_")
    # Longer keys first so ``pi_plus_s`` wins over ``pi_plus``.
    for key in sorted(TO_PLAYBACK_ROBOTS, key=len, reverse=True):
        if name.startswith(key) or f"{key}_" in name or name.endswith(f"_{key}"):
            return key
    for key in sorted(TO_PLAYBACK_ROBOTS, key=len, reverse=True):
        if key in name:
            return key
    return None

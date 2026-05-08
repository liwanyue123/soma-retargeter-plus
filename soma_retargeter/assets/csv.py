# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import csv
from dataclasses import dataclass
from typing import Protocol, ClassVar, List

import numpy as np
import warp as wp

from scipy.spatial.transform import Rotation as R
from soma_retargeter.robotics.csv_animation_buffer import CSVAnimationBuffer


class RobotCSVConfig(Protocol):
    name: str
    csv_header: List[str]

    def to_anim_frame(self, csv_row: np.ndarray) -> np.ndarray:
        ...
    def to_csv_row(self, frame_idx: int, anim_row: np.ndarray) -> List[float]:
        ...


@dataclass
class _StandardCSVConfig:
    """Shared base for robot CSV configs whose only difference is the
    column header (joint names) and an optional unit convention.

    The CSV layout is::

        Frame, root_tx_cm, root_ty_cm, root_tz_cm,
               root_rx_deg, root_ry_deg, root_rz_deg,
               joint_0_deg, joint_1_deg, ... joint_n_deg

    Sub-classes only need to set ``name`` and ``csv_header``.
    """
    name: str = "_unnamed"
    csv_header: ClassVar[List[str]] = []

    def to_anim_frame(self, csv_row: np.ndarray) -> np.ndarray:
        """Convert one CSV row (including frame index) into one anim frame."""
        num_joint_dofs = csv_row.shape[0] - 1  # remove frame index
        anim_row = np.zeros(num_joint_dofs + 1, dtype=np.float32)

        anim_row[0:3] = csv_row[1:4] * 0.01  # cm -> m
        euler = np.deg2rad(csv_row[4:7])
        anim_row[3:7] = wp.quat_rpy(euler[0], euler[1], euler[2])
        anim_row[7:] = np.deg2rad(csv_row[7:])
        return anim_row

    def to_csv_row(self, frame_idx: int, anim_row: np.ndarray) -> List[float]:
        """Convert one anim buffer row into a CSV row."""
        t = wp.vec3(*anim_row[0:3]) * 100.0  # m -> cm
        q = wp.quat(*anim_row[3:7])
        euler = R.from_quat([q[0], q[1], q[2], q[3]]).as_euler("xyz", degrees=True)
        row = [frame_idx, t[0], t[1], t[2], euler[0], euler[1], euler[2]]
        row.extend(np.rad2deg(anim_row[7:]))
        return row


_ROOT_HEADER = [
    "Frame",
    "root_translateX", "root_translateY", "root_translateZ",
    "root_rotateX", "root_rotateY", "root_rotateZ",
]


@dataclass
class UnitreeG129DOF_CSVConfig(_StandardCSVConfig):
    name: str = "unitree_g1_29dof"
    csv_header: ClassVar[List[str]] = _ROOT_HEADER + [
        "left_hip_pitch_joint_dof", "left_hip_roll_joint_dof", "left_hip_yaw_joint_dof",
        "left_knee_joint_dof", "left_ankle_pitch_joint_dof", "left_ankle_roll_joint_dof",
        "right_hip_pitch_joint_dof", "right_hip_roll_joint_dof", "right_hip_yaw_joint_dof",
        "right_knee_joint_dof", "right_ankle_pitch_joint_dof", "right_ankle_roll_joint_dof",
        "waist_yaw_joint_dof", "waist_roll_joint_dof", "waist_pitch_joint_dof",
        "left_shoulder_pitch_joint_dof", "left_shoulder_roll_joint_dof",
        "left_shoulder_yaw_joint_dof", "left_elbow_joint_dof",
        "left_wrist_roll_joint_dof", "left_wrist_pitch_joint_dof", "left_wrist_yaw_joint_dof",
        "right_shoulder_pitch_joint_dof", "right_shoulder_roll_joint_dof",
        "right_shoulder_yaw_joint_dof", "right_elbow_joint_dof",
        "right_wrist_roll_joint_dof", "right_wrist_pitch_joint_dof",
        "right_wrist_yaw_joint_dof"]


@dataclass
class EngineAIPM01_CSVConfig(_StandardCSVConfig):
    name: str = "engineai_pm01_24dof"
    csv_header: ClassVar[List[str]] = _ROOT_HEADER + [
        "J00_HIP_PITCH_L", "J01_HIP_ROLL_L", "J02_HIP_YAW_L",
        "J03_KNEE_PITCH_L", "J04_ANKLE_PITCH_L", "J05_ANKLE_ROLL_L",
        "J06_HIP_PITCH_R", "J07_HIP_ROLL_R", "J08_HIP_YAW_R",
        "J09_KNEE_PITCH_R", "J10_ANKLE_PITCH_R", "J11_ANKLE_ROLL_R",
        "J12_WAIST_YAW",
        "J13_SHOULDER_PITCH_L", "J14_SHOULDER_ROLL_L", "J15_SHOULDER_YAW_L",
        "J16_ELBOW_PITCH_L", "J17_ELBOW_YAW_L",
        "J18_SHOULDER_PITCH_R", "J19_SHOULDER_ROLL_R", "J20_SHOULDER_YAW_R",
        "J21_ELBOW_PITCH_R", "J22_ELBOW_YAW_R",
        "J23_HEAD_YAW"]


@dataclass
class HighTorquePiPlus_CSVConfig(_StandardCSVConfig):
    """20-DOF small humanoid by HighTorque (no waist, no wrist, no head)."""
    name: str = "hightorque_pi_plus_20dof"
    csv_header: ClassVar[List[str]] = _ROOT_HEADER + [
        "l_hip_pitch_joint", "l_hip_roll_joint", "l_thigh_joint",
        "l_calf_joint", "l_ankle_pitch_joint", "l_ankle_roll_joint",
        "l_shoulder_pitch_joint", "l_shoulder_roll_joint",
        "l_upper_arm_joint", "l_elbow_joint",
        "r_hip_pitch_joint", "r_hip_roll_joint", "r_thigh_joint",
        "r_calf_joint", "r_ankle_pitch_joint", "r_ankle_roll_joint",
        "r_shoulder_pitch_joint", "r_shoulder_roll_joint",
        "r_upper_arm_joint", "r_elbow_joint"]


@dataclass
class PNDboticsAdamLite_CSVConfig(_StandardCSVConfig):
    """25-DOF full humanoid by PNDbotics (3-DOF waist, 5-DOF arms, no head)."""
    name: str = "pndbotics_adam_lite_25dof"
    csv_header: ClassVar[List[str]] = _ROOT_HEADER + [
        "hipPitch_Left", "hipRoll_Left", "hipYaw_Left",
        "kneePitch_Left", "anklePitch_Left", "ankleRoll_Left",
        "hipPitch_Right", "hipRoll_Right", "hipYaw_Right",
        "kneePitch_Right", "anklePitch_Right", "ankleRoll_Right",
        "waistRoll", "waistPitch", "waistYaw",
        "shoulderPitch_Left", "shoulderRoll_Left", "shoulderYaw_Left",
        "elbow_Left", "wristYaw_Left",
        "shoulderPitch_Right", "shoulderRoll_Right", "shoulderYaw_Right",
        "elbow_Right", "wristYaw_Right"]


_ROBOT_CSV_CONFIGS = {
    "unitree_g1":          UnitreeG129DOF_CSVConfig,
    "engineai_pm01":       EngineAIPM01_CSVConfig,
    "hightorque_pi_plus":  HighTorquePiPlus_CSVConfig,
    "pndbotics_adam_lite": PNDboticsAdamLite_CSVConfig,
}


def get_csv_config_for_robot(robot_type: str) -> RobotCSVConfig:
    """
    Resolve the CSV layout config for a given robot type.

    Args:
        robot_type: Robot type string (e.g. ``"unitree_g1"``).

    Returns:
        Instantiated CSV config.

    Raises:
        ValueError: If no CSV config is registered for ``robot_type``.
    """
    cls = _ROBOT_CSV_CONFIGS.get(robot_type)
    if cls is None:
        allowed = ", ".join(_ROBOT_CSV_CONFIGS.keys())
        raise ValueError(
            f"No CSV config registered for robot type [{robot_type}]. "
            f"Allowed values: {allowed}"
        )
    return cls()


def load_csv(file_path: str, fps: float = 120.0, csv_config: RobotCSVConfig = UnitreeG129DOF_CSVConfig()) -> CSVAnimationBuffer:
    """
    Load a robot motion CSV file into a ``CSVAnimationBuffer``.
    Args:
        file_path (str): Path to the CSV file to load.
        fps (float, optional): Frames per second for the animation. Defaults to 120.0.
        csv_config (RobotCSVConfig, optional): Configuration object that defines how to parse
            CSV rows into animation frames. Defaults to ``UnitreeG129DOF_CSVConfig``.
    Returns:
        CSVAnimationBuffer: An animation buffer containing the loaded and converted animation data.
    Raises:
        FileNotFoundError: If the CSV file at file_path does not exist.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        print(f"[INFO]: Loading CSV [{file_path}] for robot [{csv_config.name}]")
        csv_data = np.loadtxt(f, delimiter=",", skiprows=1)
        num_frames = csv_data.shape[0]

        # Each anim row is derived by config, so infer size from first row
        first_row_anim = csv_config.to_anim_frame(csv_data[0])
        anim_data = np.zeros((num_frames, first_row_anim.shape[0]), dtype=np.float32)
        anim_data[0, :] = first_row_anim

        for i in range(1, num_frames):
            anim_data[i, :] = csv_config.to_anim_frame(csv_data[i])

        return CSVAnimationBuffer.create_from_raw_data(anim_data, fps)


def save_csv(file_path: str, buffer: CSVAnimationBuffer, csv_config: RobotCSVConfig = UnitreeG129DOF_CSVConfig()) -> None:
    """
    Save a ``CSVAnimationBuffer`` to a robot motion CSV file.

    Args:
        file_path (str): The path where the CSV file will be saved.
        buffer (CSVAnimationBuffer): The animation buffer containing frame data to be saved.
        csv_config (RobotCSVConfig, optional): Configuration object that defines CSV format and headers.
            Defaults to ``UnitreeG129DOF_CSVConfig``.

    Raises:
        RuntimeError: If the buffer is empty or invalid.
        OSError: If the file cannot be opened or written.
    """
    if buffer is None or buffer.num_frames == 0:
        raise RuntimeError("[ERROR]: Empty or invalid buffer.")

    with open(file_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(csv_config.csv_header)

        for i in range(buffer.num_frames):
            data = buffer.get_data(i)
            row = csv_config.to_csv_row(i, data)
            writer.writerow(row)

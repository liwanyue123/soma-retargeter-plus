# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Trajectory-optimization (TO) result playback for the Newton viewer.

Loads archived CIO history runs (``.../result/history/<robot>_history/<run>/``)
and feeds origin (reference) + TO trajectories into dual robot articulations,
mirroring ``humanoid_visualization/scripts/show_TO_data.py``.
"""

from soma_retargeter.to_playback.loader import (
    ToPlaybackData,
    load_history_run,
    load_terrain_boxes,
)
from soma_retargeter.to_playback.robots import (
    TO_PLAYBACK_ROBOTS,
    get_to_robot_spec,
    list_to_robot_keys,
)

__all__ = [
    "ToPlaybackData",
    "load_history_run",
    "load_terrain_boxes",
    "TO_PLAYBACK_ROBOTS",
    "get_to_robot_spec",
    "list_to_robot_keys",
]

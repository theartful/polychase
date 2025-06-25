# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2025 Ahmed Essam <aessam.dahy@gmail.com>

import bpy
import bpy.utils

from .operators.analysis import PC_OT_AnalyzeVideo, PC_OT_CancelAnalysis
from .operators.keyframe_management import (
    PC_OT_AddKeyFrame,
    PC_OT_ClearKeyFrames,
    PC_OT_KeyFrameClearBackwards,
    PC_OT_KeyFrameClearForwards,
    PC_OT_KeyFrameClearSegment,
    PC_OT_NextKeyFrame,
    PC_OT_PrevKeyFrame,
    PC_OT_RemoveKeyFrame)
from .operators.open_clip import PC_OT_OpenClip
from .operators.pin_mode import (
    PC_OT_ClearPins, PC_OT_KeymapFilter, PC_OT_PinMode)
from .operators.refiner import PC_OT_CancelRefining, PC_OT_RefineSequence
from .operators.refresh_geometry import PC_OT_RefreshGeometry
from .operators.scene_operations import (
    PC_OT_CenterGeometry, PC_OT_ConvertAnimation, PC_OT_TransformScene)
from .operators.tracker_management import (
    PC_OT_CreateTracker, PC_OT_DeleteTracker, PC_OT_SelectTracker)
from .operators.tracking import PC_OT_CancelTracking, PC_OT_TrackSequence
from .properties import PolychaseData, PolychaseTracker
from .ui.panels import (
    PC_PT_PolychasePanel,
    PC_PT_TrackerAppearancePanel,
    PC_PT_TrackerCameraPanel,
    PC_PT_TrackerInputsPanel,
    PC_PT_TrackerOpticalFlowPanel,
    PC_PT_TrackerPinModePanel,
    PC_PT_TrackerScenePanel,
    PC_PT_TrackerTrackingPanel)

is_registered = False

classes = [
    # Properties
    PolychaseTracker,
    PolychaseData,

    # Operators
    PC_OT_CreateTracker,
    PC_OT_SelectTracker,
    PC_OT_DeleteTracker,
    PC_OT_PinMode,
    PC_OT_ClearPins,
    PC_OT_KeymapFilter,
    PC_OT_TrackSequence,
    PC_OT_CancelTracking,
    PC_OT_AnalyzeVideo,
    PC_OT_CancelAnalysis,
    PC_OT_RefineSequence,
    PC_OT_CancelRefining,
    PC_OT_OpenClip,
    PC_OT_RefreshGeometry,
    PC_OT_CenterGeometry,
    PC_OT_ConvertAnimation,
    PC_OT_TransformScene,
    PC_OT_KeyFrameClearBackwards,
    PC_OT_KeyFrameClearForwards,
    PC_OT_KeyFrameClearSegment,
    PC_OT_NextKeyFrame,
    PC_OT_PrevKeyFrame,
    PC_OT_AddKeyFrame,
    PC_OT_RemoveKeyFrame,
    PC_OT_ClearKeyFrames,

    # Panels
    PC_PT_PolychasePanel,
    PC_PT_TrackerInputsPanel,
    PC_PT_TrackerCameraPanel,
    PC_PT_TrackerOpticalFlowPanel,
    PC_PT_TrackerPinModePanel,
    PC_PT_TrackerTrackingPanel,
    PC_PT_TrackerScenePanel,
    PC_PT_TrackerAppearancePanel,
]


def register():
    global is_registered

    if is_registered:
        return

    for cls in classes:
        bpy.utils.register_class(cls)

    is_registered = True


def unregister():
    global is_registered

    if not is_registered:
        return

    for cls in classes:
        bpy.utils.unregister_class(cls)

    is_registered = False

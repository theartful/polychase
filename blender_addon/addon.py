import bpy
import bpy.props
import bpy.types
import bpy.utils

from .operators.pin_mode import PC_OT_PinMode, PC_OT_KeymapFilter
from .operators.tracker_management import (
    PC_OT_CreateTracker, PC_OT_DeleteTracker, PC_OT_SelectTracker)
from .operators.analysis import PC_OT_AnalyzeVideo, PC_OT_CancelAnalysis
from .operators.tracking import PC_OT_TrackSequence, PC_OT_CancelTracking
from .operators.refiner import PC_OT_RefineSequence, PC_OT_CancelRefining
from .properties import PolychaseClipTracking, PolychaseData
from .ui.panels import (
    PC_PT_PolychasePanel,
    PC_PT_TrackerInputsPanel,
    PC_PT_TrackerOpticalFlowPanel,
    PC_PT_TrackerTrackingPanel,
    PC_PT_TrackerPinModePanel,
    PC_PT_TrackerAppearancePanel,
    PC_PT_TrackerCameraPanel)


def add_addon_var(name: str, settings_type) -> None:
    setattr(
        bpy.types.MovieClip,
        name,
        bpy.props.PointerProperty(type=settings_type))


def remove_addon_var(name: str) -> None:
    delattr(bpy.types.MovieClip, name)


def get_addon_var(name: str, context):
    return getattr(context.space_data.clip, name, None)


is_registered = False

classes = [
    PolychaseClipTracking,
    PolychaseData,
    PC_OT_CreateTracker,
    PC_OT_SelectTracker,
    PC_OT_DeleteTracker,
    PC_OT_PinMode,
    PC_OT_KeymapFilter,
    PC_OT_TrackSequence,
    PC_OT_CancelTracking,
    PC_OT_AnalyzeVideo,
    PC_OT_CancelAnalysis,
    PC_OT_RefineSequence,
    PC_OT_CancelRefining,
    PC_PT_PolychasePanel,
    PC_PT_TrackerInputsPanel,
    PC_PT_TrackerCameraPanel,
    PC_PT_TrackerOpticalFlowPanel,
    PC_PT_TrackerPinModePanel,
    PC_PT_TrackerTrackingPanel,
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

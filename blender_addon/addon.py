import bpy
import bpy.props
import bpy.types
import bpy.utils

from .operators.pin_mode import OT_PinMode
from .operators.tracker_management import (OT_CreateTracker, OT_DeleteTracker, OT_SelectTracker)
from .operators.analysis import OT_AnalyzeVideo, OT_CancelAnalysis
from .operators.tracking import OT_TrackForwards, OT_CancelTracking
from .properties import PolychaseClipTracking, PolychaseData
from .ui.panels import (PT_PolychasePanel, PT_TrackerInputsPanel, PT_TrackerOpticalFlowPanel, PT_TrackerTrackingPanel)


def add_addon_var(name: str, settings_type) -> None:
    setattr(bpy.types.MovieClip, name, bpy.props.PointerProperty(type=settings_type))


def remove_addon_var(name: str) -> None:
    delattr(bpy.types.MovieClip, name)


def get_addon_var(name: str, context):
    return getattr(context.space_data.clip, name, None)


is_registered = False

classes = [
    PolychaseClipTracking,
    PolychaseData,
    PT_PolychasePanel,
    PT_TrackerInputsPanel,
    PT_TrackerOpticalFlowPanel,
    PT_TrackerTrackingPanel,
    OT_CreateTracker,
    OT_SelectTracker,
    OT_DeleteTracker,
    OT_PinMode,
    OT_TrackForwards,
    OT_CancelTracking,
    OT_AnalyzeVideo,
    OT_CancelAnalysis
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

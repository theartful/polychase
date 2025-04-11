import bpy
import bpy.types

from ..operators.tracker_management import OT_CreateTracker, OT_SelectTracker, OT_DeleteTracker
from ..operators.pin_mode import OT_PinMode
from ..operators.tracking import OT_TrackForwards
from ..operators.analysis import OT_AnalyzeVideo, OT_CancelAnalysis
from ..properties import PolychaseData


class PT_PolychasePanel(bpy.types.Panel):
    bl_label = "Polychase Trackers"
    bl_category = "Polychase"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"

    def draw(self, context: bpy.types.Context):
        layout = self.layout
        assert layout

        state = PolychaseData.from_context(context)
        if not state:
            layout.label(text="Polychase data not found on scene.")
            return

        col = layout.column(align=True)
        trackers = state.trackers

        if not state.trackers:
            col.label(text="No trackers are created yet.", icon='INFO')
        else:
            for idx, tracker in enumerate(trackers):
                is_active_tracker = (idx == state.active_tracker_idx)

                row = col.row(align=True)

                op = row.operator(
                    OT_SelectTracker.bl_idname,
                    text=tracker.name,
                    depress=is_active_tracker,
                    icon="CAMERA_DATA" if tracker.tracking_target == "CAMERA" else "MESH_DATA")

                assert hasattr(op, "idx")
                setattr(op, "idx", idx)

                op = row.operator(OT_DeleteTracker.bl_idname, text="", icon="X")
                assert hasattr(op, "idx")
                setattr(op, "idx", idx)

        row = layout.row(align=True)
        row.operator(OT_CreateTracker.bl_idname, icon="ADD")


# Base class for panels that require an active tracker
class PT_PolychaseActiveTrackerBase(bpy.types.Panel):
    bl_category = "Polychase"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        state = PolychaseData.from_context(context)
        # Check if state exists and tracker is active
        return state is not None and state.is_tracking_active()


class PT_TrackerInputsPanel(PT_PolychaseActiveTrackerBase):
    bl_label = "Inputs"
    bl_category = "Polychase"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"

    def draw(self, context: bpy.types.Context):
        state = PolychaseData.from_context(context)
        if not state:
            return

        tracker = state.active_tracker
        if not tracker:
            return

        layout = self.layout
        assert layout

        row = layout.row(align=True)
        layout.prop(tracker, "name")

        row = layout.row(align=True)
        row.alert = not tracker.clip
        row.prop(tracker, "clip")

        row = layout.row(align=True)
        row.alert = not tracker.geometry
        row.prop(tracker, "geometry")

        row = layout.row(align=True)
        row.alert = not tracker.camera
        row.prop(tracker, "camera")


class PT_TrackerTrackingPanel(PT_PolychaseActiveTrackerBase):
    bl_label = "Tracking"
    bl_category = "Polychase"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"

    def draw(self, context: bpy.types.Context):
        state = PolychaseData.from_context(context)
        if not state:
            return

        tracker = state.active_tracker
        if not tracker:
            return

        layout = self.layout
        assert layout

        row = layout.row(align=True)
        row.prop(tracker, "tracking_target", text="Target")

        row = layout.row(align=True)
        row.operator(OT_PinMode.bl_idname, depress=state.in_pinmode)

        row = layout.row(align=True)
        row.operator(OT_TrackForwards.bl_idname)


class PT_TrackerOpticalFlowPanel(PT_PolychaseActiveTrackerBase):
    bl_label = "Optical Flow"
    bl_category = "Polychase"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"

    def draw(self, context: bpy.types.Context):
        state = PolychaseData.from_context(context)
        if not state:
            return

        tracker = state.active_tracker
        if not tracker:
            return

        layout = self.layout
        assert layout

        row = layout.row(align=True)
        row.prop(tracker, "database_path")

        # Show Analyze or Cancel button based on state
        if tracker.is_preprocessing:
            row = layout.row(align=True)
            row.operator(OT_CancelAnalysis.bl_idname, text="Cancel")
            row = layout.row()
            row.progress(factor=tracker.preprocessing_progress, text=tracker.preprocessing_message, type="BAR")
        else:
            row = layout.row(align=True)
            row.operator(OT_AnalyzeVideo.bl_idname)

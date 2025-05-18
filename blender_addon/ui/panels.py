import typing

import bpy
import bpy.types

from ..operators.analysis import OT_AnalyzeVideo, OT_CancelAnalysis
from ..operators.pin_mode import OT_PinMode
from ..operators.refiner import OT_RefineSequence, OT_CancelRefining
from ..operators.tracker_management import (OT_CreateTracker, OT_DeleteTracker, OT_SelectTracker)
from ..operators.tracking import OT_CancelTracking, OT_TrackSequence
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

        row = layout.row(align=True)
        row.prop(tracker, "tracking_target", text="Target")


class PT_TrackerPinModePanel(PT_PolychaseActiveTrackerBase):
    bl_label = "Pin Mode"
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

        col = layout.column(align=True)
        col.prop(tracker, "pinmode_optimize_focal_length", text="Optimize Focal Length")
        col.prop(tracker, "pinmode_optimize_principal_point", text="Optimize Principal Point")

        row = layout.row(align=True)
        row.operator(OT_PinMode.bl_idname, depress=state.in_pinmode)


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

        col = layout.column(align=True)
        col.prop(tracker, "tracking_optimize_focal_length", text="Optimize Focal Length")
        col.prop(tracker, "tracking_optimize_principal_point", text="Optimize Principal Point")

        # Show Track or Cancel button and progress based on state
        if tracker.is_tracking:
            row = layout.row()
            row.progress(factor=tracker.tracking_progress, text=tracker.tracking_message)
            row = layout.row(align=True)
            row.operator(OT_CancelTracking.bl_idname, text="Cancel")
        elif tracker.is_refining:
            row = layout.row()
            row.progress(factor=tracker.refining_progress, text=tracker.refining_message)
            row = layout.row(align=True)
            row.operator(OT_CancelRefining.bl_idname, text="Cancel")
        else:
            # Create a row for the tracking buttons
            row = layout.row(align=True)

            # Split the row into two equal quarters
            split = row.split(factor=0.25, align=True)

            # Backwards single
            col = split.column(align=True)
            op = col.operator(OT_TrackSequence.bl_idname, text="", icon='TRACKING_BACKWARDS_SINGLE')
            op_casted = typing.cast(OT_TrackSequence, op)
            op_casted.direction = "BACKWARD"
            op_casted.single_frame = True

            # Backwards all the way
            col = split.column(align=True)
            op = col.operator(OT_TrackSequence.bl_idname, text="", icon='TRACKING_BACKWARDS')
            op_casted = typing.cast(OT_TrackSequence, op)
            op_casted.direction = "BACKWARD"
            op_casted.single_frame = False

            # Forwards all the way
            col = split.column(align=True)
            op = col.operator(OT_TrackSequence.bl_idname, text="", icon='TRACKING_FORWARDS')
            op_casted = typing.cast(OT_TrackSequence, op)
            op_casted.direction = "FORWARD"
            op_casted.single_frame = False

            # Forwards single
            col = split.column(align=True)
            op = col.operator(OT_TrackSequence.bl_idname, text="", icon='TRACKING_FORWARDS_SINGLE')
            op_casted = typing.cast(OT_TrackSequence, op)
            op_casted.direction = "FORWARD"
            op_casted.single_frame = True

            row = layout.row(align=True)
            split = row.split(factor=0.5, align=True)

            # Refine
            col = split.column(align=True)
            op = col.operator(OT_RefineSequence.bl_idname, text="Refine")

            # Refine all
            col = split.column(align=True)
            op = col.operator(OT_RefineSequence.bl_idname, text="Refine All")


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

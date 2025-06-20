import typing

import bpy
import bpy.types

from ..operators.analysis import PC_OT_AnalyzeVideo, PC_OT_CancelAnalysis
from ..operators.keyframe_management import (
    PC_OT_AddKeyFrame,
    PC_OT_ClearKeyFrames,
    PC_OT_KeyFrameClearBackwards,
    PC_OT_KeyFrameClearForwards,
    PC_OT_KeyFrameClearSegment,
    PC_OT_NextKeyFrame,
    PC_OT_PrevKeyFrame,
    PC_OT_RemoveKeyFrame)
from ..operators.open_clip import PC_OT_OpenClip
from ..operators.pin_mode import PC_OT_PinMode
from ..operators.refiner import PC_OT_CancelRefining, PC_OT_RefineSequence
from ..operators.refresh_geometry import PC_OT_RefreshGeometry
from ..operators.scene_operations import PC_OT_CenterGeometry, PC_OT_ConvertAnimation
from ..operators.tracker_management import (
    PC_OT_CreateTracker, PC_OT_DeleteTracker, PC_OT_SelectTracker)
from ..operators.tracking import PC_OT_CancelTracking, PC_OT_TrackSequence
from ..properties import PolychaseData


class PC_PT_PolychasePanel(bpy.types.Panel):
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
            col.label(text="No trackers are created yet.", icon="INFO")
        else:
            for idx, tracker in enumerate(trackers):
                is_active_tracker = (idx == state.active_tracker_idx)

                row = col.row(align=True)

                op = row.operator(
                    PC_OT_SelectTracker.bl_idname,
                    text=tracker.name,
                    depress=is_active_tracker,
                    icon="CAMERA_DATA"
                    if tracker.tracking_target == "CAMERA" else "MESH_DATA")

                assert hasattr(op, "idx")
                setattr(op, "idx", idx)

                op = row.operator(
                    PC_OT_DeleteTracker.bl_idname, text="", icon="X")
                assert hasattr(op, "idx")
                setattr(op, "idx", idx)

        row = layout.row(align=True)
        row.operator(PC_OT_CreateTracker.bl_idname, icon="ADD")


# Base class for panels that require an active tracker
class PC_PT_PolychaseActiveTrackerBase(bpy.types.Panel):
    bl_category = "Polychase"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        state = PolychaseData.from_context(context)
        # Check if state exists and tracker is active
        return state is not None and state.is_tracking_active()


class PC_PT_TrackerInputsPanel(PC_PT_PolychaseActiveTrackerBase):
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

        split = layout.split(factor=0.3, align=True)
        col1 = split.column(align=False)
        col2 = split.column(align=False)

        col1.label(text="Name:")
        col2.prop(tracker, "name", text="")

        col1.label(text="Clip:")
        col2_row = col2.row(align=True)
        col2_row.alert = not tracker.clip
        col2_row.prop(tracker, "clip", text="")
        col2_row.operator(PC_OT_OpenClip.bl_idname, text="", icon="FILEBROWSER")

        col1.label(text="Geometry:")
        col2_row = col2.row(align=True)
        col2_row.alert = not tracker.geometry
        col2_row.prop(tracker, "geometry", text="")
        col2_row.operator(
            PC_OT_RefreshGeometry.bl_idname, text="", icon="FILE_REFRESH")

        col1.label(text="Camera:")
        col2_row = col2.row(align=True)
        col2_row.alert = not tracker.camera
        col2_row.prop(tracker, "camera", text="")

        col1.label(text="Target:")
        col2.prop(tracker, "tracking_target", text="")


class PC_PT_TrackerPinModePanel(PC_PT_PolychaseActiveTrackerBase):
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
        col.prop(
            tracker,
            "pinmode_optimize_focal_length",
            text="Estimate Focal Length")
        col.prop(
            tracker,
            "pinmode_optimize_principal_point",
            text="Estimate Principal Point")

        col = layout.column(align=True)
        col.operator(PC_OT_PinMode.bl_idname, depress=state.in_pinmode)
        col.operator(PC_OT_CenterGeometry.bl_idname)
        col.operator(PC_OT_ConvertAnimation.bl_idname)


class PC_PT_TrackerTrackingPanel(PC_PT_PolychaseActiveTrackerBase):
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

        # Show Track or Cancel button and progress based on state
        if tracker.is_tracking:
            row = layout.row()
            row.progress(
                factor=tracker.tracking_progress, text=tracker.tracking_message)
            row = layout.row(align=True)
            row.operator(PC_OT_CancelTracking.bl_idname, text="Cancel")
        elif tracker.is_refining:
            row = layout.row()
            row.progress(
                factor=tracker.refining_progress, text=tracker.refining_message)
            row = layout.row(align=True)
            row.operator(PC_OT_CancelRefining.bl_idname, text="Cancel")
        else:
            # Create a row for the tracking buttons
            row = layout.row(align=True)
            split = row.split(factor=0.5, align=True)

            col_left = split.column(align=True)
            col_right = split.column(align=True)

            split_left = col_left.split(factor=0.5, align=True)
            split_right = col_right.split(factor=0.5, align=True)

            # Backwards single
            op = split_left.operator(
                PC_OT_TrackSequence.bl_idname,
                text="",
                icon="TRACKING_BACKWARDS_SINGLE")
            op_casted = typing.cast(PC_OT_TrackSequence, op)
            op_casted.direction = "BACKWARD"
            op_casted.single_frame = True

            # Backwards all the way
            op = split_left.operator(
                PC_OT_TrackSequence.bl_idname,
                text="",
                icon="TRACKING_BACKWARDS")
            op_casted = typing.cast(PC_OT_TrackSequence, op)
            op_casted.direction = "BACKWARD"
            op_casted.single_frame = False

            # Forwards all the way
            op = split_right.operator(
                PC_OT_TrackSequence.bl_idname,
                text="",
                icon="TRACKING_FORWARDS")
            op_casted = typing.cast(PC_OT_TrackSequence, op)
            op_casted.direction = "FORWARD"
            op_casted.single_frame = False

            # Forwards single
            col = split_right.column(align=True)
            op = col.operator(
                PC_OT_TrackSequence.bl_idname,
                text="",
                icon="TRACKING_FORWARDS_SINGLE")
            op_casted = typing.cast(PC_OT_TrackSequence, op)
            op_casted.direction = "FORWARD"
            op_casted.single_frame = True

            # Refine
            op = col_left.operator(
                PC_OT_RefineSequence.bl_idname, text="Refine")
            op_casted = typing.cast(PC_OT_RefineSequence, op)
            op_casted.refine_all_segments = False

            # Refine all
            op = col_right.operator(
                PC_OT_RefineSequence.bl_idname, text="Refine All")
            op_casted = typing.cast(PC_OT_RefineSequence, op)
            op_casted.refine_all_segments = True

            # Create a row for the keyframe buttons
            row = layout.row(align=True)
            split = row.split(factor=0.25, align=True)

            col1 = split.column(align=True)
            col2 = split.column(align=True)
            col3 = split.column(align=True)
            col4 = split.column(align=True)

            col1.operator(
                PC_OT_PrevKeyFrame.bl_idname, text="", icon="PREV_KEYFRAME")
            col2.operator(
                PC_OT_NextKeyFrame.bl_idname, text="", icon="NEXT_KEYFRAME")
            col3.operator(PC_OT_AddKeyFrame.bl_idname, text="", icon="KEY_HLT")
            col4.operator(
                PC_OT_RemoveKeyFrame.bl_idname, text="", icon="KEY_DEHLT")

            col1.operator(
                PC_OT_KeyFrameClearBackwards.bl_idname,
                text="",
                icon="TRACKING_CLEAR_BACKWARDS")
            col2.operator(PC_OT_KeyFrameClearSegment.bl_idname, text="|-X-|")
            col3.operator(
                PC_OT_KeyFrameClearForwards.bl_idname,
                text="",
                icon="TRACKING_CLEAR_FORWARDS")
            col4.operator(PC_OT_ClearKeyFrames.bl_idname, text="", icon="X")


class PC_PT_TrackerOpticalFlowPanel(PC_PT_PolychaseActiveTrackerBase):
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
            row = layout.row()
            row.progress(
                factor=tracker.preprocessing_progress,
                text=tracker.preprocessing_message,
                type="BAR")
            row = layout.row(align=True)
            row.operator(PC_OT_CancelAnalysis.bl_idname, text="Cancel")
        else:
            row = layout.row(align=True)
            row.operator(PC_OT_AnalyzeVideo.bl_idname)


class PC_PT_TrackerAppearancePanel(PC_PT_PolychaseActiveTrackerBase):
    bl_label = "Appearance"
    bl_category = "Polychase"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_options = {"DEFAULT_CLOSED"}

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
        row.label(text="Wireframe:")

        row = layout.row(align=True)
        row.prop(tracker, "wireframe_color", text="")
        row.prop(tracker, "wireframe_width", text="Width")

        row = layout.row(align=True)
        split = row.split(factor=0.5, align=True)
        col = split.column(align=True)
        col.label(text="Default Pin Color")
        col = split.column(align=True)
        col.label(text="Selected Pin Color")

        row = layout.row(align=True)
        split = row.split(factor=0.5, align=True)
        col = split.column(align=True)
        col.prop(tracker, "default_pin_color", text="")
        col = split.column(align=True)
        col.prop(tracker, "selected_pin_color", text="")

        row = layout.row(align=True)
        row.prop(tracker, "pin_radius", text="Pin Radius")


class PC_PT_TrackerCameraPanel(PC_PT_PolychaseActiveTrackerBase):
    bl_label = "Camera"
    bl_category = "Polychase"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_options = {"DEFAULT_CLOSED"}

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        state = PolychaseData.from_context(context)
        if not state:
            return False
        tracker = state.active_tracker
        if not tracker or not tracker.camera:
            return False

        # Not needed, but ok
        return super().poll(context)

    def draw(self, context: bpy.types.Context):
        state = PolychaseData.from_context(context)
        if not state:
            return

        tracker = state.active_tracker
        if not tracker:
            return

        camera = tracker.camera
        if not camera or not isinstance(camera.data, bpy.types.Camera):
            return

        layout = self.layout
        assert layout

        # Sensor Size
        col = layout.column(align=True)
        col.label(text="Sensor:")
        row = col.row(align=True)

        if camera.data.sensor_fit == "VERTICAL":
            row.prop(camera.data, "sensor_height", text="Height")
        else:
            row.prop(camera.data, "sensor_width", text="Width")
        row.prop(camera.data, "sensor_fit", text="")

        # Focal Length
        col = layout.column(align=True)
        col.label(text="Focal Length:")
        row = col.row(align=True)
        if camera.data.lens_unit == "FOV":
            row.prop(camera.data, "angle", text="")
        else:
            row.prop(camera.data, "lens", text="")

        row.prop(camera.data, "lens_unit", text="")

        # Principal point
        col = layout.column(align=True)
        col.label(text="Principal Point:")
        row = col.row(align=True)
        row.prop(camera.data, "shift_x", text="X")
        row.prop(camera.data, "shift_y", text="Y")

        row = layout.row()
        row.prop(tracker, "variable_focal_length")

        row = layout.row()
        row.prop(tracker, "variable_principal_point")

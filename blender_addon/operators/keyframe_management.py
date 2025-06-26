# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2025 Ahmed Essam <aessam.dahy@gmail.com>

import bpy
import bpy.types

from .. import keyframes, utils
from ..properties import PolychaseState


class PC_OT_PrevKeyFrame(bpy.types.Operator):
    bl_idname = "polychase.prev_keyframe"
    bl_label = "Previous Keyframe"
    bl_description = "Go to previous keyframe"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context: bpy.types.Context) -> set:
        state = PolychaseState.from_context(context)
        if not state:
            self.report({"ERROR"}, "No polychase data found")
            return {"CANCELLED"}

        tracker = state.active_tracker
        if not tracker:
            self.report({"ERROR"}, "No active tracker found")
            return {"CANCELLED"}

        target_object = tracker.get_target_object()
        if not target_object:
            self.report({"ERROR"}, "No target object found")
            return {"CANCELLED"}

        if not target_object.animation_data or not target_object.animation_data.action:
            self.report({"INFO"}, "No animation data found for target object")
            return {"CANCELLED"}

        assert context.scene
        current_frame = context.scene.frame_current

        prev_keyframe = keyframes.find_prev_keyframe(
            target_object, current_frame, "location")

        if prev_keyframe is not None:
            frame = int(prev_keyframe.co[0])
            context.scene.frame_set(frame)
        else:
            self.report({"INFO"}, "No previous keyframe found")

        return {"FINISHED"}


class PC_OT_NextKeyFrame(bpy.types.Operator):
    bl_idname = "polychase.next_keyframe"
    bl_label = "Next Keyframe"
    bl_description = "Go to next keyframe"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context: bpy.types.Context) -> set:
        state = PolychaseState.from_context(context)
        if not state:
            self.report({"ERROR"}, "No polychase data found")
            return {"CANCELLED"}

        tracker = state.active_tracker
        if not tracker:
            self.report({"ERROR"}, "No active tracker found")
            return {"CANCELLED"}

        target_object = tracker.get_target_object()
        if not target_object:
            self.report({"ERROR"}, "No target object found")
            return {"CANCELLED"}

        if not target_object.animation_data or not target_object.animation_data.action:
            self.report({"INFO"}, "No animation data found for target object")
            return {"CANCELLED"}

        assert context.scene
        current_frame = context.scene.frame_current
        next_keyframe = keyframes.find_next_keyframe(
            target_object, current_frame, "location")

        if not next_keyframe:
            next_keyframe = keyframes.find_last_keyframe(target_object, "location", "GENERATED")
            if next_keyframe and next_keyframe.co[0] <= current_frame:
                next_keyframe = None

        if next_keyframe is not None:
            frame = int(next_keyframe.co[0])
            context.scene.frame_set(frame)
        else:
            self.report({"INFO"}, "No next keyframe found")

        return {"FINISHED"}


class PC_OT_KeyFrameClearBackwards(bpy.types.Operator):
    bl_idname = "polychase.keyframe_clear_backwards"
    bl_label = "Clear Keyframes Backwards"
    bl_description = "Clear all keyframes before the current frame"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context: bpy.types.Context) -> set:
        state = PolychaseState.from_context(context)
        if not state:
            self.report({"ERROR"}, "No polychase data found")
            return {"CANCELLED"}

        tracker = state.active_tracker
        if not tracker:
            self.report({"ERROR"}, "No active tracker found")
            return {"CANCELLED"}

        target_object = tracker.get_target_object()
        if not target_object:
            self.report({"ERROR"}, "No target object found")
            return {"CANCELLED"}

        if not tracker.clip:
            self.report({"ERROR"}, "No clip found")
            return {"CANCELLED"}

        assert context.scene
        current_frame = context.scene.frame_current
        clip = tracker.clip

        keyframes.clear_keyframes_in_range(
            obj=target_object,
            frame_start=clip.frame_start,
            frame_end_inclusive=current_frame - 1,
        )

        if tracker.camera:
            assert isinstance(tracker.camera.data, bpy.types.Camera)
            keyframes.clear_keyframes_in_range(
                obj=tracker.camera.data,
                frame_start=clip.frame_start,
                frame_end_inclusive=current_frame - 1,
                data_paths=keyframes.CAMERA_DATAPATHS,
            )

        # FCurve graph editor doesn"t get redrawn for some reason, even after
        # using tag_redraw, but resetting the current frame works.
        context.scene.frame_set(current_frame)

        return {"FINISHED"}


class PC_OT_KeyFrameClearForwards(bpy.types.Operator):
    bl_idname = "polychase.keyframe_clear_forwards"
    bl_label = "Clear Keyframes Backwards"
    bl_description = "Clear all keyframes before the current frame"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context: bpy.types.Context) -> set:
        state = PolychaseState.from_context(context)
        if not state:
            self.report({"ERROR"}, "No polychase data found")
            return {"CANCELLED"}

        tracker = state.active_tracker
        if not tracker:
            self.report({"ERROR"}, "No active tracker found")
            return {"CANCELLED"}

        if not tracker.clip:
            self.report({"ERROR"}, "No clip found")
            return {"CANCELLED"}

        assert context.scene
        current_frame = context.scene.frame_current
        clip = tracker.clip

        frame_end = clip.frame_start + clip.frame_duration - 1

        if tracker.geometry:
            keyframes.clear_keyframes_in_range(
                obj=tracker.geometry,
                frame_start=current_frame + 1,
                frame_end_inclusive=frame_end,
            )
        if tracker.camera:
            keyframes.clear_keyframes_in_range(
                obj=tracker.camera,
                frame_start=current_frame + 1,
                frame_end_inclusive=frame_end,
            )

        # FCurve graph editor doesn"t get redrawn for some reason, even after
        # using tag_redraw, but resetting the current frame works.
        context.scene.frame_set(current_frame)

        return {"FINISHED"}


class PC_OT_KeyFrameClearSegment(bpy.types.Operator):
    bl_idname = "polychase.keyframe_clear_segment"
    bl_label = "Clear Animation Segment"
    bl_description = "Clear tracked animation segment"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context: bpy.types.Context) -> set:
        state = PolychaseState.from_context(context)
        if not state:
            self.report({"ERROR"}, "No polychase data found")
            return {"CANCELLED"}

        tracker = state.active_tracker
        if not tracker:
            self.report({"ERROR"}, "No active tracker found")
            return {"CANCELLED"}

        if not tracker.clip:
            self.report({"ERROR"}, "No clip found")
            return {"CANCELLED"}

        target_object = tracker.get_target_object()
        if not target_object:
            self.report({"ERROR"}, "No target object found")
            return {"CANCELLED"}

        assert context.scene
        current_frame = context.scene.frame_current
        clip = tracker.clip
        frame_start = clip.frame_start
        frame_end = clip.frame_start + clip.frame_duration - 1

        prev = keyframes.find_prev_keyframe(
            obj=target_object,
            frame=current_frame + 1,
            data_path="location",
        )
        next = keyframes.find_next_keyframe(
            obj=target_object,
            frame=current_frame,
            data_path="location",
        )
        left_boundary = int(prev.co[0]) + 1 if prev else frame_start
        right_boundary = int(next.co[0]) - 1 if next else frame_end

        keyframes.clear_keyframes_in_range(
            obj=target_object,
            frame_start=left_boundary,
            frame_end_inclusive=right_boundary,
        )

        if tracker.camera:
            assert isinstance(tracker.camera.data, bpy.types.Camera)
            keyframes.clear_keyframes_in_range(
                obj=tracker.camera.data,
                frame_start=left_boundary,
                frame_end_inclusive=right_boundary,
                data_paths=keyframes.CAMERA_DATAPATHS,
            )

        # FCurve graph editor doesn"t get redrawn for some reason, even after
        # using tag_redraw, but resetting the current frame works.
        context.scene.frame_set(context.scene.frame_current)

        return {"FINISHED"}


class PC_OT_AddKeyFrame(bpy.types.Operator):
    bl_idname = "polychase.add_keyframe"
    bl_label = "Add Keyframe"
    bl_description = "Add keyframe for the target object"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context: bpy.types.Context) -> set:
        state = PolychaseState.from_context(context)
        if not state:
            self.report({"ERROR"}, "No polychase data found")
            return {"CANCELLED"}

        tracker = state.active_tracker
        if not tracker:
            self.report({"ERROR"}, "No active tracker found")
            return {"CANCELLED"}

        target_object = tracker.get_target_object()
        if not target_object:
            self.report({"ERROR"}, "No target object found")
            return {"CANCELLED"}

        assert context.scene
        current_frame = context.scene.frame_current

        rotation_data_path = utils.get_rotation_data_path(target_object)

        keyframes.insert_keyframe(
            obj=target_object,
            frame=current_frame,
            data_paths=["location", rotation_data_path],
            keytype="KEYFRAME",
        )

        if tracker.camera:
            assert isinstance(tracker.camera.data, bpy.types.Camera)

            if tracker.variable_focal_length:
                keyframes.insert_keyframe(
                    obj=tracker.camera.data,
                    frame=current_frame,
                    data_paths=["lens"],
                    keytype="KEYFRAME",
                )

            if tracker.variable_principal_point:
                keyframes.insert_keyframe(
                    obj=tracker.camera.data,
                    frame=current_frame,
                    data_paths=["shift_x", "shift_y"],
                    keytype="KEYFRAME",
                )

        return {"FINISHED"}


class PC_OT_RemoveKeyFrame(bpy.types.Operator):
    bl_idname = "polychase.remove_keyframe"
    bl_label = "Remove Keyframe"
    bl_description = "Remove keyframe at the current frame"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context: bpy.types.Context) -> set:
        state = PolychaseState.from_context(context)
        if not state:
            self.report({"ERROR"}, "No polychase data found")
            return {"CANCELLED"}

        tracker = state.active_tracker
        if not tracker:
            self.report({"ERROR"}, "No active tracker found")
            return {"CANCELLED"}

        target_object = tracker.get_target_object()
        if not target_object:
            self.report({"ERROR"}, "No target object found")
            return {"CANCELLED"}

        assert context.scene
        current_frame = context.scene.frame_current

        keyframes.remove_keyframes_at_frame(
            obj=target_object, frame=current_frame)

        if tracker.camera:
            assert isinstance(tracker.camera.data, bpy.types.Camera)
            keyframes.remove_keyframes_at_frame(
                obj=tracker.camera.data,
                frame=current_frame,
                data_paths=keyframes.CAMERA_DATAPATHS)

        return {"FINISHED"}


class PC_OT_ClearKeyFrames(bpy.types.Operator):
    bl_idname = "polychase.clear_keyframes"
    bl_label = "Clear Keyframes"
    bl_description = "Clear keyframes from the target object"
    bl_options = {"REGISTER", "UNDO"}

    clear_tracked_only: bpy.props.BoolProperty(
        name="Clear tracked keyframes only",
        description=
        "Clear only non-KEYFRAME type keyframes (tracked/generated), preserve manual keyframes",
        default=True)

    def execute(self, context: bpy.types.Context) -> set:
        state = PolychaseState.from_context(context)
        if not state:
            self.report({"ERROR"}, "No polychase data found")
            return {"CANCELLED"}

        tracker = state.active_tracker
        if not tracker:
            self.report({"ERROR"}, "No active tracker found")
            return {"CANCELLED"}

        if not tracker.clip:
            self.report({"ERROR"}, "No clip found")
            return {"CANCELLED"}

        assert context.scene

        clip = tracker.clip
        frame_start = clip.frame_start
        frame_end = clip.frame_start + clip.frame_duration - 1

        def predicate(keyframe: bpy.types.Keyframe):
            frame = int(keyframe.co[0])
            return keyframe.type == "GENERATED" and frame_start <= frame <= frame_end

        for obj in [tracker.geometry, tracker.camera]:
            if not obj:
                continue

            if self.clear_tracked_only:
                keyframes.clear_keyframes(obj, predicate)

            else:
                keyframes.clear_keyframes_in_range(
                    obj=obj,
                    frame_start=frame_start,
                    frame_end_inclusive=frame_end,
                )

        if tracker.camera:
            assert isinstance(tracker.camera.data, bpy.types.Camera)

            if self.clear_tracked_only:
                keyframes.clear_keyframes(
                    obj=tracker.camera.data,
                    predicate=predicate,
                    data_paths=keyframes.CAMERA_DATAPATHS,
                )

            else:
                keyframes.clear_keyframes_in_range(
                    obj=tracker.camera.data,
                    frame_start=frame_start,
                    frame_end_inclusive=frame_end,
                )

        # FCurve graph editor doesn"t get redrawn for some reason, even after
        # using tag_redraw, but resetting the current frame works.
        context.scene.frame_set(context.scene.frame_current)
        return {"FINISHED"}

    def invoke(self, context: bpy.types.Context, event: bpy.types.Event) -> set:
        assert context.window_manager
        return context.window_manager.invoke_props_dialog(self)

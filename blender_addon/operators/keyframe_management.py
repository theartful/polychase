import bpy
import bpy.types
from ..properties import PolychaseData
from .. import utils


class PC_OT_PrevKeyFrame(bpy.types.Operator):
    bl_idname = "polychase.prev_keyframe"
    bl_label = "Previous Keyframe"
    bl_description = "Go to previous keyframe"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context: bpy.types.Context) -> set:
        state = PolychaseData.from_context(context)
        if not state:
            self.report({'ERROR'}, "No polychase data found")
            return {'CANCELLED'}

        tracker = state.active_tracker
        if not tracker:
            self.report({'ERROR'}, "No active tracker found")
            return {'CANCELLED'}

        target_object = tracker.get_target_object()
        if not target_object:
            self.report({'ERROR'}, "No target object found")
            return {'CANCELLED'}

        if not target_object.animation_data or not target_object.animation_data.action:
            self.report({'INFO'}, "No animation data found for target object")
            return {'CANCELLED'}

        assert context.scene
        current_frame = context.scene.frame_current
        prev_frame = None

        # Find previous keyframe from location fcurves
        for fcurve in target_object.animation_data.action.fcurves:
            if fcurve.data_path != "location":
                continue

            fcurve.keyframe_points.sort()
            # Doesn't work unless I remove in reverse order
            for keyframe in fcurve.keyframe_points:
                if keyframe.type == 'KEYFRAME':
                    frame = int(keyframe.co[0])
                    if frame < current_frame:
                        if prev_frame is None or frame > prev_frame:
                            prev_frame = frame
                    else:
                        break    # Since sorted, no more previous frames

        if prev_frame is not None:
            context.scene.frame_set(prev_frame)
        else:
            self.report({'INFO'}, "No previous keyframe found")

        return {'FINISHED'}


class PC_OT_NextKeyFrame(bpy.types.Operator):
    bl_idname = "polychase.next_keyframe"
    bl_label = "Next Keyframe"
    bl_description = "Go to next keyframe"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context: bpy.types.Context) -> set:
        state = PolychaseData.from_context(context)
        if not state:
            self.report({'ERROR'}, "No polychase data found")
            return {'CANCELLED'}

        tracker = state.active_tracker
        if not tracker:
            self.report({'ERROR'}, "No active tracker found")
            return {'CANCELLED'}

        target_object = tracker.get_target_object()
        if not target_object:
            self.report({'ERROR'}, "No target object found")
            return {'CANCELLED'}

        if not target_object.animation_data or not target_object.animation_data.action:
            self.report({'INFO'}, "No animation data found for target object")
            return {'CANCELLED'}

        assert context.scene
        current_frame = context.scene.frame_current
        next_frame = None

        # Find next keyframe from location fcurves
        for fcurve in target_object.animation_data.action.fcurves:
            if fcurve.data_path != "location":
                continue

            fcurve.keyframe_points.sort()
            for keyframe in fcurve.keyframe_points:
                if keyframe.type == 'KEYFRAME':
                    frame = int(keyframe.co[0])
                    if frame > current_frame:
                        next_frame = frame
                        break

        if next_frame is not None:
            context.scene.frame_set(next_frame)
        else:
            self.report({'INFO'}, "No next keyframe found")

        return {'FINISHED'}


class PC_OT_KeyFrameClearBackwards(bpy.types.Operator):
    bl_idname = "polychase.keyframe_clear_backwards"
    bl_label = "Clear Keyframes Backwards"
    bl_description = "Clear all keyframes before the current frame"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context: bpy.types.Context) -> set:
        state = PolychaseData.from_context(context)
        if not state:
            self.report({'ERROR'}, "No polychase data found")
            return {'CANCELLED'}

        tracker = state.active_tracker
        if not tracker:
            self.report({'ERROR'}, "No active tracker found")
            return {'CANCELLED'}

        target_object = tracker.get_target_object()
        if not target_object:
            self.report({'ERROR'}, "No target object found")
            return {'CANCELLED'}

        if not tracker.clip:
            self.report({'ERROR'}, "No clip found")
            return {'CANCELLED'}

        if not target_object.animation_data or not target_object.animation_data.action:
            self.report({'INFO'}, "No animation data found for target object")
            return {'CANCELLED'}

        assert context.scene
        current_frame = context.scene.frame_current
        clip = tracker.clip

        frame_start = clip.frame_start
        frame_end = clip.frame_start + clip.frame_duration - 1

        # Clear keyframes before current frame from location fcurves
        for fcurve in target_object.animation_data.action.fcurves:
            if fcurve.data_path not in ["location",
                                        "rotation_quaternion",
                                        "rotation_euler",
                                        "rotation_axis_angle",
                                        "lens",
                                        "shift_x",
                                        "shift_y"]:
                continue

            fcurve.keyframe_points.sort()

            for keyframe in reversed(fcurve.keyframe_points):
                frame = int(keyframe.co[0])
                if frame < current_frame and frame_start <= frame <= frame_end:
                    fcurve.keyframe_points.remove(keyframe)

        # FCurve graph editor doesn't get redrawn for some reason, even after
        # using tag_redraw, but resetting the current frame works.
        context.scene.frame_set(current_frame)

        return {'FINISHED'}


class PC_OT_KeyFrameClearForwards(bpy.types.Operator):
    bl_idname = "polychase.keyframe_clear_forwards"
    bl_label = "Clear Keyframes Backwards"
    bl_description = "Clear all keyframes before the current frame"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context: bpy.types.Context) -> set:
        state = PolychaseData.from_context(context)
        if not state:
            self.report({'ERROR'}, "No polychase data found")
            return {'CANCELLED'}

        tracker = state.active_tracker
        if not tracker:
            self.report({'ERROR'}, "No active tracker found")
            return {'CANCELLED'}

        target_object = tracker.get_target_object()
        if not target_object:
            self.report({'ERROR'}, "No target object found")
            return {'CANCELLED'}

        if not tracker.clip:
            self.report({'ERROR'}, "No clip found")
            return {'CANCELLED'}

        if not target_object.animation_data or not target_object.animation_data.action:
            self.report({'INFO'}, "No animation data found for target object")
            return {'CANCELLED'}

        assert context.scene
        current_frame = context.scene.frame_current
        clip = tracker.clip

        frame_start = clip.frame_start
        frame_end = clip.frame_start + clip.frame_duration - 1

        # Clear keyframes before current frame from location fcurves
        for fcurve in target_object.animation_data.action.fcurves:
            if fcurve.data_path not in ["location",
                                        "rotation_quaternion",
                                        "rotation_euler",
                                        "rotation_axis_angle",
                                        "lens",
                                        "shift_x",
                                        "shift_y"]:
                continue

            fcurve.keyframe_points.sort()

            for keyframe in reversed(fcurve.keyframe_points):
                frame = int(keyframe.co[0])
                if frame > current_frame and frame_start <= frame <= frame_end:
                    fcurve.keyframe_points.remove(keyframe)

        # FCurve graph editor doesn't get redrawn for some reason, even after
        # using tag_redraw, but resetting the current frame works.
        context.scene.frame_set(current_frame)

        return {'FINISHED'}


class PC_OT_KeyFrameClearSegment(bpy.types.Operator):
    bl_idname = "polychase.keyframe_clear_segment"
    bl_label = "Clear Animation Segment"
    bl_description = "Clear tracked animation segment"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context: bpy.types.Context) -> set:
        state = PolychaseData.from_context(context)
        if not state:
            self.report({'ERROR'}, "No polychase data found")
            return {'CANCELLED'}

        tracker = state.active_tracker
        if not tracker:
            self.report({'ERROR'}, "No active tracker found")
            return {'CANCELLED'}

        target_object = tracker.get_target_object()
        if not target_object:
            self.report({'ERROR'}, "No target object found")
            return {'CANCELLED'}

        if not tracker.clip:
            self.report({'ERROR'}, "No clip found")
            return {'CANCELLED'}

        if not target_object.animation_data or not target_object.animation_data.action:
            self.report({'INFO'}, "No animation data found for target object")
            return {'CANCELLED'}

        assert context.scene
        current_frame = context.scene.frame_current
        clip = tracker.clip
        frame_start = clip.frame_start
        frame_end = clip.frame_start + clip.frame_duration - 1

        # Clear non-keyframe types in segment from relevant fcurves
        for fcurve in target_object.animation_data.action.fcurves:
            if fcurve.data_path not in ["location",
                                        "rotation_quaternion",
                                        "rotation_euler",
                                        "rotation_axis_angle",
                                        "lens",
                                        "shift_x",
                                        "shift_y"]:
                continue

            fcurve.keyframe_points.sort()

            # Find backwards boundary
            backward_boundary = frame_start - 1
            for keyframe in reversed(fcurve.keyframe_points):
                frame = int(keyframe.co[0])
                if frame < current_frame and frame_start <= frame <= frame_end:
                    if keyframe.type == 'KEYFRAME':
                        backward_boundary = frame
                        break

            # Find forwards boundary
            forward_boundary = frame_end + 1
            for keyframe in fcurve.keyframe_points:
                frame = int(keyframe.co[0])
                if frame > current_frame and frame_start <= frame <= frame_end:
                    if keyframe.type == 'KEYFRAME':
                        forward_boundary = frame
                        break

            for keyframe in reversed(fcurve.keyframe_points):
                frame = int(keyframe.co[0])
                if backward_boundary < frame < forward_boundary:
                    assert keyframe.type != 'KEYFRAME'
                    fcurve.keyframe_points.remove(keyframe)

        # FCurve graph editor doesn't get redrawn for some reason, even after
        # using tag_redraw, but resetting the current frame works.
        context.scene.frame_set(current_frame)

        return {'FINISHED'}


class PC_OT_AddKeyFrame(bpy.types.Operator):
    bl_idname = "polychase.add_keyframe"
    bl_label = "Add Keyframe"
    bl_description = "Add keyframe for the target object"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context: bpy.types.Context) -> set:
        from .. import utils

        state = PolychaseData.from_context(context)
        if not state:
            self.report({'ERROR'}, "No polychase data found")
            return {'CANCELLED'}

        tracker = state.active_tracker
        if not tracker:
            self.report({'ERROR'}, "No active tracker found")
            return {'CANCELLED'}

        target_object = tracker.get_target_object()
        if not target_object:
            self.report({'ERROR'}, "No target object found")
            return {'CANCELLED'}

        camera = tracker.camera
        if not camera:
            self.report({'ERROR'}, "No camera found")
            return {'CANCELLED'}

        assert context.scene
        assert isinstance(camera.data, bpy.types.Camera)

        current_frame = context.scene.frame_current
        rotation_data_path = utils.get_rotation_data_path(target_object)

        # Delete existing keyframes to ensure they become KEYFRAME type
        try:
            target_object.keyframe_delete(
                data_path="location", frame=current_frame)
        except:
            pass
        try:
            target_object.keyframe_delete(
                data_path=rotation_data_path, frame=current_frame)
        except:
            pass

        # Insert keyframes
        target_object.keyframe_insert(
            data_path="location", frame=current_frame, keytype="KEYFRAME")
        target_object.keyframe_insert(
            data_path=rotation_data_path,
            frame=current_frame,
            keytype="KEYFRAME")

        # Handle focal length optimization
        if tracker.pinmode_optimize_focal_length:
            try:
                camera.data.keyframe_delete(
                    data_path="lens", frame=current_frame)
            except:
                pass
            camera.data.keyframe_insert(
                data_path="lens", frame=current_frame, keytype="KEYFRAME")

        # Handle principal point optimization
        if tracker.pinmode_optimize_principal_point:
            try:
                camera.data.keyframe_delete(
                    data_path="shift_x", frame=current_frame)
            except:
                pass
            try:
                camera.data.keyframe_delete(
                    data_path="shift_y", frame=current_frame)
            except:
                pass
            camera.data.keyframe_insert(
                data_path="shift_x", frame=current_frame, keytype="KEYFRAME")
            camera.data.keyframe_insert(
                data_path="shift_y", frame=current_frame, keytype="KEYFRAME")

        return {'FINISHED'}


class PC_OT_RemoveKeyFrame(bpy.types.Operator):
    bl_idname = "polychase.remove_keyframe"
    bl_label = "Remove Keyframe"
    bl_description = "Remove keyframe for the target object"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context: bpy.types.Context) -> set:
        state = PolychaseData.from_context(context)
        if not state:
            self.report({'ERROR'}, "No polychase data found")
            return {'CANCELLED'}

        tracker = state.active_tracker
        if not tracker:
            self.report({'ERROR'}, "No active tracker found")
            return {'CANCELLED'}

        target_object = tracker.get_target_object()
        if not target_object:
            self.report({'ERROR'}, "No target object found")
            return {'CANCELLED'}

        camera = tracker.camera
        if not camera:
            self.report({'ERROR'}, "No camera found")
            return {'CANCELLED'}

        assert context.scene
        assert isinstance(camera.data, bpy.types.Camera)

        current_frame = context.scene.frame_current
        rotation_data_path = utils.get_rotation_data_path(target_object)

        # Remove keyframes for target object
        try:
            target_object.keyframe_delete(
                data_path="location", frame=current_frame)
        except:
            pass
        try:
            target_object.keyframe_delete(
                data_path=rotation_data_path, frame=current_frame)
        except:
            pass

        # Remove focal length keyframe if it exists
        try:
            camera.data.keyframe_delete(data_path="lens", frame=current_frame)
        except:
            pass

        # Remove principal point keyframes if they exist
        try:
            camera.data.keyframe_delete(
                data_path="shift_x", frame=current_frame)
        except:
            pass
        try:
            camera.data.keyframe_delete(
                data_path="shift_y", frame=current_frame)
        except:
            pass

        return {'FINISHED'}


class PC_OT_ClearKeyFrames(bpy.types.Operator):
    bl_idname = "polychase.clear_keyframes"
    bl_label = "Clear Keyframes"
    bl_description = "Clear keyframes from the target object"
    bl_options = {'REGISTER', 'UNDO'}

    clear_tracked_only: bpy.props.BoolProperty(
        name="Clear tracked keyframes only",
        description=
        "Clear only non-KEYFRAME type keyframes (tracked/generated), preserve manual keyframes",
        default=True)

    def execute(self, context: bpy.types.Context) -> set:
        state = PolychaseData.from_context(context)
        if not state:
            self.report({'ERROR'}, "No polychase data found")
            return {'CANCELLED'}

        tracker = state.active_tracker
        if not tracker:
            self.report({'ERROR'}, "No active tracker found")
            return {'CANCELLED'}

        target_object = tracker.get_target_object()
        if not target_object:
            self.report({'ERROR'}, "No target object found")
            return {'CANCELLED'}

        camera = tracker.camera
        if not camera:
            self.report({'ERROR'}, "No camera found")
            return {'CANCELLED'}

        if not tracker.clip:
            self.report({'ERROR'}, "No clip found")
            return {'CANCELLED'}

        if not target_object.animation_data or not target_object.animation_data.action:
            self.report({'INFO'}, "No animation data found for target object")
            return {'CANCELLED'}

        assert context.scene
        assert isinstance(camera.data, bpy.types.Camera)

        clip = tracker.clip
        frame_start = clip.frame_start
        frame_end = clip.frame_start + clip.frame_duration - 1

        # Clear keyframes from relevant fcurves
        for fcurve in target_object.animation_data.action.fcurves:
            if fcurve.data_path not in ["location",
                                        "rotation_quaternion",
                                        "rotation_euler",
                                        "rotation_axis_angle"]:
                continue

            fcurve.keyframe_points.sort()

            for keyframe in reversed(fcurve.keyframe_points):
                frame = int(keyframe.co[0])
                if frame_start <= frame <= frame_end:
                    should_remove = False

                    if self.clear_tracked_only:
                        # Only remove non-KEYFRAME types
                        if keyframe.type != 'KEYFRAME':
                            should_remove = True
                    else:
                        # Remove all keyframes
                        should_remove = True

                    if should_remove:
                        fcurve.keyframe_points.remove(keyframe)

        # Also clear camera keyframes if they exist
        if camera.data.animation_data and camera.data.animation_data.action:
            for fcurve in camera.data.animation_data.action.fcurves:
                if fcurve.data_path not in ["lens", "shift_x", "shift_y"]:
                    continue

                fcurve.keyframe_points.sort()

                for keyframe in reversed(fcurve.keyframe_points):
                    frame = int(keyframe.co[0])
                    if frame_start <= frame <= frame_end:
                        should_remove = False

                        if self.clear_tracked_only:
                            # Only remove non-KEYFRAME types
                            if keyframe.type != 'KEYFRAME':
                                should_remove = True
                        else:
                            # Remove all keyframes
                            should_remove = True

                        if should_remove:
                            fcurve.keyframe_points.remove(keyframe)

        # FCurve graph editor doesn't get redrawn for some reason, even after
        # using tag_redraw, but resetting the current frame works.
        context.scene.frame_set(context.scene.frame_current)
        return {'FINISHED'}

    def invoke(self, context: bpy.types.Context, event: bpy.types.Event) -> set:
        assert context.window_manager
        return context.window_manager.invoke_props_dialog(self)

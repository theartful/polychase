import math
import os
import typing

import bpy
import bpy.types
import mathutils
import numpy as np

from .. import core
from ..properties import PolychaseClipTracking, PolychaseData


class OT_RefineSequence(bpy.types.Operator):
    bl_idname = "polychase.refine_sequence"
    bl_options = {"REGISTER", "UNDO"}
    bl_label = "Refine Sequence"

    @classmethod
    def poll(cls, context):
        state = PolychaseData.from_context(context)
        if not state:
            return False

        tracker = state.active_tracker
        return (
            tracker is not None and tracker.clip is not None and tracker.camera is not None
            and tracker.geometry is not None and tracker.database_path != "")

    def execute(self, context: bpy.types.Context) -> set:
        assert context.scene

        scene = context.scene

        state = PolychaseData.from_context(context)
        if not state:
            return {"CANCELLED"}    # Should not happen due to poll

        tracker = state.active_tracker
        if not tracker:
            return {"CANCELLED"}    # Should not happen due to poll

        # Check if already tracking
        if tracker.is_tracking:
            self.report({'WARNING'}, "Tracking is already in progress.")
            return {"CANCELLED"}

        database_path = bpy.path.abspath(tracker.database_path)
        if not database_path:
            self.report({"ERROR"}, "Database path is not set.")
            return {"CANCELLED"}

        if not os.path.isfile(database_path):
            self.report({"ERROR"}, "Analyze the video first.")
            return {"CANCELLED"}

        clip = tracker.clip
        if not clip:
            self.report({"ERROR"}, "Clip is not set.")
            return {"CANCELLED"}

        camera = tracker.camera
        if not camera or not isinstance(camera.data, bpy.types.Camera):
            self.report({"ERROR"}, "Camera is not set.")
            return {"CANCELLED"}

        geometry = tracker.geometry
        if not geometry:
            self.report({"ERROR"}, "Geometry is not set.")
            return {"CANCELLED"}

        self._trans_type = core.TransformationType.Model if tracker.tracking_target == "GEOMETRY" else core.TransformationType.Camera
        self._tracker_id = tracker.id

        target_object = tracker.get_target_object()
        assert target_object

        # Get segment boundaries
        clip_start_frame = clip.frame_start
        clip_end_frame = clip.frame_start + clip.frame_duration - 1

        if scene.frame_current < clip_start_frame or scene.frame_current > clip_end_frame:
            self.report({'ERROR'}, "Current frame is outside the range of the clip.")
            return {"CANCELLED"}

        # We want to find the largest frame_from that is less than or equal to the current frame
        # And to find the smallest frame_to that is greatest than the current frame
        frame_current = scene.frame_current
        frame_from = None
        frame_to = None

        frame_from_keyframe_set = False
        frame_to_keyframe_set = False

        anim_data = target_object.animation_data

        if not anim_data or not anim_data.action:
            self.report({"ERROR"}, "Target object has no animation data to refine")
            return {"CANCELLED"}

        for fcurve in anim_data.action.fcurves:
            if fcurve.data_path not in {"location", "rotation_quaternion"}:
                continue

            fcurve.keyframe_points.sort()
            for kf in fcurve.keyframe_points:
                frame = kf.co[0]

                if frame <= frame_current:
                    if kf.type == "KEYFRAME":
                        frame_from = max(frame, frame_from) if frame_from else frame
                        frame_from_keyframe_set = True
                    elif not frame_from_keyframe_set:
                        frame_from = min(frame, frame_from) if frame_from else frame

                if frame > frame_current:
                    if kf.type == "KEYFRAME":
                        frame_to = min(frame, frame_to) if frame_to else frame
                        frame_to_keyframe_set = True
                        break
                    elif not frame_to_keyframe_set:
                        frame_to = max(frame, frame_to) if frame_to else frame

            break

        if not frame_from or not frame_to:
            self.report({"ERROR"}, "Could not detect the segment to refine")
            return {"CANCELLED"}

        frame_from = int(frame_from)
        frame_to = int(frame_to)
        num_frames = frame_to - frame_from + 1

        # Create camera trajectory
        camera_traj = core.CameraTrajectory(first_frame_id=frame_from, count=num_frames)
        pose_obj = core.CameraPose()
        cam_state_obj = core.CameraState()

        depsgraph = context.evaluated_depsgraph_get()

        geom_rot_mode = geometry.rotation_mode
        cam_rot_mode = camera.rotation_mode

        # Set rotation mode to quaternion for convenience
        geometry.rotation_mode = "QUATERNION"
        camera.rotation_mode = "QUATERNION"

        for frame in range(frame_from, frame_to + 1):
            scene.frame_set(frame)

            geometry_evaled = geometry.evaluated_get(depsgraph)
            camera_evaled = camera.evaluated_get(depsgraph)
            assert isinstance(camera_evaled.data, bpy.types.Camera)

            model_quat = geometry_evaled.rotation_quaternion
            model_t = geometry_evaled.location

            view_quat = camera_evaled.rotation_quaternion.inverted()
            view_t = -(view_quat @ camera_evaled.location)

            model_view_quat = view_quat @ model_quat
            model_view_loc = view_t + view_quat @ model_t

            pose_obj.q = typing.cast(np.ndarray, model_view_quat)
            pose_obj.t = typing.cast(np.ndarray, model_view_loc)

            intrins_obj = core.camera_intrinsics_expanded(
                lens=camera_evaled.data.lens,
                shift_x=camera_evaled.data.shift_x,
                shift_y=camera_evaled.data.shift_y,
                sensor_width=camera_evaled.data.sensor_width,
                sensor_height=camera_evaled.data.sensor_height,
                sensor_fit=camera_evaled.data.sensor_fit,
                width=clip.size[0],
                height=clip.size[1],
            )

            cam_state_obj.intrinsics = intrins_obj
            cam_state_obj.pose = pose_obj
            camera_traj.set(frame, cam_state_obj)

        # Restore original rotation mode
        geometry.rotation_mode = geom_rot_mode
        camera.rotation_mode = cam_rot_mode

        # Restore scene frame
        scene.frame_set(frame_current)

        return {"CANCELLED"}

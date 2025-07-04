# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2025 Ahmed Essam <aessam.dahy@gmail.com>

import os
import typing

import bpy
import bpy.types
import mathutils

from .. import core, utils, keyframes
from ..properties import PolychaseTracker, PolychaseState


class PC_OT_TrackSequence(bpy.types.Operator):
    bl_idname = "polychase.track_sequence"
    bl_options = {"REGISTER", "UNDO"}
    bl_label = "Track Sequence"

    direction: bpy.props.EnumProperty(
        name="Direction",
        items=[
            ("FORWARD", "Forward", "Track forwards from the current frame"),
            ("BACKWARD", "Backward", "Track backwards from the current frame"),
        ],
        default="FORWARD",
    )
    single_frame: bpy.props.BoolProperty(name="Single Frame", default=False)

    _cpp_thread: core.TrackerThread | None = None
    _timer: bpy.types.Timer | None = None
    _tracker_id: int = -1
    _frame_from: int = -1
    _frame_to_inclusive: int = -1

    @classmethod
    def poll(cls, context):
        state = PolychaseState.from_context(context)
        if not state:
            return False

        tracker = state.active_tracker
        return (
            tracker is not None and tracker.clip is not None
            and tracker.camera is not None and tracker.geometry is not None
            and tracker.database_path != "")

    def execute(self, context: bpy.types.Context) -> set:
        assert context.window_manager
        assert context.scene

        scene = context.scene

        transient = PolychaseState.get_transient_state()
        state = PolychaseState.from_context(context)
        if not state:
            return {"CANCELLED"}    # Should not happen due to poll

        tracker = state.active_tracker
        if not tracker:
            return {"CANCELLED"}    # Should not happen due to poll

        # Check if already tracking
        if transient.is_tracking:
            self.report({"WARNING"}, "Tracking is already in progress.")
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
        if not camera:
            self.report({"ERROR"}, "Camera is not set.")
            return {"CANCELLED"}

        geometry = tracker.geometry
        if not geometry:
            self.report({"ERROR"}, "Geometry is not set.")
            return {"CANCELLED"}

        width, height = clip.size
        self._trans_type = core.TransformationType.Model if tracker.tracking_target == "GEOMETRY" else core.TransformationType.Camera
        self._tracker_id = tracker.id

        target_object = tracker.get_target_object()
        assert target_object

        # Determine the actual start and end frames for tracking
        clip_start_frame = clip.frame_start
        clip_end_frame = clip.frame_start + clip.frame_duration - 1
        self._frame_from = scene.frame_current
        boundary_keyframe = self._find_boundary_keyframe(
            target_object=target_object,
            current_frame=scene.frame_current,
            start_frame=clip_start_frame,
            end_frame=clip_end_frame,
            direction=self.direction,
        )

        # Sanity checks
        if scene.frame_current < clip_start_frame or scene.frame_current > clip_end_frame:
            self.report(
                {"ERROR"}, "Current frame is outside the range of the clip.")
            return {"CANCELLED"}

        if self.direction == "FORWARD" and self._frame_from == clip_end_frame:
            self.report(
                {"ERROR"}, "Current frame is already at the edge of the clip.")
            return {"CANCELLED"}

        if self.direction == "BACKWARD" and self._frame_from == clip_start_frame:
            self.report(
                {"ERROR"}, "Current frame is already at the edge of the clip.")
            return {"CANCELLED"}

        if self.direction == "FORWARD":
            if self.single_frame:
                self._frame_to_inclusive = self._frame_from + 1
            else:
                self._frame_to_inclusive = boundary_keyframe
        else:
            if self.single_frame:
                self._frame_to_inclusive = self._frame_from - 1
            else:
                self._frame_to_inclusive = boundary_keyframe

        num_frames = abs(self._frame_to_inclusive - self._frame_from) + 1
        if num_frames <= 1:
            self.report({"INFO"}, "Already at or past the next keyframe.")
            return {"CANCELLED"}

        # Ensure a keyframe exists at the starting frame
        self._ensure_keyframe_at_start(tracker, scene.frame_current)

        self._cpp_thread = self._create_cpp_thread(
            tracker=tracker,
            database_path=database_path,
            frame_from=self._frame_from,
            frame_to_inclusive=self._frame_to_inclusive,
            width=width,
            height=height,
            camera=camera,
        )

        if self.single_frame:
            # FIXME: YUCK? Maybe don't create the thread in the first place.
            self._cpp_thread.join()

            # FIXME: YUCK? Maybe refactor modal so that we don't have to call it?
            result = self.modal(context, None)
            assert result != {"PASS_THROUGH"}
            return result

        else:
            transient.is_tracking = True
            transient.should_stop_tracking = False
            transient.tracking_progress = 0.0
            transient.tracking_message = "Starting..."

            context.window_manager.modal_handler_add(self)
            self._timer = context.window_manager.event_timer_add(
                0.1, window=context.window)
            context.window_manager.progress_begin(0.0, 1.0)
            context.window_manager.progress_update(0)

            return {"RUNNING_MODAL"}

    def _create_cpp_thread(
        self,
        tracker: PolychaseTracker,
        database_path: str,
        frame_from: int,
        frame_to_inclusive: int,
        width: int,
        height: int,
        camera: bpy.types.Object,
    ) -> core.TrackerThread:

        tracker_core = core.Tracker.get(tracker)

        assert tracker.geometry
        assert tracker_core

        model_matrix = tracker.geometry.matrix_world
        view_matrix = utils.get_camera_view_matrix(camera)

        scale_matrix = mathutils.Matrix.Diagonal(
            model_matrix.to_scale().to_4d())
        model_view_matrix = view_matrix @ model_matrix

        loc, rot, _ = model_view_matrix.decompose()
        model_view_matrix_no_scale = mathutils.Matrix.LocRotScale(
            loc, rot, None)

        accel_mesh = tracker_core.accel_mesh

        bundle_opts = core.BundleOptions()
        bundle_opts.loss_type = core.LossType.Cauchy
        bundle_opts.loss_scale = 1.0

        return core.TrackerThread(
            database_path=database_path,
            frame_from=frame_from,
            frame_to_inclusive=frame_to_inclusive,
            scene_transform=core.SceneTransformations(
                model_matrix=typing.cast(core.np.ndarray, scale_matrix),
                view_matrix=typing.cast(
                    core.np.ndarray, model_view_matrix_no_scale),
                intrinsics=core.camera_intrinsics(camera, width, height),
            ),
            accel_mesh=accel_mesh,
            bundle_opts=bundle_opts,
            optimize_focal_length=tracker.variable_focal_length,
            optimize_principal_point=tracker.variable_principal_point,
        )

    def modal(
            self, context: bpy.types.Context,
            event: bpy.types.Event | None) -> set:
        try:
            return self._modal_impl(context, event)
        except Exception as e:
            return self._cleanup(context, False, str(e))

    def _modal_impl(
        self,
        context: bpy.types.Context,
        event: bpy.types.Event | None,
    ) -> set:
        assert context.window_manager
        assert self._cpp_thread
        assert context.scene

        state = PolychaseState.from_context(context)
        transient = PolychaseState.get_transient_state()
        tracker = state.active_tracker if state else None

        if not tracker or tracker.id != self._tracker_id:
            return self._cleanup(
                context,
                success=False,
                message="Tracker was either switched or deleted")
        if not tracker.geometry or not tracker.clip or not tracker.camera:
            return self._cleanup(
                context, success=False, message="Tracking input changed")
        if transient.should_stop_tracking:
            return self._cleanup(
                context, success=False, message="Cancelled by user")
        if event is not None and event.type in {"ESC"}:
            return self._cleanup(
                context, success=False, message="Cancelled by user (ESC)")

        assert isinstance(tracker.camera.data, bpy.types.Camera)

        while not self._cpp_thread.empty():
            message = self._cpp_thread.try_pop()
            assert message

            if isinstance(message, bool):
                return self._cleanup(
                    context,
                    success=True,
                    message="Tracking finished successfully")

            elif isinstance(message, core.CppException):
                return self._cleanup(
                    context, success=False, message=message.what())

            assert isinstance(message, core.FrameTrackingResult)

            num_frames = abs(self._frame_to_inclusive - self._frame_from) + 1
            frame_id = message.frame
            frames_processed = abs(frame_id - self._frame_from) + 1

            if message.inlier_ratio < 0.25:
                error_message = f"Error: Could not predict pose at frame #{frame_id} from optical flow data due to low inlier ratio ({message.inlier_ratio*100:.02f}%)"
                return self._cleanup(
                    context, success=False, message=error_message)

            progress = frames_processed / num_frames
            progress_message = f"Tracking frame {frame_id} ({self.direction.lower()})"

            transient.tracking_progress = progress
            transient.tracking_message = progress_message
            context.window_manager.progress_update(progress)

            pose = message.pose

            context.scene.frame_set(frame_id)
            geometry = tracker.geometry
            camera = tracker.camera

            depsgraph = context.evaluated_depsgraph_get()
            evaluated_geometry = geometry.evaluated_get(depsgraph)
            evaluated_camera = camera.evaluated_get(depsgraph)

            Rmv = mathutils.Quaternion(
                typing.cast(typing.Sequence[float], pose.q))
            tmv = mathutils.Vector(typing.cast(typing.Sequence[float], pose.t))

            # If tracking geometry
            if self._trans_type == core.TransformationType.Model:
                tv, Rv = utils.get_camera_view_matrix_loc_rot(evaluated_camera)
                Rv_inv = Rv.inverted()

                Rm = Rv_inv @ Rmv
                tm = Rv_inv @ (tmv - tv)
                utils.set_object_model_matrix(geometry, tm, Rm)

                keyframes.insert_keyframe(
                    obj=geometry,
                    frame=frame_id,
                    data_paths=[
                        "location", utils.get_rotation_data_path(geometry)
                    ],
                    keytype="GENERATED",
                )
            else:
                tm, Rm, _ = utils.get_object_model_matrix_loc_rot_scale(evaluated_geometry)
                Rm_inv = Rm.inverted()

                Rv = Rmv @ Rm_inv
                tv = tmv - Rv @ tm

                utils.set_camera_view_matrix(camera, tv, Rv)

                keyframes.insert_keyframe(
                    obj=camera,
                    frame=frame_id,
                    data_paths=[
                        "location", utils.get_rotation_data_path(camera)
                    ],
                    keytype="GENERATED",
                )

            if tracker.variable_focal_length or tracker.variable_principal_point:
                core.set_camera_intrinsics(tracker.camera, message.intrinsics)

                keyframes.insert_keyframe(
                    obj=tracker.camera.data,
                    frame=frame_id,
                    data_paths=keyframes.CAMERA_DATAPATHS,
                    keytype="GENERATED",
                )

        return {"PASS_THROUGH"}

    def _cleanup(self, context: bpy.types.Context, success: bool, message: str):
        assert context.window_manager

        if self._timer:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None

        if self._cpp_thread:
            self._cpp_thread.request_stop()
            self._cpp_thread.join()

        self._cpp_thread = None
        self._tracker_id = -1

        transient = PolychaseState.get_transient_state()
        transient.is_tracking = False
        transient.should_stop_tracking = False
        transient.tracking_message = ""

        tracker = PolychaseState.get_tracker_by_id(self._tracker_id, context)
        if tracker:
            tracker.store_geom_cam_transform()

        context.window_manager.progress_end()
        if context.area:
            context.area.tag_redraw()

        if success:
            self.report({"INFO"}, message)
            return {"FINISHED"}
        else:
            self.report(
                {"WARNING"} if message.startswith("Cancelled") else {"ERROR"},
                message)
            # Return finished even though we failed, so that undoing works.
            return {"FINISHED"}

    def _find_boundary_keyframe(
        self,
        target_object: bpy.types.Object,
        current_frame: int,
        start_frame: int,
        end_frame: int,
        direction: str,
    ) -> int:
        if direction == "FORWARD":
            next_keyframe = keyframes.find_next_keyframe(
                obj=target_object, frame=current_frame, data_path="location")
            return int(next_keyframe.co[0]) - 1 if next_keyframe else end_frame

        else:
            assert direction == "BACKWARD"
            prev_keyframe = keyframes.find_prev_keyframe(
                obj=target_object, frame=current_frame, data_path="location")

            return int(
                prev_keyframe.co[0]) + 1 if prev_keyframe else start_frame

    def _ensure_keyframe_at_start(
        self,
        tracker: PolychaseTracker,
        frame: int,
    ):
        target_object = tracker.get_target_object()
        assert target_object
        assert tracker.camera

        # Handle camera intrinsics keyframes
        if tracker.variable_focal_length or tracker.variable_principal_point:
            existing_keyframes = keyframes.get_keyframes(
                obj=target_object, frame=frame, data_paths=["lens"])

            if not existing_keyframes:
                assert isinstance(tracker.camera.data, bpy.types.Camera)

                keyframes.insert_keyframe(
                    obj=tracker.camera.data,
                    frame=frame,
                    data_paths=keyframes.CAMERA_DATAPATHS,
                    keytype="KEYFRAME",
                )

        # Check if keyframe already exists at this frame
        existing_keyframes = keyframes.get_keyframes(
            obj=target_object, frame=frame, data_paths=["location"])

        if not existing_keyframes:
            keyframes.insert_keyframe(
                obj=target_object,
                frame=frame,
                data_paths=[
                    "location", utils.get_rotation_data_path(target_object)
                ],
                keytype="KEYFRAME",
            )


class PC_OT_CancelTracking(bpy.types.Operator):
    bl_idname = "polychase.cancel_tracking"
    bl_label = "Cancel Tracking"
    bl_options = {"INTERNAL"}

    @classmethod
    def poll(cls, context):
        state = PolychaseState.from_context(context)
        transient = PolychaseState.get_transient_state()
        return state is not None and state.active_tracker is not None and transient.is_tracking

    def execute(self, context) -> set:
        transient = PolychaseState.get_transient_state()
        if transient.is_tracking:
            transient.should_stop_tracking = True
        else:
            # Defensive check
            self.report({"WARNING"}, "Tracking is not running.")

        return {"FINISHED"}

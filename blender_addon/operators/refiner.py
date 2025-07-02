# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2025 Ahmed Essam <aessam.dahy@gmail.com>

import os
import typing

import bpy
import bpy.types
import mathutils
import numpy as np

from .. import core, utils, keyframes
from ..properties import PolychaseTracker, PolychaseState


def refine_sequence_lazy(
    tracker: PolychaseTracker,
    database_path: str,
    camera_traj: core.CameraTrajectory,
    optimize_focal_length: bool,
    optimize_principal_point: bool,
):
    tracker_core = core.Tracker.get(tracker)
    geometry = tracker.geometry

    assert geometry
    assert tracker_core

    accel_mesh = tracker_core.accel_mesh
    model_matrix = mathutils.Matrix.Diagonal(
        geometry.matrix_world.to_scale().to_4d())

    bundle_opts = core.BundleOptions()
    bundle_opts.loss_type = core.LossType.Cauchy
    bundle_opts.loss_scale = 1.0

    def inner(callback: typing.Callable[[core.RefineTrajectoryUpdate], bool]):
        return core.refine_trajectory(
            database_path=database_path,
            camera_trajectory=camera_traj,
            model_matrix=typing.cast(np.ndarray, model_matrix),
            mesh=accel_mesh,
            optimize_focal_length=optimize_focal_length,
            optimize_principal_point=optimize_principal_point,
            callback=callback,
            bundle_opts=bundle_opts,
        )

    return inner


class PC_OT_RefineSequence(bpy.types.Operator):
    bl_idname = "polychase.refine_sequence"
    bl_options = {"REGISTER", "UNDO"}
    bl_label = "Refine Sequence"

    refine_all_segments: bpy.props.BoolProperty(
        name="Refine All Segments",
        description=
        "Refine all animation segments instead of just the current one",
        default=False)

    _camera_traj: core.CameraTrajectory | None = None
    _cpp_thread: core.RefinerThread | None = None
    _timer: bpy.types.Timer | None = None
    _tracker_id: int = -1
    _optimize_focal_length: bool = False
    _optimize_principal_point: bool = False

    _segments: list[tuple[int, int]] = []
    _current_segment_index: int = 0
    _initial_scene_frame: int = 0

    _database_path: str = ""
    _clip_name: str = ""
    _clip_size: tuple[int, int] = (0, 0)
    _clip_start_frame: int = 0
    _clip_end_frame: int = 0
    _camera_name: str = ""
    _geometry_name: str = ""
    _target_object_name: str = ""
    _trans_type: core.TransformationType | None = None

    @classmethod
    def poll(cls, context):
        state = PolychaseState.from_context(context)
        if not state:
            return False

        tracker = state.active_tracker
        return tracker is not None and tracker.clip is not None and \
               tracker.camera is not None and tracker.geometry is not None and \
               tracker.database_path != ""

    def _get_current_segment(
            self,
            scene: bpy.types.Scene,
            target_object: bpy.types.Object,
            clip: bpy.types.MovieClip) -> list[tuple[int, int]]:
        clip_start_frame = clip.frame_start
        clip_end_frame = clip.frame_start + clip.frame_duration - 1

        if scene.frame_current < clip_start_frame or scene.frame_current > clip_end_frame:
            return []

        frame_current = scene.frame_current

        if not target_object.animation_data or not target_object.animation_data.action:
            return []

        prev = keyframes.find_prev_keyframe(
            obj=target_object,
            frame=frame_current,
            data_path="location",
        )
        cur = keyframes.get_keyframe(
            obj=target_object,
            frame=frame_current,
            data_path="location",
        )
        next = keyframes.find_next_keyframe(
            obj=target_object,
            frame=frame_current,
            data_path="location",
        )

        # Determine frame boundaries
        frame_from = int(prev.co[0]) if prev else clip_start_frame
        frame_to = int(next.co[0]) if next else clip_end_frame

        if not cur or cur.type == "GENERATED":
            return [(frame_from, frame_to)]
        else:
            return [(frame_from, frame_current), (frame_current, frame_to)]

    def _collect_all_segments(
        self,
        target_object: bpy.types.Object,
        clip: bpy.types.MovieClip,
    ) -> list[tuple[int, int]]:
        clip_start_frame = clip.frame_start
        clip_end_frame = clip.frame_start + clip.frame_duration - 1

        if not target_object.animation_data or not target_object.animation_data.action:
            return []

        all_keyframes = keyframes.collect_keyframes_of_type(
            obj=target_object,
            keyframe_type="KEYFRAME",
            data_path="location",
            frame_start=clip_start_frame,
            frame_end_inclusive=clip_end_frame,
        )

        if not all_keyframes:
            return []

        # Create segments between consecutive keyframes
        segments = []

        # Add segment from clip start to first keyframe if needed
        if all_keyframes[0] > clip_start_frame:
            segments.append((clip_start_frame, all_keyframes[0]))

        # Add segments between consecutive keyframes
        for i in range(len(all_keyframes) - 1):
            segments.append((all_keyframes[i], all_keyframes[i + 1]))

        # Add segment from last keyframe to clip end if needed
        if all_keyframes[-1] < clip_end_frame:
            segments.append((all_keyframes[-1], clip_end_frame))

        return segments

    def _setup_current_segment_and_worker(
            self, context: bpy.types.Context) -> bool:
        assert context.scene
        assert self._segments
        assert self._current_segment_index < len(self._segments)
        assert self._camera_name
        assert self._geometry_name
        assert self._clip_name

        # Validate objects still exist
        if self._camera_name not in bpy.data.objects:
            return False
        if self._geometry_name not in bpy.data.objects:
            return False
        if self._clip_name not in bpy.data.movieclips:
            return False

        camera = bpy.data.objects[self._camera_name]
        geometry = bpy.data.objects[self._geometry_name]

        depsgraph = context.evaluated_depsgraph_get()
        evaluated_geometry = geometry.evaluated_get(depsgraph)
        evaluated_camera = camera.evaluated_get(depsgraph)

        if not isinstance(camera.data, bpy.types.Camera):
            return False

        segment = self._segments[self._current_segment_index]
        frame_from, frame_to = segment
        num_frames = frame_to - frame_from + 1

        # Create camera trajectory for this segment
        self._camera_traj = core.CameraTrajectory(
            first_frame_id=frame_from, count=num_frames)
        pose_obj = core.CameraPose()
        cam_state_obj = core.CameraState()

        for frame in range(frame_from, frame_to + 1):
            context.scene.frame_set(frame)

            tm, Rm, _ = utils.get_object_model_matrix_loc_rot_scale(evaluated_geometry)
            tv, Rv = utils.get_camera_view_matrix_loc_rot(evaluated_camera)

            Rmv = Rv @ Rm
            tmv = tv + Rv @ tm

            pose_obj.q = typing.cast(np.ndarray, Rmv)
            pose_obj.t = typing.cast(np.ndarray, tmv)

            intrins_obj = core.camera_intrinsics_expanded(
                lens=camera.data.lens,
                shift_x=camera.data.shift_x,
                shift_y=camera.data.shift_y,
                sensor_width=camera.data.sensor_width,
                sensor_height=camera.data.sensor_height,
                sensor_fit=camera.data.sensor_fit,
                width=self._clip_size[0],
                height=self._clip_size[1],
            )

            cam_state_obj.intrinsics = intrins_obj
            cam_state_obj.pose = pose_obj
            self._camera_traj.set(frame, cam_state_obj)

        # Set current frame to the middle of the segment to indicate what segment we're working on.
        if self.refine_all_segments:
            context.scene.frame_set((frame_from + frame_to) // 2)
        else:
            context.scene.frame_set(self._initial_scene_frame)

        # Get tracker for creating lazy function
        state = PolychaseState.from_context(context)
        tracker = state.active_tracker if state else None
        if not tracker or tracker.id != self._tracker_id:
            return False

        # Start worker thread for this segment
        if self._cpp_thread:
            self._cpp_thread.join()

        self._cpp_thread = self._create_refiner_thread(
            tracker=tracker,
            database_path=self._database_path,
            camera_traj=self._camera_traj,
            optimize_focal_length=self._optimize_focal_length,
            optimize_principal_point=self._optimize_principal_point,
        )
        return True

    def _start_next_segment_or_finish(self, context: bpy.types.Context) -> set:
        """Start processing the next segment or finish if all segments are done."""
        self._current_segment_index += 1

        if self._current_segment_index >= len(self._segments):
            # All segments processed
            return self._cleanup(
                context,
                success=True,
                message=f"Refined {len(self._segments)} segment(s) successfully"
            )

        # Start next segment
        if not self._setup_current_segment_and_worker(context):
            return self._cleanup(
                context, success=False, message="Failed to setup next segment")

        return {"PASS_THROUGH"}

    def execute(self, context: bpy.types.Context) -> set:
        assert context.scene
        assert context.window_manager

        scene = context.scene
        self._initial_scene_frame = scene.frame_current

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

        # Store essential data and object identifiers
        self._database_path = database_path
        self._clip_name = clip.name
        self._clip_size = typing.cast(tuple[int, int], clip.size)
        self._clip_start_frame = clip.frame_start
        self._clip_end_frame = clip.frame_start + clip.frame_duration - 1
        self._camera_name = camera.name
        self._geometry_name = geometry.name
        self._target_object_name = target_object.name

        # Collect segments based on user choice
        if self.refine_all_segments:
            self._segments = self._collect_all_segments(target_object, clip)
        else:
            self._segments = self._get_current_segment(
                scene, target_object, clip)

        # Keep segments that have more than 2 frames
        self._segments = list(
            filter(lambda s: s[1] - s[0] + 1 > 2, self._segments))

        if len(self._segments) == 0:
            self.report({"ERROR"}, "Could not detect the segment to refine")
            return {"CANCELLED"}

        self._current_segment_index = 0

        # Store optimization settings
        self._optimize_focal_length = tracker.variable_focal_length
        self._optimize_principal_point = tracker.variable_principal_point

        # Setup first segment and worker
        if not self._setup_current_segment_and_worker(context):
            self.report({"ERROR"}, "Failed to setup first segment")
            return {"CANCELLED"}

        # Setup modal operation
        context.window_manager.modal_handler_add(self)
        self._timer = context.window_manager.event_timer_add(
            0.1, window=context.window)
        context.window_manager.progress_begin(0.0, 1.0)
        context.window_manager.progress_update(0)

        transient.is_refining = True
        transient.should_stop_refining = False
        transient.refining_progress = 0.0
        transient.refining_message = f"Starting segment 1 of {len(self._segments)}..."

        return {"RUNNING_MODAL"}

    def _create_refiner_thread(
        self,
        tracker: PolychaseTracker,
        database_path: str,
        camera_traj: core.CameraTrajectory,
        optimize_focal_length: bool,
        optimize_principal_point: bool,
    ) -> core.RefinerThread:
        tracker_core = core.Tracker.get(tracker)
        geometry = tracker.geometry

        assert geometry
        assert tracker_core

        accel_mesh = tracker_core.accel_mesh
        model_matrix = mathutils.Matrix.Diagonal(
            geometry.matrix_world.to_scale().to_4d())

        bundle_opts = core.BundleOptions()
        bundle_opts.loss_type = core.LossType.Cauchy
        bundle_opts.loss_scale = 1.0

        return core.RefinerThread(
            database_path=database_path,
            camera_trajectory=camera_traj,
            model_matrix=typing.cast(np.ndarray, model_matrix),
            mesh=accel_mesh,
            optimize_focal_length=optimize_focal_length,
            optimize_principal_point=optimize_principal_point,
            bundle_opts=bundle_opts,
        )

    def _apply_camera_traj(
            self, context: bpy.types.Context, tracker: PolychaseTracker):
        assert context.scene
        assert self._camera_traj

        geometry = tracker.geometry
        camera = tracker.camera

        assert geometry
        assert camera
        assert isinstance(camera.data, bpy.types.Camera)
        assert geometry.name == self._geometry_name
        assert camera.name == self._camera_name

        # Validate object names match what we expect (objects haven't been replaced)
        if geometry.name != self._geometry_name or camera.name != self._camera_name:
            return

        is_tracking_geometry = self._trans_type == core.TransformationType.Model

        # Assuming poses at frame_from and frame_to are constants.
        frame_current = context.scene.frame_current

        segment = self._segments[self._current_segment_index]
        frame_from, frame_to = segment

        depsgraph = context.evaluated_depsgraph_get()
        evaluated_geometry = geometry.evaluated_get(depsgraph)
        evaluated_camera = camera.evaluated_get(depsgraph)

        # Exclude first and last frames
        for frame in range(frame_from + 1, frame_to):
            cam_state = self._camera_traj.get(frame)
            assert cam_state

            context.scene.frame_set(frame)

            Rmv = mathutils.Quaternion(
                typing.cast(typing.Sequence[float], cam_state.pose.q))
            tmv = mathutils.Vector(
                typing.cast(typing.Sequence[float], cam_state.pose.t))

            if is_tracking_geometry:
                tv, Rv = utils.get_camera_view_matrix_loc_rot(evaluated_camera)
                Rv_inv = Rv.inverted()

                Rm = Rv_inv @ Rmv
                tm = Rv_inv @ (tmv - tv)
                utils.set_object_model_matrix(geometry, tm, Rm)

                keyframes.insert_keyframe(
                    obj=geometry,
                    frame=frame,
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
                    frame=frame,
                    data_paths=[
                        "location", utils.get_rotation_data_path(camera)
                    ],
                    keytype="GENERATED",
                )

            if self._optimize_focal_length or self._optimize_principal_point:
                core.set_camera_intrinsics(camera, cam_state.intrinsics)
                keyframes.insert_keyframe(
                    obj=camera.data,
                    frame=frame,
                    data_paths=keyframes.CAMERA_DATAPATHS,
                    keytype="GENERATED",
                )

        # Restore scene frame
        context.scene.frame_set(frame_current)

    def modal(self, context: bpy.types.Context, event: bpy.types.Event) -> set:
        try:
            return self._modal_impl(context, event)
        except Exception as e:
            return self._cleanup(context, False, str(e))

    def _modal_impl(
            self, context: bpy.types.Context, event: bpy.types.Event) -> set:
        assert context.window_manager
        assert self._cpp_thread
        assert context.scene
        assert context.area

        transient = PolychaseState.get_transient_state()
        state = PolychaseState.from_context(context)
        tracker = state.active_tracker if state else None
        if not tracker or tracker.id != self._tracker_id:
            return self._cleanup(
                context,
                success=False,
                message="Tracker was switched or deleted")

        # Validate that tracker objects still exist and match our stored names
        if not tracker.geometry or not tracker.clip or not tracker.camera:
            return self._cleanup(
                context, success=False, message="Tracking input changed")
        if (tracker.geometry.name != self._geometry_name
                or tracker.camera.name != self._camera_name
                or tracker.clip.name != self._clip_name):
            return self._cleanup(
                context, success=False, message="Tracking objects changed")

        if transient.should_stop_refining:
            return self._cleanup(
                context, success=False, message="Cancelled by user")
        if event is not None and event.type in {"ESC"}:
            return self._cleanup(
                context, success=False, message="Cancelled by user (ESC)")

        work_finished = False
        while not self._cpp_thread.empty():
            message = self._cpp_thread.try_pop()
            assert message

            if isinstance(message, bool):
                work_finished = True
                break

            elif isinstance(message, core.CppException):
                return self._cleanup(
                    context, success=False, message=message.what())

            assert isinstance(message, core.RefineTrajectoryUpdate)

            # Calculate overall progress across all segments
            segment_progress = message.progress
            overall_progress = (self._current_segment_index
                                + segment_progress) / len(self._segments)

            transient.refining_progress = overall_progress
            transient.refining_message = message.message
            context.area.tag_redraw()

        if work_finished:
            # Apply camera trajectory for the completed segment
            self._apply_camera_traj(context, tracker)

            # Clean up worker thread for this segment
            self._cpp_thread.request_stop()
            self._cpp_thread.join()
            self._cpp_thread = None

            # Start next segment or finish
            return self._start_next_segment_or_finish(context)

        return {"PASS_THROUGH"}

    def _cleanup(self, context: bpy.types.Context, success: bool, message: str):
        assert context.window_manager
        assert context.scene

        if self._timer:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None

        if self._cpp_thread:
            self._cpp_thread.request_stop()
            self._cpp_thread.join()

        # Restore scene frame
        context.scene.frame_set(self._initial_scene_frame)

        transient = PolychaseState.get_transient_state()
        transient.is_refining = False
        transient.should_stop_refining = False
        transient.refining_message = ""

        tracker = PolychaseState.get_tracker_by_id(self._tracker_id, context)
        if tracker:
            # Apply camera trajectory even if we failed, since we might have applied a couple of
            # optimization iterations for the current segment.
            if self._camera_traj and self._current_segment_index < len(
                    self._segments):
                self._apply_camera_traj(context, tracker)

            tracker.store_geom_cam_transform()

        # Reset segment tracking variables
        self._segments = []
        self._current_segment_index = 0
        self._database_path = ""
        self._clip_name = ""
        self._clip_size = (0, 0)
        self._clip_start_frame = 0
        self._clip_end_frame = 0
        self._camera_name = ""
        self._geometry_name = ""
        self._target_object_name = ""
        self._trans_type = None
        self._should_stop = None
        self._tracker_id = -1

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


class PC_OT_CancelRefining(bpy.types.Operator):
    bl_idname = "polychase.cancel_refining"
    bl_label = "Cancel Refining"
    bl_options = {"INTERNAL"}

    @classmethod
    def poll(cls, context):
        transient = PolychaseState.get_transient_state()
        state = PolychaseState.from_context(context)
        return state is not None and state.active_tracker is not None and transient.is_refining

    def execute(self, context) -> set:
        transient = PolychaseState.get_transient_state()
        if transient.is_refining:
            transient.should_stop_refining = True
        else:
            # Defensive check
            self.report({"WARNING"}, "Refining is not running.")

        return {"FINISHED"}

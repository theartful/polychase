import os
import queue
import threading
import traceback
import typing

import bpy
import bpy.types
import mathutils
import numpy as np

from .. import core, utils
from ..properties import PolychaseClipTracking, PolychaseData

WorkerMessage = core.RefineTrajectoryUpdate | Exception | None


def refine_sequence_lazy(
    tracker: PolychaseClipTracking,
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
        [geometry.scale[0], geometry.scale[1], geometry.scale[2], 1.0])

    bundle_opts = core.BundleOptions()
    bundle_opts.loss_type = core.LossType.Huber
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
    _worker_thread: threading.Thread | None = None
    _from_worker_queue: queue.Queue[WorkerMessage] | None = None
    _timer: bpy.types.Timer | None = None
    _tracker_id: int = -1
    _should_stop: threading.Event | None = None
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
        state = PolychaseData.from_context(context)
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
            clip: bpy.types.MovieClip) -> tuple[int, int] | None:
        """Get the animation segment around the current frame."""
        clip_start_frame = clip.frame_start
        clip_end_frame = clip.frame_start + clip.frame_duration - 1

        if scene.frame_current < clip_start_frame or scene.frame_current > clip_end_frame:
            return None

        frame_current = scene.frame_current
        frame_from: float | None = None
        frame_to: float | None = None

        frame_from_keyframe_set = False

        anim_data = target_object.animation_data
        if not anim_data or not anim_data.action:
            return None

        for fcurve in anim_data.action.fcurves:
            if fcurve.data_path != "location":
                continue

            fcurve.keyframe_points.sort()
            for kf in fcurve.keyframe_points:
                frame = kf.co[0]

                if frame <= frame_current:
                    if kf.type == "KEYFRAME":
                        frame_from = max(
                            frame, frame_from) if frame_from else frame
                        frame_from_keyframe_set = True
                    elif not frame_from_keyframe_set:
                        frame_from = min(
                            frame, frame_from) if frame_from else frame

                if frame > frame_current:
                    if kf.type == "KEYFRAME":
                        frame_to = frame
                        break
                    else:
                        frame_to = max(frame, frame_to) if frame_to else frame

            break

        if not frame_from or not frame_to:
            return None

        return (int(frame_from), int(frame_to))

    def _collect_all_segments(
        self,
        target_object: bpy.types.Object,
        clip: bpy.types.MovieClip,
    ) -> list[tuple[int, int]]:
        clip_start_frame = clip.frame_start
        clip_end_frame = clip.frame_start + clip.frame_duration - 1

        anim_data = target_object.animation_data
        if not anim_data or not anim_data.action:
            return []

        # Collect all KEYFRAME keyframes within clip range
        keyframes = set()
        for fcurve in anim_data.action.fcurves:
            if fcurve.data_path != "location":
                continue

            fcurve.keyframe_points.sort()
            for kf in fcurve.keyframe_points:
                frame = int(kf.co[0])
                if clip_start_frame <= frame <= clip_end_frame and kf.type == "KEYFRAME":
                    keyframes.add(frame)

        # Sort keyframes
        sorted_keyframes = sorted(keyframes)

        # If no keyframes found, treat entire clip as one segment
        if not sorted_keyframes:
            return [(clip_start_frame, clip_end_frame)]

        # Create segments between consecutive keyframes
        segments = []

        # Add segment from clip start to first keyframe if needed
        if sorted_keyframes[0] > clip_start_frame:
            segments.append((clip_start_frame, sorted_keyframes[0]))

        # Add segments between consecutive keyframes
        for i in range(len(sorted_keyframes) - 1):
            segments.append((sorted_keyframes[i], sorted_keyframes[i + 1]))

        # Add segment from last keyframe to clip end if needed
        if sorted_keyframes[-1] < clip_end_frame:
            segments.append((sorted_keyframes[-1], clip_end_frame))

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

        depsgraph = context.evaluated_depsgraph_get()

        for frame in range(frame_from, frame_to + 1):
            context.scene.frame_set(frame)

            geometry_evaled = geometry.evaluated_get(depsgraph)
            camera_evaled = camera.evaluated_get(depsgraph)
            assert isinstance(camera_evaled.data, bpy.types.Camera)

            model_quat = utils.get_rotation_quat(geometry_evaled)
            model_t = geometry_evaled.location

            view_quat = utils.get_rotation_quat(camera_evaled).inverted()
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
        tracker = PolychaseData.get_tracker_by_id(self._tracker_id, context)
        if not tracker:
            return False

        # Start worker thread for this segment
        lazy_func = refine_sequence_lazy(
            tracker=tracker,
            database_path=self._database_path,
            camera_traj=self._camera_traj,
            optimize_focal_length=self._optimize_focal_length,
            optimize_principal_point=self._optimize_principal_point,
        )

        self._from_worker_queue = queue.Queue()
        self._should_stop = threading.Event()

        self._worker_thread = threading.Thread(
            target=self._work,
            args=(
                lazy_func,
                self._from_worker_queue,
                self._should_stop,
            ),
            daemon=True,
        )

        self._worker_thread.start()
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

        state = PolychaseData.from_context(context)
        if not state:
            return {"CANCELLED"}    # Should not happen due to poll

        tracker = state.active_tracker
        if not tracker:
            return {"CANCELLED"}    # Should not happen due to poll

        # Check if already tracking
        if tracker.is_tracking:
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
            if not self._segments:
                self.report({"ERROR"}, "No animation segments found to refine")
                return {"CANCELLED"}
        else:
            # Get current segment
            current_segment = self._get_current_segment(
                scene, target_object, clip)
            if not current_segment:
                self.report({"ERROR"}, "Could not detect the segment to refine")
                return {"CANCELLED"}
            self._segments = [current_segment]

        self._current_segment_index = 0

        # Store optimization settings
        self._optimize_focal_length = tracker.variable_focal_length
        self._optimize_principal_point = tracker.variable_principal_point

        # Initialize tracker state
        tracker.is_refining = True
        tracker.should_stop_refining = False
        tracker.refining_progress = 0.0
        tracker.refining_message = f"Starting segment 1 of {len(self._segments)}..."

        # Setup first segment and worker
        if not self._setup_current_segment_and_worker(context):
            tracker.is_refining = False
            self.report({"ERROR"}, "Failed to setup first segment")
            return {"CANCELLED"}

        # Setup modal operation
        context.window_manager.modal_handler_add(self)
        self._timer = context.window_manager.event_timer_add(
            0.1, window=context.window)
        context.window_manager.progress_begin(0.0, 1.0)
        context.window_manager.progress_update(0)

        return {"RUNNING_MODAL"}

    def _work(
        self,
        fn: typing.Callable,
        from_worker_queue: queue.Queue[WorkerMessage],
        should_stop: threading.Event,
    ):

        def _callback(progress: core.RefineTrajectoryUpdate):
            # Send progress update
            print(progress.stats)
            from_worker_queue.put(progress)
            return not should_stop.is_set()

        try:
            success = fn(_callback)
            if not success and not should_stop.is_set():
                from_worker_queue.put(
                    RuntimeError(f"Refining failed unexpectedly"))

            from_worker_queue.put(None)    # Signal completion (even if stopped)

        except Exception as e:
            traceback.print_exc()
            from_worker_queue.put(e)

    def _apply_camera_traj(
            self, context: bpy.types.Context, tracker: PolychaseClipTracking):
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

        depsgraph = context.evaluated_depsgraph_get()

        segment = self._segments[self._current_segment_index]
        frame_from, frame_to = segment

        for frame in range(frame_from, frame_to + 1):
            cam_state = self._camera_traj.get(frame)
            assert cam_state

            context.scene.frame_set(frame)

            geometry_evaled = geometry.evaluated_get(depsgraph)
            camera_evaled = camera.evaluated_get(depsgraph)
            assert isinstance(camera_evaled.data, bpy.types.Camera)

            model_view_quat = mathutils.Quaternion(
                typing.cast(typing.Sequence[float], cam_state.pose.q))
            model_view_t = mathutils.Vector(
                typing.cast(typing.Sequence[float], cam_state.pose.t))

            if is_tracking_geometry:
                utils.set_rotation_quat(
                    geometry,
                    camera_evaled.rotation_quaternion @ model_view_quat)
                geometry.location = camera_evaled.rotation_quaternion @ model_view_t + camera_evaled.location

                geometry.keyframe_insert(
                    data_path="location", frame=frame, keytype="GENERATED")
                geometry.keyframe_insert(
                    data_path=utils.get_rotation_data_path(geometry),
                    frame=frame,
                    keytype="GENERATED")

            else:
                geom_quat_inv = utils.get_rotation_quat(
                    geometry_evaled).inverted()
                geom_loc_inv = -(geom_quat_inv @ geometry_evaled.location)

                view_quat = model_view_quat @ geom_quat_inv
                view_t = model_view_quat @ geom_loc_inv + model_view_t

                view_quat_inv = view_quat.inverted()

                camera.rotation_quaternion = view_quat_inv
                camera.location = -(view_quat_inv @ view_t)

                camera.keyframe_insert(
                    data_path="location", frame=frame, keytype="GENERATED")
                camera.keyframe_insert(
                    data_path=utils.get_rotation_data_path(camera),
                    frame=frame,
                    keytype="GENERATED")

            if self._optimize_focal_length or self._optimize_principal_point:
                core.set_camera_intrinsics(camera, cam_state.intrinsics)
                camera.data.keyframe_insert(
                    data_path="lens", frame=frame, keytype="GENERATED")
                camera.data.keyframe_insert(
                    data_path="shift_x", frame=frame, keytype="GENERATED")
                camera.data.keyframe_insert(
                    data_path="shift_y", frame=frame, keytype="GENERATED")

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
        assert self._from_worker_queue
        assert self._worker_thread
        assert context.scene
        assert context.area

        tracker = PolychaseData.get_tracker_by_id(self._tracker_id, context)
        if not tracker:
            return self._cleanup(
                context, success=False, message="Tracker was deleted")

        # Validate that tracker objects still exist and match our stored names
        if not tracker.geometry or not tracker.clip or not tracker.camera:
            return self._cleanup(
                context, success=False, message="Tracking input changed")
        if (tracker.geometry.name != self._geometry_name
                or tracker.camera.name != self._camera_name
                or tracker.clip.name != self._clip_name):
            return self._cleanup(
                context, success=False, message="Tracking objects changed")

        if tracker.should_stop_refining:
            return self._cleanup(
                context, success=False, message="Cancelled by user")
        if event is not None and event.type in {"ESC"}:
            return self._cleanup(
                context, success=False, message="Cancelled by user (ESC)")

        work_finished = False
        while not self._from_worker_queue.empty():
            try:
                message: WorkerMessage = self._from_worker_queue.get_nowait()

                if message is None:
                    work_finished = True
                    break

                elif isinstance(message, core.RefineTrajectoryUpdate):
                    # Calculate overall progress across all segments
                    segment_progress = message.progress
                    overall_progress = (
                        self._current_segment_index + segment_progress) / len(
                            self._segments)

                    tracker.refining_progress = overall_progress
                    tracker.refining_message = f"{message.message}"
                    context.area.tag_redraw()

                elif isinstance(message, Exception):
                    return self._cleanup(
                        context, success=False, message=f"Error: {message}")
                else:
                    # This should never happen
                    return self._cleanup(
                        context,
                        success=False,
                        message=f"Unknown message: {str(message)}")

            except queue.Empty:
                pass
            except Exception as e:
                traceback.print_exc()
                return self._cleanup(
                    context, success=False, message=f"Internal error: {e}")

        if work_finished:
            # Apply camera trajectory for the completed segment
            self._apply_camera_traj(context, tracker)

            # Clean up worker thread for this segment
            if self._worker_thread and self._worker_thread.is_alive():
                self._worker_thread.join()
            self._worker_thread = None
            self._from_worker_queue = None
            if self._should_stop:
                self._should_stop.set()
            self._should_stop = None

            # Start next segment or finish
            return self._start_next_segment_or_finish(context)

        if not self._worker_thread.is_alive():
            return self._cleanup(
                context,
                success=False,
                message="Refiner worker thread stopped unexpectedly")

        return {"PASS_THROUGH"}

    def _cleanup(self, context: bpy.types.Context, success: bool, message: str):
        assert context.window_manager
        assert context.scene

        if self._timer:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None

        if self._should_stop:
            self._should_stop.set()

        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join()

        # Restore scene frame
        context.scene.frame_set(self._initial_scene_frame)

        tracker = PolychaseData.get_tracker_by_id(self._tracker_id, context)
        if tracker:
            # Apply camera trajectory even if we failed, since we might have applied a couple of
            # optimization iterations for the current segment.
            if self._camera_traj and self._current_segment_index < len(
                    self._segments):
                self._apply_camera_traj(context, tracker)

            tracker.is_refining = False
            tracker.should_stop_refining = False    # Ensure it's reset
            tracker.refining_message = ""

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
        self._worker_thread = None
        self._from_worker_queue = None
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
        state = PolychaseData.from_context(context)
        return state is not None and state.active_tracker is not None and state.active_tracker.is_refining

    def execute(self, context) -> set:
        state = PolychaseData.from_context(context)
        if not state or not state.active_tracker:
            return {"CANCELLED"}

        tracker = state.active_tracker
        if tracker.is_refining:
            tracker.should_stop_refining = True
        else:
            # Defensive check
            self.report({"WARNING"}, "Refining is not running.")

        return {"FINISHED"}

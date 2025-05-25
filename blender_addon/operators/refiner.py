import os
import queue
import threading
import traceback
import typing

import bpy
import bpy.types
import mathutils
import numpy as np

from .. import core
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
    model_matrix = mathutils.Matrix.Diagonal([geometry.scale[0], geometry.scale[1], geometry.scale[2], 1.0])

    def inner(callback: typing.Callable[[core.RefineTrajectoryUpdate], bool]):
        return core.refine_trajectory(
            database_path=database_path,
            camera_trajectory=camera_traj,
            model_matrix=typing.cast(np.ndarray, model_matrix),
            mesh=accel_mesh,
            optimize_focal_length=optimize_focal_length,
            optimize_principal_point=optimize_principal_point,
            callback=callback,
        )

    return inner


class OT_RefineSequence(bpy.types.Operator):
    bl_idname = "polychase.refine_sequence"
    bl_options = {"REGISTER", "UNDO"}
    bl_label = "Refine Sequence"

    _camera_traj: core.CameraTrajectory | None = None
    _worker_thread: threading.Thread | None = None
    _from_worker_queue: queue.Queue[WorkerMessage] | None = None
    _timer: bpy.types.Timer | None = None
    _tracker_id: int = -1
    _should_stop: threading.Event | None = None
    _optimize_focal_length: bool = False
    _optimize_principal_point: bool = False


    _frame_from: int | None = None
    _frame_to: int | None = None

    @classmethod
    def poll(cls, context):
        state = PolychaseData.from_context(context)
        if not state:
            return False

        tracker = state.active_tracker
        return tracker is not None and tracker.clip is not None and \
               tracker.camera is not None and tracker.geometry is not None and \
               tracker.database_path != ""

    def execute(self, context: bpy.types.Context) -> set:
        assert context.scene
        assert context.window_manager

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

        frame_current = scene.frame_current
        frame_from: float | None = None
        frame_to: float | None = None

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
                        frame_to = frame
                        frame_to_keyframe_set = True
                        break
                    elif not frame_to_keyframe_set:
                        frame_to = max(frame, frame_to) if frame_to else frame

            break

        if not frame_from or not frame_to:
            self.report({"ERROR"}, "Could not detect the segment to refine")
            return {"CANCELLED"}

        self._frame_from = int(frame_from)
        self._frame_to = int(frame_to)
        num_frames = self._frame_to - self._frame_from + 1

        # Create camera trajectory
        self._camera_traj = core.CameraTrajectory(first_frame_id=self._frame_from, count=num_frames)
        pose_obj = core.CameraPose()
        cam_state_obj = core.CameraState()

        depsgraph = context.evaluated_depsgraph_get()

        # Set rotation mode to quaternion for convenience
        geom_rot_mode = geometry.rotation_mode
        cam_rot_mode = camera.rotation_mode
        geometry.rotation_mode = "QUATERNION"
        camera.rotation_mode = "QUATERNION"

        for frame in range(self._frame_from, self._frame_to + 1):
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
            self._camera_traj.set(frame, cam_state_obj)

        # Restore original rotation mode
        geometry.rotation_mode = geom_rot_mode
        camera.rotation_mode = cam_rot_mode

        # Restore scene frame
        scene.frame_set(frame_current)

        # Just storing these variables so that if they change while tracking, everything still works
        self._optimize_focal_length = tracker.tracking_optimize_focal_length
        self._optimize_principal_point = tracker.tracking_optimize_principal_point

        # Start worker thread
        lazy_func = refine_sequence_lazy(
            tracker=tracker,
            database_path=database_path,
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

        tracker.is_refining = True
        tracker.should_stop_refining = False
        tracker.refining_progress = 0.0
        tracker.refining_message = "Starting..."

        context.window_manager.modal_handler_add(self)
        self._timer = context.window_manager.event_timer_add(0.1, window=context.window)
        context.window_manager.progress_begin(0.0, 1.0)
        context.window_manager.progress_update(0)

        self._worker_thread.start()

        return {"RUNNING_MODAL"}

    def _work(
        self,
        fn: typing.Callable,
        from_worker_queue: queue.Queue[WorkerMessage],
        should_stop: threading.Event,
    ):

        def _callback(progress: core.RefineTrajectoryUpdate):
            # Send progress update
            from_worker_queue.put(progress)
            return not should_stop.is_set()

        try:
            success = fn(_callback)
            if not success and not should_stop.is_set():
                from_worker_queue.put(RuntimeError(f"Refining failed unexpectedly"))

            from_worker_queue.put(None)    # Signal completion (even if stopped)

        except Exception as e:
            traceback.print_exc()
            from_worker_queue.put(e)

    def _apply_camera_traj(self, context: bpy.types.Context, tracker: PolychaseClipTracking):
        geometry = tracker.geometry
        camera = tracker.camera

        assert camera
        assert geometry
        assert isinstance(camera.data, bpy.types.Camera)
        assert self._camera_traj
        assert self._frame_from
        assert self._frame_to
        assert context.scene

        frame_from = int(self._frame_from)
        frame_to = int(self._frame_to)

        is_tracking_geometry = tracker.tracking_target == "GEOMETRY"

        # Assuming poses at frame_from and frame_to are constants.
        frame_current = context.scene.frame_current

        depsgraph = context.evaluated_depsgraph_get()

        # Set rotation mode to quaternion for convenience
        geom_rot_mode = geometry.rotation_mode
        cam_rot_mode = camera.rotation_mode
        geometry.rotation_mode = "QUATERNION"
        camera.rotation_mode = "QUATERNION"

        for frame in range(frame_from, frame_to + 1):
            cam_state = self._camera_traj.get(frame)
            assert cam_state

            context.scene.frame_set(frame)

            geometry_evaled = geometry.evaluated_get(depsgraph)
            camera_evaled = camera.evaluated_get(depsgraph)
            assert isinstance(camera_evaled.data, bpy.types.Camera)

            model_view_quat = mathutils.Quaternion(typing.cast(typing.Sequence[float], cam_state.pose.q))
            model_view_t = mathutils.Vector(typing.cast(typing.Sequence[float], cam_state.pose.t))

            if is_tracking_geometry:
                geometry.rotation_quaternion = camera_evaled.rotation_quaternion @ model_view_quat
                geometry.location = camera_evaled.rotation_quaternion @ model_view_t + camera_evaled.location

                geometry.keyframe_insert(data_path="location", frame=frame, keytype="GENERATED")
                geometry.keyframe_insert(data_path="rotation_quaternion", frame=frame, keytype="GENERATED")

            else:
                geom_quat_inv = geometry_evaled.rotation_quaternion.inverted()
                geom_loc_inv = -(geom_quat_inv @ geometry_evaled.location)

                view_quat = model_view_quat @ geom_quat_inv
                view_t = model_view_quat @ geom_loc_inv + model_view_t

                view_quat_inv = view_quat.inverted()

                camera.rotation_quaternion = view_quat_inv
                camera.location = -(view_quat_inv @ view_t)

                camera.keyframe_insert(data_path="location", frame=frame, keytype="GENERATED")
                camera.keyframe_insert(data_path="rotation_quaternion", frame=frame, keytype="GENERATED")

            if self._optimize_focal_length or self._optimize_principal_point:
                core.set_camera_intrinsics(camera, cam_state.intrinsics)
                camera.data.keyframe_insert(data_path="lens", frame=frame, keytype="GENERATED")
                camera.data.keyframe_insert(data_path="shift_x", frame=frame, keytype="GENERATED")
                camera.data.keyframe_insert(data_path="shift_y", frame=frame, keytype="GENERATED")

        # Restore original rotation mode
        geometry.rotation_mode = geom_rot_mode
        camera.rotation_mode = cam_rot_mode

        # Restore scene frame
        context.scene.frame_set(frame_current)

        pass

    def modal(self, context: bpy.types.Context, event: bpy.types.Event) -> set:
        try:
            return self._modal_impl(context, event)
        except Exception as e:
            return self._cleanup(context, False, str(e))

    def _modal_impl(self, context: bpy.types.Context, event: bpy.types.Event) -> set:
        assert context.window_manager
        assert self._from_worker_queue
        assert self._worker_thread
        assert context.scene
        assert context.area

        state = PolychaseData.from_context(context)
        if not state:
            return self._cleanup(context, success=False, message="State data lost")
        tracker = state.get_tracker_by_id(self._tracker_id)
        if not tracker:
            return self._cleanup(context, success=False, message="Tracker was deleted")
        if not tracker.geometry or not tracker.clip or not tracker.camera:
            return self._cleanup(context, success=False, message="Tracking input changed")
        if tracker.should_stop_refining:
            return self._cleanup(context, success=False, message="Cancelled by user")
        if event is not None and event.type in {'ESC'}:
            return self._cleanup(context, success=False, message="Cancelled by user (ESC)")

        work_finished = False
        while not self._from_worker_queue.empty():
            try:
                message: WorkerMessage = self._from_worker_queue.get_nowait()

                if message is None:
                    work_finished = True
                    break

                elif isinstance(message, core.RefineTrajectoryUpdate):
                    tracker.refining_progress = message.progress
                    tracker.refining_message = message.message
                    context.area.tag_redraw()

                elif isinstance(message, Exception):
                    return self._cleanup(context, success=False, message=f"Error: {message}")

            except queue.Empty:
                pass
            except Exception as e:
                traceback.print_exc()
                return self._cleanup(context, success=False, message=f"Internal error: {e}")

        if work_finished:
            return self._cleanup(context, success=True, message="Refiner finished successfully")

        if not self._worker_thread.is_alive():
            return self._cleanup(context, success=False, message="Refiner worker thread stopped unexpectedly")

        return {"PASS_THROUGH"}

    def _cleanup(self, context: bpy.types.Context, success: bool, message: str):
        assert context.window_manager

        if self._timer:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None

        if self._should_stop:
            self._should_stop.set()

        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join()

        self._worker_thread = None
        self._to_worker_queue = None
        self._from_worker_queue = None
        self._should_stop = None
        current_tracker_id = self._tracker_id    # Store before resetting
        self._tracker_id = -1

        state = PolychaseData.from_context(context)
        if state:
            tracker = state.get_tracker_by_id(current_tracker_id)
            if tracker:
                # Apply camera trajectory even if we failed, since we might have applied a couple of
                # optimization iterations.
                self._apply_camera_traj(context, tracker)

                tracker.is_refining = False
                tracker.should_stop_refining = False    # Ensure it's reset
                tracker.refining_message = ""

        context.window_manager.progress_end()
        if context.area:
            context.area.tag_redraw()

        if success:
            self.report({"INFO"}, message)
            return {"FINISHED"}
        else:
            self.report({"WARNING"} if message.startswith("Cancelled") else {"ERROR"}, message)
            # Return finished even though we failed, so that undoing works.
            return {"FINISHED"}


class OT_CancelRefining(bpy.types.Operator):
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

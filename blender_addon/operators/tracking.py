import dataclasses
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


@dataclasses.dataclass
class ProgressUpdate:
    progress: float
    message: str


@dataclasses.dataclass
class TrackingResult:
    frame: int
    matrix: np.ndarray


WorkerMessage = ProgressUpdate | TrackingResult | Exception | None    # Types of messages from worker


def solve_forwards_lazy(
    tracker: PolychaseClipTracking,
    database_path: str,
    frame_from: int,
    num_frames: int,
    width: int,
    height: int,
    camera: bpy.types.Object,
    trans_type: core.TransformationType,
) -> typing.Callable[[typing.Callable[[int, np.ndarray], bool]], bool]:

    viewport_matrix = mathutils.Matrix(
        [
            [width / 2, 0, 0, width / 2],
            [0, height / 2, 0, height / 2],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])

    projection_matrix = viewport_matrix @ camera.calc_matrix_camera(
        bpy.context.evaluated_depsgraph_get(),
        x=width,
        y=height,
    )

    model_matrix = tracker.geometry.matrix_world.copy()
    view_matrix = camera.matrix_world.inverted()
    accel_mesh = tracker.core().accel_mesh

    def inner(callback: typing.Callable[[int, np.ndarray], bool]):
        return core.solve_forwards(
            database_path=database_path,
            frame_from=frame_from,
            num_frames=num_frames,
            scene_transform=core.SceneTransformations(
                model_matrix=model_matrix,    # type: ignore
                view_matrix=view_matrix,    # type: ignore
                projection_matrix=projection_matrix,    # type: ignore
            ),
            accel_mesh=accel_mesh,
            trans_type=trans_type,
            callback=callback,
        )

    return inner


class OT_TrackForwards(bpy.types.Operator):
    bl_idname = "polychase.track_forwards"
    bl_options = {"REGISTER", "UNDO"}
    bl_label = "Track Forwards"

    _worker_thread: threading.Thread | None = None
    _from_worker_queue: queue.Queue[WorkerMessage] | None = None
    _timer: bpy.types.Timer | None = None
    _tracker_id: int = -1
    _should_stop: threading.Event | None = None
    _trans_type: core.TransformationType = core.TransformationType.Camera

    @classmethod
    def poll(cls, context):
        state = PolychaseData.from_context(context)
        if not state:
            return False

        tracker = state.active_tracker
        return (
            tracker is not None and tracker.clip is not None and tracker.camera is not None
            and tracker.geometry is not None and tracker.database_path != "")

    def execute(self, context: bpy.types.Context):    # type: ignore
        assert context.window_manager
        scene = context.scene
        if not scene:
            self.report({"ERROR"}, "Scene context not found.")
            return {"CANCELLED"}

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
        if not camera:
            self.report({"ERROR"}, "Camera is not set.")
            return {"CANCELLED"}

        geometry = tracker.geometry
        if not geometry:
            self.report({"ERROR"}, "Geometry is not set.")
            return {"CANCELLED"}

        frame_from = scene.frame_current
        num_frames = clip.frame_duration - frame_from + clip.frame_start
        width, height = clip.size
        self._trans_type = core.TransformationType.Model if tracker.tracking_target == "GEOMETRY" else core.TransformationType.Camera
        self._tracker_id = tracker.id

        self._from_worker_queue = queue.Queue()
        self._should_stop = threading.Event()

        fn = solve_forwards_lazy(
            tracker=tracker,
            database_path=database_path,
            frame_from=frame_from,
            num_frames=num_frames,
            width=width,
            height=height,
            camera=camera,
            trans_type=self._trans_type,
        )

        self._worker_thread = threading.Thread(
            target=self._work,
            args=(
                fn,
                self._from_worker_queue,
                frame_from,
                num_frames,
                self._should_stop,
            ),
            daemon=True)

        tracker.is_tracking = True
        tracker.should_stop_tracking = False
        tracker.tracking_progress = 0.0
        tracker.tracking_message = "Starting..."

        context.window_manager.modal_handler_add(self)
        self._timer = context.window_manager.event_timer_add(0.05, window=context.window)
        context.window_manager.progress_begin(0.0, 1.0)
        context.window_manager.progress_update(0)

        self._worker_thread.start()
        return {"RUNNING_MODAL"}

    def _work(
            self,
            fn: typing.Callable,
            from_worker_queue: queue.Queue[WorkerMessage],
            frame_from: int,
            num_frames: int,
            should_stop: threading.Event):
        total_frames_to_process = num_frames
        start_frame = frame_from

        def _callback(frame_id: int, matrix: np.ndarray):
            current_frame_index = frame_id - start_frame
            progress = (current_frame_index + 1) / total_frames_to_process if total_frames_to_process > 0 else 1.0
            message = f"Tracking frame {frame_id}"

            # Send progress update
            progress_update = ProgressUpdate(progress, message)
            from_worker_queue.put(progress_update)

            # Send result
            result = TrackingResult(frame_id, matrix)
            from_worker_queue.put(result)

            return not should_stop.is_set()

        try:
            success = fn(_callback)
            if not success and not should_stop.is_set():
                # If solve_forwards returned false and we weren't stopped, it's an error
                raise RuntimeError("solve_forwards failed unexpectedly")

            from_worker_queue.put(None)    # Signal completion (even if stopped)

        except Exception as e:
            traceback.print_exc()
            from_worker_queue.put(e)    # Send exception back to main thread

    def modal(self, context: bpy.types.Context, event: bpy.types.Event):    # type: ignore
        assert context.window_manager
        assert self._from_worker_queue
        assert self._worker_thread
        assert context.scene

        state = PolychaseData.from_context(context)
        if not state:
            return self._cleanup(context, success=False, message="State data lost")
        tracker = state.get_tracker_by_id(self._tracker_id)
        if not tracker:
            return self._cleanup(context, success=False, message="Tracker lost")
        if tracker.should_stop_tracking:
            return self._cleanup(context, success=False, message="Cancelled by user")
        if event.type in {'ESC'}:
            return self._cleanup(context, success=False, message="Cancelled by user (ESC)")

        work_finished = False
        while not self._from_worker_queue.empty():
            try:
                message: WorkerMessage = self._from_worker_queue.get_nowait()

                if message is None:
                    # Worker signaled completion
                    work_finished = True
                    break    # Finish processing queue before cleanup

                if isinstance(message, ProgressUpdate):
                    tracker.tracking_progress = message.progress
                    tracker.tracking_message = message.message
                    context.window_manager.progress_update(message.progress)

                elif isinstance(message, TrackingResult):
                    matrix = mathutils.Matrix(message.matrix)    # type: ignore
                    frame = message.frame

                    context.scene.frame_set(frame)

                    target_object: bpy.types.Object = tracker.geometry if self._trans_type == core.TransformationType.Model else tracker.camera
                    if not target_object:
                        return self._cleanup(context, success=False, message=f"Error: Target object not found")

                    if self._trans_type == core.TransformationType.Model:
                        loc, rot, _ = matrix.decompose()
                        target_object.location = loc
                        target_object.rotation_mode = "QUATERNION"
                        target_object.rotation_quaternion = rot
                        target_object.keyframe_insert(data_path="location", frame=frame, keytype="GENERATED")
                        target_object.keyframe_insert(data_path="rotation_quaternion", frame=frame, keytype="GENERATED")
                    else:
                        loc, rot, _ = matrix.inverted().decompose()
                        target_object.location = loc
                        target_object.rotation_mode = "QUATERNION"
                        target_object.rotation_quaternion = rot
                        target_object.keyframe_insert(data_path="location", frame=frame, keytype="GENERATED")
                        target_object.keyframe_insert(data_path="rotation_quaternion", frame=frame, keytype="GENERATED")

                elif isinstance(message, Exception):
                    self.report({"ERROR"}, f"Error during tracking: {message}")
                    return self._cleanup(context, success=False, message=f"Error: {message}")

            except queue.Empty:
                pass    # Continue modal loop
            except Exception as e:
                traceback.print_exc()
                return self._cleanup(context, success=False, message=f"Internal error: {e}")

        if work_finished:
            return self._cleanup(context, success=True, message="Tracking finished successfully")

        if not self._worker_thread.is_alive():
            return self._cleanup(context, success=False, message="Tracking worker thread stopped unexpectedly")

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
                tracker.is_tracking = False
                tracker.should_stop_tracking = False    # Ensure it's reset
                tracker.tracking_message = ""

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


class OT_CancelTracking(bpy.types.Operator):
    bl_idname = "polychase.cancel_tracking"
    bl_label = "Cancel Tracking"
    bl_options = {"INTERNAL"}

    @classmethod
    def poll(cls, context):
        state = PolychaseData.from_context(context)
        return state is not None and state.active_tracker is not None and state.active_tracker.is_tracking

    def execute(self, context):    # type: ignore
        state = PolychaseData.from_context(context)
        if not state or not state.active_tracker:
            return {"CANCELLED"}

        tracker = state.active_tracker
        if tracker.is_tracking:
            tracker.should_stop_tracking = True
        else:
            # Defensive check
            self.report({"WARNING"}, "Tracking is not running.")

        return {"FINISHED"}

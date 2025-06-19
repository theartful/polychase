import dataclasses
import os
import queue
import threading
import traceback
import typing

import bpy
import bpy.types
import mathutils

from .. import core, utils, keyframes
from ..properties import PolychaseTracker, PolychaseData


@dataclasses.dataclass
class ProgressUpdate:
    progress: float
    message: str


WorkerMessage = ProgressUpdate | core.FrameTrackingResult | Exception | None


def track_sequence_lazy(
    tracker: PolychaseTracker,
    database_path: str,
    frame_from: int,
    frame_to_inclusive: int,
    width: int,
    height: int,
    camera: bpy.types.Object,
    trans_type: core.TransformationType,
) -> typing.Callable[[typing.Callable[[core.FrameTrackingResult], bool]], bool]:

    tracker_core = core.Tracker.get(tracker)

    assert tracker.geometry
    assert tracker_core

    model_matrix = tracker.geometry.matrix_world.copy()
    view_matrix = camera.matrix_world.inverted()
    accel_mesh = tracker_core.accel_mesh

    bundle_opts = core.BundleOptions()
    bundle_opts.loss_type = core.LossType.Cauchy
    bundle_opts.loss_scale = 1.0

    def inner(callback: typing.Callable[[core.FrameTrackingResult], bool]):
        return core.track_sequence(
            database_path=database_path,
            frame_from=frame_from,
            frame_to_inclusive=frame_to_inclusive,
            scene_transform=core.SceneTransformations(
                model_matrix=typing.cast(core.np.ndarray, model_matrix),
                view_matrix=typing.cast(core.np.ndarray, view_matrix),
                intrinsics=core.camera_intrinsics(camera, width, height),
            ),
            accel_mesh=accel_mesh,
            trans_type=trans_type,
            callback=callback,
            bundle_opts=bundle_opts,
            optimize_focal_length=tracker.variable_focal_length,
            optimize_principal_point=tracker.variable_principal_point,
        )

    return inner


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
            tracker is not None and tracker.clip is not None
            and tracker.camera is not None and tracker.geometry is not None
            and tracker.database_path != "")

    def execute(self, context: bpy.types.Context) -> set:
        assert context.window_manager
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
        frame_from = scene.frame_current
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

        if self.direction == "FORWARD" and frame_from == clip_end_frame:
            self.report(
                {"ERROR"}, "Current frame is already at the edge of the clip.")
            return {"CANCELLED"}

        if self.direction == "BACKWARD" and frame_from == clip_start_frame:
            self.report(
                {"ERROR"}, "Current frame is already at the edge of the clip.")
            return {"CANCELLED"}

        if self.direction == "FORWARD":
            if self.single_frame:
                frame_to_inclusive = frame_from + 1
            else:
                frame_to_inclusive = boundary_keyframe
        else:
            if self.single_frame:
                frame_to_inclusive = frame_from - 1
            else:
                frame_to_inclusive = boundary_keyframe

        num_frames = abs(frame_to_inclusive - frame_from) + 1
        if num_frames <= 1:
            self.report({"INFO"}, "Already at or past the next keyframe.")
            return {"CANCELLED"}

        # Ensure a keyframe exists at the starting frame
        self._ensure_keyframe_at_start(tracker, scene.frame_current)

        lazy_func = track_sequence_lazy(
            tracker=tracker,
            database_path=database_path,
            frame_from=frame_from,
            frame_to_inclusive=frame_to_inclusive,
            width=width,
            height=height,
            camera=camera,
            trans_type=self._trans_type,
        )

        self._from_worker_queue = queue.Queue()
        self._should_stop = threading.Event()

        self._worker_thread = threading.Thread(
            target=self._work,
            args=(
                lazy_func,
                self._from_worker_queue,
                frame_from,
                frame_to_inclusive,
                self.direction,
                self._should_stop,
            ),
            daemon=True,
        )

        if self.single_frame:
            # FIXME: YUCK? Maybe don't create the thread in the first place.
            self._worker_thread.start()
            self._worker_thread.join()

            # FIXME: YUCK? Maybe refactor modal so that we don't have to call it?
            result = self.modal(context, None)
            assert result != {"PASS_THROUGH"}
            return result

        else:
            tracker.is_tracking = True
            tracker.should_stop_tracking = False
            tracker.tracking_progress = 0.0
            tracker.tracking_message = "Starting..."

            context.window_manager.modal_handler_add(self)
            self._timer = context.window_manager.event_timer_add(
                0.1, window=context.window)
            context.window_manager.progress_begin(0.0, 1.0)
            context.window_manager.progress_update(0)

            self._worker_thread.start()
            return {"RUNNING_MODAL"}

    def _work(
        self,
        fn: typing.Callable,
        from_worker_queue: queue.Queue[WorkerMessage],
        frame_from: int,
        frame_to_inclusive: int,
        direction: str,
        should_stop: threading.Event,
    ):
        num_frames = abs(frame_to_inclusive - frame_from) + 1
        if num_frames <= 1:
            # Should have been caught in execute, but handle defensively
            from_worker_queue.put(
                Exception("Invalid frame range for tracking."))
            return

        def _callback(result: core.FrameTrackingResult):
            frame_id = result.frame
            frames_processed = abs(frame_id - frame_from) + 1

            progress = frames_processed / num_frames
            message = f"Tracking frame {frame_id} ({direction.lower()})"

            if result.inlier_ratio < 0.5:
                from_worker_queue.put(
                    Exception(
                        f"Could not predict pose at frame #{frame_id} from optical flow data due to low inlier ratio ({result.inlier_ratio*100:.02f}%)"
                    ))
                return False

            else:
                # Send progress update
                progress_update = ProgressUpdate(progress, message)
                from_worker_queue.put(progress_update)

                # Send result
                from_worker_queue.put(result)
                return not should_stop.is_set()

        try:
            success = fn(_callback)
            if not success and not should_stop.is_set():
                # If tracking function returned false and we weren't stopped, it's an error
                from_worker_queue.put(
                    RuntimeError(
                        f"Tracking ({direction.lower()}) failed unexpectedly"))

            from_worker_queue.put(None)    # Signal completion (even if stopped)

        except Exception as e:
            traceback.print_exc()
            from_worker_queue.put(e)    # Send exception back to main thread

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
        assert self._from_worker_queue
        assert self._worker_thread
        assert context.scene

        tracker = PolychaseData.get_tracker_by_id(self._tracker_id, context)
        if not tracker:
            return self._cleanup(
                context, success=False, message="Tracker was deleted")
        if not tracker.geometry or not tracker.clip or not tracker.camera:
            return self._cleanup(
                context, success=False, message="Tracking input changed")
        if tracker.should_stop_tracking:
            return self._cleanup(
                context, success=False, message="Cancelled by user")
        if event is not None and event.type in {"ESC"}:
            return self._cleanup(
                context, success=False, message="Cancelled by user (ESC)")

        assert isinstance(tracker.camera.data, bpy.types.Camera)

        work_finished = False
        while not self._from_worker_queue.empty():
            try:
                message: WorkerMessage = self._from_worker_queue.get_nowait()

                if message is None:
                    # Worker signaled completion
                    work_finished = True
                    break    # Finish processing queue before cleanup

                elif isinstance(message, ProgressUpdate):
                    tracker.tracking_progress = message.progress
                    tracker.tracking_message = message.message
                    context.window_manager.progress_update(message.progress)

                elif isinstance(message, core.FrameTrackingResult):
                    frame = message.frame
                    pose = message.pose
                    if self._trans_type == core.TransformationType.Camera:
                        pose = pose.inverse()

                    translation = mathutils.Vector(
                        typing.cast(typing.Sequence[float], pose.t))
                    quat = mathutils.Quaternion(
                        typing.cast(typing.Sequence[float], pose.q))

                    # We're not using tracker.get_target_object, because
                    # tracker.tracking_target might have changed
                    target_object = tracker.geometry if self._trans_type == core.TransformationType.Model else tracker.camera

                    context.scene.frame_set(frame)
                    target_object.location = translation
                    utils.set_rotation_quat(target_object, quat)

                    keyframes.insert_keyframe(
                        obj=target_object,
                        frame=frame,
                        data_paths=[
                            "location",
                            utils.get_rotation_data_path(target_object)
                        ],
                        keytype="GENERATED")

                    if tracker.variable_focal_length or tracker.variable_principal_point:
                        core.set_camera_intrinsics(
                            tracker.camera, message.intrinsics)

                        keyframes.insert_keyframe(
                            obj=tracker.camera.data,
                            frame=frame,
                            data_paths=keyframes.CAMERA_DATAPATHS,
                            keytype="GENERATED")

                elif isinstance(message, Exception):
                    return self._cleanup(
                        context, success=False, message=f"Error: {message}")

            except queue.Empty:
                pass    # Continue modal loop
            except Exception as e:
                traceback.print_exc()
                return self._cleanup(
                    context, success=False, message=f"Internal error: {e}")

        if work_finished:
            return self._cleanup(
                context, success=True, message="Tracking finished successfully")

        if not self._worker_thread.is_alive():
            return self._cleanup(
                context,
                success=False,
                message="Tracking worker thread stopped unexpectedly")

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

        tracker = PolychaseData.get_tracker_by_id(current_tracker_id, context)
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
        assert tracker.camera.data

        # Handle camera intrinsics keyframes
        if tracker.variable_focal_length or tracker.variable_principal_point:
            existing_keyframes = keyframes.get_keyframes(
                obj=target_object, frame=frame, data_paths=["lens"])

            if not existing_keyframes:
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
        state = PolychaseData.from_context(context)
        return state is not None and state.active_tracker is not None and state.active_tracker.is_tracking

    def execute(self, context) -> set:
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

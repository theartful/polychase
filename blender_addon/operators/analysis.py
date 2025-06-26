# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2025 Ahmed Essam <aessam.dahy@gmail.com>

import os
import queue
import threading
import traceback
import typing

import bpy
import bpy.props
import bpy.types
import numpy as np

from .. import background_images, core
from ..properties import PolychaseTracker, PolychaseState

ProgressUpdate = tuple[float, str]
FrameRequest = int
FrameData = np.ndarray | None    # Frame data or None on error/stop
WorkerMessage = ProgressUpdate | FrameRequest | Exception | None


def generate_database(
        first_frame: int,
        num_frames: int,
        width: int,
        height: int,
        frame_accessor: typing.Callable[[int], np.ndarray | None],
        callback: typing.Callable[[float, str], bool],
        database_path: str,
        write_images: bool = False):

    core.generate_optical_flow_database(
        video_info=core.VideoInfo(
            width=width,
            height=height,
            first_frame=first_frame,
            num_frames=num_frames),
        frame_accessor_function=frame_accessor,
        callback=callback,
        database_path=database_path,
        write_images=write_images,
    )


class PC_OT_AnalyzeVideo(bpy.types.Operator):
    bl_idname = "polychase.analyze_video"
    bl_options = {"INTERNAL"}
    bl_label = "Analyze Video"

    frame_from: bpy.props.IntProperty(name="First Frame", default=1, min=1)
    frame_to_inclusive: bpy.props.IntProperty(
        name="Last Frame", default=2, min=1)
    write_debug_images: bpy.props.BoolProperty(
        name="Write Debug Images",
        description=
        "Write images with detected features overlayed on top on disk for debugging.",
        default=False)

    _worker_thread: threading.Thread | None = None
    _to_worker_queue: queue.Queue | None = None
    _from_worker_queue: queue.Queue | None = None

    _timer: bpy.types.Timer | None = None
    _requested_frame: int | None = None
    _tracker_id: int = -1
    _image_source_name: str = ""
    _should_stop: threading.Event | None = None

    @classmethod
    def poll(cls, context):
        state = PolychaseState.from_context(context)
        if not state:
            return False

        tracker = state.active_tracker
        return tracker is not None and tracker.clip is not None and \
               tracker.camera is not None and tracker.database_path != ""

    def draw(self, context: bpy.types.Context):
        layout = self.layout
        assert layout

        should_alert = self.frame_from >= self.frame_to_inclusive

        layout.use_property_split = True

        row = layout.row(align=True)
        row.alert = should_alert
        row.prop(self, "frame_from")

        row = layout.row(align=True)
        row.alert = should_alert
        row.prop(self, "frame_to_inclusive")

        if should_alert:
            row = layout.row(align=True)
            row.alert = True
            row.label(text="Last Frame must be > First Frame.")

        layout.prop(self, "write_debug_images")

    def invoke(self, context: bpy.types.Context, event: bpy.types.Event) -> set:
        assert context.window_manager

        state = PolychaseState.from_context(context)
        if not state:
            return {"CANCELLED"}

        tracker = state.active_tracker
        if not tracker or not tracker.camera or not tracker.clip:
            return {"CANCELLED"}

        self.frame_from = tracker.clip.frame_start
        self.frame_to_inclusive = tracker.clip.frame_start + tracker.clip.frame_duration - 1

        return context.window_manager.invoke_props_dialog(self)

    def _prepare_image_source(
            self, tracker: PolychaseTracker) -> bpy.types.Image | None:
        assert tracker.clip
        assert tracker.camera
        assert isinstance(tracker.camera.data, bpy.types.Camera)

        camera_data: bpy.types.Camera = tracker.camera.data
        clip = tracker.clip

        camera_background = background_images.find_background_image_for_clip(
            camera_data, clip)

        if not camera_background:
            self.report(
                {"ERROR"},
                "Please set the selected clip as the background for this camera."
            )
            return None

        # If the background source is not IMAGE, create a new IMAGE background
        if camera_background.source != "IMAGE":
            # Create a new background image backed by an image (which is backed by the clip)
            # Set alpha to 0 to not interfere with existing source
            _, image_source = background_images.create_background_image_for_clip(
                camera_data, clip, alpha=0.0)
        else:
            assert camera_background.image
            image_source = camera_background.image

        self._image_source_name = image_source.name
        return image_source

    def execute(self, context: bpy.types.Context) -> set:
        assert context.window_manager

        state = PolychaseState.from_context(context)
        if not state:
            return {"CANCELLED"}

        tracker = state.active_tracker
        if not tracker:
            return {"CANCELLED"}

        if not tracker.clip:
            return {"CANCELLED"}

        if tracker.is_preprocessing:
            self.report({"WARNING"}, "Analysis is already in progress.")
            return {"CANCELLED"}

        image_source = self._prepare_image_source(tracker)
        if not image_source:
            return {"CANCELLED"}

        database_path = bpy.path.abspath(tracker.database_path)
        database_dir = os.path.dirname(database_path)

        if not os.path.isdir(database_dir):
            try:
                os.makedirs(database_dir, exist_ok=True)
            except Exception as e:
                self.report(
                    {"ERROR"}, f"Cannot create directory for daatabse: {e}")
                return {"CANCELLED"}

        if os.path.isdir(database_path):
            self.report(
                {"ERROR"}, f"Database path is a directory: '{database_path}'")
            return {"CANCELLED"}

        self._to_worker_queue = queue.Queue()
        self._from_worker_queue = queue.Queue()
        self._tracker_id = tracker.id
        self._should_stop = threading.Event()

        self._worker_thread = threading.Thread(
            target=self._work,
            args=(
                self._from_worker_queue,
                self._to_worker_queue,
                self.frame_from,
                self.frame_to_inclusive,
                tracker.clip.size[0],
                tracker.clip.size[1],
                database_path,
                self.write_debug_images,
                self._should_stop),
            daemon=True)

        tracker.is_preprocessing = True
        tracker.should_stop_preprocessing = False
        tracker.preprocessing_progress = 0.0

        # Now we're ready to pass the images, and generate the database
        context.window_manager.modal_handler_add(self)
        self._timer = context.window_manager.event_timer_add(
            0.05, window=context.window)

        context.window_manager.progress_begin(0.0, 1.0)
        context.window_manager.progress_update(0)

        self._worker_thread.start()
        return {"RUNNING_MODAL"}

    def _work(
        self,
        from_worker_queue: queue.Queue,
        to_worker_queue: queue.Queue,
        frame_from: int,
        frame_to_inclusive: int,
        width: int,
        height: int,
        database_path: str,
        write_debug_images: bool,
        should_stop: threading.Event,
    ):

        def callback(progress: float, msg: str):
            update_msg: ProgressUpdate = (progress, msg)
            from_worker_queue.put(update_msg)
            return not should_stop.is_set()

        def frame_accessor(frame: int):
            request_msg: FrameRequest = frame
            from_worker_queue.put(request_msg)

            for _ in range(10):
                if should_stop.is_set():
                    return None
                try:
                    return to_worker_queue.get(block=True, timeout=1)
                except queue.Empty:
                    pass
                except Exception:
                    traceback.print_exc()
                    return None

            return None

        generate_database(
            first_frame=frame_from,
            num_frames=frame_to_inclusive - frame_from + 1,
            width=width,
            height=height,
            frame_accessor=frame_accessor,
            callback=callback,
            database_path=database_path,
            write_images=write_debug_images,
        )

    def _provide_frame(self, context: bpy.types.Context, frame_id: int):
        if not self._to_worker_queue:
            return

        if not context.scene:
            self._to_worker_queue.put(None)
            return

        tracker = PolychaseState.get_tracker_by_id(self._tracker_id, context)
        if not tracker or not tracker.camera:
            self._to_worker_queue.put(None)    # Send error signal
            return

        image_source = bpy.data.images.get(self._image_source_name)
        if not image_source:
            self._to_worker_queue.put(None)    # Send error signal
            return

        camera: bpy.types.Object = tracker.camera
        if not camera or not isinstance(camera.data, bpy.types.Camera):
            self._to_worker_queue.put(None)    # Send error signal
            return

        camera_data: bpy.types.Camera = camera.data

        # Find image user for the image source
        image_user = background_images.get_image_user_for_image(
            camera_data, image_source)

        if not image_user:
            self._to_worker_queue.put(None)
            return

        # Blender is weird, so wait until frame_current in both the scene and
        # the camera background image are stable.
        if image_user.frame_current != frame_id or context.scene.frame_current != frame_id:
            context.scene.frame_set(frame_id)
            self._requested_frame = frame_id
            return

        # Even after making sure that we're at the correct frame, it might still
        # be the case that the image is not refreshed. So redraw the entire UI.
        bpy.ops.wm.redraw_timer(type="DRAW_WIN_SWAP", iterations=1)

        width, height = image_source.size
        channels = image_source.channels

        image_data = np.empty((height, width, channels), dtype=np.float32)
        image_source.pixels.foreach_get(    # type: ignore
            image_data.ravel())

        # TODO: Handle gray images?
        if image_source.channels == 4:
            image_data = image_data[:, :, :3]

        # Convert to uint8
        image_data = (image_data * 255).astype(np.uint8)
        self._to_worker_queue.put(image_data)

    def modal(self, context: bpy.types.Context, event: bpy.types.Event) -> set:
        try:
            return self._modal_impl(context, event)
        except Exception:
            return self._cleanup(context, success=False)

    def _modal_impl(
            self, context: bpy.types.Context, event: bpy.types.Event) -> set:
        assert context.window_manager
        assert self._from_worker_queue
        assert self._worker_thread

        # Stop processing if we were signaled to stop.
        tracker = PolychaseState.get_tracker_by_id(self._tracker_id, context)
        if not tracker:
            return self._cleanup(context, success=False)
        if tracker.should_stop_preprocessing:
            return self._cleanup(context, success=False)

        if event.type in {"ESC"}:
            return self._cleanup(context, success=False)

        while not self._from_worker_queue.empty():
            try:
                request: WorkerMessage = self._from_worker_queue.get_nowait()

                if isinstance(request, int):
                    self._requested_frame = request

                elif isinstance(request, tuple) and len(request) == 2:
                    progress, msg = request
                    tracker.preprocessing_progress = progress
                    tracker.preprocessing_message = msg
                    context.window_manager.progress_update(progress)

                elif isinstance(request, Exception):
                    self.report({"ERROR"}, str(request))
                    return self._cleanup(context, success=False)

            except queue.Empty:
                pass

            except Exception as e:
                traceback.print_exc()
                self.report(
                    {"ERROR"},
                    f"Internal error processing analysis update: {e}")
                return self._cleanup(context, success=False)

        if self._requested_frame is not None:
            frame_to_get = self._requested_frame
            # Clear request before running _provide_frame, because it can fail
            # and reset _requested_frame to the same frame again
            self._requested_frame = None
            self._provide_frame(context, frame_to_get)

        if tracker.preprocessing_progress >= 1.0:
            return self._cleanup(context, success=True)

        if not self._worker_thread.is_alive():
            # Worker died unexpectedly without sending 1.0 progress or error
            self.report(
                {"ERROR"}, "Analysis worker thread stopped unexpectedly.")
            return self._cleanup(context, success=False)

        return {"PASS_THROUGH"}

    def _cleanup(self, context: bpy.types.Context, success: bool):
        assert context.window_manager
        assert self._should_stop

        if self._timer:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None

        tracker = PolychaseState.get_tracker_by_id(self._tracker_id, context)

        if tracker:
            tracker.is_preprocessing = False
            tracker.should_stop_preprocessing = False
            tracker.preprocessing_progress = 0.0
            tracker.preprocessing_message = ""

        if self._worker_thread and self._worker_thread.is_alive():
            self._should_stop.set()
            self._worker_thread.join()

        self._worker_thread = None
        self._to_worker_queue = None
        self._from_worker_queue = None
        self._should_stop = None

        context.window_manager.progress_end()
        if context.area:
            context.area.tag_redraw()

        if success:
            self.report({"INFO"}, "Video analysis completed successfully.")
            return {"FINISHED"}
        else:
            self.report({"INFO"}, "Video analysis was cancelled.")
            return {"CANCELLED"}


class PC_OT_CancelAnalysis(bpy.types.Operator):
    bl_idname = "polychase.cancel_analysis"
    bl_label = "Cancel Analysis"
    bl_options = {"INTERNAL"}

    @classmethod
    def poll(cls, context):
        state = PolychaseState.from_context(context)
        return state is not None and state.active_tracker is not None and state.active_tracker.is_preprocessing

    def execute(self, context) -> set:
        state = PolychaseState.from_context(context)
        if not state or not state.active_tracker:
            return {"CANCELLED"}

        tracker = state.active_tracker
        if tracker.is_preprocessing:
            tracker.should_stop_preprocessing = True
        else:
            # This shouldn't happen due to poll, but handle defensively
            self.report({"WARNING"}, "Analysis is not running.")

        return {"FINISHED"}

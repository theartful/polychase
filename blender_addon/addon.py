import functools
import os
import queue
import threading
import traceback

import bpy
import bpy.props
import bpy.types
import bpy.utils
import gpu
import mathutils
import numpy as np
from bpy_extras.view3d_utils import location_3d_to_region_2d
from gpu_extras.batch import batch_for_shader

from . import core


@functools.cache
def get_points_shader():
    shader_info = gpu.types.GPUShaderCreateInfo()
    shader_info.vertex_source(
        """
    void main()
    {
        gl_Position = ModelViewProjectionMatrix * objectToWorld * vec4(position, 1.0f);
        finalColor = vec4(color, 1.0f);
    }
    """)
    shader_info.fragment_source(
        """
    void main()
    {
        vec2 cxy = 2.0 * gl_PointCoord - 1.0;
        float r = dot(cxy, cxy);
        float delta = fwidth(r);
        float alpha = 1.0 - smoothstep(1.0 - delta, 1.0 + delta, r);
        if (alpha <= 0.0) discard;
        fragColor = finalColor * alpha;
    }
    """)
    vert_out = gpu.types.GPUStageInterfaceInfo("point_interface")
    vert_out.smooth("VEC4", "finalColor")

    shader_info.vertex_in(0, "VEC3", "position")
    shader_info.vertex_in(1, "VEC3", "color")
    shader_info.vertex_out(vert_out)
    shader_info.fragment_out(0, "VEC4", "fragColor")
    shader_info.push_constant("MAT4", "ModelViewProjectionMatrix")
    shader_info.push_constant("MAT4", "objectToWorld")

    return gpu.shader.create_from_info(shader_info)


def bpy_poll_is_mesh(self, obj: bpy.types.Object) -> bool:
    return obj and obj.type == "MESH"


def bpy_poll_is_camera(self, obj: bpy.types.Object) -> bool:
    return obj and obj.type == "CAMERA"


class PolychaseClipTracking(bpy.types.PropertyGroup):
    id: bpy.props.IntProperty(default=0)
    name: bpy.props.StringProperty(name="Name")
    clip: bpy.props.PointerProperty(name="Clip", type=bpy.types.MovieClip)
    geometry: bpy.props.PointerProperty(name="Geometry", type=bpy.types.Object, poll=bpy_poll_is_mesh)
    camera: bpy.props.PointerProperty(name="Camera", type=bpy.types.Object, poll=bpy_poll_is_camera)
    tracking_target: bpy.props.EnumProperty(
        name="Tracking Target",
        items=(("CAMERA", "Camera", "Track Camera"), ("GEOMETRY", "Geometry", "Track Geometry")))
    database_path: bpy.props.StringProperty(
        name="Database", description="Optical flow database path", subtype="FILE_PATH")

    # State for database generation
    is_preprocessing: bpy.props.BoolProperty(default=False)
    should_stop_preprocessing: bpy.props.BoolProperty(default=False)

    def core(self) -> core.Tracker:
        return core.Trackers.get_tracker(self.id, self.geometry)


class PolychaseData(bpy.types.PropertyGroup):
    trackers: bpy.props.CollectionProperty(type=PolychaseClipTracking, name="Trackers")
    active_tracker_idx: bpy.props.IntProperty(default=-1)
    num_created_trackers: bpy.props.IntProperty(default=0)
    in_pinmode: bpy.props.BoolProperty(default=False)

    @classmethod
    def register(cls):
        setattr(bpy.types.Scene, "polychase_data", bpy.props.PointerProperty(type=cls))

    @classmethod
    def unregister(cls):
        delattr(bpy.types.Scene, "polychase_data")

    @classmethod
    def from_context(cls, context=None):
        return getattr(context.scene if context else bpy.context.scene, "polychase_data", None)

    @property
    def active_tracker(self):
        if self.active_tracker_idx == -1:
            return None
        return self.trackers[self.active_tracker_idx]

    def is_tracking_active(self):
        return self.active_tracker_idx != -1


class OT_CreateTracker(bpy.types.Operator):
    bl_idname = "polychase.create_tracker"
    bl_options = {"REGISTER"}
    bl_label = "Create Tracker"

    def execute(self, context):
        state = PolychaseData.from_context(context)
        state.num_created_trackers += 1
        state.active_tracker_idx = len(state.trackers)

        tracker = state.trackers.add()
        tracker.id = state.num_created_trackers
        tracker.name = f"Polychase Tracker #{state.num_created_trackers:04}"

        return {"FINISHED"}


class OT_SelectTracker(bpy.types.Operator):
    bl_idname = "polychase.select_tracker"
    bl_options = {"REGISTER"}
    bl_label = "Select Tracker"

    idx: bpy.props.IntProperty(default=0)

    def execute(self, context):
        state = PolychaseData.from_context(context)
        state.active_tracker_idx = self.idx
        return {"FINISHED"}


class OT_DeleteTracker(bpy.types.Operator):
    bl_idname = "polychase.delete_tracker"
    bl_options = {"REGISTER"}
    bl_label = "Select Tracker"

    idx: bpy.props.IntProperty(default=0)

    def execute(self, context):
        state = PolychaseData.from_context(context)
        if state.active_tracker_idx >= self.idx:
            state.active_tracker_idx -= 1

        state.trackers.remove(self.idx)
        return {"FINISHED"}


class PT_PolychasePanel(bpy.types.Panel):
    bl_label = "Polychase"
    bl_category = "Polychase"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        return True

    def draw(self, context: bpy.types.Context):
        state = PolychaseData.from_context(context)
        trackers = state.trackers

        layout = self.layout
        col = layout.column(align=True)
        for idx, tracker in enumerate(trackers):
            is_active_tracker = (idx == state.active_tracker_idx)

            row = col.row(align=True)
            op = row.operator(
                OT_SelectTracker.bl_idname,
                text=tracker.name,
                depress=is_active_tracker,
                icon="CAMERA_DATA" if tracker.tracking_target == "CAMERA" else "MESH_DATA")
            op.idx = idx

            op = row.operator(OT_DeleteTracker.bl_idname, text="", icon="CANCEL")
            op.idx = idx

        row = layout.row(align=True)
        row.operator(OT_CreateTracker.bl_idname, icon="ADD")


class PT_TrackerInputsPanel(bpy.types.Panel):
    bl_label = "Inputs"
    bl_category = "Polychase"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        return PolychaseData.from_context(context).is_tracking_active()

    def draw(self, context: bpy.types.Context):
        state = PolychaseData.from_context(context)
        tracker = state.active_tracker

        layout = self.layout

        row = layout.row(align=True)
        row.prop(tracker, "name")

        row = layout.row(align=True)
        row.alert = not tracker.clip
        row.prop(tracker, "clip")

        row = layout.row(align=True)
        row.alert = not tracker.geometry
        row.prop(tracker, "geometry")

        row = layout.row(align=True)
        row.alert = not tracker.camera
        row.prop(tracker, "camera")


class PT_TrackerTrackingPanel(bpy.types.Panel):
    bl_label = "Tracking"
    bl_category = "Polychase"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        return PolychaseData.from_context(context).is_tracking_active

    def draw(self, context: bpy.types.Context):
        state = PolychaseData.from_context(context)
        tracker = state.active_tracker

        layout = self.layout
        row = layout.row(align=True)
        row.prop(tracker, "tracking_target", text="Target")

        row = layout.row(align=True)
        row.operator(OT_PinMode.bl_idname, depress=state.in_pinmode)

        row = layout.row(align=True)
        row.operator(OT_TrackForwards.bl_idname)


class PT_TrackerOpticalFlowPanel(bpy.types.Panel):
    bl_label = "Optical Flow"
    bl_category = "Polychase"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        return PolychaseData.from_context(context).is_tracking_active()

    def draw(self, context: bpy.types.Context):
        state = PolychaseData.from_context(context)
        tracker = state.active_tracker

        layout = self.layout

        row = layout.row(align=True)
        row.prop(tracker, "database_path")

        row = layout.row(align=True)
        row.operator(OT_AnalyzeVideo.bl_idname)


class OT_PinMode(bpy.types.Operator):
    bl_idname = "polychase.start_pinmode"
    bl_options = {"UNDO", "INTERNAL"}
    bl_label = "Start Pin Mode"

    tracker: PolychaseClipTracking = None
    draw_handler = None
    shader = None
    batch = None
    point_radius = 15.0
    is_left_mouse_clicked = False

    old_shading_type = None
    old_show_axis_x = None
    old_show_axis_y = None
    old_show_axis_z = None
    old_show_floor = None
    old_show_xray_wireframe = None

    def get_pin_mode_data(self):
        return self.tracker.core().pin_mode

    def create_batch(self):
        pin_mode_data: core.PinModeData = self.get_pin_mode_data()
        self.batch = batch_for_shader(
            self.shader,
            "POINTS",
            {
                "position": pin_mode_data.points if len(pin_mode_data.points) else [],
                "color": pin_mode_data.colors if len(pin_mode_data.colors) else []
            })

    def draw_callback(self):
        geometry = self.tracker.geometry
        object_to_world = geometry.matrix_world

        gpu.state.point_size_set(self.point_radius)
        self.shader.uniform_float("objectToWorld", object_to_world)
        self.batch.draw(self.shader)

    def invoke(self, context: bpy.types.Context, event: bpy.types.Event):
        state = PolychaseData.from_context(context)

        if state.in_pinmode:
            state.in_pinmode = False
            return {"CANCELLED"}

        state.in_pinmode = True

        self.tracker = state.active_tracker
        self.tracker.core().pin_mode.unselect_pin()

        camera = self.tracker.camera
        geometry = self.tracker.geometry

        bpy.ops.object.select_all(action="DESELECT")
        bpy.ops.object.mode_set(mode="OBJECT", toggle=False)
        camera.hide_set(False)
        geometry.hide_set(False)
        geometry.select_set(True)

        context.view_layer.objects.active = camera if self.tracker.tracking_target == "CAMERA" else geometry
        context.area.spaces.active.camera = camera
        context.area.spaces.active.region_3d.view_perspective = "CAMERA"

        self.old_shading_type = context.area.spaces.active.shading.type
        self.old_show_xray_wireframe = context.area.spaces.active.shading.show_xray_wireframe
        self.old_show_axis_x = context.area.spaces.active.overlay.show_axis_x
        self.old_show_axis_y = context.area.spaces.active.overlay.show_axis_y
        self.old_show_axis_z = context.area.spaces.active.overlay.show_axis_z
        self.old_show_floor = context.area.spaces.active.overlay.show_floor

        context.area.spaces.active.shading.type = 'WIREFRAME'
        context.area.spaces.active.shading.show_xray_wireframe = False
        context.area.spaces.active.overlay.show_axis_x = False
        context.area.spaces.active.overlay.show_axis_y = False
        context.area.spaces.active.overlay.show_axis_z = False
        context.area.spaces.active.overlay.show_floor = False

        context.window_manager.modal_handler_add(self)

        self.shader = get_points_shader()
        self.create_batch()

        self.draw_handler = bpy.types.SpaceView3D.draw_handler_add(self.draw_callback, (), "WINDOW", "POST_VIEW")
        return {"RUNNING_MODAL"}

    def find_clicked_pin(
        self,
        event: bpy.types.Event,
        geometry: bpy.types.Object,
        region: bpy.types.Region,
        rv3d: bpy.types.RegionView3D,
    ):
        pin_mode_data: core.PinModeData = self.get_pin_mode_data()

        mouse_x, mouse_y = event.mouse_region_x, event.mouse_region_y
        object_to_world = geometry.matrix_world

        # TODO: Vectorize
        for idx, point in enumerate(pin_mode_data.points):
            screen_point = location_3d_to_region_2d(region, rv3d, object_to_world @ mathutils.Vector(point))
            if not screen_point:
                continue
            dist = ((mouse_x - screen_point.x)**2 + (mouse_y - screen_point.y)**2)**0.5
            if dist < self.point_radius:
                return idx

        return None

    def raycast(
        self,
        event: bpy.types.Event,
        geometry: bpy.types.Object,
        region: bpy.types.Region,
        rv3d: bpy.types.RegionView3D,
        tracker_id: int,
    ):
        mouse_x, mouse_y = event.mouse_region_x, event.mouse_region_y
        result = self.tracker.core().ray_cast(region, rv3d, mouse_x, mouse_y)

        return result.pos if result else None

    def select_pin(self, pin_idx: int):
        pin_mode_data: core.PinModeData = self.get_pin_mode_data()
        pin_mode_data.select_pin(pin_idx)

    def update_initial_scene_transformation(self, region, rv3d):
        self.tracker.core().update_initial_scene_transformation(region, rv3d)

    def create_pin(self, location: np.ndarray, context: bpy.types.Context):
        pin_mode_data: core.PinModeData = self.get_pin_mode_data()
        pin_mode_data.create_pin(location, select=True)

    def delete_pin(self, pin_idx: int):
        pin_mode_data: core.PinModeData = self.get_pin_mode_data()
        pin_mode_data.delete_pin(pin_idx)

    def unselect_pin(self):
        pin_mode_data: core.PinModeData = self.get_pin_mode_data()
        pin_mode_data.unselect_pin()

    def redraw(self, context):
        # This is horrible
        self.create_batch()
        context.area.tag_redraw()

    def is_dragging_pin(self, event):
        return event.type == "MOUSEMOVE" and self.is_left_mouse_clicked and self.get_pin_mode_data(
        ).selected_pin_idx >= 0

    def modal(self, context: bpy.types.Context, event: bpy.types.Event):
        try:
            return self.modal_impl(context, event)
        except:
            traceback.print_exc()

            self.cancel(context)
            return {"CANCELLED"}

    def modal_impl(self, context: bpy.types.Context, event: bpy.types.Event):
        state = PolychaseData.from_context(context)
        tracker = state.active_tracker
        geometry = tracker.geometry
        camera = tracker.camera
        region = context.region
        rv3d = context.space_data.region_3d

        if event.type == "LEFTMOUSE" and event.value == "PRESS":
            self.is_left_mouse_clicked = True

            pin_idx = self.find_clicked_pin(event, geometry, region, rv3d)
            if pin_idx is not None:
                self.select_pin(pin_idx)
                self.update_initial_scene_transformation(region, rv3d)
                # FIXME: don't recreate the batch every time a selection is made
                # Instead add a uniform indicating the selected vertex, and check
                # if selected in the shader.
                self.redraw(context)
                return {"RUNNING_MODAL"}

            location = self.raycast(event, geometry, region, rv3d, tracker.id)
            if location is not None:
                self.create_pin(location, context)
                self.update_initial_scene_transformation(region, rv3d)
                self.redraw(context)
                return {"RUNNING_MODAL"}

            self.unselect_pin()
            self.redraw(context)
            return {"RUNNING_MODAL"}

        elif event.type == "RIGHTMOUSE" and event.value == "PRESS":
            pin_idx = self.find_clicked_pin(event, geometry, region, rv3d)
            if pin_idx is not None:
                self.delete_pin(pin_idx)
                self.redraw(context)
            return {"RUNNING_MODAL"}

        elif event.type == "LEFTMOUSE" and event.value == "RELEASE":
            self.is_left_mouse_clicked = False

        elif self.is_dragging_pin(event):
            if self.tracker.tracking_target == "GEOMETRY":
                matrix_world = self.tracker.core().find_transformation(
                    region, rv3d, event.mouse_region_x, event.mouse_region_y, core.TransformationType.Model)

                if matrix_world is not None:
                    geometry.matrix_world = mathutils.Matrix(matrix_world)
            else:
                view_matrix = self.tracker.core().find_transformation(
                    region, rv3d, event.mouse_region_x, event.mouse_region_y, core.TransformationType.Camera)

                if view_matrix is not None:
                    camera.matrix_world = mathutils.Matrix(view_matrix).inverted()

            return {"RUNNING_MODAL"}

        elif event.type == "ESC":
            self.cancel(context)
            return {"CANCELLED"}

        return {"PASS_THROUGH"}

    def cancel(self, context):
        state = PolychaseData.from_context(context)
        state.in_pinmode = False

        bpy.types.SpaceView3D.draw_handler_remove(self.draw_handler, "WINDOW")
        context.area.tag_redraw()

        self.unselect_pin()
        context.area.spaces.active.shading.type = self.old_shading_type
        context.area.spaces.active.shading.show_xray_wireframe = self.old_show_xray_wireframe
        context.area.spaces.active.overlay.show_axis_x = self.old_show_axis_x
        context.area.spaces.active.overlay.show_axis_y = self.old_show_axis_y
        context.area.spaces.active.overlay.show_axis_z = self.old_show_axis_z
        context.area.spaces.active.overlay.show_floor = self.old_show_floor


class OT_TrackForwards(bpy.types.Operator):
    bl_idname = "polychase.track_forwards"
    bl_options = {"REGISTER", "BLOCKING", "INTERNAL"}
    bl_label = "Track Forwards"

    def invoke(self, context: bpy.types.Context, event: bpy.types.Event):
        state = PolychaseData.from_context(context)
        tracker = state.active_tracker
        clip = tracker.clip
        camera = tracker.camera
        geometry = tracker.geometry

        trans_type = core.TransformationType.Model if tracker.tracking_target == "GEOMETRY" else core.TransformationType.Camera

        def callback(frame_id, matrix):
            context.scene.frame_set(frame_id)
            if trans_type == core.TransformationType.Model:
                loc, rot, scale = mathutils.Matrix(matrix).decompose()
                geometry.location = loc
                geometry.rotation_mode = "QUATERNION"
                geometry.rotation_quaternion = rot
                geometry.keyframe_insert(data_path="location")
                geometry.keyframe_insert(data_path="rotation_quaternion")
            else:
                loc, rot, scale = mathutils.Matrix(matrix).inverted().decompose()
                camera.location = loc
                camera.rotation_mode = "QUATERNION"
                camera.rotation_quaternion = rot
                camera.keyframe_insert(data_path="location")
                camera.keyframe_insert(data_path="rotation_quaternion")
            return True

        frame_from = context.scene.frame_current
        num_frames = clip.frame_duration - frame_from + 1
        width, height = clip.size

        result = tracker.core().solve_forwards(
            database_path=bpy.path.abspath(tracker.database_path),
            frame_from=frame_from,
            num_frames=num_frames,
            width=width,
            height=height,
            camera=camera,
            trans_type=trans_type,
            callback=callback,
        )
        print("RESULT = ", result)

        return {"FINISHED"}


class OT_AnalyzeVideo(bpy.types.Operator):
    bl_idname = "polychase.analyze_video"
    bl_options = {"REGISTER", "BLOCKING", "INTERNAL"}
    bl_label = "Analyze Video"

    frame_from: bpy.props.IntProperty(name="First Frame", default=1, min=1)
    frame_to_inclusive: bpy.props.IntProperty(name="Last Frame", default=2, min=1)

    _worker_thread: threading.Thread | None = None
    _frames_queue: queue.Queue | None = None
    _requests_queue: queue.Queue | None = None

    _timer: bpy.types.Timer | None = None

    def draw(self, context: bpy.types.Context):
        layout = self.layout

        row = layout.row(align=True)
        row.alert = (self.frame_from >= self.frame_to_inclusive)
        row.prop(self, "frame_from")

        row = layout.row(align=True)
        row.alert = (self.frame_from >= self.frame_to_inclusive)
        row.prop(self, "frame_to_inclusive")

    def invoke(self, context: bpy.types.Context, event: bpy.types.Event):
        return context.window_manager.invoke_props_dialog(self)

    def execute(self, context: bpy.types.Context):
        state = PolychaseData.from_context(context)
        tracker = state.active_tracker

        assert tracker is not None

        camera = tracker.camera
        clip = tracker.clip

        # Find background image object for camera
        camera_background = None
        for bg in camera.data.background_images:
            if bg.clip == tracker.clip or (bg.source == "IMAGE" and bg.image is not None
                                           and bg.image.filepath == clip.filepath):
                camera_background = bg

        # If no background image was found relating to the clip, then return error. Otherwise it will be confusing to the user.
        # as the results will be to a different clip.
        if not camera_background:
            self.report({"ERROR"}, "Please set the selected clip as the background for this camera.")
            return {"CANCELLED"}

        # I can't find a way to read the frames pixel if the background source is a movieclip
        if camera_background.source != "IMAGE":
            # Create a new background image backed by an image (which is backed by the clip)
            camera_background = camera.data.background_images.new()

            image_source = bpy.data.images.new(
                f"polychase_{os.path.basename(clip.filepath)}",
                clip.size[0],
                clip.size[1],
                alpha=True,
                float_buffer=False)
            image_source.source = clip.source
            image_source.filepath = clip.filepath

            camera_background.image = image_source
            camera_background.image_user.frame_start = clip.frame_start
            camera_background.image_user.frame_duration = clip.frame_duration
            camera_background.image_user.use_auto_refresh = True
            # Set the alpha to 0 so that we don't interfere with the current MOVIE_CLIP source
            camera_background.alpha = 0.0

            # TODO: Support the rest of the options
            # Specifically: offset, and flip_x / flip_y, rotation and scale

        else:
            image_source = camera_background.image

        # Now we're ready to pass the images, and generate the database
        context.window_manager.modal_handler_add(self)
        self._timer = context.window_manager.event_timer_add(0.05, window=context.window)
        self._frames_queue = queue.Queue()
        self._requests_queue = queue.Queue()

        self._worker_thread = threading.Thread(
            target=self._work,
            args=(self._requests_queue, self._frames_queue, self.frame_from, self.frame_to_inclusive))

        tracker.is_preprocessing = True
        tracker.should_stop_preprocessing = False

        self._worker_thread.start()

        return {"RUNNING_MODAL"}

    def _work(self, requests_queue: queue.Queue, frames_queue: queue.Queue, frame_from: int, frame_to_inclusive: int):
        tracker = PolychaseData.from_context().active_tracker

        def callback(progress: float, msg: str):
            print(f"{int(progress*100)}: {msg}")
            requests_queue.put((progress, msg))
            return tracker.is_preprocessing and not tracker.should_stop_preprocessing

        def frame_accessor(frame: int):
            requests_queue.put(frame)

            while True:
                try:
                    return frames_queue.get(block=True, timeout=5)
                except queue.Empty:
                    if not tracker.is_preprocessing:
                        return np.empty((0, 0), dtype=np.uint8)
                except Exception:
                    traceback.print_exc()
                    return None

        tracker.core().generate_database(
            first_frame=frame_from,
            num_frames=frame_to_inclusive - frame_from + 1,
            width=tracker.clip.size[0],
            height=tracker.clip.size[1],
            frame_accessor=frame_accessor,
            callback=callback,
            database_path=bpy.path.abspath(tracker.database_path))

    def _get_frame(self, frame_id: int):
        tracker = PolychaseData.from_context().active_tracker
        camera = tracker.camera
        clip = tracker.clip

        # Find background image object for camera
        camera_background = None
        for bg in camera.data.background_images:
            if bg.source == "IMAGE" and bg.image is not None and bg.image.filepath == clip.filepath:
                camera_background = bg

        if not camera_background:
            self._frames_queue.put(None)
            return None

        # Blender is weird, so wait until frame_current in both the scene and the camera background image are stable.
        if camera_background.image_user.frame_current != frame_id or bpy.context.scene.frame_current != frame_id:
            bpy.context.scene.frame_set(frame_id)
            self._requests_queue.put(frame_id)
            return

        image_source = camera_background.image

        image_data = np.empty((image_source.size[1], image_source.size[0], image_source.channels), dtype=np.float32)
        image_source.pixels.foreach_get(image_data.ravel())

        # TODO: Handle gray images?
        if image_source.channels == 4:
            image_data = image_data[:, :, :3]

        # Convert to uint8
        image_data = (image_data * 255).astype(np.uint8)
        self._frames_queue.put(image_data)

    def _report_progress(self, progress: float, msg: str):
        pass

    def modal(self, context: bpy.types.Context, event: bpy.types.Event):
        context.area.tag_redraw()

        try:
            request = self._requests_queue.get_nowait()
            if isinstance(request, int):
                self._get_frame(request)

            elif isinstance(request, tuple):
                assert len(request) == 2
                progress, msg = request
                self._report_progress(progress, msg)
                if progress == 1.0:
                    self._cleanup(context)
                    return {"FINISHED"}

            elif isinstance(request, Exception):
                self.report({"ERROR"}, str(request))
                self._cleanup(context)
                return {"CANCELLED"}

        except queue.Empty:
            if not self._worker_thread.is_alive():
                self.report({"ERROR"}, "Unknown error!")
                self._cleanup(context)
                return {"CANCELLED"}

        except Exception as e:
            traceback.print_exc()
            self.report({"ERROR"}, str(e))
            self._cleanup(context)
            return {"CANCELLED"}

        return {"RUNNING_MODAL"}

    def _cleanup(self, context):
        tracker = PolychaseData.from_context().active_tracker
        tracker.is_preprocessing = False

        if self._worker_thread:
            self._worker_thread.join()
            self._worker_thread = None

        context.window_manager.event_timer_remove(self._timer)


def add_addon_var(name: str, settings_type) -> None:
    setattr(bpy.types.MovieClip, name, bpy.props.PointerProperty(type=settings_type))


def remove_addon_var(name: str) -> None:
    delattr(bpy.types.MovieClip, name)


def get_addon_var(name: str, context):
    return getattr(context.space_data.clip, name, None)


is_registered = False

classes = [
    PolychaseClipTracking,
    PolychaseData,
    PT_PolychasePanel,
    PT_TrackerInputsPanel,
    PT_TrackerOpticalFlowPanel,
    PT_TrackerTrackingPanel,
    OT_CreateTracker,
    OT_SelectTracker,
    OT_DeleteTracker,
    OT_PinMode,
    OT_TrackForwards,
    OT_AnalyzeVideo,
]


def register():
    global is_registered

    if is_registered:
        return

    for cls in classes:
        bpy.utils.register_class(cls)

    is_registered = True


def unregister():
    global is_registered

    if not is_registered:
        return

    for cls in classes:
        bpy.utils.unregister_class(cls)

    is_registered = False

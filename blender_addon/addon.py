import functools
import math

import bpy
import bpy.props
import bpy.types
import bpy.utils
import cv2
import gpu
import mathutils
import numpy as np
from bpy_extras.view3d_utils import (
    location_3d_to_region_2d, region_2d_to_location_3d, region_2d_to_origin_3d, region_2d_to_vector_3d)
from gpu_extras.batch import batch_for_shader


@functools.cache
def get_points_shader():
    shader_info = gpu.types.GPUShaderCreateInfo()
    shader_info.vertex_source(
        """
    void main()
    {
        gl_Position = ModelViewProjectionMatrix * objectToWorld * position;
        finalColor = color;
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

    shader_info.vertex_in(0, "VEC4", "position")
    shader_info.vertex_in(1, "VEC4", "color")
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
    name: bpy.props.StringProperty(name="Name")
    clip: bpy.props.PointerProperty(name="Clip", type=bpy.types.MovieClip)
    geometry: bpy.props.PointerProperty(name="Geometry", type=bpy.types.Object, poll=bpy_poll_is_mesh)
    camera: bpy.props.PointerProperty(name="Camera", type=bpy.types.Object, poll=bpy_poll_is_camera)


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
        return getattr(context.scene, "polychase_data", None)

    @property
    def active_tracker(self):
        return self.trackers[self.active_tracker_idx]


class OT_CreateTracker(bpy.types.Operator):
    bl_idname = "polychase.create_tracker"
    bl_options = {"REGISTER"}
    bl_label = "Create Tracker"

    def execute(self, context):
        state = PolychaseData.from_context(context)
        state.num_created_trackers += 1
        state.active_tracker_idx = len(state.trackers)
        tracker = state.trackers.add()
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
                OT_SelectTracker.bl_idname, text=tracker.name, depress=is_active_tracker, icon="CAMERA_DATA")
            op.idx = idx

            op = row.operator(OT_DeleteTracker.bl_idname, text="", icon="CANCEL")
            op.idx = idx

        row = layout.row(align=True)
        row.operator(OT_CreateTracker.bl_idname, icon="ADD")

        if state.active_tracker_idx != -1:
            tracker = state.trackers[state.active_tracker_idx]


class PT_TrackerInputsPanel(bpy.types.Panel):
    bl_label = "Inputs"
    bl_category = "Polychase"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        state = PolychaseData.from_context(context)
        return state.active_tracker_idx != -1

    def draw(self, context: bpy.types.Context):
        state = PolychaseData.from_context(context)
        tracker = state.trackers[state.active_tracker_idx]

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
        state = PolychaseData.from_context(context)
        return state.active_tracker_idx != -1

    def draw(self, context: bpy.types.Context):
        state = PolychaseData.from_context(context)
        # tracker = state.trackers[state.active_tracker_idx]

        layout = self.layout

        row = layout.row(align=True)
        row.operator(OT_PinMode.bl_idname, depress=state.in_pinmode)


class PT_TrackerOpticalFlowPanel(bpy.types.Panel):
    bl_label = "Optical Flow"
    bl_category = "Polychase"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        state = PolychaseData.from_context(context)
        return state.active_tracker_idx != -1

    def draw(self, context: bpy.types.Context):
        state = PolychaseData.from_context(context)
        # tracker = state.trackers[state.active_tracker_idx]

        layout = self.layout

        row = layout.row(align=True)
        row.operator(OT_PinMode.bl_idname, depress=state.in_pinmode)


DEFAULT_PIN_COLOR = mathutils.Vector((0, 0, 1, 1))
SELECTED_PIN_COLOR = mathutils.Vector((1, 0, 0, 1))
NEGATIVE_Z = mathutils.Matrix(
    [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])


class OT_PinMode(bpy.types.Operator):
    bl_idname = "polychase.start_pinmode"
    bl_options = {"REGISTER", "BLOCKING", "INTERNAL"}
    bl_label = "Start Pin Mode"

    draw_handler = None
    points = []
    colors = []
    shader = None
    batch = None
    point_radius = 15.0
    selected_pin_idx = None
    is_left_mouse_clicked = False

    old_shading_type = None
    old_show_axis_x = None
    old_show_axis_y = None
    old_show_axis_z = None
    old_show_floor = None
    old_show_xray_wireframe = None
    old_matrix_world = None

    def create_batch(self):
        self.batch = batch_for_shader(self.shader, "POINTS", {"position": self.points, "color": self.colors})

    def draw_callback(self):
        state = PolychaseData.from_context(bpy.context)
        tracker = state.trackers[state.active_tracker_idx]
        geometry = tracker.geometry

        object_to_world = geometry.matrix_world

        gpu.state.point_size_set(self.point_radius)
        self.shader.uniform_float("objectToWorld", object_to_world)
        self.batch.draw(self.shader)

    def invoke(self, context: bpy.types.Context, event):
        state = PolychaseData.from_context(context)

        if state.in_pinmode:
            state.in_pinmode = False
            return {"CANCELLED"}

        state.in_pinmode = True

        tracker = state.active_tracker
        camera = tracker.camera
        geometry = tracker.geometry

        bpy.ops.object.select_all(action="DESELECT")
        bpy.ops.object.mode_set(mode="OBJECT", toggle=False)
        camera.hide_set(False)
        geometry.hide_set(False)
        geometry.select_set(True)

        context.view_layer.objects.active = geometry
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
        mouse_x, mouse_y = event.mouse_region_x, event.mouse_region_y
        object_to_world = geometry.matrix_world

        # TODO: Vectorize
        for idx, point in enumerate(self.points):
            screen_point = location_3d_to_region_2d(region, rv3d, object_to_world @ point)
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
    ):
        mouse_x, mouse_y = event.mouse_region_x, event.mouse_region_y
        view_vector_worldspace = region_2d_to_vector_3d(region, rv3d, (mouse_x, mouse_y))
        view_origin_worldspace = region_2d_to_origin_3d(region, rv3d, (mouse_x, mouse_y))

        # matrix_world is object to world, so we invert it to be world to object
        world_to_object = geometry.matrix_world.inverted()
        view_origin_objectspace = world_to_object @ view_origin_worldspace
        view_vector_objectspace = (world_to_object.to_3x3() @ view_vector_worldspace).normalized()

        result, location, normal, index = geometry.ray_cast(view_origin_objectspace, view_vector_objectspace)
        return location.to_4d() if result else None

    def select_pin(self, pin_idx: int, context: bpy.types.Context):
        if self.selected_pin_idx is not None:
            self.colors[self.selected_pin_idx] = DEFAULT_PIN_COLOR

        self.selected_pin_idx = pin_idx
        self.colors[self.selected_pin_idx] = SELECTED_PIN_COLOR

    def create_pin(self, location: mathutils.Vector, context: bpy.types.Context):
        self.points.append(location)
        self.colors.append(DEFAULT_PIN_COLOR)
        self.select_pin(len(self.points) - 1, context)

    def delete_pin(self, idx: int):
        if idx < 0 or idx >= len(self.points):
            return

        if self.selected_pin_idx == idx:
            self.selected_pin_idx = 0 if len(self.points) > 0 else None
        elif self.selected_pin_idx > idx:
            self.selected_pin_idx -= 1

        del self.colors[idx]
        del self.points[idx]

    def unselect_pin(self):
        if self.selected_pin_idx is not None:
            self.colors[self.selected_pin_idx] = DEFAULT_PIN_COLOR

        self.selected_pin_idx = None

    def redraw(self, context):
        # This is horrible
        self.create_batch()
        context.area.tag_redraw()

    def is_dragging_pin(self, event):
        return event.type == "MOUSEMOVE" and self.is_left_mouse_clicked and self.selected_pin_idx != None

    def modal(self, context: bpy.types.Context, event: bpy.types.Event):
        state = PolychaseData.from_context(context)
        tracker = state.trackers[state.active_tracker_idx]
        geometry = tracker.geometry
        region = context.region
        rv3d = context.space_data.region_3d

        if event.type == "LEFTMOUSE" and event.value == "PRESS":
            self.is_left_mouse_clicked = True
            self.old_matrix_world = geometry.matrix_world.copy()

            pin_idx = self.find_clicked_pin(event, geometry, region, rv3d)
            if pin_idx is not None:
                self.select_pin(pin_idx, context)
                self.redraw(context)
                return {"RUNNING_MODAL"}

            location = self.raycast(event, geometry, region, rv3d)
            if location is not None:
                self.create_pin(location, context)
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
            mouse_x, mouse_y = event.mouse_region_x, event.mouse_region_y
            view_vector_worldspace = region_2d_to_vector_3d(region, rv3d, (mouse_x, mouse_y))
            view_origin_worldspace = region_2d_to_origin_3d(region, rv3d, (mouse_x, mouse_y))

            if len(self.points) == 1:
                # In this case, simply drag
                point_object_space = self.points[0].to_3d()
                point_world_space = geometry.matrix_world @ point_object_space
                depth = (point_world_space - view_origin_worldspace).length

                translated_pos = view_origin_worldspace + depth * view_vector_worldspace
                translation = translated_pos - point_world_space

                geometry.matrix_world.translation += translation

            elif len(self.points) == 2:
                # In this case, scale and rotate
                moving_point_object_space = self.points[self.selected_pin_idx].to_3d()
                anchor_point_object_space = self.points[1 - self.selected_pin_idx].to_3d()

                moving_point_world_space = geometry.matrix_world @ moving_point_object_space
                anchor_point_world_space = geometry.matrix_world @ anchor_point_object_space
                depth = (moving_point_world_space - view_origin_worldspace).length

                translated_moving_point = view_origin_worldspace + depth * view_vector_worldspace
                du = (moving_point_world_space - anchor_point_world_space)
                dv = (translated_moving_point - anchor_point_world_space)

                scale = dv.length / du.length

                du.normalize()
                dv.normalize()
                dn = du.cross(dv)

                # TODO: Do I really have to use "asin"? I can easily compute the sin and cosine instead using dot and cross product,
                # but mathutils.Matrix.Rotation requires an angle Q_Q
                angle = math.asin(dn.length)
                rotation = mathutils.Matrix.Rotation(angle, 4, dn)

                geometry.matrix_world = \
                        mathutils.Matrix.Translation(anchor_point_world_space) @ \
                        mathutils.Matrix.Scale(scale, 4) @ \
                        rotation @ \
                        mathutils.Matrix.Translation(-anchor_point_world_space) @ \
                        geometry.matrix_world

            else:
                points = np.array(self.points)
                model_view_matrix = np.array(rv3d.view_matrix @ self.old_matrix_world)
                window_matrix = np.array(rv3d.window_matrix)
                width = region.width
                height = region.height

                fx = window_matrix[0][0] * width / 2.0
                fy = window_matrix[1][1] * height / 2.0
                cx = width * (1.0 - window_matrix[2][0]) / 2.0
                cy = height * (window_matrix[2][1] + 1.0) / 2.0
                intrinsics_mat = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])

                object_points = (model_view_matrix @ points.T).T
                object_points = object_points[:, 0:3] / object_points[:, 3].reshape(-1, 1)

                # intrinsics_mat uses OpenCV convention, which makes the camera
                # look at positive z direction, while blender uses OpenGL convention,
                # with the camera looking at negative z direction.
                object_points[:, 2] *= -1.0

                image_points = (intrinsics_mat @ object_points.T).T
                image_points = image_points[:, 0:2] / image_points[:, 2].reshape(-1, 1)

                image_points[self.selected_pin_idx, 0] = mouse_x
                image_points[self.selected_pin_idx, 1] = mouse_y

                retval, rvec, tvec = cv2.solvePnP(
                    object_points,
                    image_points,
                    intrinsics_mat,
                    None,
                    rvec=np.zeros(3),
                    tvec=np.zeros(3),
                    useExtrinsicGuess=True,
                    flags=cv2.
                    SOLVEPNP_ITERATIVE,  # We want to be as near to the extrinsic guess as possible
                )

                if retval > 0:
                    rot, _ = cv2.Rodrigues(rvec)

                    transform = mathutils.Matrix(rot).to_4x4()
                    transform.translation = tvec

                    geometry.matrix_world = rv3d.view_matrix.inverted(
                    ) @ NEGATIVE_Z @ transform @ NEGATIVE_Z @ rv3d.view_matrix @ self.old_matrix_world

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
        if self.old_shading_type:
            context.area.spaces.active.shading.type = self.old_shading_type
        if self.old_show_xray_wireframe:
            context.area.spaces.active.shading.show_xray_wireframe = self.old_show_xray_wireframe
        if self.old_show_axis_x:
            context.area.spaces.active.overlay.show_axis_x = self.old_show_axis_x
        if self.old_show_axis_y:
            context.area.spaces.active.overlay.show_axis_y = self.old_show_axis_y
        if self.old_show_axis_z:
            context.area.spaces.active.overlay.show_axis_z = self.old_show_axis_z
        if self.old_show_floor:
            context.area.spaces.active.overlay.show_floor = self.old_show_floor


def add_addon_var(name: str, settings_type) -> None:
    setattr(bpy.types.MovieClip, name, bpy.props.PointerProperty(type=settings_type))


def remove_addon_var(name: str) -> None:
    delattr(bpy.types.MovieClip, name)


def get_addon_var(name: str, context):
    return getattr(context.space_data.clip, name, None)


is_registered = False


def register():
    global is_registered

    if is_registered:
        return

    bpy.utils.register_class(PolychaseClipTracking)
    bpy.utils.register_class(PolychaseData)
    bpy.utils.register_class(PT_PolychasePanel)
    bpy.utils.register_class(PT_TrackerInputsPanel)
    bpy.utils.register_class(PT_TrackerTrackingPanel)
    bpy.utils.register_class(OT_CreateTracker)
    bpy.utils.register_class(OT_SelectTracker)
    bpy.utils.register_class(OT_DeleteTracker)
    bpy.utils.register_class(OT_PinMode)

    is_registered = True


def unregister():
    global is_registered

    if not is_registered:
        return

    bpy.utils.unregister_class(PolychaseClipTracking)
    bpy.utils.unregister_class(PolychaseData)
    bpy.utils.unregister_class(PT_PolychasePanel)
    bpy.utils.unregister_class(PT_TrackerInputsPanel)
    bpy.utils.unregister_class(PT_TrackerTrackingPanel)
    bpy.utils.unregister_class(OT_CreateTracker)
    bpy.utils.unregister_class(OT_SelectTracker)
    bpy.utils.unregister_class(OT_DeleteTracker)
    bpy.utils.unregister_class(OT_PinMode)

    is_registered = False

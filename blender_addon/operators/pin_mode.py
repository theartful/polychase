import bpy
import bpy.types
import gpu
import mathutils
import numpy as np
import traceback
from gpu_extras.batch import batch_for_shader
from bpy_extras.view3d_utils import location_3d_to_region_2d

from .. import core
from ..properties import PolychaseData, PolychaseClipTracking
from ..utils import get_points_shader


class OT_PinMode(bpy.types.Operator):
    bl_idname = "polychase.start_pinmode"
    bl_options = {"UNDO", "INTERNAL"}
    bl_label = "Start Pin Mode"

    tracker: PolychaseClipTracking | None = None
    draw_handler = None
    shader: gpu.types.GPUShader | None = None
    batch = None
    point_radius = 15.0
    is_left_mouse_clicked = False

    space_view: bpy.types.SpaceView3D | None = None
    old_shading_type = None
    old_show_axis_x = None
    old_show_axis_y = None
    old_show_axis_z = None
    old_show_floor = None
    old_show_xray_wireframe = None

    def get_pin_mode_data(self):
        assert self.tracker
        return self.tracker.core().pin_mode

    def create_batch(self):
        assert self.shader

        pin_mode_data: core.PinModeData = self.get_pin_mode_data()
        points = pin_mode_data.points if len(pin_mode_data.points) > 0 else []
        colors = pin_mode_data.colors if len(pin_mode_data.colors) > 0 else []

        self.batch = batch_for_shader(self.shader, "POINTS", {"position": points, "color": colors}) # type:ignore

    def draw_callback(self):
        assert self.tracker
        assert self.shader
        assert self.batch

        tracker = self.tracker

        geometry = tracker.geometry
        object_to_world = geometry.matrix_world

        gpu.state.point_size_set(self.point_radius)
        self.shader.uniform_float("objectToWorld", object_to_world)
        self.batch.draw(self.shader)

    def invoke(self, context: bpy.types.Context, event: bpy.types.Event):
        assert context.view_layer is not None
        assert context.area
        assert context.area.spaces.active
        assert isinstance(context.area.spaces.active, bpy.types.SpaceView3D)
        assert context.window_manager

        state = PolychaseData.from_context(context)
        if not state:
            return {"CANCELLED"}

        if state.in_pinmode:
            state.in_pinmode = False
            return {"CANCELLED"}

        state.in_pinmode = True

        self.tracker = state.active_tracker
        if not self.tracker:
            return {"CANCELLED"}

        self.tracker.core().pin_mode.unselect_pin()

        camera = self.tracker.camera
        geometry = self.tracker.geometry

        bpy.ops.object.select_all(action="DESELECT")
        bpy.ops.object.mode_set(mode="OBJECT", toggle=False)
        camera.hide_set(False)
        geometry.hide_set(False)
        geometry.select_set(True)

        context.view_layer.objects.active = camera if self.tracker.tracking_target == "CAMERA" else geometry

        self.space_view = context.area.spaces.active
        assert self.space_view.region_3d

        self.space_view.camera = camera
        self.space_view.region_3d.view_perspective = "CAMERA"

        self.old_shading_type = self.space_view.shading.type
        self.old_show_xray_wireframe = self.space_view.shading.show_xray_wireframe
        self.old_show_axis_x = self.space_view.overlay.show_axis_x
        self.old_show_axis_y = self.space_view.overlay.show_axis_y
        self.old_show_axis_z = self.space_view.overlay.show_axis_z
        self.old_show_floor = self.space_view.overlay.show_floor

        self.space_view.shading.type = 'WIREFRAME'
        self.space_view.shading.show_xray_wireframe = False
        self.space_view.overlay.show_axis_x = False
        self.space_view.overlay.show_axis_y = False
        self.space_view.overlay.show_axis_z = False
        self.space_view.overlay.show_floor = False

        context.window_manager.modal_handler_add(self)

        self.shader = get_points_shader()
        self.create_batch()

        self.draw_handler = bpy.types.SpaceView3D.draw_handler_add(self.draw_callback, (), "WINDOW", "POST_VIEW") # type: ignore
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
        region: bpy.types.Region,
        rv3d: bpy.types.RegionView3D,
    ):
        mouse_x, mouse_y = event.mouse_region_x, event.mouse_region_y

        assert self.tracker
        result = self.tracker.core().ray_cast(region, rv3d, mouse_x, mouse_y)

        return result.pos if result else None

    def select_pin(self, pin_idx: int):
        pin_mode_data: core.PinModeData = self.get_pin_mode_data()
        pin_mode_data.select_pin(pin_idx)

    def update_initial_scene_transformation(self, region, rv3d):
        assert self.tracker
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

    def modal(self, context: bpy.types.Context, event: bpy.types.Event): # type: ignore
        try:
            return self.modal_impl(context, event)
        except:
            traceback.print_exc()
            return self.cancel(context)

    def modal_impl(self, context: bpy.types.Context, event: bpy.types.Event):
        if not self.tracker:
            return self.cancel(context)

        geometry = self.tracker.geometry
        camera = self.tracker.camera
        region = context.region

        if not region or not isinstance(context.space_data, bpy.types.SpaceView3D):
            return self.cancel(context)

        rv3d = context.space_data.region_3d

        if not rv3d:
            return self.cancel(context)

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

            location = self.raycast(event, region, rv3d)
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
                    geometry.matrix_world = mathutils.Matrix(matrix_world) # type: ignore
            else:
                view_matrix = self.tracker.core().find_transformation(
                    region, rv3d, event.mouse_region_x, event.mouse_region_y, core.TransformationType.Camera)

                if view_matrix is not None:
                    camera.matrix_world = mathutils.Matrix(view_matrix).inverted() # type: ignore

            return {"RUNNING_MODAL"}

        elif event.type == "ESC":
            self.cancel(context)
            return {"CANCELLED"}

        return {"PASS_THROUGH"}

    def cancel(self, context):
        state = PolychaseData.from_context(context)
        assert state

        state.in_pinmode = False

        bpy.types.SpaceView3D.draw_handler_remove(self.draw_handler, "WINDOW")
        if context.area:
            context.area.tag_redraw()

        self.unselect_pin()
        assert self.space_view

        self.space_view.shading.type = self.old_shading_type # type: ignore
        self.space_view.shading.show_xray_wireframe = self.old_show_xray_wireframe # type: ignore
        self.space_view.overlay.show_axis_x = self.old_show_axis_x # type: ignore
        self.space_view.overlay.show_axis_y = self.old_show_axis_y # type: ignore
        self.space_view.overlay.show_axis_z = self.old_show_axis_z # type: ignore
        self.space_view.overlay.show_floor = self.old_show_floor # type: ignore
        
        return {"CANCELLED"}



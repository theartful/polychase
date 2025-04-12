import traceback

import bpy
import bpy.types
import gpu
import mathutils
import numpy as np
from bpy_extras.view3d_utils import location_3d_to_region_2d
from gpu_extras.batch import batch_for_shader

from .. import core
from ..properties import PolychaseClipTracking, PolychaseData
from ..utils import get_points_shader


class OT_PinMode(bpy.types.Operator):
    bl_idname = "polychase.start_pinmode"
    bl_options = {"UNDO", "INTERNAL"}
    bl_label = "Start Pin Mode"

    _tracker: PolychaseClipTracking | None = None
    _draw_handler = None
    _shader: gpu.types.GPUShader | None = None
    _batch = None
    _point_radius = 15.0
    _is_left_mouse_clicked = False

    _space_view: bpy.types.SpaceView3D | None = None
    _area: bpy.types.Area | None = None
    _old_shading_type = None
    _old_show_axis_x = None
    _old_show_axis_y = None
    _old_show_axis_z = None
    _old_show_floor = None
    _old_show_xray_wireframe = None

    def get_pin_mode_data(self):
        assert self._tracker
        return self._tracker.core().pin_mode

    def create_batch(self):
        assert self._shader

        pin_mode_data: core.PinModeData = self.get_pin_mode_data()
        points = pin_mode_data.points if len(pin_mode_data.points) > 0 else []
        colors = pin_mode_data.colors if len(pin_mode_data.colors) > 0 else []

        self._batch = batch_for_shader(self._shader, "POINTS", {"position": points, "color": colors})    # type:ignore

    def _update_initial_scene_transformation(self, rv3d: bpy.types.RegionView3D):
        assert self._tracker
        core_tracker = self._tracker.core()
        self._tracker.core().pin_mode.initial_scene_transform = core.SceneTransformations(
            model_matrix=core_tracker.geom.matrix_world,    # type: ignore
            projection_matrix=rv3d.window_matrix,    # type: ignore
            view_matrix=rv3d.view_matrix,    # type: ignore
        )

    def _find_transformation(
        self,
        region: bpy.types.Region,
        rv3d: bpy.types.RegionView3D,
        region_x: int,
        region_y: int,
        trans_type: core.TransformationType,
    ):
        assert self._tracker
        core_tracker = self._tracker.core()
        ndc_pos = core_tracker.ndc(region, region_x, region_y)
        return core.find_transformation(
            core_tracker.pin_mode.points,
            core_tracker.pin_mode.initial_scene_transform,    # type: ignore
            core.SceneTransformations(
                model_matrix=core_tracker.geom.matrix_world,    # type: ignore
                view_matrix=rv3d.view_matrix,    # type: ignore
                projection_matrix=rv3d.window_matrix,    # type: ignore
            ),
            core.PinUpdate(pin_idx=core_tracker.pin_mode.selected_pin_idx, pin_pos=ndc_pos),    # type: ignore
            trans_type,
        )

    def draw_callback(self):
        assert self._tracker
        assert self._shader
        assert self._batch

        tracker = self._tracker

        geometry = tracker.geometry
        object_to_world = geometry.matrix_world

        gpu.state.point_size_set(self._point_radius)
        self._shader.uniform_float("objectToWorld", object_to_world)
        self._batch.draw(self._shader)

    def invoke(self, context: bpy.types.Context, event: bpy.types.Event):    # type: ignore
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

        self._tracker = state.active_tracker
        if not self._tracker:
            return {"CANCELLED"}

        self._tracker.core().pin_mode.unselect_pin()

        camera = self._tracker.camera
        geometry = self._tracker.geometry

        bpy.ops.object.select_all(action="DESELECT")
        bpy.ops.object.mode_set(mode="OBJECT", toggle=False)
        camera.hide_set(False)
        geometry.hide_set(False)
        geometry.select_set(True)

        context.view_layer.objects.active = camera if self._tracker.tracking_target == "CAMERA" else geometry

        self._space_view = context.area.spaces.active
        assert self._space_view.region_3d

        self._space_view.camera = camera
        self._space_view.region_3d.view_perspective = "CAMERA"

        self._old_shading_type = self._space_view.shading.type
        self._old_show_xray_wireframe = self._space_view.shading.show_xray_wireframe
        self._old_show_axis_x = self._space_view.overlay.show_axis_x
        self._old_show_axis_y = self._space_view.overlay.show_axis_y
        self._old_show_axis_z = self._space_view.overlay.show_axis_z
        self._old_show_floor = self._space_view.overlay.show_floor

        self._space_view.shading.type = 'WIREFRAME'
        self._space_view.shading.show_xray_wireframe = False
        self._space_view.overlay.show_axis_x = False
        self._space_view.overlay.show_axis_y = False
        self._space_view.overlay.show_axis_z = False
        self._space_view.overlay.show_floor = False

        context.window_manager.modal_handler_add(self)

        self._shader = get_points_shader()
        self.create_batch()

        self._draw_handler = bpy.types.SpaceView3D.draw_handler_add(
            self.draw_callback, (), "WINDOW", "POST_VIEW")    # type: ignore
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
            if dist < self._point_radius:
                return idx

        return None

    def raycast(
        self,
        event: bpy.types.Event,
        region: bpy.types.Region,
        rv3d: bpy.types.RegionView3D,
    ):
        mouse_x, mouse_y = event.mouse_region_x, event.mouse_region_y

        assert self._tracker
        result = self._tracker.core().ray_cast(region, rv3d, mouse_x, mouse_y)

        return result.pos if result else None

    def select_pin(self, pin_idx: int):
        pin_mode_data: core.PinModeData = self.get_pin_mode_data()
        pin_mode_data.select_pin(pin_idx)

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
        return event.type == "MOUSEMOVE" and self._is_left_mouse_clicked and self.get_pin_mode_data(
        ).selected_pin_idx >= 0

    def modal(self, context: bpy.types.Context, event: bpy.types.Event):    # type: ignore
        try:
            return self.modal_impl(context, event)
        except:
            traceback.print_exc()
            return self._cleanup(context)

    def modal_impl(self, context: bpy.types.Context, event: bpy.types.Event):
        if not self._tracker:
            return self._cleanup(context)

        geometry = self._tracker.geometry
        camera = self._tracker.camera
        region = self._space_view.region_3d
        region = context.region

        if not region or not isinstance(context.space_data, bpy.types.SpaceView3D):
            return self._cleanup(context)

        rv3d = context.space_data.region_3d

        if not rv3d:
            return self._cleanup(context)

        if event.type == "LEFTMOUSE" and event.value == "PRESS":
            self._is_left_mouse_clicked = True

            pin_idx = self.find_clicked_pin(event, geometry, region, rv3d)
            if pin_idx is not None:
                self.select_pin(pin_idx)
                self._update_initial_scene_transformation(rv3d)
                # FIXME: don't recreate the batch every time a selection is made
                # Instead add a uniform indicating the selected vertex, and check
                # if selected in the shader.
                self.redraw(context)
                return {"RUNNING_MODAL"}

            location = self.raycast(event, region, rv3d)
            if location is not None:
                self.create_pin(location, context)
                self._update_initial_scene_transformation(rv3d)
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
            self._is_left_mouse_clicked = False

        elif self.is_dragging_pin(event):
            if self._tracker.tracking_target == "GEOMETRY":
                matrix_world = self._find_transformation(
                    region, rv3d, event.mouse_region_x, event.mouse_region_y, core.TransformationType.Model)

                if matrix_world is not None:
                    geometry.matrix_world = mathutils.Matrix(matrix_world)    # type: ignore
            else:
                view_matrix = self._find_transformation(
                    region, rv3d, event.mouse_region_x, event.mouse_region_y, core.TransformationType.Camera)

                if view_matrix is not None:
                    camera.matrix_world = mathutils.Matrix(view_matrix).inverted()    # type: ignore

            return {"RUNNING_MODAL"}

        elif event.type == "ESC":
            self._cleanup(context)
            return {"CANCELLED"}

        return {"PASS_THROUGH"}

    def _cleanup(self, context):
        state = PolychaseData.from_context(context)
        assert state

        state.in_pinmode = False

        bpy.types.SpaceView3D.draw_handler_remove(self._draw_handler, "WINDOW")
        if context.area:
            context.area.tag_redraw()

        self.unselect_pin()
        assert self._space_view

        self._space_view.shading.type = self._old_shading_type    # type: ignore
        self._space_view.shading.show_xray_wireframe = self._old_show_xray_wireframe    # type: ignore
        self._space_view.overlay.show_axis_x = self._old_show_axis_x    # type: ignore
        self._space_view.overlay.show_axis_y = self._old_show_axis_y    # type: ignore
        self._space_view.overlay.show_axis_z = self._old_show_axis_z    # type: ignore
        self._space_view.overlay.show_floor = self._old_show_floor    # type: ignore

        return {"CANCELLED"}

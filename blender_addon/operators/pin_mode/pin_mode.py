# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2025 Ahmed Essam <aessam.dahy@gmail.com>

import traceback
import typing

import bpy
import bpy.types
import bpy_extras.view3d_utils as view3d_utils
import mathutils
import numpy as np

from ... import core, utils, keyframes
from ...properties import PolychaseTracker, PolychaseData
from . import keymaps, rendering, masking_3d

actually_in_pin_mode: bool = False


class PC_OT_PinMode(bpy.types.Operator):
    bl_idname = "polychase.start_pinmode"
    bl_options = {"REGISTER", "INTERNAL"}
    bl_label = "Enter Pin Mode"

    _tracker_id: int = -1
    _tracker: PolychaseTracker | None = None
    _draw_handler = None
    _is_left_mouse_clicked = False
    _is_right_mouse_clicked = False

    _renderer: rendering.PinModeRenderer | None = None

    _space_view_pointer: int = 0
    _initial_scene_transform: core.SceneTransformations | None = None
    _current_scene_transform: core.SceneTransformations | None = None

    # For masking polygons
    _is_drawing_3d_mask: bool = False
    _mask_selector: masking_3d.Masking3DSelector | None = None

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        state = PolychaseData.from_context(context)
        if state is None:
            return False
        tracker = state.active_tracker
        # Check if state exists and tracker is active
        return tracker is not None and tracker.camera is not None and tracker.geometry is not None

    def get_pin_mode_data(self) -> core.PinModeData:
        assert self._tracker
        tracker_core = core.Tracker.get(self._tracker)
        assert tracker_core

        return tracker_core.pin_mode

    def update_initial_scene_transformation(self, rv3d: bpy.types.RegionView3D):
        assert self._tracker
        assert self._tracker.geometry
        assert self._tracker.camera

        camera = self._tracker.camera
        geom = self._tracker.geometry

        self._initial_scene_transform = core.SceneTransformations(
            model_matrix=typing.cast(np.ndarray, geom.matrix_world),
            view_matrix=typing.cast(np.ndarray, rv3d.view_matrix),
            intrinsics=core.camera_intrinsics(camera),
        )

        # We check that _current_scene_transform is None to detect that no transformation
        # has happened yet
        self._current_scene_transform = None

    def update_current_scene_transformation(
        self,
        context: bpy.types.Context,
        scene_transform: core.SceneTransformations,
    ):
        assert self._tracker
        assert self._tracker.geometry
        assert self._tracker.camera
        assert context.scene

        geom = self._tracker.geometry
        camera = self._tracker.camera

        assert isinstance(camera.data, bpy.types.Camera)

        if self._tracker.tracking_target == "GEOMETRY":
            geom.matrix_world = mathutils.Matrix(
                typing.cast(typing.Sequence, scene_transform.model_matrix))
        else:
            camera.matrix_world = mathutils.Matrix(
                typing.cast(typing.Sequence,
                            scene_transform.view_matrix)).inverted()

        if self._tracker.pinmode_optimize_focal_length or self._tracker.pinmode_optimize_principal_point:
            core.set_camera_intrinsics(camera, scene_transform.intrinsics)

        self._current_scene_transform = scene_transform

    def insert_keyframe(self, context: bpy.types.Context):
        assert self._tracker
        assert context.scene

        camera = self._tracker.camera
        target_object = self._tracker.get_target_object()

        assert target_object
        assert camera
        assert isinstance(camera.data, bpy.types.Camera)

        current_frame = context.scene.frame_current

        keyframes.insert_keyframe(
            obj=target_object,
            frame=current_frame,
            data_paths=[
                "location", utils.get_rotation_data_path(target_object)
            ],
            keytype="KEYFRAME")

        # Insert camera intrinsic keyframes if needed
        if self._tracker.variable_focal_length or self._tracker.variable_principal_point:
            keyframes.insert_keyframe(
                obj=camera.data,
                frame=current_frame,
                data_paths=keyframes.CAMERA_DATAPATHS,
                keytype="KEYFRAME")

    def find_transformation(
        self,
        region: bpy.types.Region,
        rv3d: bpy.types.RegionView3D,
        region_x: int,
        region_y: int,
        trans_type: core.TransformationType,
        optimize_focal_length: bool,
        optimize_principal_point: bool,
    ) -> core.SceneTransformations:
        assert self._tracker
        assert self._tracker.camera
        assert self._tracker.geometry
        assert self._initial_scene_transform

        camera: bpy.types.Object = self._tracker.camera
        geom: bpy.types.Object = self._tracker.geometry
        pin_mode = self.get_pin_mode_data()

        projection_matrix = utils.calc_camera_proj_mat_pixels(camera)
        ndc_pos = utils.ndc(region, region_x, region_y)
        pos = projection_matrix @ rv3d.window_matrix.inverted(
        ) @ mathutils.Vector((ndc_pos[0], ndc_pos[1], 0.5, 1.0))
        pos = mathutils.Vector((pos[0] / pos[3], pos[1] / pos[3]))

        return core.find_transformation(
            object_points=pin_mode.points,
            initial_scene_transform=self._initial_scene_transform,
            current_scene_transform=core.SceneTransformations(
                model_matrix=typing.cast(np.ndarray, geom.matrix_world),
                view_matrix=typing.cast(
                    np.ndarray, camera.matrix_world.inverted()),
                intrinsics=core.camera_intrinsics(camera),
            ),
            update=core.PinUpdate(
                pin_idx=self._tracker.selected_pin_idx,
                pin_pos=typing.cast(np.ndarray, pos)),
            trans_type=trans_type,
            optimize_focal_length=optimize_focal_length,
            optimize_principal_point=optimize_principal_point,
        )

    def invoke(self, context: bpy.types.Context, event: bpy.types.Event) -> set:
        # General checks
        assert context.view_layer is not None
        assert context.area
        assert context.area.spaces.active
        assert context.space_data
        assert isinstance(context.space_data, bpy.types.SpaceView3D)
        assert context.space_data.region_3d
        assert context.region
        assert context.window_manager
        assert context.window_manager.keyconfigs.addon
        assert context.window_manager.keyconfigs.user

        global keymap
        global keymap_items
        global active_region_pointer

        active_region_pointer = context.region.as_pointer()

        state = PolychaseData.from_context(context)
        if not state:
            return {"CANCELLED"}

        global actually_in_pin_mode
        if state.in_pinmode or state.should_stop_pin_mode:
            if actually_in_pin_mode:
                state.should_stop_pin_mode = True
            else:
                state.should_stop_pin_mode = False
                state.in_pinmode = False
            return {"CANCELLED"}

        self._tracker = state.active_tracker
        if not self._tracker:
            return {"CANCELLED"}
        self._tracker_id = self._tracker.id

        pin_mode_data = self.get_pin_mode_data()
        pin_mode_data.unselect_pin()

        camera = self._tracker.camera
        geometry = self._tracker.geometry

        if not camera or not geometry:
            return {"CANCELLED"}

        space_view = context.space_data
        self._space_view_pointer = space_view.as_pointer()

        # Exit local view if we're in it
        if space_view.local_view:
            bpy.ops.view3d.localview()

        target_object = self._tracker.get_target_object()
        assert target_object

        # Go to object mode, and deselect all objects
        bpy.ops.object.mode_set(mode="OBJECT", toggle=False)
        bpy.ops.object.select_all(action="DESELECT")

        # Select camera, and switch to local view, so that all other objects are hidden.
        camera.select_set(True)

        # Enter local view
        bpy.ops.view3d.localview()

        # Deselect camera, and select target object
        camera.select_set(False)

        context.view_layer.objects.active = target_object
        target_object.select_set(True)

        # Hide objects and axes
        assert space_view.region_3d

        space_view.camera = camera
        space_view.region_3d.view_perspective = "CAMERA"

        # Create renderer object which will add a draw handler for rendering pins.
        self._renderer = rendering.PinModeRenderer(self._tracker_id)

        # Create mask selector, which will contain the logic for masking 3d polygons.
        self._mask_selector = masking_3d.Masking3DSelector(
            self._tracker, self._renderer, context)

        keymaps.setup_keymaps(context)

        # Listen to events
        context.window_manager.modal_handler_add(self)

        state.in_pinmode = True
        actually_in_pin_mode = True

        bpy.ops.ed.undo_push(message="Pinmode Start")
        return {"RUNNING_MODAL"}

    def find_clicked_pin(
        self,
        event: bpy.types.Event,
        geometry: bpy.types.Object,
        region: bpy.types.Region,
        rv3d: bpy.types.RegionView3D,
    ):
        assert self._tracker
        pin_mode_data: core.PinModeData = self.get_pin_mode_data()

        mouse_x, mouse_y = event.mouse_region_x, event.mouse_region_y
        object_to_world = geometry.matrix_world

        # TODO: Vectorize
        for idx, point in enumerate(pin_mode_data.points):
            point_2d = view3d_utils.location_3d_to_region_2d(
                region, rv3d, object_to_world @ mathutils.Vector(point))
            if not point_2d:
                continue
            dist_sq = (mouse_x - point_2d.x)**2 + (mouse_y - point_2d.y)**2
            if dist_sq < self._tracker.pin_radius**2:
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
        tracker_core = core.Tracker.get(self._tracker)
        assert tracker_core

        return tracker_core.ray_cast(region, rv3d, mouse_x, mouse_y, False)

    def select_pin(self, pin_idx: int):
        pin_mode_data: core.PinModeData = self.get_pin_mode_data()
        pin_mode_data.select_pin(pin_idx)
        bpy.ops.ed.undo_push(message="Pin Selected")

    def create_pin(self, location: np.ndarray):
        pin_mode_data: core.PinModeData = self.get_pin_mode_data()
        pin_mode_data.create_pin(location, select=True)
        bpy.ops.ed.undo_push(message="Pin Created")

    def delete_pin(self, pin_idx: int):
        pin_mode_data: core.PinModeData = self.get_pin_mode_data()
        pin_mode_data.delete_pin(pin_idx)
        bpy.ops.ed.undo_push(message="Pin Removed")

    def unselect_pin(self):
        pin_mode_data: core.PinModeData = self.get_pin_mode_data()
        pin_mode_data.unselect_pin()
        bpy.ops.ed.undo_push(message="Pin Unselected")

    def is_dragging_pin(self, event):
        assert self._tracker
        return event.type == "MOUSEMOVE" and self._is_left_mouse_clicked and self._tracker.selected_pin_idx >= 0

    def handle_left_mouse_release(self, context: bpy.types.Context):
        self._is_left_mouse_clicked = False
        self.insert_keyframe(context)
        if self._current_scene_transform is not None:
            bpy.ops.ed.undo_push(message="Transformation Stopped")

    def handle_apply_mask(
        self,
        context: bpy.types.Context,
        event: bpy.types.Event,
        clear=False,
    ):
        """Apply mask at mouse position using the mask selector."""
        assert self._mask_selector
        assert self._tracker
        assert self._tracker.geometry
        assert self._tracker.camera
        assert self._renderer

        camera = self._tracker.camera
        geometry = self._tracker.geometry

        # Apply mask using the dedicated mask selector
        success = self._mask_selector.apply_mask_at_position(
            context=context,
            event=event,
            camera=camera,
            geometry=geometry,
            selection_radius=self._tracker.mask_selection_radius,
            clear=clear,
        )

        if success:
            # Update wireframe rendering
            tracker_core = core.Tracker.get(self._tracker)
            assert tracker_core
            self._renderer.update_wireframe_mask(
                tracker_core.accel_mesh.inner().masked_triangles, context)

    def modal(self, context: bpy.types.Context, event: bpy.types.Event) -> set:
        # It's dangerous to keep self._tracker alive between invocations of modal,
        # since it might die, and cause blender to crash if accessed. So we reset it here.
        # FIXME: Maybe just don't hold self._tracker at all?
        self._tracker = PolychaseData.get_tracker_by_id(
            self._tracker_id, context)
        if not self._tracker:
            return self.cleanup(context)

        try:
            return self.modal_impl(context, event)
        except:
            traceback.print_exc()
            return self.cleanup(context)

    def is_event_in_region(
            self,
            area: bpy.types.Area,
            region: bpy.types.Region,
            event: bpy.types.Event):
        x, y = event.mouse_x, event.mouse_y

        if x < region.x or x > region.x + region.width or y < region.y or y > region.y + region.height:
            return False

        # Also check that we're not intersecting with any other region in the area except the region we're interested in.
        for other_region in area.regions:
            if other_region != region and \
                    x >= other_region.x and x < other_region.x + other_region.width and \
                    y >= other_region.y and y < other_region.y + other_region.height:
                return False

        return True

    def handle_mask_drawing_events(
            self, context: bpy.types.Context, event: bpy.types.Event):
        assert context.region
        assert context.area
        assert self._renderer

        region = context.region
        area = context.area

        if not self.is_event_in_region(area, region, event):
            return {"PASS_THROUGH"}

        # Update mouse position for rendering
        self._renderer.set_mouse_pos(
            (event.mouse_region_x, event.mouse_region_y))

        if event.type == "MOUSEMOVE":
            region.tag_redraw()

            if self._is_left_mouse_clicked:
                self.handle_apply_mask(context, event, clear=False)
                return {"RUNNING_MODAL"}

            elif self._is_right_mouse_clicked:
                self.handle_apply_mask(context, event, clear=True)
                return {"RUNNING_MODAL"}

        elif event.type == "LEFTMOUSE" and event.value == "PRESS":
            self._is_left_mouse_clicked = True
            self.handle_apply_mask(context, event, clear=False)
            return {"RUNNING_MODAL"}

        elif event.type == "LEFTMOUSE" and event.value == "RELEASE":
            self._is_left_mouse_clicked = False

        elif event.type == "RIGHTMOUSE" and event.value == "PRESS":
            self._is_right_mouse_clicked = True
            self.handle_apply_mask(context, event, clear=True)
            return {"RUNNING_MODAL"}

        elif event.type == "RIGHTMOUSE" and event.value == "RELEASE":
            self._is_right_mouse_clicked = False

        return {"PASS_THROUGH"}

    def handle_pin_manipulation_events(
            self, context: bpy.types.Context, event: bpy.types.Event):
        assert context.area
        assert context.region
        assert self._tracker
        assert self._tracker.geometry
        assert self._tracker.camera
        assert self._renderer
        assert isinstance(context.space_data, bpy.types.SpaceView3D)
        assert context.space_data.region_3d

        region = context.region
        rv3d = context.space_data.region_3d
        geometry = self._tracker.geometry
        area = context.area

        # If the user is dragging a pin, we don't care if he's still
        # in the region or not

        if self.is_dragging_pin(event):
            scene_transform = self.find_transformation(
                region=region,
                rv3d=rv3d,
                region_x=event.mouse_region_x,
                region_y=event.mouse_region_y,
                trans_type=core.TransformationType.Model
                if self._tracker.tracking_target == "GEOMETRY" else
                core.TransformationType.Camera,
                optimize_focal_length=self._tracker.
                pinmode_optimize_focal_length,
                optimize_principal_point=self._tracker.
                pinmode_optimize_principal_point,
            )
            self.update_current_scene_transformation(context, scene_transform)
            return {"RUNNING_MODAL"}

        elif event.type == "LEFTMOUSE" and event.value == "RELEASE":
            # It might not be true if the user clicked the mouse outside the region
            if self._is_left_mouse_clicked:
                self.handle_left_mouse_release(context)
                return {"RUNNING_MODAL"}

        # Now we check that the event happened in the region
        if not self.is_event_in_region(area, region, event):
            return {"PASS_THROUGH"}

        if event.type == "LEFTMOUSE" and event.value == "PRESS":
            self._is_left_mouse_clicked = True

            pin_idx = self.find_clicked_pin(event, geometry, region, rv3d)
            if pin_idx is not None:
                self.select_pin(pin_idx)
                self.update_initial_scene_transformation(rv3d)
                # FIXME: Find a way so that we don't recreate the batch every
                # time a selection is made
                self._renderer.update_pins(context)
                return {"RUNNING_MODAL"}

            rayhit = self.raycast(event, region, rv3d)
            if rayhit is not None:
                self.update_initial_scene_transformation(rv3d)
                self.create_pin(rayhit.pos)
                self._renderer.update_pins(context)
                return {"RUNNING_MODAL"}

            self.unselect_pin()
            self._renderer.update_pins(context)
            return {"RUNNING_MODAL"}

        elif event.type == "RIGHTMOUSE" and event.value == "PRESS":
            pin_idx = self.find_clicked_pin(event, geometry, region, rv3d)
            if pin_idx is not None:
                self.delete_pin(pin_idx)
                self._renderer.update_pins(context)
            return {"RUNNING_MODAL"}

        return {"PASS_THROUGH"}

    def reset_input_state(self, context: bpy.types.Context):
        assert context.region
        assert self._mask_selector

        if self._is_left_mouse_clicked and not self._is_drawing_3d_mask:
            self.handle_left_mouse_release(context)
        self._is_right_mouse_clicked = False
        self._is_drawing_3d_mask = False

        # Invalidate triangle buffer to force re-render
        self._mask_selector.invalidate_triangle_buffer()
        context.region.tag_redraw()

    def modal_impl(self, context: bpy.types.Context, event: bpy.types.Event):
        assert self._renderer

        state = PolychaseData.from_context(context)
        if not state or state.should_stop_pin_mode or not state.in_pinmode:
            return self.cleanup(context)

        if event.type == "ESC" and event.value == "PRESS":
            return self.cleanup(context)

        space_view = context.space_data
        if not space_view or space_view.as_pointer() != self._space_view_pointer or \
                not isinstance(space_view, bpy.types.SpaceView3D) or \
                not space_view.region_3d or space_view.region_3d.view_perspective != "CAMERA":
            return self.cleanup(context)

        if not self._tracker:
            return self.cleanup(context)

        if not context.scene:
            return self.cleanup(context)

        geometry = self._tracker.geometry
        camera = self._tracker.camera
        region = context.region
        area = context.area

        if not geometry or not camera:
            return self.cleanup(context)

        # This should never happen AFAIK.
        if not region or not area or not isinstance(context.space_data,
                                                    bpy.types.SpaceView3D):
            return self.cleanup(context)

        rv3d = context.space_data.region_3d

        # This should never happen AFAIK.
        if not rv3d:
            return self.cleanup(context)

        # Redraw if necessary
        # Version numbers may not match in case the user pressed Ctrl-Z for example.
        if self.get_pin_mode_data().is_out_of_date():
            self._renderer.update_pins(context)

        if event.type == "M" and event.value == "PRESS":
            is_drawing_3d_mask = self._is_drawing_3d_mask

            # Reset input state when switching modes
            self.reset_input_state(context)

            self._is_drawing_3d_mask = not is_drawing_3d_mask
            self._renderer.set_mouse_pos(
                (event.mouse_region_x, event.mouse_region_y))
            self._renderer.set_mask_mode(self._is_drawing_3d_mask, context)

            return {"RUNNING_MODAL"}

        if self._is_drawing_3d_mask:
            return self.handle_mask_drawing_events(context, event)
        else:
            return self.handle_pin_manipulation_events(context, event)

    def cleanup(self, context: bpy.types.Context):
        assert context.window_manager

        state = PolychaseData.from_context(context)

        if state:
            global actually_in_pin_mode
            state.in_pinmode = False
            actually_in_pin_mode = False
            state.should_stop_pin_mode = False

        if self._draw_handler:
            bpy.types.SpaceView3D.draw_handler_remove(
                self._draw_handler, "WINDOW")
            self._draw_handler = None

        if context.area:
            context.area.tag_redraw()

        keymaps.remove_keymaps(context)

        # Exit local view if we're in it
        space_view = context.space_data
        if space_view and space_view.as_pointer() == self._space_view_pointer and \
                isinstance(space_view, bpy.types.SpaceView3D) and space_view.local_view:
            bpy.ops.view3d.localview()

        bpy.ops.ed.undo_push(message="Pinmode End")

        if self._tracker:
            # Save mask
            tracker_core = core.Tracker.get(self._tracker)
            if tracker_core:
                self._tracker.masked_triangles = \
                        tracker_core.accel_mesh.inner().masked_triangles.tobytes()

            # Store object scale
            self._tracker.store_geom_cam_transform()

        if self._renderer:
            self._renderer.cleanup()

        if self._mask_selector:
            self._mask_selector.cleanup()

        return {"FINISHED"}

class PC_OT_ClearPins(bpy.types.Operator):
    bl_idname = "polychase.clear_pins"
    bl_options = {"REGISTER", "UNDO", "INTERNAL"}
    bl_label = "Clear Pins"

    def execute(self, context: bpy.types.Context) -> set:
        state = PolychaseData.from_context(context)
        if state is None:
            return {"CANCELLED"}
        tracker = state.active_tracker
        if not tracker:
            return {"CANCELLED"}
        
        tracker.points = b""
        tracker.points_version_number += 1
        tracker.selected_pin_idx = -1
        return {"FINISHED"}

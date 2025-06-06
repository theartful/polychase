import functools
import traceback
import typing

import bpy
import bpy.types
import gpu
import mathutils
import numpy as np
from bpy_extras.view3d_utils import location_3d_to_region_2d
from gpu_extras.batch import batch_for_shader

from .. import core, utils
from ..properties import PolychaseClipTracking, PolychaseData

# Storing blender objects might lead to crashes, so instead we store pointers
active_region_pointer: int = 0
keymap: bpy.types.KeyMap | None = None
keymap_items: list[bpy.types.KeyMapItem] = []
actually_in_pin_mode: bool = False


@functools.cache
def get_points_shader():
    shader_info = gpu.types.GPUShaderCreateInfo()
    shader_info.vertex_source(
        """
    void main()
    {
        gl_Position = mvp * vec4(position, 1.0f);
        is_selected_out = is_selected;
    }
    """)
    shader_info.fragment_source(
        """
    void main()
    {
        // Get coordinates in range [-1, 1], with (0,0) at center
        vec2 coord = gl_PointCoord * 2.0 - 1.0;
        float dist = dot(coord, coord); // Squared distance from center

        // Soft edge: anti-aliasing
        float alpha = smoothstep(1.0, 0.9, dist);

        if (dist > 1.0)
            discard; // Outside the circle

        if (is_selected_out == 1)
            fragColor = selected_color * alpha;
        else
            fragColor = default_color * alpha;
    }
    """)
    vert_out = gpu.types.GPUStageInterfaceInfo(
        "polychase_point_interface")    # type: ignore
    vert_out.flat("UINT", "is_selected_out")

    shader_info.vertex_in(0, "VEC3", "position")
    shader_info.vertex_in(1, "UINT", "is_selected")
    shader_info.vertex_out(vert_out)
    shader_info.fragment_out(0, "VEC4", "fragColor")
    shader_info.push_constant("MAT4", "mvp")
    shader_info.push_constant("VEC4", "default_color")
    shader_info.push_constant("VEC4", "selected_color")

    return gpu.shader.create_from_info(shader_info)


@functools.cache
def get_wireframe_shader():
    shader_info = gpu.types.GPUShaderCreateInfo()
    shader_info.vertex_source(
        """
    void main()
    {
        gl_Position = mvp * vec4(position, 1.0f);
        gl_Position.z += bias * gl_Position.w;
    }
    """)
    shader_info.fragment_source(
        """
    void main()
    {
        fragColor = color;
    }
    """)

    shader_info.vertex_in(0, "VEC3", "position")
    shader_info.fragment_out(0, "VEC4", "fragColor")
    shader_info.push_constant("MAT4", "mvp")
    shader_info.push_constant("VEC4", "color")
    shader_info.push_constant("FLOAT", "bias")

    return gpu.shader.create_from_info(shader_info)


class OT_KeymapFilter(bpy.types.Operator):
    bl_idname = "polychase.keymap_filter"
    bl_label = "Keymap Filter"
    bl_options = {"INTERNAL"}

    keymap_idx: bpy.props.IntProperty(default=-1)

    def execute(self, context) -> set:
        global keymap_items

        # This should never happen
        if self.keymap_idx < 0 or self.keymap_idx >= len(keymap_items):
            return {"PASS_THROUGH"}

        state = PolychaseData.from_context(context)
        active = state is not None and context.region is not None and \
                context.region.as_pointer() == active_region_pointer

        old_active = keymap_items[self.keymap_idx].active
        keymap_items[self.keymap_idx].active = active

        # Block the event if the keymap behavior changed at this instance.
        if old_active != active:
            return {'FINISHED'}
        else:
            return {"PASS_THROUGH"}


class OT_PinMode(bpy.types.Operator):
    bl_idname = "polychase.start_pinmode"
    bl_options = {"REGISTER", "INTERNAL"}
    bl_label = "Start Pin Mode"

    _tracker_id: int = -1
    _tracker: PolychaseClipTracking | None = None
    _draw_handler = None
    _pins_shader: gpu.types.GPUShader | None = None
    _wireframe_shader: gpu.types.GPUShader | None = None
    _pins_batch: gpu.types.GPUBatch | None = None
    _wireframe_batch: gpu.types.GPUBatch | None = None
    _wireframe_depth_batch: gpu.types.GPUBatch | None = None
    _point_radius = 15.0
    _is_left_mouse_clicked = False

    # FIXME: Storing a blender object is risky
    _space_view_pointer: int = 0
    _initial_scene_transform: core.SceneTransformations | None = None
    _current_scene_transform: core.SceneTransformations | None = None

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        state = PolychaseData.from_context(context)
        if state is None:
            return False
        tracker = state.active_tracker
        # Check if state exists and tracker is active
        return tracker is not None and tracker.camera is not None and tracker.geometry is not None

    def get_pin_mode_data(self):
        assert self._tracker
        tracker_core = core.Tracker.get(self._tracker)
        assert tracker_core

        return tracker_core.pin_mode

    def create_pins_batch(self):
        assert self._pins_shader

        pin_mode_data: core.PinModeData = self.get_pin_mode_data()
        points = pin_mode_data.points if len(pin_mode_data.points) > 0 else []
        is_selected = pin_mode_data.is_selected if len(
            pin_mode_data.is_selected) > 0 else []

        self._pins_batch = batch_for_shader(
            self._pins_shader,
            "POINTS",
            {
                "position": typing.cast(typing.Sequence, points),
                "is_selected": typing.cast(typing.Sequence, is_selected)
            })

    def create_wireframe_batches(self):
        assert self._wireframe_shader
        assert self._tracker

        tracker_core = core.Tracker.get(self._tracker)
        assert tracker_core

        self._wireframe_batch = batch_for_shader(
            self._wireframe_shader,
            "LINES",
            {
                "position":
                    typing.cast(
                        typing.Sequence,
                        tracker_core.accel_mesh.inner().vertices)
            },
            indices=typing.cast(
                typing.Sequence, tracker_core.edges_indices.ravel()))

        self._wireframe_depth_batch = batch_for_shader(
            self._wireframe_shader,
            "TRIS",
            {
                "position":
                    typing.cast(
                        typing.Sequence,
                        tracker_core.accel_mesh.inner().vertices)
            },
            indices=typing.cast(
                typing.Sequence,
                tracker_core.accel_mesh.inner().indices.ravel()))

    def _update_initial_scene_transformation(
            self, rv3d: bpy.types.RegionView3D):
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

    def _update_current_scene_transformation(
            self,
            context: bpy.types.Context,
            scene_transform: core.SceneTransformations):
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

    def _insert_keyframe(self, context: bpy.types.Context):
        assert self._tracker
        assert context.scene

        camera = self._tracker.camera
        target_object = self._tracker.get_target_object()

        assert target_object
        assert camera
        assert isinstance(camera.data, bpy.types.Camera)

        # Insert Keyframes
        target_object.rotation_mode = "QUATERNION"
        current_frame = context.scene.frame_current

        # FIXME? This is disgusting to be honest
        # I want to delete the keyframe so that if it already existed with a keytype GENERATED
        # it would change to KEYFRAME
        # Too lazy to use fcurves, and search through the keyframes.
        try:
            target_object.keyframe_delete(
                data_path="location", frame=current_frame)
        except:
            pass
        try:
            target_object.keyframe_delete(
                data_path="rotation_quaternion", frame=current_frame)
        except:
            pass

        target_object.keyframe_insert(
            data_path="location", frame=current_frame, keytype="KEYFRAME")
        target_object.keyframe_insert(
            data_path="rotation_quaternion",
            frame=current_frame,
            keytype="KEYFRAME")

        if self._tracker.pinmode_optimize_focal_length:
            try:
                target_object.keyframe_delete(
                    data_path="lens", frame=current_frame)
            except:
                pass
            camera.data.keyframe_insert(
                data_path="lens", frame=current_frame, keytype="KEYFRAME")

        if self._tracker.pinmode_optimize_principal_point:
            try:
                target_object.keyframe_delete(
                    data_path="shift_x", frame=current_frame)
            except:
                pass
            try:
                target_object.keyframe_delete(
                    data_path="shift_y", frame=current_frame)
            except:
                pass
            camera.data.keyframe_insert(
                data_path="shift_x", frame=current_frame, keytype="KEYFRAME")
            camera.data.keyframe_insert(
                data_path="shift_y", frame=current_frame, keytype="KEYFRAME")

    def _find_transformation(
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

    def draw_callback(self):
        if not bpy.context.region_data:
            return

        state = PolychaseData.from_context()
        if not state or not state.in_pinmode:
            return

        tracker = state.get_tracker_by_id(self._tracker_id)

        if not tracker or not tracker.geometry:
            return

        if not self._pins_shader or not self._pins_batch:
            return

        if not self._wireframe_shader or not self._wireframe_batch or not self._wireframe_depth_batch:
            return

        geometry = tracker.geometry
        mvp = bpy.context.region_data.perspective_matrix @ geometry.matrix_world

        # Prepare Z-buffer
        gpu.state.color_mask_set(False, False, False, False)
        gpu.state.depth_mask_set(True)
        gpu.state.depth_test_set("LESS_EQUAL")

        self._wireframe_shader.bind()
        self._wireframe_shader.uniform_float(
            "mvp", typing.cast(typing.Sequence, mvp))
        self._wireframe_shader.uniform_float("bias", 0)
        self._wireframe_shader.uniform_float(
            "color",
            [0.0, 0.0, 0.0, 0.0],
        )
        self._wireframe_depth_batch.draw(self._wireframe_shader)

        # Draw wireframe
        gpu.state.color_mask_set(True, True, True, True)
        gpu.state.depth_mask_set(False)
        gpu.state.depth_test_set("LESS_EQUAL")
        gpu.state.line_width_set(tracker.wireframe_width)

        self._wireframe_shader.uniform_float(
            "mvp", typing.cast(typing.Sequence, mvp))
        self._wireframe_shader.uniform_float(
            "bias", -1e-4)    # Prevent Z-fighting
        self._wireframe_shader.uniform_float("color", tracker.wireframe_color)
        self._wireframe_batch.draw(self._wireframe_shader)

        # Draw pins
        gpu.state.blend_set("ALPHA")
        gpu.state.point_size_set(tracker.pin_radius)
        gpu.state.depth_test_set("NONE")

        self._pins_shader.bind()
        self._pins_shader.uniform_float(
            "mvp", typing.cast(typing.Sequence, mvp))
        self._pins_shader.uniform_float(
            "default_color", tracker.default_pin_color)
        self._pins_shader.uniform_float(
            "selected_color", tracker.selected_pin_color)
        gpu.state.depth_mask_set(False)
        self._pins_batch.draw(self._pins_shader)

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

        # Go to object mode, and deselect all objects
        bpy.ops.object.select_all(action="DESELECT")
        camera.hide_set(False)
        geometry.hide_set(False)

        # Select camera, and switch to local view, so that all other objects are hidden.
        camera.select_set(True)

        # Enter local view
        bpy.ops.view3d.localview()

        # Select target object, and switch to object mode
        target_object = self._tracker.get_target_object()
        assert target_object

        context.view_layer.objects.active = target_object
        target_object.select_set(True)
        bpy.ops.object.mode_set(mode="OBJECT", toggle=False)

        # Hide objects and axes
        assert space_view.region_3d

        space_view.camera = camera
        space_view.region_3d.view_perspective = "CAMERA"

        # Add a draw handler for rendering pins.
        self._pins_shader = get_points_shader()
        self._wireframe_shader = get_wireframe_shader()
        self.create_pins_batch()
        self.create_wireframe_batches()

        self._draw_handler = bpy.types.SpaceView3D.draw_handler_add(
            typing.cast(typing.Callable, self.draw_callback), (),
            "WINDOW",
            "POST_PIXEL")

        # Lock rotation
        # Strategy: Find all keymaps under "3D View" that perform action "view3d.rotate", and replace it with "view3d.move"
        # But add a filter before each view3d.move that checks that we're in the right region.
        keyconfigs_user = context.window_manager.keyconfigs.user
        current_keymaps = keyconfigs_user.keymaps.get("3D View")
        assert current_keymaps

        keyconfigs_addon = context.window_manager.keyconfigs.addon
        keymap = keyconfigs_addon.keymaps.new(
            name="3D View", space_type="VIEW_3D")
        keymap_items = []

        for keymap_item in current_keymaps.keymap_items:
            if keymap_item.idname != "view3d.rotate":
                continue

            filter_keymap_item = keymap.keymap_items.new(
                idname=OT_KeymapFilter.bl_idname,
                type=keymap_item.type,
                value=keymap_item.value,
                any=keymap_item.any,
                shift=keymap_item.shift,
                ctrl=keymap_item.ctrl,
                alt=keymap_item.alt,
                oskey=keymap_item.oskey,
                key_modifier=keymap_item.key_modifier,
                direction=keymap_item.direction,
                repeat=keymap_item.repeat,
                head=True,
            )

            keymap_item = keymap.keymap_items.new(
                idname="view3d.move",
                type=keymap_item.type,
                value=keymap_item.value,
                any=keymap_item.any,
                shift=keymap_item.shift,
                ctrl=keymap_item.ctrl,
                alt=keymap_item.alt,
                oskey=keymap_item.oskey,
                key_modifier=keymap_item.key_modifier,
                direction=keymap_item.direction,
                repeat=keymap_item.repeat,
                head=True,
            )
            op_props = typing.cast(
                OT_KeymapFilter, filter_keymap_item.properties)
            op_props.keymap_idx = len(keymap_items)

            keymap_items.append(keymap_item)
            keymap_items.append(filter_keymap_item)

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
        pin_mode_data: core.PinModeData = self.get_pin_mode_data()

        mouse_x, mouse_y = event.mouse_region_x, event.mouse_region_y
        object_to_world = geometry.matrix_world

        # TODO: Vectorize
        for idx, point in enumerate(pin_mode_data.points):
            screen_point = location_3d_to_region_2d(
                region, rv3d, object_to_world @ mathutils.Vector(point))
            if not screen_point:
                continue
            dist = (
                (mouse_x - screen_point.x)**2 +
                (mouse_y - screen_point.y)**2)**0.5
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
        tracker_core = core.Tracker.get(self._tracker)
        assert tracker_core

        result = tracker_core.ray_cast(region, rv3d, mouse_x, mouse_y)

        return result.pos if result else None

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

    def redraw_pins(self, context):
        # This is horrible
        self.create_pins_batch()
        context.area.tag_redraw()

    def is_dragging_pin(self, event):
        assert self._tracker
        return event.type == "MOUSEMOVE" and self._is_left_mouse_clicked and self._tracker.selected_pin_idx >= 0

    def modal(self, context: bpy.types.Context, event: bpy.types.Event) -> set:
        # It's dangerous to keep self._tracker alive between invocations of modal,
        # since it might die, and cause blender to crash if accessed. So we reset it here.
        # FIXME: Maybe just don't hold self._tracker at all?
        state = PolychaseData.from_context(context)
        if not state:
            return self._cleanup(context)
        self._tracker = state.get_tracker_by_id(self._tracker_id)
        if not self._tracker:
            return self._cleanup(context)

        try:
            return self.modal_impl(context, event)
        except:
            traceback.print_exc()
            return self._cleanup(context)

    def _is_event_in_region(
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

    def modal_impl(self, context: bpy.types.Context, event: bpy.types.Event):
        state = PolychaseData.from_context(context)
        if not state or state.should_stop_pin_mode or not state.in_pinmode:
            return self._cleanup(context)

        if event.type == "ESC":
            return self._cleanup(context)

        space_view = context.space_data
        if not space_view or space_view.as_pointer() != self._space_view_pointer or \
                not isinstance(space_view, bpy.types.SpaceView3D) or \
                not space_view.region_3d or space_view.region_3d.view_perspective != "CAMERA":
            return self._cleanup(context)

        if not self._tracker:
            return self._cleanup(context)

        if not context.scene:
            return self._cleanup(context)

        geometry = self._tracker.geometry
        camera = self._tracker.camera
        region = context.region
        area = context.area

        if not geometry or not camera:
            return self._cleanup(context)

        # This should never happen AFAIK.
        if not region or not area or not isinstance(context.space_data,
                                                    bpy.types.SpaceView3D):
            return self._cleanup(context)

        rv3d = context.space_data.region_3d

        # This should never happen AFAIK.
        if not rv3d:
            return self._cleanup(context)

        if not self._is_event_in_region(area, region, event):
            return {"PASS_THROUGH"}

        # Redraw if necessary
        # Version numbers may not match in case the user pressed Ctrl-Z for example.
        if self.get_pin_mode_data(
        )._points_version_number != self._tracker.points_version_number:
            self.redraw_pins(context)

        if event.type == "LEFTMOUSE" and event.value == "PRESS":
            self._is_left_mouse_clicked = True

            pin_idx = self.find_clicked_pin(event, geometry, region, rv3d)
            if pin_idx is not None:
                self.select_pin(pin_idx)
                self._update_initial_scene_transformation(rv3d)
                # FIXME: Find a way so that we don't recreate the batch every
                # time a selection is made
                self.redraw_pins(context)
                return {"RUNNING_MODAL"}

            location = self.raycast(event, region, rv3d)
            if location is not None:
                self.create_pin(location)
                self._update_initial_scene_transformation(rv3d)
                self.redraw_pins(context)
                return {"RUNNING_MODAL"}

            self.unselect_pin()
            self.redraw_pins(context)
            return {"RUNNING_MODAL"}

        elif event.type == "RIGHTMOUSE" and event.value == "PRESS":
            pin_idx = self.find_clicked_pin(event, geometry, region, rv3d)
            if pin_idx is not None:
                self.delete_pin(pin_idx)
                self.redraw_pins(context)
            return {"RUNNING_MODAL"}

        elif event.type == "LEFTMOUSE" and event.value == "RELEASE":
            self._is_left_mouse_clicked = False
            self._insert_keyframe(context)
            if self._current_scene_transform is not None:
                bpy.ops.ed.undo_push(message="Transformation Stopped")

        elif self.is_dragging_pin(event):
            scene_transform = self._find_transformation(
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
            self._update_current_scene_transformation(context, scene_transform)
            return {"RUNNING_MODAL"}

        return {"PASS_THROUGH"}

    def _cleanup(self, context: bpy.types.Context):
        assert context.window_manager

        state = PolychaseData.from_context(context)
        assert state

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

        keyconfigs_addon = context.window_manager.keyconfigs.addon
        assert keyconfigs_addon

        global keymap
        global keymap_items
        if keymap:
            for item in keymap_items:
                keymap.keymap_items.remove(item)

        keymap = None
        keymap_items = []

        # Exit local view if we're in it
        space_view = context.space_data
        if space_view and space_view.as_pointer() == self._space_view_pointer and \
                isinstance(space_view, bpy.types.SpaceView3D) and space_view.local_view:
            bpy.ops.view3d.localview()

        bpy.ops.ed.undo_push(message="Pinmode End")
        return {"FINISHED"}

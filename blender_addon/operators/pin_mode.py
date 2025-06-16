import functools
import traceback
import typing

import bpy
import bpy.types
import gpu
import mathutils
import numpy as np
import bpy_extras.view3d_utils as view3d_utils
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
        if (!useMask) {
            fragColor = color;
        } else {
            int vec_idx = gl_PrimitiveID / 128;
            int component_idx = (gl_PrimitiveID % 128) / 32;
            int bit_idx = (gl_PrimitiveID % 128) % 32;

            bool is_masked =
                (u_maskData.data[vec_idx][component_idx] & (1u << bit_idx)) != 0;

            if (is_masked) {
                fragColor = mask_color;
            } else {
                fragColor = color;
            }
        }
    }
    """)

    shader_info.typedef_source("struct MaskData { uvec4 data[1024]; };")
    shader_info.vertex_in(0, "VEC3", "position")
    shader_info.fragment_out(0, "VEC4", "fragColor")
    shader_info.push_constant("MAT4", "mvp")
    shader_info.push_constant("VEC4", "color")
    shader_info.push_constant("VEC4", "mask_color")
    shader_info.push_constant("FLOAT", "bias")
    shader_info.push_constant("BOOL", "useMask")
    shader_info.uniform_buf(0, "MaskData", "u_maskData")

    return gpu.shader.create_from_info(shader_info)


@functools.cache
def get_selection_circle_shader():
    shader_info = gpu.types.GPUShaderCreateInfo()
    shader_info.vertex_source(
        """
    void main()
    {
        gl_Position = vec4(position, 0.0, 1.0);
    }
    """)
    shader_info.fragment_source(
        """
    void main()
    {
        const float width = 1.5f;
        float d = abs(distance(vec2(gl_FragCoord), center) - radius);
        if (d < width) {
            fragColor = vec4(1.0, 1.0, 1.0, 1.0 - d/width);
        } else {
            discard;
        }
    }
    """)

    shader_info.vertex_in(0, "VEC2", "position")
    shader_info.push_constant("VEC2", "center")
    shader_info.push_constant("FLOAT", "radius")
    shader_info.fragment_out(0, "VEC4", "fragColor")

    return gpu.shader.create_from_info(shader_info)


@functools.cache
def get_triangle_idx_shader():
    shader_info = gpu.types.GPUShaderCreateInfo()
    shader_info.vertex_source(
        """
    void main()
    {
        gl_Position = mvp * vec4(position, 1.0f);
    }
    """)
    shader_info.fragment_source(
        """
    void main()
    {
        uint b1 =  gl_PrimitiveID & 0x000000FF;
        uint b2 = (gl_PrimitiveID & 0x0000FF00) >> 8;
        uint b3 = (gl_PrimitiveID & 0x00FF0000) >> 16;
        uint b4 = (gl_PrimitiveID & 0xFF000000) >> 24;

        fragColor = vec4(b1/255.0f, b2/255.0f, b3/255.0f, b4/255.0f);
    }
    """)

    shader_info.vertex_in(0, "VEC3", "position")
    shader_info.fragment_out(0, "VEC4", "fragColor")
    shader_info.push_constant("MAT4", "mvp")

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
            return {"FINISHED"}
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
    _wireframe_mask_ubo: gpu.types.GPUUniformBuf | None = None
    _is_left_mouse_clicked = False
    _is_right_mouse_clicked = False

    _space_view_pointer: int = 0
    _initial_scene_transform: core.SceneTransformations | None = None
    _current_scene_transform: core.SceneTransformations | None = None

    # For masking polygons
    _is_drawing_3d_mask: bool = False
    _selection_circle_shader: gpu.types.GPUShader | None = None
    _selection_circle_batch: gpu.types.GPUBatch | None = None
    _mouse_pos: tuple[float, float] = (0, 0)

    # For storing raytracing results using rasterization
    _triangle_idx_shader: gpu.types.GPUShader | None = None
    _triangle_idx_batch: gpu.types.GPUBatch | None = None
    _offscreen: gpu.types.GPUOffScreen | None = None
    _offscreen_buffer: gpu.types.Buffer | None = None
    _triangle_buffer_frame: int | None = None

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

    def create_pins_batch(self):
        assert self._pins_shader

        pin_mode_data = self.get_pin_mode_data()
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
            indices=tracker_core.edges_indices.ravel())

        self._wireframe_depth_batch = batch_for_shader(
            self._wireframe_shader,
            "TRIS",
            {
                "position":
                    typing.cast(
                        typing.Sequence,
                        tracker_core.accel_mesh.inner().vertices)
            },
            indices=tracker_core.accel_mesh.inner().triangles.ravel())

        self._wireframe_mask_ubo = gpu.types.GPUUniformBuf(
            tracker_core.accel_mesh.inner().masked_triangles)    # type: ignore

    def create_triangle_idx_batch(self):
        assert self._wireframe_depth_batch
        self._triangle_idx_batch = self._wireframe_depth_batch

    def create_selection_circle_batch(self):
        assert self._selection_circle_shader

        self._selection_circle_batch = batch_for_shader(
            self._selection_circle_shader,
            "TRIS",
            {"position": [[-1.0, -1.0], [3.0, -1.0], [-1.0, 3.0]]},
            indices=np.array([0, 1, 2], dtype=np.uint32),
        )

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

    def _insert_keyframe(self, context: bpy.types.Context):
        assert self._tracker
        assert context.scene

        camera = self._tracker.camera
        target_object = self._tracker.get_target_object()

        assert target_object
        assert camera
        assert isinstance(camera.data, bpy.types.Camera)

        # Insert Keyframes
        current_frame = context.scene.frame_current
        rotation_data_path = utils.get_rotation_data_path(target_object)

        # FIXME? This is disgusting to be honest
        # I want to delete the keyframe so that if it already existed with
        # a keytype GENERATED it would change to KEYFRAME.
        # Too lazy to use fcurves, and search through the keyframes.
        try:
            target_object.keyframe_delete(
                data_path="location", frame=current_frame)
        except:
            pass
        try:
            target_object.keyframe_delete(
                data_path=rotation_data_path, frame=current_frame)
        except:
            pass

        target_object.keyframe_insert(
            data_path="location", frame=current_frame, keytype="KEYFRAME")
        target_object.keyframe_insert(
            data_path=rotation_data_path,
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

        # Draw solid geometry to prepare Z-buffer, and highlight masked polygons
        gpu.state.depth_mask_set(True)
        gpu.state.depth_test_set("LESS_EQUAL")
        gpu.state.blend_set("ALPHA")

        self._wireframe_shader.bind()
        self._wireframe_shader.uniform_float(
            "mvp", typing.cast(typing.Sequence, mvp))
        self._wireframe_shader.uniform_float("bias", 0)
        self._wireframe_shader.uniform_float(
            "color",
            [0.0, 0.0, 0.0, 0.0],
        )
        self._wireframe_shader.uniform_float("mask_color", tracker.mask_color)
        self._wireframe_shader.uniform_bool("useMask", True)
        self._wireframe_shader.uniform_block(
            "u_maskData", self._wireframe_mask_ubo)
        self._wireframe_depth_batch.draw(self._wireframe_shader)

        # Draw wireframe
        gpu.state.depth_mask_set(False)
        gpu.state.depth_test_set("LESS_EQUAL")
        gpu.state.line_width_set(tracker.wireframe_width)

        self._wireframe_shader.uniform_float(
            "bias", -1e-4)    # Prevent Z-fighting
        self._wireframe_shader.uniform_float("color", tracker.wireframe_color)
        self._wireframe_shader.uniform_bool("useMask", False)
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

        if self._is_drawing_3d_mask:
            assert self._selection_circle_shader
            assert self._selection_circle_batch

            self._selection_circle_shader.bind()
            self._selection_circle_shader.uniform_float(
                "center", self._mouse_pos)
            self._selection_circle_shader.uniform_float(
                "radius", tracker.mask_selection_radius)
            self._selection_circle_batch.draw(self._selection_circle_shader)

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

        # Deselect camera, and select target object, and switch to object mode
        camera.select_set(False)

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
        self._selection_circle_shader = get_selection_circle_shader()
        self._triangle_idx_shader = get_triangle_idx_shader()
        self.create_pins_batch()
        self.create_wireframe_batches()
        self.create_selection_circle_batch()
        self.create_triangle_idx_batch()

        if self._tracker.clip:
            w, h = self._tracker.clip.size
        else:
            assert context.scene
            w, h = context.scene.render.resolution_x, context.scene.render.resolution_y

        self._offscreen = gpu.types.GPUOffScreen(w, h)    # type: ignore
        self._triangle_buffer_frame = None

        self._draw_handler = bpy.types.SpaceView3D.draw_handler_add(
            typing.cast(typing.Callable, self.draw_callback), (),
            "WINDOW",
            "POST_PIXEL")

        # Lock rotation
        # Strategy: Find all keymaps under "3D View" that perform action "view3d.rotate",
        # and replace it with "view3d.move".
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

    def redraw_pins(self, context):
        # This is horrible
        self.create_pins_batch()
        context.area.tag_redraw()

    def is_dragging_pin(self, event):
        assert self._tracker
        return event.type == "MOUSEMOVE" and self._is_left_mouse_clicked and self._tracker.selected_pin_idx >= 0

    def handle_left_mouse_release(self, context: bpy.types.Context):
        self._is_left_mouse_clicked = False
        self._insert_keyframe(context)
        if self._current_scene_transform is not None:
            bpy.ops.ed.undo_push(message="Transformation Stopped")

    def handle_apply_mask(self, context: bpy.types.Context, clear=False):
        assert self._offscreen
        assert self._triangle_idx_shader
        assert self._triangle_idx_batch
        assert self._tracker
        assert self._tracker.geometry
        assert self._tracker.camera
        assert context.scene
        assert context.region
        assert isinstance(context.space_data, bpy.types.SpaceView3D)
        assert context.space_data.region_3d
        assert self._wireframe_mask_ubo

        region = context.region
        camera = self._tracker.camera
        geometry = self._tracker.geometry
        rv3d = context.space_data.region_3d

        w, h = self._offscreen.width, self._offscreen.height

        depsgraph = bpy.context.evaluated_depsgraph_get()
        proj_matrix = camera.calc_matrix_camera(depsgraph, x=w, y=h)

        if self._triangle_buffer_frame != context.scene.frame_current:
            with self._offscreen.bind():
                gpu.state.color_mask_set(True, True, True, True)
                gpu.state.depth_mask_set(True)
                gpu.state.depth_test_set("LESS_EQUAL")

                fb = gpu.state.active_framebuffer_get()    # type: ignore
                fb.clear(color=(1.0, 1.0, 1.0), depth=1.0)

                model_matrix = geometry.matrix_world
                view_matrix = camera.matrix_world.inverted_safe()

                mvp = typing.cast(
                    typing.Sequence[float],
                    proj_matrix @ view_matrix @ model_matrix)

                self._triangle_idx_shader.bind()
                self._triangle_idx_shader.uniform_float("mvp", mvp)
                self._triangle_idx_batch.draw(self._triangle_idx_shader)

                if self._offscreen_buffer is None:
                    self._offscreen_buffer = gpu.types.Buffer(
                        "UBYTE", w * h * 4)    # type: ignore

                fb.read_color(
                    0, 0, w, h, 4, 0, "UBYTE", data=self._offscreen_buffer)

            self._triangle_buffer_frame = context.scene.frame_current

        assert self._offscreen_buffer
        pixels = np.frombuffer(
            typing.cast(bytes, self._offscreen_buffer),
            dtype=np.uint32).reshape((h, w))

        out = mathutils.Vector(
            (
                (2.0 * self._mouse_pos[0] / region.width) - 1.0,
                (2.0 * self._mouse_pos[1] / region.height) - 1.0,
                0.0,
                1.0,
            ))

        pos_ndc = proj_matrix @ rv3d.window_matrix.inverted() @ out
        pos_ndc /= pos_ndc.w
        pos_ndc = pos_ndc.to_2d()
        x, y = round(0.5 * (pos_ndc.x + 1.0) * w), round(0.5 * (pos_ndc.y + 1.0) * h)

        half_radius = round(self._tracker.mask_selection_radius / 2)
        min_x = max(x - half_radius, 0)
        max_x = min(x + half_radius, w)
        min_y = max(y - half_radius, 0)
        max_y = min(y + half_radius, h)

        # FIXME: We're masking a square instead of a triangle, but maybe good enough
        triangle_indices = pixels[min_y:max_y, min_x:max_x]

        # FIXME: This can be made much faster
        tracker_core = core.Tracker.get(self._tracker)
        assert tracker_core

        for triangle_idx in triangle_indices.ravel():
            if triangle_idx != 0xFFFFFFFF:
                if not clear:
                    tracker_core.set_polygon_mask_using_triangle_idx(
                        int(triangle_idx))
                else:
                    tracker_core.clear_polygon_mask_using_triangle_idx(
                        int(triangle_idx))

        self._wireframe_mask_ubo.update(
            tracker_core.accel_mesh.inner().masked_triangles)

        region.tag_redraw()

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
        if self.get_pin_mode_data().is_out_of_date():
            self.redraw_pins(context)

        if event.type == "M" and event.value == "PRESS":
            # TODO: Maybe create a function called "reset_state", that handles
            # resetting the state if left/right mouse is clicked
            if self._is_left_mouse_clicked:
                self.handle_left_mouse_release(context)
            self._is_right_mouse_clicked = False

            self._is_drawing_3d_mask = not self._is_drawing_3d_mask
            self._triangle_buffer_frame = None
            self._mouse_pos = (event.mouse_region_x, event.mouse_region_y)
            region.tag_redraw()

            if not self._is_drawing_3d_mask:
                # Save mask
                tracker_core = core.Tracker.get(self._tracker)
                assert tracker_core
                self._tracker.masked_triangles = tracker_core.accel_mesh.inner(
                ).masked_triangles.tobytes()

            return {"RUNNING_MODAL"}

        if self._is_drawing_3d_mask:
            self._mouse_pos = (event.mouse_region_x, event.mouse_region_y)

            if event.type == "MOUSEMOVE":
                region.tag_redraw()

                if self._is_left_mouse_clicked:
                    self.handle_apply_mask(context, clear=False)
                    return {"RUNNING_MODAL"}

                elif self._is_right_mouse_clicked:
                    self.handle_apply_mask(context, clear=True)
                    return {"RUNNING_MODAL"}

            elif event.type == "LEFTMOUSE" and event.value == "PRESS":
                assert self._offscreen
                assert self._triangle_idx_shader

                self._is_left_mouse_clicked = True
                self.handle_apply_mask(context, clear=False)
                return {"RUNNING_MODAL"}

            elif event.type == "LEFTMOUSE" and event.value == "RELEASE":
                self._is_left_mouse_clicked = False

            elif event.type == "RIGHTMOUSE" and event.value == "PRESS":
                assert self._offscreen
                assert self._triangle_idx_shader

                self._is_right_mouse_clicked = True
                self.handle_apply_mask(context, clear=True)
                return {"RUNNING_MODAL"}

            elif event.type == "RIGHTMOUSE" and event.value == "RELEASE":
                self._is_right_mouse_clicked = False

            return {"PASS_THROUGH"}

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

            rayhit = self.raycast(event, region, rv3d)
            if rayhit is not None:
                self._update_initial_scene_transformation(rv3d)
                self.create_pin(rayhit.pos)
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
            self.handle_left_mouse_release(context)

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

        # Save mask
        if self._tracker:
            tracker_core = core.Tracker.get(self._tracker)
            if tracker_core:
                self._tracker.masked_triangles = tracker_core.accel_mesh.inner(
                ).masked_triangles.tobytes()

        return {"FINISHED"}

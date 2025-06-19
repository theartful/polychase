import functools
import typing

import bpy
import gpu
import mathutils
import numpy as np

from ... import core
from ...properties import PolychaseData, PolychaseTracker
from . import rendering


@functools.cache
def get_triangle_idx_shader() -> gpu.types.GPUShader:
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


class Masking3DSelector:

    def __init__(
        self,
        tracker: PolychaseTracker,
        renderer: rendering.PinModeRenderer,
        context: bpy.types.Context,
    ):
        if tracker.clip:
            w, h = tracker.clip.size
        else:
            assert context.scene
            w, h = context.scene.render.resolution_x, context.scene.render.resolution_y

        self.tracker_id = tracker.id
        self.width = w
        self.height = h

        # GPU resources
        self._triangle_idx_shader = get_triangle_idx_shader()
        self._triangle_idx_batch = renderer.wireframe_depth_batch
        self._offscreen = gpu.types.GPUOffScreen(
            self.width, self.height)    # type: ignore
        self._offscreen_buffer: gpu.types.Buffer | None = None
        self._triangle_buffer_frame: int | None = None

    def _render_triangle_ids(
            self,
            context: bpy.types.Context,
            camera: bpy.types.Object,
            geometry: bpy.types.Object):
        if not self._triangle_idx_batch or not context.scene:
            return

        # Only update buffer if frame changed
        if self._triangle_buffer_frame == context.scene.frame_current:
            return

        depsgraph = bpy.context.evaluated_depsgraph_get()
        proj_matrix = camera.calc_matrix_camera(
            depsgraph, x=self.width, y=self.height)

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
                    "UBYTE", self.width * self.height * 4)    # type: ignore

            fb.read_color(
                0,
                0,
                self.width,
                self.height,
                4,
                0,
                "UBYTE",
                data=self._offscreen_buffer)

        self._triangle_buffer_frame = context.scene.frame_current

    def _get_triangle_indices_at_position(
            self,
            context: bpy.types.Context,
            event: bpy.types.Event,
            camera: bpy.types.Object,
            selection_radius: float) -> np.ndarray:
        """Get triangle indices at mouse position within selection radius."""
        if not self._offscreen_buffer or not context.region:
            return np.array([], dtype=np.uint32)

        assert isinstance(context.space_data, bpy.types.SpaceView3D)
        assert context.space_data.region_3d

        region = context.region
        rv3d = context.space_data.region_3d

        # Convert pixels to uint32 triangle IDs
        pixels = np.frombuffer(
            typing.cast(bytes, self._offscreen_buffer),
            dtype=np.uint32).reshape((self.height, self.width))

        # Convert mouse position to texture coordinates
        depsgraph = bpy.context.evaluated_depsgraph_get()
        proj_matrix = camera.calc_matrix_camera(
            depsgraph, x=self.width, y=self.height)

        out = mathutils.Vector(
            (
                (2.0 * event.mouse_region_x / region.width) - 1.0,
                (2.0 * event.mouse_region_y / region.height) - 1.0,
                0.0,
                1.0,
            ))

        pos_ndc = proj_matrix @ rv3d.window_matrix.inverted() @ out
        pos_ndc /= pos_ndc.w
        pos_ndc = pos_ndc.to_2d()

        x = round(0.5 * (pos_ndc.x + 1.0) * self.width)
        y = round(0.5 * (pos_ndc.y + 1.0) * self.height)

        # Get triangles within selection radius
        half_radius = round(selection_radius / 2)
        min_x = max(x - half_radius, 0)
        max_x = min(x + half_radius, self.width)
        min_y = max(y - half_radius, 0)
        max_y = min(y + half_radius, self.height)

        triangle_indices = pixels[min_y:max_y, min_x:max_x]

        return triangle_indices.ravel()

    def apply_mask_at_position(
            self,
            context: bpy.types.Context,
            event: bpy.types.Event,
            camera: bpy.types.Object,
            geometry: bpy.types.Object,
            selection_radius: float,
            clear: bool = False) -> bool:
        if not self._triangle_idx_batch:
            return False

        # Render triangle IDs to offscreen buffer
        self._render_triangle_ids(context, camera, geometry)

        # Get triangle indices at mouse position
        triangle_indices = self._get_triangle_indices_at_position(
            context, event, camera, selection_radius)

        if len(triangle_indices) == 0:
            return False

        # Apply mask to triangles
        tracker = PolychaseData.get_tracker_by_id(self.tracker_id, context)
        if not tracker:
            return False

        tracker_core = core.Tracker.get(tracker)
        if not tracker_core:
            return False

        for triangle_idx in triangle_indices:
            if triangle_idx != 0xFFFFFFFF:
                if not clear:
                    tracker_core.set_polygon_mask_using_triangle_idx(
                        int(triangle_idx))
                else:
                    tracker_core.clear_polygon_mask_using_triangle_idx(
                        int(triangle_idx))

        return True

    def invalidate_triangle_buffer(self):
        self._triangle_buffer_frame = None

    def cleanup(self):
        self._triangle_idx_batch = None
        self._offscreen_buffer = None

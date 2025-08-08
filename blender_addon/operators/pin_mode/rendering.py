# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2025 Ahmed Essam <aessam.dahy@gmail.com>

import functools
import typing

import bpy
import gpu
import numpy as np
from gpu_extras.batch import batch_for_shader

from ... import core, properties


@functools.cache
def get_points_shader() -> gpu.types.GPUShader:
    shader_info = gpu.types.GPUShaderCreateInfo()
    shader_info.vertex_source(
        """
    void main()
    {
        gl_Position = mvp * vec4(position, 1.0f);
        gl_PointSize = point_size;
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
    shader_info.push_constant("FLOAT", "point_size")
    shader_info.push_constant("MAT4", "mvp")
    shader_info.push_constant("VEC4", "default_color")
    shader_info.push_constant("VEC4", "selected_color")

    return gpu.shader.create_from_info(shader_info)


@functools.cache
def get_wireframe_shader() -> gpu.types.GPUShader:
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
def get_selection_circle_shader() -> gpu.types.GPUShader:
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


class PinModeRenderer:

    def __init__(self, context: bpy.types.Context, tracker_id: int):
        assert context.region

        tracker = properties.PolychaseState.get_tracker_by_id(tracker_id)
        assert tracker
        tracker_core = core.Tracker.get(tracker)
        assert tracker_core

        self.region_pointer = context.region.as_pointer()
        self.tracker_id: int = tracker_id
        self.pins_shader = get_points_shader()
        self.wireframe_shader = get_wireframe_shader()
        self.selection_circle_shader = get_selection_circle_shader()

        self.pins_batch = self._create_pins_batch(self.pins_shader)
        self.wireframe_batch = self._create_wireframe_batch(tracker_core)
        self.wireframe_depth_batch = self._create_wireframe_depth_batch(
            tracker_core)
        self.wireframe_mask_ubo = self._create_wireframe_mask_ubo(tracker_core)
        self.selection_circle_batch = self._create_selection_circle_batch(
            self.selection_circle_shader)
        self.triangle_idx_batch = None
        self.mask_mode = False
        self.mouse_pos = (0, 0)

        self.draw_handler = bpy.types.SpaceView3D.draw_handler_add(
            typing.cast(typing.Callable, self._draw_callback), (),
            "WINDOW",
            "POST_PIXEL")

    def set_mask_mode(self, val: bool, context: bpy.types.Context):
        assert context.region
        self.mask_mode = val
        context.region.tag_redraw()

    def set_mouse_pos(self, mouse_pos: tuple[float, float]):
        self.mouse_pos = mouse_pos

    def update_wireframe_mask(self, mask_data, context: bpy.types.Context):
        assert context.region
        self.wireframe_mask_ubo.update(mask_data)
        context.region.tag_redraw()

    def cleanup(self):
        bpy.types.SpaceView3D.draw_handler_remove(self.draw_handler, "WINDOW")

    def update_pins(self, context: bpy.types.Context):
        assert context.region
        self.pins_batch = self._create_pins_batch(self.pins_shader)
        context.region.tag_redraw()

    def _get_pin_mode_data(self) -> core.PinModeData:
        tracker = properties.PolychaseState.get_tracker_by_id(self.tracker_id)
        assert tracker
        tracker_core = core.Tracker.get(tracker)
        assert tracker_core

        return tracker_core.pin_mode

    def _create_pins_batch(self, pins_shader: gpu.types.GPUShader):
        data = self._get_pin_mode_data()
        points = data.points if len(data.points) > 0 else []
        is_selected = data.is_selected if len(data.is_selected) > 0 else []

        return batch_for_shader(
            pins_shader,
            "POINTS",
            {
                "position": typing.cast(typing.Sequence, points),
                "is_selected": typing.cast(typing.Sequence, is_selected)
            })

    def _create_wireframe_batch(self, tracker_core: core.Tracker):
        return batch_for_shader(
            self.wireframe_shader,
            "LINES",
            {
                "position":
                    typing.cast(
                        typing.Sequence,
                        tracker_core.accel_mesh.inner().vertices)
            },
            indices=tracker_core.edges_indices.ravel())

    def _create_wireframe_depth_batch(self, tracker_core: core.Tracker):
        return batch_for_shader(
            self.wireframe_shader,
            "TRIS",
            {
                "position":
                    typing.cast(
                        typing.Sequence,
                        tracker_core.accel_mesh.inner().vertices)
            },
            indices=tracker_core.accel_mesh.inner().triangles.ravel())

    def _create_wireframe_mask_ubo(
            self, tracker_core) -> gpu.types.GPUUniformBuf:
        return gpu.types.GPUUniformBuf(
            tracker_core.accel_mesh.inner().masked_triangles)    # type: ignore

    def _create_selection_circle_batch(self, shader: gpu.types.GPUShader):
        return batch_for_shader(
            shader,
            "TRIS",
            {"position": [[-1.0, -1.0], [3.0, -1.0], [-1.0, 3.0]]},
            indices=np.array([0, 1, 2], dtype=np.uint32),
        )

    def _draw_callback(self):
        tracker = properties.PolychaseState.get_tracker_by_id(self.tracker_id)
        if not tracker:
            return
        if not tracker.geometry or not tracker.camera:
            return
        if not bpy.context.region or bpy.context.region.as_pointer() != self.region_pointer:
            return
        if not bpy.context.region_data:
            return

        mvp = bpy.context.region_data.perspective_matrix @ tracker.geometry.matrix_world
        mvp = typing.cast(typing.Sequence[float], mvp)

        self._draw_wireframe(mvp, tracker)
        if not self.mask_mode:
            self._draw_pins(mvp, tracker)
        else:
            self._draw_selection_circle(tracker)

    def _draw_pins(
        self,
        mvp: typing.Sequence,
        tracker: properties.PolychaseTracker,
    ):
        if not self.pins_shader or not self.pins_batch:
            return

        gpu.state.blend_set("ALPHA")
        gpu.state.point_size_set(tracker.pin_radius)
        gpu.state.depth_test_set("NONE")

        self.pins_shader.bind()
        self.pins_shader.uniform_float("point_size", tracker.pin_radius)
        self.pins_shader.uniform_float("mvp", mvp)
        self.pins_shader.uniform_float(
            "default_color", tracker.default_pin_color)
        self.pins_shader.uniform_float(
            "selected_color", tracker.selected_pin_color)
        gpu.state.depth_mask_set(False)
        self.pins_batch.draw(self.pins_shader)

    def _draw_wireframe(
        self,
        mvp: typing.Sequence[float],
        tracker: properties.PolychaseTracker,
    ):
        # Draw solid geometry to prepare Z-buffer, and highlight masked polygons
        gpu.state.depth_mask_set(True)
        gpu.state.depth_test_set("LESS_EQUAL")
        gpu.state.blend_set("ALPHA")

        self.wireframe_shader.bind()
        self.wireframe_shader.uniform_float("mvp", mvp)
        self.wireframe_shader.uniform_float("bias", 0)
        self.wireframe_shader.uniform_float("color", [0.0, 0.0, 0.0, 0.0])
        self.wireframe_shader.uniform_float("mask_color", tracker.mask_color)
        self.wireframe_shader.uniform_bool("useMask", True)
        self.wireframe_shader.uniform_block(
            "u_maskData", self.wireframe_mask_ubo)
        self.wireframe_depth_batch.draw(self.wireframe_shader)

        # Draw wireframe
        gpu.state.depth_mask_set(False)
        gpu.state.depth_test_set("LESS_EQUAL")
        gpu.state.line_width_set(tracker.wireframe_width)
        # Prevent Z-fighting
        self.wireframe_shader.uniform_float("bias", -1e-4)
        self.wireframe_shader.uniform_float("color", tracker.wireframe_color)
        self.wireframe_shader.uniform_bool("useMask", False)
        self.wireframe_batch.draw(self.wireframe_shader)

    def _draw_selection_circle(self, tracker: properties.PolychaseTracker):
        self.selection_circle_shader.bind()
        self.selection_circle_shader.uniform_float("center", self.mouse_pos)
        self.selection_circle_shader.uniform_float(
            "radius", tracker.mask_selection_radius)
        self.selection_circle_batch.draw(self.selection_circle_shader)

import functools

import bpy
import bpy.types
import gpu
import mathutils


@functools.cache
def get_points_shader():
    shader_info = gpu.types.GPUShaderCreateInfo()
    shader_info.vertex_source(
        """
    void main()
    {
        gl_Position = viewProjectionMatrix * objectToWorld * vec4(position, 1.0f);
        finalColor = vec4(color, 1.0f);
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

        fragColor = finalColor * alpha;
    }
    """)
    vert_out = gpu.types.GPUStageInterfaceInfo("polychase_point_interface")    # type: ignore
    vert_out.flat("VEC4", "finalColor")

    shader_info.vertex_in(0, "VEC3", "position")
    shader_info.vertex_in(1, "VEC3", "color")
    shader_info.vertex_out(vert_out)
    shader_info.fragment_out(0, "VEC4", "fragColor")
    shader_info.push_constant("MAT4", "viewProjectionMatrix")
    shader_info.push_constant("MAT4", "objectToWorld")

    return gpu.shader.create_from_info(shader_info)


def bpy_poll_is_mesh(_self: bpy.types.bpy_struct, obj: bpy.types.ID) -> bool:
    return isinstance(obj, bpy.types.Object) and obj.type == "MESH"


def bpy_poll_is_camera(_self: bpy.types.bpy_struct, obj: bpy.types.ID) -> bool:
    return isinstance(obj, bpy.types.Object) and obj.type == "CAMERA"


def ndc(region: bpy.types.Region, x: int | float, y: int | float) -> tuple[float, float]:
    return (2.0 * (x / region.width) - 1.0, 2.0 * (y / region.height) - 1.0)


def calc_camera_proj_mat(camera: bpy.types.Object, width: int, height: int):
    return camera.calc_matrix_camera(bpy.context.evaluated_depsgraph_get(), x=width, y=height)


# Following rna_Object_calc_matrix_camera, BKE_camera_params_compute_viewplane and BKE_camera_params_compute_matrix
# from blender source code. But instead we're computing directly in pixel coordinates.
def calc_camera_params(
    camera: bpy.types.Object,
    width: float,
    height: float,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
) -> (float, float, float, float):
    assert isinstance(camera.data, bpy.types.Camera)

    ycor = scale_y / scale_x

    if camera.data.sensor_fit == "HORIZONTAL":
        sensor_size = camera.data.sensor_width
        extent = width
    elif camera.data.sensor_fit == "VERTICAL":
        sensor_size = camera.data.sensor_height
        extent = height
    else:
        assert camera.data.sensor_fit == "AUTO"
        sensor_size = camera.data.sensor_width
        if width > height:
            extent = width
        else:
            extent = height * ycor

    fx = camera.data.lens * extent / sensor_size
    fy = fx / ycor    # I'm not sure about this division

    cx = camera.data.shift_x * extent - width / 2.0
    cy = camera.data.shift_y * extent - height / 2.0

    return fx, fy, cx, cy


def set_camera_params(
    camera: bpy.types.Object,
    width: float,
    height: float,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
):
    assert isinstance(camera.data, bpy.types.Camera)
    assert fx == fy, f"fx = {fx}, fy = {fy}"    # For now

    ycor = scale_y / scale_x

    if camera.data.sensor_fit == "HORIZONTAL":
        sensor_size = camera.data.sensor_width
        extent = width
    elif camera.data.sensor_fit == "VERTICAL":
        sensor_size = camera.data.sensor_height
        extent = height
    else:
        assert camera.data.sensor_fit == "AUTO"
        sensor_size = camera.data.sensor_width
        if width > height:
            extent = width
        else:
            extent = height * ycor

    camera.data.lens = fx * sensor_size / extent
    camera.data.shift_x = (cx + width / 2.0) / extent
    camera.data.shift_y = (cy + height / 2.0) / extent


def calc_camera_proj_mat_pixels(camera: bpy.types.Object, width: float = 1.0, height: float = 1.0) -> mathutils.Matrix:
    assert isinstance(camera.data, bpy.types.Camera)

    fx, fy, cx, cy = calc_camera_params(camera, width, height)
    n = camera.data.clip_start
    f = camera.data.clip_end
    return mathutils.Matrix(
        [[fx, 0, cx, 0], [0, fy, cy, 0], [0, 0, -(f + n) / (f - n), -2 * (f * n) / (f - n)], [0, 0, -1, 0]])


def calc_camera_params_from_proj(proj: mathutils.Matrix) -> (float, float, float, float):
    return proj[0][0], proj[1][1], proj[0][2], proj[1][2]

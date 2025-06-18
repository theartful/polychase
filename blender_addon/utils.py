import bpy
import bpy.types
import mathutils


def bpy_poll_is_mesh(_self: bpy.types.bpy_struct, obj: bpy.types.ID) -> bool:
    return isinstance(obj, bpy.types.Object) and obj.type == "MESH"


def bpy_poll_is_camera(_self: bpy.types.bpy_struct, obj: bpy.types.ID) -> bool:
    return isinstance(obj, bpy.types.Object) and obj.type == "CAMERA"


def ndc(region: bpy.types.Region, x: int | float,
        y: int | float) -> tuple[float, float]:
    return (2.0 * (x / region.width) - 1.0, 2.0 * (y / region.height) - 1.0)


def calc_camera_proj_mat(camera: bpy.types.Object, width: int, height: int):
    return camera.calc_matrix_camera(
        bpy.context.evaluated_depsgraph_get(), x=width, y=height)


def calc_camera_params(
    camera: bpy.types.Object,
    width: float,
    height: float,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
) -> (float, float, float, float):
    assert isinstance(camera.data, bpy.types.Camera)

    return calc_camera_params_expanded(
        lens=camera.data.lens,
        shift_x=camera.data.shift_x,
        shift_y=camera.data.shift_y,
        sensor_width=camera.data.sensor_width,
        sensor_height=camera.data.sensor_height,
        sensor_fit=camera.data.sensor_fit,
        width=width,
        height=height,
        scale_x=scale_x,
        scale_y=scale_y,
    )


# Following rna_Object_calc_matrix_camera, BKE_camera_params_compute_viewplane
# and BKE_camera_params_compute_matrix from blender source code.
# But instead we're computing directly in pixel coordinates.
def calc_camera_params_expanded(
    lens: float,
    shift_x: float,
    shift_y: float,
    sensor_width: float,
    sensor_height: float,
    sensor_fit: str,
    width: float,
    height: float,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
) -> (float, float, float, float):
    ycor = scale_y / scale_x

    if sensor_fit == "HORIZONTAL":
        sensor_size = sensor_width
        extent = width
    elif sensor_fit == "VERTICAL":
        sensor_size = sensor_height
        extent = height
    else:
        assert sensor_fit == "AUTO"
        sensor_size = sensor_width
        if width > height:
            extent = width
        else:
            extent = height * ycor

    fx = lens * extent / sensor_size
    fy = fx / ycor    # I'm not sure about this division

    cx = shift_x * extent - width / 2.0
    cy = shift_y * extent - height / 2.0

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


def calc_camera_proj_mat_pixels(
        camera: bpy.types.Object,
        width: float = 1.0,
        height: float = 1.0) -> mathutils.Matrix:
    assert isinstance(camera.data, bpy.types.Camera)

    fx, fy, cx, cy = calc_camera_params(camera, width, height)
    n = camera.data.clip_start
    f = camera.data.clip_end
    return mathutils.Matrix(
        [
            [fx, 0, cx, 0], [0, fy, cy, 0],
            [0, 0, -(f + n) / (f - n), -2 * (f * n) / (f - n)], [0, 0, -1, 0]
        ])


def calc_camera_params_from_proj(
        proj: mathutils.Matrix) -> (float, float, float, float):
    return proj[0][0], proj[1][1], proj[0][2], proj[1][2]


def get_rotation_quat(obj: bpy.types.Object):
    if obj.rotation_mode == "QUATERNION":
        return obj.rotation_quaternion
    elif obj.rotation_mode in ("XYZ", "XZY", "YXZ", "YZX", "ZXY", "ZYX"):
        return obj.rotation_euler.to_quaternion()
    elif obj.rotation_mode == "AXIS_ANGLE":
        angle, x, y, z = obj.rotation_axis_angle
        return mathutils.Matrix.Rotation(angle, 3, [x, y, z]).to_quaternion()
    else:
        raise Exception(f"Unknown rotation mode {obj.rotation_mode}")


def set_rotation_quat(obj: bpy.types.Object, quat: mathutils.Quaternion):
    if obj.rotation_mode == "QUATERNION":
        obj.rotation_quaternion = quat
    elif obj.rotation_mode in ("XYZ", "XZY", "YXZ", "YZX", "ZXY", "ZYX"):
        obj.rotation_euler = quat.to_euler(obj.rotation_mode)
    elif obj.rotation_mode == "AXIS_ANGLE":
        axis, angle = quat.to_axis_angle()
        obj.rotation_axis_angle = (angle, axis[0], axis[1], axis[2])
    else:
        raise Exception(f"Unknown rotation mode {obj.rotation_mode}")

def get_rotation_data_path(obj: bpy.types.Object):
    if obj.rotation_mode == "QUATERNION":
        return "rotation_quaternion"
    elif obj.rotation_mode in ("XYZ", "XZY", "YXZ", "YZX", "ZXY", "ZYX"):
        return "rotation_euler"
    elif obj.rotation_mode == "AXIS_ANGLE":
        return "rotation_axis_angle"
    else:
        raise Exception(f"Unknown rotation mode {obj.rotation_mode}")

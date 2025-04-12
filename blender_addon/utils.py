import functools
import bpy
import bpy.types
import gpu


@functools.cache
def get_points_shader():
    shader_info = gpu.types.GPUShaderCreateInfo()
    shader_info.vertex_source(
        """
    void main()
    {
        gl_Position = ModelViewProjectionMatrix * objectToWorld * vec4(position, 1.0f);
        finalColor = vec4(color, 1.0f);
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
    vert_out = gpu.types.GPUStageInterfaceInfo("polychase_point_interface") # type: ignore

    vert_out.smooth("VEC4", "finalColor")

    shader_info.vertex_in(0, "VEC3", "position")
    shader_info.vertex_in(1, "VEC3", "color")
    shader_info.vertex_out(vert_out)
    shader_info.fragment_out(0, "VEC4", "fragColor")
    shader_info.push_constant("MAT4", "ModelViewProjectionMatrix")
    shader_info.push_constant("MAT4", "objectToWorld")

    return gpu.shader.create_from_info(shader_info)


def bpy_poll_is_mesh(_self: bpy.types.bpy_struct, obj: bpy.types.ID) -> bool:
    return isinstance(obj, bpy.types.Object) and obj.type == "MESH"


def bpy_poll_is_camera(_self: bpy.types.bpy_struct, obj: bpy.types.ID) -> bool:
    return isinstance(obj, bpy.types.Object) and obj.type == "CAMERA"

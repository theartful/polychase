#include "ray_casting.h"

#include <embree4/rtcore.h>
#include <fmt/format.h>
#include <spdlog/spdlog.h>

#include <Eigen/LU>
#include <limits>
#include <stdexcept>

#include "utils.h"

void ErrorFunction([[maybe_unused]] void* userPtr, enum RTCError error,
                   const char* str) {
    spdlog::error("RTC Error {}: {}", static_cast<int>(error), str);
}

AcceleratedMesh::AcceleratedMesh(RowMajorArrayX3f vertices,
                                 RowMajorArrayX3u triangles,
                                 ArrayXu masked_triangles)
    : mesh_{std::move(vertices), std::move(triangles),
            std::move(masked_triangles)} {
    // Initialize device
    rtc_device_ = rtcNewDevice(NULL);

    if (!rtc_device_) {
        // blender links to embree, which might not have the new
        // rtcGetErrorString function.
        throw std::runtime_error(
            fmt::format("ERROR: Could not create RTC device: {}",
                        /*rtcGetErrorString*/ (int)(rtcGetDeviceError(NULL))));
    }

    rtcSetDeviceErrorFunction(rtc_device_, ErrorFunction, NULL);

    // Initialize scene
    rtc_scene_ = rtcNewScene(rtc_device_);

    RTCGeometry rtc_geom =
        rtcNewGeometry(rtc_device_, RTC_GEOMETRY_TYPE_TRIANGLE);

    const size_t num_vertices = static_cast<size_t>(mesh_.vertices.rows());
    const size_t num_triangles = static_cast<size_t>(mesh_.triangles.rows());
    const float* vertex_buffer = mesh_.vertices.data();
    const uint32_t* index_buffer = mesh_.triangles.data();

    rtcSetSharedGeometryBuffer(rtc_geom, RTC_BUFFER_TYPE_VERTEX, 0,
                               RTC_FORMAT_FLOAT3, vertex_buffer, 0,
                               3 * sizeof(float), num_vertices);

    rtcSetSharedGeometryBuffer(rtc_geom, RTC_BUFFER_TYPE_INDEX, 0,
                               RTC_FORMAT_UINT3, index_buffer, 0,
                               3 * sizeof(uint32_t), num_triangles);

    rtcCommitGeometry(rtc_geom);
    rtcAttachGeometry(rtc_scene_, rtc_geom);
    rtcReleaseGeometry(rtc_geom);

    rtcCommitScene(rtc_scene_);
}

std::optional<RayHit> AcceleratedMesh::RayCast(Eigen::Vector3f origin,
                                               Eigen::Vector3f direction,
                                               bool check_mask) const {
    RTCRayHit rtc_rayhit = {
        .ray =
            {
                .org_x = origin.x(),
                .org_y = origin.y(),
                .org_z = origin.z(),
                .tnear = 0.0,
                .dir_x = direction.x(),
                .dir_y = direction.y(),
                .dir_z = direction.z(),
                .time = 0.0,
                .tfar = std::numeric_limits<float>::infinity(),
                .mask = 0xFFFFFFFF,
                .id = 0,
                .flags = 0,
            },
        .hit = {.Ng_x = 0.0f,
                .Ng_y = 0.0f,
                .Ng_z = 0.0f,
                .u = 0.0f,
                .v = 0.0f,
                .primID = RTC_INVALID_GEOMETRY_ID,
                .geomID = RTC_INVALID_GEOMETRY_ID,
                .instID = {RTC_INVALID_GEOMETRY_ID},
                .instPrimID = {RTC_INVALID_GEOMETRY_ID}}};

    rtcIntersect1(rtc_scene_, &rtc_rayhit, nullptr);

    if (rtc_rayhit.hit.geomID == RTC_INVALID_GEOMETRY_ID) {
        return std::nullopt;
    }

    CHECK(rtc_rayhit.hit.geomID == 0);

    if (check_mask && mesh_.IsTriangleMasked(rtc_rayhit.hit.primID)) {
        return std::nullopt;
    }

    return RayHit{
        .pos = mesh_.GetTriangle(rtc_rayhit.hit.primID)
                   .Barycentric(rtc_rayhit.hit.u, rtc_rayhit.hit.v),
        .normal = Eigen::Vector3f(rtc_rayhit.hit.Ng_x, rtc_rayhit.hit.Ng_y,
                                  rtc_rayhit.hit.Ng_z)
                      .normalized(),
        .barycentric_coordinate =
            Eigen::Vector2f(rtc_rayhit.hit.u, rtc_rayhit.hit.v),
        .t = rtc_rayhit.ray.tfar,
        .primitive_id = rtc_rayhit.hit.primID,
    };
}

AcceleratedMesh::~AcceleratedMesh() {
    rtcReleaseScene(rtc_scene_);
    rtcReleaseDevice(rtc_device_);
}

std::optional<RayHit> RayCast(const AcceleratedMesh& accel_mesh,
                              const SceneTransformations& scene_transform,
                              Eigen::Vector2f pos, bool check_mask) {
    const Ray ray = GetRayObjectSpace(scene_transform, pos);
    return accel_mesh.RayCast(ray.origin, ray.dir, check_mask);
}

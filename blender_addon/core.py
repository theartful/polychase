# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2025 Ahmed Essam <aessam.dahy@gmail.com>

import typing

import bpy
import mathutils
import numpy as np

from . import properties, utils

if typing.TYPE_CHECKING:
    # Import generated C++ stubs to make development easier
    from .lib.polychase_core import *
else:
    try:
        # This import should work when polychase_core is bundled as a wheel
        from polychase_core import *
    except:
        # When developing, I don't bundle as wheel everytime
        # Instead, I install the python module directly in .lib
        from .lib.polychase_core import *


class _Trackers:

    def __init__(self):
        self.trackers = {}

    def get_tracker(self, id: int, geom: bpy.types.Object):
        geom_id = geom.id_data.name_full

        if id not in self.trackers or self.trackers[id].geom_id != geom_id:
            self.trackers[id] = Tracker(id, geom)

        assert id in self.trackers
        tracker = self.trackers[id]
        tracker.geom = geom    # Update geom as well
        return tracker

    def delete_tracker(self, id: int):
        if id in self.trackers:
            del self.trackers[id]


Trackers = _Trackers()


class PinModeData:

    _tracker_id: int
    _points: np.ndarray
    _is_selected: np.ndarray
    _points_version_number: int
    _selected_pin_idx: int

    def __init__(self, tracker_id: int):
        self._tracker_id = tracker_id
        self._points = np.empty((0, 3), dtype=np.float32)
        self._is_selected = np.empty((0,), dtype=np.uint32)
        self._points_version_number = 0
        self._selected_pin_idx = -1

    def reset_points_if_necessary(self, tracker: properties.PolychaseTracker):
        if tracker.points_version_number != self._points_version_number:
            if tracker.points_version_number == 0:
                assert tracker.selected_pin_idx == -1
                self._points = np.empty((0, 3), dtype=np.float32)
                self._is_selected = np.empty((0,), dtype=np.uint32)
                self._selected_pin_idx = -1
            else:
                self._points = np.frombuffer(
                    tracker.points, dtype=np.float32).reshape((-1, 3))
                self._is_selected = np.zeros(
                    (self._points.shape[0],), dtype=np.uint32)
                self._selected_pin_idx = tracker.selected_pin_idx
                if self._selected_pin_idx > 0:
                    self._is_selected[self._selected_pin_idx] = 1

            self._points_version_number = tracker.points_version_number

        if tracker.selected_pin_idx != self._selected_pin_idx:
            self._is_selected[self._selected_pin_idx] = 0
            self._is_selected[tracker.selected_pin_idx] = 1
            self._selected_pin_idx = tracker.selected_pin_idx

    def _update_points(self, tracker: properties.PolychaseTracker):
        assert self._points_version_number == tracker.points_version_number

        self._points_version_number += 1
        tracker.points_version_number += 1

        tracker.points = self._points.tobytes()

    def _update_selected_pin_idx(
            self, idx, tracker: properties.PolychaseTracker):
        assert self._selected_pin_idx == tracker.selected_pin_idx

        self._selected_pin_idx = idx
        tracker.selected_pin_idx = idx

    @property
    def points(self) -> np.ndarray:
        tracker = properties.PolychaseState.get_tracker_by_id(self._tracker_id)
        assert tracker

        self.reset_points_if_necessary(tracker)
        return self._points

    @property
    def is_selected(self) -> np.ndarray:
        tracker = properties.PolychaseState.get_tracker_by_id(self._tracker_id)
        assert tracker

        self.reset_points_if_necessary(tracker)
        return self._is_selected

    def is_out_of_date(self) -> bool:
        tracker = properties.PolychaseState.get_tracker_by_id(self._tracker_id)
        assert tracker

        return self._points_version_number != tracker.points_version_number

    def create_pin(self, point: np.ndarray, select: bool = False):
        tracker = properties.PolychaseState.get_tracker_by_id(self._tracker_id)
        assert tracker

        self.reset_points_if_necessary(tracker)

        self._points = np.append(
            self._points, np.array([point], dtype=np.float32), axis=0)
        self._is_selected = np.append(
            self._is_selected, np.array([0], dtype=np.uint32), axis=0)
        self._update_points(tracker)

        if select:
            self.select_pin(len(self._points) - 1)

    def delete_pin(self, idx: int):
        tracker = properties.PolychaseState.get_tracker_by_id(self._tracker_id)
        assert tracker

        self.reset_points_if_necessary(tracker)

        if idx < 0 or idx >= len(self._points):
            return

        if self._selected_pin_idx == idx:
            self._update_selected_pin_idx(-1, tracker)
        elif self._selected_pin_idx > idx:
            self._update_selected_pin_idx(self._selected_pin_idx - 1, tracker)

        self._points = np.delete(self._points, idx, axis=0)
        self._is_selected = np.delete(self._is_selected, idx, axis=0)

        self._update_points(tracker)

    def select_pin(self, pin_idx: int):
        tracker = properties.PolychaseState.get_tracker_by_id(self._tracker_id)
        assert tracker

        self.reset_points_if_necessary(tracker)

        self.unselect_pin()
        self._update_selected_pin_idx(pin_idx, tracker)
        self._is_selected[self._selected_pin_idx] = 1

    def unselect_pin(self):
        tracker = properties.PolychaseState.get_tracker_by_id(self._tracker_id)
        assert tracker

        self.reset_points_if_necessary(tracker)

        if self._selected_pin_idx != -1:
            self._is_selected[self._selected_pin_idx] = 0
        self._update_selected_pin_idx(-1, tracker)


class Tracker:

    def __init__(self, tracker_id: int, geom: bpy.types.Object):
        geom_id = geom.id_data.name_full

        self.tracker_id = tracker_id
        self.geom_id = geom_id
        self.geom = geom
        self.pin_mode = PinModeData(tracker_id=self.tracker_id)

        self.init_accel_mesh()

    def init_accel_mesh(self):
        tracker = properties.PolychaseState.get_tracker_by_id(self.tracker_id)
        assert tracker

        geom = tracker.geometry
        assert geom

        depsgraph = bpy.context.evaluated_depsgraph_get()
        evaluated_geom = geom.evaluated_get(depsgraph)
        mesh = evaluated_geom.to_mesh()

        mesh.calc_loop_triangles()

        num_vertices = len(mesh.vertices)
        num_triangles = len(mesh.loop_triangles)

        vertices: np.ndarray = np.empty((num_vertices, 3), dtype=np.float32)
        triangles: np.ndarray = np.empty((num_triangles, 3), dtype=np.uint32)
        triangle_polygons: np.ndarray = np.empty(
            (num_triangles,), dtype=np.uint32)

        assert len(mesh.loop_triangles) == len(mesh.loop_triangle_polygons)

        mesh.vertices.foreach_get("co", vertices.ravel())
        mesh.loop_triangles.foreach_get("vertices", triangles.ravel())
        mesh.loop_triangle_polygons.foreach_get(
            "value", triangle_polygons.ravel())

        # Sort triangles and triangle_polygons
        sort_indices = np.argsort(triangle_polygons, axis=0)
        triangles = triangles[sort_indices]
        triangle_polygons = triangle_polygons[sort_indices]

        masked_triangles: np.ndarray
        if hasattr(self, "accel_mesh"):
            masked_triangles = self.accel_mesh.inner().masked_triangles
        else:
            masked_triangles = np.frombuffer(
                tracker.masked_triangles, dtype=np.uint32)

        try:
            self.accel_mesh = AcceleratedMesh(
                vertices, triangles, masked_triangles)
        except:
            self.accel_mesh = AcceleratedMesh(vertices, triangles)
            tracker.masked_triangles = self.accel_mesh.inner(
            ).masked_triangles.tobytes()

        # Are we sure we want to store edges here?
        self.edges_indices = np.empty((len(mesh.edges), 2), dtype=np.uint32)
        mesh.edges.foreach_get("vertices", self.edges_indices.ravel())

        # Are we also sure that we want to handle polygon/triangle mapping here,
        # and not in C++ land?
        self.triangle_polygons = triangle_polygons

    def ray_cast(
            self,
            region: bpy.types.Region,
            rv3d: bpy.types.RegionView3D,
            region_x: int,
            region_y: int,
            check_mask: bool):
        return ray_cast(
            accel_mesh=self.accel_mesh,
            scene_transform=SceneTransformations(
                model_matrix=typing.cast(np.ndarray, self.geom.matrix_world),
                view_matrix=typing.cast(np.ndarray, rv3d.view_matrix),
                intrinsics=camera_intrinsics_from_proj(rv3d.window_matrix),
            ),
            pos=typing.cast(np.ndarray, utils.ndc(region, region_x, region_y)),
            check_mask=check_mask,
        )

    def set_polygon_mask_using_triangle_idx(self, tri_idx: int):
        polygon = self.triangle_polygons[tri_idx]
        idx = tri_idx
        while idx < len(self.triangle_polygons
                       ) and self.triangle_polygons[idx] == polygon:
            self.accel_mesh.inner_mut().mask_triangle(idx)
            idx += 1

        idx = tri_idx - 1
        while idx >= 0 and self.triangle_polygons[idx] == polygon:
            self.accel_mesh.inner_mut().mask_triangle(idx)
            idx -= 1

    def clear_polygon_mask_using_triangle_idx(self, tri_idx: int):
        polygon = self.triangle_polygons[tri_idx]
        idx = tri_idx
        while idx < len(self.triangle_polygons
                       ) and self.triangle_polygons[idx] == polygon:
            self.accel_mesh.inner_mut().unmask_triangle(idx)
            idx += 1

        idx = tri_idx - 1
        while idx >= 0 and self.triangle_polygons[idx] == polygon:
            self.accel_mesh.inner_mut().unmask_triangle(idx)
            idx -= 1

    @classmethod
    def get(
        cls,
        tracker: properties.PolychaseTracker,
    ) -> typing.Self | None:
        return Trackers.get_tracker(
            tracker.id, tracker.geometry) if tracker.geometry else None


# TODO: Remove these from here?
def camera_intrinsics(
    camera: bpy.types.Object,
    width: float = 1.0,
    height: float = 1.0,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
) -> CameraIntrinsics:
    assert isinstance(camera.data, bpy.types.Camera)

    return camera_intrinsics_expanded(
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


def camera_intrinsics_expanded(
    lens: float,
    shift_x: float,
    shift_y: float,
    sensor_width: float,
    sensor_height: float,
    sensor_fit: str,
    width: float = 1.0,
    height: float = 1.0,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
):
    fx, fy, cx, cy = utils.calc_camera_params_expanded(
        lens=lens,
        shift_x=shift_x,
        shift_y=shift_y,
        sensor_width=sensor_width,
        sensor_height=sensor_height,
        sensor_fit=sensor_fit,
        width=width,
        height=height,
        scale_x=scale_x,
        scale_y=scale_y,
    )
    return CameraIntrinsics(
        fx=-fx,
        fy=-fy,
        cx=-cx,
        cy=-cy,
        aspect_ratio=fx / fy,
        width=width,
        height=height,
        convention=CameraConvention.OpenGL,
    )


def set_camera_intrinsics(
        camera: bpy.types.Object, intrinsics: CameraIntrinsics):
    utils.set_camera_params(
        camera,
        intrinsics.width,
        intrinsics.height,
        -intrinsics.fx,
        -intrinsics.fy,
        -intrinsics.cx,
        -intrinsics.cy,
    )


def camera_intrinsics_from_proj(
        proj: mathutils.Matrix,
        width: float = 2.0,
        height: float = 2.0) -> CameraIntrinsics:
    fx, fy, cx, cy = utils.calc_camera_params_from_proj(proj)
    return CameraIntrinsics(
        fx=-fx,
        fy=-fy,
        cx=-cx,
        cy=-cy,
        aspect_ratio=fx / fy,
        width=width,
        height=height,
        convention=CameraConvention.OpenGL,
    )

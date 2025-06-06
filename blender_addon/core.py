import sys
import typing

import bpy
import mathutils
import numpy as np

from . import properties, utils

if sys.platform == "win32":
    from .bin.polychase_core import *
else:
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


Trackers = _Trackers()

DEFAULT_PIN_COLOR = (0.0, 0.0, 1.0)
SELECTED_PIN_COLOR = (1.0, 0.0, 0.0)


class PinModeData:

    _tracker_id: int
    _points: np.ndarray
    _colors: np.ndarray
    _points_version_number: int
    _selected_pin_idx: int

    def __init__(self, tracker_id: int):
        self._tracker_id = tracker_id
        self._points = np.empty((0, 3), dtype=np.float32)
        self._colors = np.empty((0, 3), dtype=np.float32)
        self._points_version_number = 0
        self._selected_pin_idx = -1

    def _reset_points_if_necessary(self):
        state = properties.PolychaseData.from_context(bpy.context)
        assert state
        tracker = state.get_tracker_by_id(self._tracker_id)
        assert tracker

        if tracker.points_version_number != self._points_version_number:
            if tracker.points_version_number == 0:
                assert tracker.selected_pin_idx == -1
                self._points = np.empty((0, 3), dtype=np.float32)
                self._colors = np.empty((0, 3), dtype=np.float32)
                self._selected_pin_idx = -1
            else:
                self._points = np.frombuffer(
                    tracker.points, dtype=np.float32).reshape((-1, 3))
                self._colors = np.empty_like(self._points)
                self._colors[:] = DEFAULT_PIN_COLOR
                self._selected_pin_idx = tracker.selected_pin_idx
                if self._selected_pin_idx > 0:
                    self._colors[self._selected_pin_idx] = SELECTED_PIN_COLOR

            self._points_version_number = tracker.points_version_number

        if tracker.selected_pin_idx != self._selected_pin_idx:
            self._colors[self._selected_pin_idx] = DEFAULT_PIN_COLOR
            self._colors[tracker.selected_pin_idx] = SELECTED_PIN_COLOR
            self._selected_pin_idx = tracker.selected_pin_idx

    def _update_points(self):
        state = properties.PolychaseData.from_context(bpy.context)
        assert state
        tracker = state.get_tracker_by_id(self._tracker_id)
        assert tracker
        assert self._points_version_number == tracker.points_version_number

        self._points_version_number += 1
        tracker.points_version_number += 1

        tracker.points = self._points.tobytes()

    def _update_selected_pin_idx(self, idx):
        state = properties.PolychaseData.from_context(bpy.context)
        assert state
        tracker = state.get_tracker_by_id(self._tracker_id)
        assert tracker
        assert self._selected_pin_idx == tracker.selected_pin_idx

        self._selected_pin_idx = idx
        tracker.selected_pin_idx = idx

    @property
    def points(self) -> np.ndarray:
        self._reset_points_if_necessary()
        return self._points

    @property
    def colors(self) -> np.ndarray:
        self._reset_points_if_necessary()
        return self._colors

    def create_pin(self, point: np.ndarray, select: bool = False):
        self._reset_points_if_necessary()

        self._points = np.append(
            self._points, np.array([point], dtype=np.float32), axis=0)
        self._colors = np.append(
            self._colors,
            np.array([DEFAULT_PIN_COLOR], dtype=np.float32),
            axis=0)
        self._update_points()

        if select:
            self.select_pin(len(self._points) - 1)

    def delete_pin(self, idx: int):
        self._reset_points_if_necessary()

        if idx < 0 or idx >= len(self._points):
            return

        if self._selected_pin_idx == idx:
            self._update_selected_pin_idx(-1)
        elif self._selected_pin_idx > idx:
            self._update_selected_pin_idx(self._selected_pin_idx - 1)

        self._points = np.delete(self._points, idx, axis=0)
        self._colors = np.delete(self._colors, idx, axis=0)

        self._update_points()

    def select_pin(self, pin_idx: int):
        self._reset_points_if_necessary()

        self.unselect_pin()
        self._update_selected_pin_idx(pin_idx)
        self._colors[self._selected_pin_idx] = SELECTED_PIN_COLOR

    def unselect_pin(self):
        self._reset_points_if_necessary()

        if self._selected_pin_idx != -1:
            self._colors[self._selected_pin_idx] = DEFAULT_PIN_COLOR
        self._update_selected_pin_idx(-1)


class Tracker:

    def __init__(self, tracker_id: int, geom: bpy.types.Object):
        geom_id = geom.id_data.name_full
        depsgraph = bpy.context.evaluated_depsgraph_get()
        evaluated_geom = geom.evaluated_get(depsgraph)
        mesh = evaluated_geom.to_mesh()

        mesh.calc_loop_triangles()

        num_vertices = len(mesh.vertices)
        num_triangles = len(mesh.loop_triangles)

        vertices: np.ndarray = np.empty((num_vertices, 3), dtype=np.float32)
        triangles: np.ndarray = np.empty((num_triangles, 3), dtype=np.uint32)

        mesh.vertices.foreach_get("co", vertices.ravel())
        mesh.loop_triangles.foreach_get("vertices", triangles.ravel())

        self.tracker_id = tracker_id
        self.geom_id = geom_id
        self.geom = geom
        self.accel_mesh = AcceleratedMesh(vertices, triangles)
        self.pin_mode = PinModeData(tracker_id=self.tracker_id)

        # FIXME: Are we sure we want to store edges here?
        self.edges_indices = np.empty((len(mesh.edges), 2), dtype=np.uint32)
        mesh.edges.foreach_get("vertices", self.edges_indices.ravel())


    def ray_cast(
            self,
            region: bpy.types.Region,
            rv3d: bpy.types.RegionView3D,
            region_x: int,
            region_y: int):
        return ray_cast(
            accel_mesh=self.accel_mesh,
            scene_transform=SceneTransformations(
                model_matrix=typing.cast(np.ndarray, self.geom.matrix_world),
                view_matrix=typing.cast(np.ndarray, rv3d.view_matrix),
                intrinsics=camera_intrinsics_from_proj(rv3d.window_matrix),
            ),
            pos=typing.cast(np.ndarray, utils.ndc(region, region_x, region_y)),
        )

    @classmethod
    def get(
            cls,
            tracker: properties.PolychaseClipTracking) -> typing.Self | None:
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

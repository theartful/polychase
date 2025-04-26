import base64
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

    def get_tracker(self, tracker_id: int, geom: bpy.types.Object):
        geom_id = geom.id_data.name_full

        if tracker_id not in self.trackers or self.trackers[tracker_id].geom_id != geom_id:
            self.trackers[tracker_id] = Tracker(tracker_id, geom)

        assert tracker_id in self.trackers
        tracker = self.trackers[tracker_id]
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
                buffer = base64.b64decode(tracker.points)
                self._points = np.frombuffer(buffer, dtype=np.float32).reshape((-1, 3))
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

        tracker.points = base64.b64encode(self._points.tobytes())

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

        self._points = np.append(self._points, np.array([point], dtype=np.float32), axis=0)
        self._colors = np.append(self._colors, np.array([DEFAULT_PIN_COLOR], dtype=np.float32), axis=0)
        self._update_points()

        if select:
            self.select_pin(len(self._points) - 1)

    def delete_pin(self, idx: int):
        self._reset_points_if_necessary()

        if idx < 0 or idx >= len(self._points):
            return

        if self._selected_pin_idx == idx:
            self._update_selected_pin_idx(0 if len(self._points) > 0 else -1)
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

        assert isinstance(evaluated_geom.data, bpy.types.Mesh)

        evaluated_geom.data.calc_loop_triangles()

        num_vertices = len(evaluated_geom.data.vertices)
        num_triangles = len(evaluated_geom.data.loop_triangles)

        vertices: np.ndarray = np.empty((num_vertices, 3), dtype=np.float32)
        triangles: np.ndarray = np.empty((num_triangles, 3), dtype=np.uint32)

        evaluated_geom.data.vertices.foreach_get("co", vertices.ravel())
        evaluated_geom.data.loop_triangles.foreach_get("vertices", triangles.ravel())

        self.tracker_id = tracker_id
        self.geom_id = geom_id
        self.geom = geom
        self.accel_mesh = create_accelerated_mesh(vertices, triangles)
        self.pin_mode = PinModeData(tracker_id=self.tracker_id)

    def ray_cast(self, region: bpy.types.Region, rv3d: bpy.types.RegionView3D, region_x: int, region_y: int):
        return ray_cast(
            self.accel_mesh,
            SceneTransformations(
                model_matrix=self.geom.matrix_world,    # type: ignore
                view_matrix=rv3d.view_matrix,    # type: ignore
                intrinsics=camera_intrinsics_from_proj(rv3d.window_matrix),
            ),
            utils.ndc(region, region_x, region_y))    # type: ignore

    @classmethod
    def get(cls, tracker: properties.PolychaseClipTracking) -> typing.Self | None:
        return Trackers.get_tracker(tracker.id, tracker.geometry) if tracker.geometry else None


# TODO: Remove these from here?
def camera_intrinsics(
        camera: bpy.types.Object,
        width: int,
        height: int,
        scale_x: float = 1.0,
        scale_y: float = 1.0) -> CameraIntrinsics:
    fx, fy, cx, cy = utils.calc_camera_params(camera, width, height, scale_x, scale_y)
    return CameraIntrinsics(-fx, -fy, -cx, -cy, fx / fy, CameraConvention.OpenGL)


def set_camera_intrinsics(camera: bpy.types.Object, width: int, height: int, intrinsics: CameraIntrinsics):
    utils.set_camera_params(
        camera,
        width,
        height,
        -intrinsics.fx,
        -intrinsics.fy,
        -intrinsics.cx,
        -intrinsics.cy,
    )


def camera_intrinsics_from_proj(proj: mathutils.Matrix) -> CameraIntrinsics:
    fx, fy, cx, cy = utils.calc_camera_params_from_proj(proj)
    return CameraIntrinsics(-fx, -fy, -cx, -cy, fx / fy, CameraConvention.OpenGL)

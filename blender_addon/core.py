import dataclasses
import sys

import bpy
import mathutils
import numpy as np

from . import utils
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
            self.trackers[tracker_id] = Tracker(geom)

        assert tracker_id in self.trackers
        tracker = self.trackers[tracker_id]
        tracker.geom = geom    # Update geom as well
        return tracker


Trackers = _Trackers()

DEFAULT_PIN_COLOR = (0.0, 0.0, 1.0)
SELECTED_PIN_COLOR = (1.0, 0.0, 0.0)


@dataclasses.dataclass
class PinModeData:
    # Manipulating numpy arrays as a list is inefficient, but it makes it easier to pass them to C++ land as an Eigen matrix
    points: np.ndarray = dataclasses.field(default_factory=lambda: np.empty((0, 3), dtype=np.float32))
    colors: np.ndarray = dataclasses.field(default_factory=lambda: np.empty((0, 3), dtype=np.float32))
    selected_pin_idx: int = -1
    initial_scene_transform: SceneTransformations | None = None

    def create_pin(self, point: np.ndarray, select: bool = False):
        self.points = np.append(self.points, np.array([point], dtype=np.float32), axis=0)
        self.colors = np.append(self.colors, np.array([DEFAULT_PIN_COLOR], dtype=np.float32), axis=0)

        if select:
            self.select_pin(len(self.points) - 1)

    def delete_pin(self, idx: int):
        if idx < 0 or idx >= len(self.points):
            return

        if self.selected_pin_idx == idx:
            self.selected_pin_idx = 0 if len(self.points) > 0 else -1
        elif self.selected_pin_idx > idx:
            self.selected_pin_idx -= 1

        self.points = np.delete(self.points, idx, axis=0)
        self.colors = np.delete(self.colors, idx, axis=0)

    def select_pin(self, pin_idx: int):
        self.unselect_pin()
        self.selected_pin_idx = pin_idx
        self.colors[self.selected_pin_idx] = SELECTED_PIN_COLOR

    def unselect_pin(self):
        if self.selected_pin_idx != -1:
            self.colors[self.selected_pin_idx] = DEFAULT_PIN_COLOR
        self.selected_pin_idx = -1


class Tracker:

    def __init__(self, geom: bpy.types.Object):
        id = geom.id_data.name_full
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

        self.geom_id = id
        self.geom = geom
        self.accel_mesh = create_accelerated_mesh(vertices, triangles)
        self.pin_mode = PinModeData()

    def ray_cast(self, region: bpy.types.Region, rv3d: bpy.types.RegionView3D, region_x: int, region_y: int):
        return ray_cast(
            self.accel_mesh,
            SceneTransformations(
                model_matrix=self.geom.matrix_world,    # type: ignore
                view_matrix=rv3d.view_matrix,    # type: ignore
                intrinsics=camera_intrinsics_from_proj(rv3d.window_matrix),
            ),
            utils.ndc(region, region_x, region_y))    # type: ignore


# TODO: Remove this from here?
def camera_intrinsics(
        camera: bpy.types.Object,
        width: int,
        height: int,
        scale_x: float = 1.0,
        scale_y: float = 1.0) -> CameraIntrinsics:
    fx, fy, cx, cy = utils.calc_camera_params(camera, width, height, scale_x, scale_y)
    return CameraIntrinsics(-fx, -fy, -cx, -cy, fx / fy, CameraConvention.OpenGL)


def camera_intrinsics_from_proj(proj: mathutils.Matrix) -> CameraIntrinsics:
    fx, fy, cx, cy = utils.calc_camera_params_from_proj(proj)
    return CameraIntrinsics(-fx, -fy, -cx, -cy, fx / fy, CameraConvention.OpenGL)

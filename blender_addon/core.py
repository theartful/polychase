import dataclasses
import sys

import bpy
import mathutils
import numpy as np

if sys.platform == "win32":
    from .bin import polychase_core
else:
    from .lib import polychase_core


class _Trackers:

    def __init__(self):
        self.trackers = {}

    def get_tracker(self, tracker_id: int, geom: bpy.types.Object):
        geom_id = geom.id_data.name_full

        if tracker_id not in self.trackers or self.trackers[tracker_id].geom_id != geom_id:
            self.trackers[tracker_id] = Tracker(geom)

        return self.trackers.get(tracker_id, None)


Trackers = _Trackers()


class PinMode:

    def __init__(self):
        self.points = np.array([])
        self.colors = np.array([])

    def create_pin(self, pos: np.ndarray):
        self.points.append(pos)


class Tracker:

    def __init__(self, geom: bpy.types.Object):
        id = geom.id_data.name_full
        depsgraph = bpy.context.evaluated_depsgraph_get()
        evaluated_geom = geom.evaluated_get(depsgraph)

        evaluated_geom.data.calc_loop_triangles()

        num_vertices = len(evaluated_geom.data.vertices)
        num_triangles = len(evaluated_geom.data.loop_triangles)

        vertices = np.empty((num_vertices, 3), dtype=np.float32)
        triangles = np.empty((num_triangles, 3), dtype=np.uint32)

        evaluated_geom.data.vertices.foreach_get("co", vertices.ravel())
        evaluated_geom.data.loop_triangles.foreach_get("vertices", triangles.ravel())

        self.geom_id = id
        self.geom = geom
        self.accel_mesh = polychase_core.create_accelerated_mesh(vertices, triangles)
        self.pin_mode = PinMode()

    def ray_cast(self, region: bpy.types.Region, rv3d: bpy.types.RegionView3D, mouse_x: int, mouse_y: int):
        scene_transform = polychase_core.SceneTransformations(
            model_matrix=self.geom.matrix_world,
            projection_matrix=rv3d.window_matrix,
            view_matrix=rv3d.view_matrix,
        )
        x = 2.0 * (mouse_x / region.width) - 1.0
        y = 2.0 * (mouse_y / region.height) - 1.0
        return polychase_core.ray_cast(self.accel_mesh, scene_transform, (x, y))

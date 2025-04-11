import dataclasses
import sys
import typing

import bpy
import mathutils
import numpy as np
import numpy.typing as npt

if sys.platform == "win32":
    from .bin import polychase_core
else:
    from .lib import polychase_core

TransformationType = polychase_core.TransformationType


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


def generate_database(
        first_frame: int,
        num_frames: int,
        width: int,
        height: int,
        frame_accessor: typing.Callable[[int], np.ndarray | None],
        callback: typing.Callable[[float, str], bool],
        database_path: str):

    polychase_core.generate_optical_flow_database(
        video_info=polychase_core.VideoInfo(
            width=width, height=height, first_frame=first_frame, num_frames=num_frames),
        frame_accessor_function=frame_accessor,
        callback=callback,
        database_path=database_path,
    )


@dataclasses.dataclass
class PinModeData:
    # Manipulating numpy arrays as a list is inefficient, but it makes it easier to pass them to C++ land as an Eigen matrix
    points: np.ndarray = dataclasses.field(default_factory=lambda: np.empty((0, 3), dtype=np.float32))
    colors: np.ndarray = dataclasses.field(default_factory=lambda: np.empty((0, 3), dtype=np.float32))
    selected_pin_idx: int = -1
    initial_scene_transform: polychase_core.SceneTransformations | None = None

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
        self.accel_mesh = polychase_core.create_accelerated_mesh(vertices, triangles)
        self.pin_mode = PinModeData()

    def ndc(self, region: bpy.types.Region, x: int | float, y: int | float) -> tuple[float, float]:
        return (2.0 * (x / region.width) - 1.0, 2.0 * (y / region.height) - 1.0)

    def ray_cast(self, region: bpy.types.Region, rv3d: bpy.types.RegionView3D, region_x: int, region_y: int):
        return polychase_core.ray_cast(
            self.accel_mesh,
            polychase_core.SceneTransformations(
                model_matrix=self.geom.matrix_world,    # type: ignore
                view_matrix=rv3d.view_matrix,    # type: ignore
                projection_matrix=rv3d.window_matrix,    # type: ignore
            ),
            self.ndc(region, region_x, region_y))    # type: ignore

    def find_transformation(
            self,
            region: bpy.types.Region,
            rv3d: bpy.types.RegionView3D,
            region_x: int,
            region_y: int,
            trans_type: TransformationType):

        return polychase_core.find_transformation(
            self.pin_mode.points,
            self.pin_mode.initial_scene_transform,    # type: ignore
            polychase_core.SceneTransformations(
                model_matrix=self.geom.matrix_world,    # type: ignore
                view_matrix=rv3d.view_matrix,    # type: ignore
                projection_matrix=rv3d.window_matrix,    # type: ignore
            ),
            polychase_core.PinUpdate(
                pin_idx=self.pin_mode.selected_pin_idx, pin_pos=self.ndc(region, region_x,
                                                                         region_y)),    # type: ignore
            trans_type,
        )

    def update_initial_scene_transformation(self, region: bpy.types.Region, rv3d: bpy.types.RegionView3D):
        self.pin_mode.initial_scene_transform = polychase_core.SceneTransformations(
            model_matrix=self.geom.matrix_world,    # type: ignore
            projection_matrix=rv3d.window_matrix,    # type: ignore
            view_matrix=rv3d.view_matrix,    # type: ignore
        )

    def solve_forwards(
            self,
            database_path: str,
            frame_from: int,
            num_frames: int,
            width: int,
            height: int,
            camera: bpy.types.Object,
            trans_type: TransformationType,
            callback=typing.Callable[[int, np.ndarray], bool]) -> bool:

        viewport_matrix = mathutils.Matrix(
            [
                [width / 2, 0, 0, width / 2],
                [0, height / 2, 0, height / 2],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ])

        projection_matrix = viewport_matrix @ camera.calc_matrix_camera(
            bpy.context.evaluated_depsgraph_get(),
            x=width,
            y=height,
        )

        return polychase_core.solve_forwards(
            database_path=database_path,
            frame_from=frame_from,
            num_frames=num_frames,
            scene_transform=polychase_core.SceneTransformations(
                model_matrix=self.geom.matrix_world, # type: ignore
                view_matrix=camera.matrix_world.inverted(),# type: ignore
                projection_matrix=projection_matrix, # type: ignore
            ),
            accel_mesh=self.accel_mesh,
            trans_type=trans_type,
            callback=callback,
        )

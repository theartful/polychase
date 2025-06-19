import bpy
import bpy.types
import mathutils

from .. import properties


class PC_OT_CenterGeometry(bpy.types.Operator):
    bl_idname = "polychase.center_geometry"
    bl_label = "Center Geometry"
    bl_options = {"REGISTER", "UNDO"}
    bl_description = "Center the geometry in front of the camera such that it takes about half the 2D space"

    def execute(self, context: bpy.types.Context) -> set:
        state = properties.PolychaseData.from_context(context)
        if not state:
            self.report({'ERROR'}, "No Polychase data found")
            return {"CANCELLED"}

        tracker = state.active_tracker
        if not tracker:
            self.report({'ERROR'}, "No active tracker")
            return {"CANCELLED"}

        if not tracker.geometry:
            self.report({'ERROR'}, "No geometry object assigned to tracker")
            return {"CANCELLED"}

        if not tracker.camera:
            self.report({'ERROR'}, "No camera object assigned to tracker")
            return {"CANCELLED"}

        if not context.scene:
            self.report({'ERROR'}, "Could not find associated scene")
            return {"CANCELLED"}

        geometry = tracker.geometry
        camera = tracker.camera

        assert isinstance(camera.data, bpy.types.Camera)

        depsgraph = context.evaluated_depsgraph_get()

        model_matrix = geometry.matrix_world
        view_matrix_inv = camera.matrix_world
        view_matrix = view_matrix_inv.inverted()
        model_view_matrix = view_matrix @ model_matrix
        proj_matrix = camera.calc_matrix_camera(
            depsgraph,
            x=context.scene.render.resolution_x,
            y=context.scene.render.resolution_y)

        bbox_corners_cs = [
            model_view_matrix @ mathutils.Vector(corner)
            for corner in geometry.bound_box
        ]
        bbox_center_cs = sum(
            bbox_corners_cs, mathutils.Vector(
                (0.0, 0.0, 0.0))) / len(bbox_corners_cs)

        bbox_min_cs = mathutils.Vector(
            (
                min((p.x for p in bbox_corners_cs)),
                min((p.y for p in bbox_corners_cs)),
                min((p.z for p in bbox_corners_cs)),
            ))
        bbox_max_cs = mathutils.Vector(
            (
                max((p.x for p in bbox_corners_cs)),
                max((p.y for p in bbox_corners_cs)),
                max((p.z for p in bbox_corners_cs)),
            ))
        max_extent_cs = max(
            bbox_max_cs.x - bbox_min_cs.x, bbox_max_cs.y - bbox_min_cs.y)

        forward_cs = (
            proj_matrix.inverted() @ mathutils.Vector(
                (0.0, 0.0, -1.0))).normalized()
        distance = max(proj_matrix[0][0], proj_matrix[1][1]) * max_extent_cs

        offset = view_matrix_inv @ (
            forward_cs * distance) - view_matrix_inv @ bbox_center_cs
        geometry.location += offset

        return {"FINISHED"}

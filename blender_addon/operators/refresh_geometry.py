import bpy.types

from .. import properties
from .. import core


class PC_OT_RefreshGeometry(bpy.types.Operator):
    bl_idname = "polychase.refresh_geometry"
    bl_label = "Refresh Geometry"
    bl_options = {"INTERNAL"}
    bl_description = "Refresh Polychase's view of the geometry in case it was editted"

    def execute(self, context: bpy.types.Context) -> set:
        state = properties.PolychaseData.from_context(context)
        if not state:
            return {"CANCELLED"}

        tracker = state.active_tracker
        if not tracker or not tracker.geometry:
            return {"CANCELLED"}

        tracker_core = core.Tracker.get(tracker)
        assert tracker_core # This should never happen

        tracker_core.init_accel_mesh()
        return {"FINISHED"}

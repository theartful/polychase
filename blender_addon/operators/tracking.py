import bpy
import bpy.types
import mathutils

from .. import core
from ..properties import PolychaseData


class OT_TrackForwards(bpy.types.Operator):
    bl_idname = "polychase.track_forwards"
    bl_options = {"REGISTER", "UNDO"}
    bl_label = "Track Forwards"

    @classmethod
    def poll(cls, context):
        state = PolychaseData.from_context(context)
        if not state:
            return False

        tracker = state.active_tracker
        return (
            tracker is not None and tracker.clip is not None and tracker.camera is not None
            and tracker.geometry is not None and tracker.database_path != "")

    def invoke(self, context: bpy.types.Context, event: bpy.types.Event):    # type: ignore
        state = PolychaseData.from_context(context)
        if not state:
            return {"CANCELLED"}

        tracker = state.active_tracker
        if not tracker:
            return {"CANCELLED"}

        scene = context.scene
        if not scene:
            return False

        clip = tracker.clip
        camera = tracker.camera
        geometry = tracker.geometry

        trans_type = core.TransformationType.Model if tracker.tracking_target == "GEOMETRY" else core.TransformationType.Camera

        def callback(frame_id, matrix):
            scene.frame_set(frame_id)
            if trans_type == core.TransformationType.Model:
                loc, rot, scale = mathutils.Matrix(matrix).decompose()
                geometry.location = loc
                geometry.rotation_mode = "QUATERNION"
                geometry.rotation_quaternion = rot
                geometry.keyframe_insert(data_path="location")
                geometry.keyframe_insert(data_path="rotation_quaternion")
            else:
                loc, rot, scale = mathutils.Matrix(matrix).inverted().decompose()
                camera.location = loc
                camera.rotation_mode = "QUATERNION"
                camera.rotation_quaternion = rot
                camera.keyframe_insert(data_path="location")
                camera.keyframe_insert(data_path="rotation_quaternion")
            return True

        frame_from = scene.frame_current
        num_frames = clip.frame_duration - frame_from + 1
        width, height = clip.size

        result = tracker.core().solve_forwards(
            database_path=bpy.path.abspath(tracker.database_path),
            frame_from=frame_from,
            num_frames=num_frames,
            width=width,
            height=height,
            camera=camera,
            trans_type=trans_type,
            callback=callback,
        )
        print("RESULT = ", result)

        return {"FINISHED"}

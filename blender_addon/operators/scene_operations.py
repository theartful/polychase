import bpy
import bpy.types
import mathutils

from .. import properties, keyframes, utils


class PC_OT_CenterGeometry(bpy.types.Operator):
    bl_idname = "polychase.center_geometry"
    bl_label = "Center Geometry"
    bl_options = {"REGISTER", "UNDO"}
    bl_description = "Center the geometry in front of the camera"

    @classmethod
    def poll(cls, context):
        state = properties.PolychaseData.from_context(context)
        if not state:
            return False

        tracker = state.active_tracker
        return (
            tracker is not None and tracker.camera is not None
            and tracker.geometry is not None)

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

        model_matrix = geometry.matrix_local
        view_matrix_inv = camera.matrix_local
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


class PC_OT_ConvertAnimation(bpy.types.Operator):
    bl_idname = "polychase.convert_animation"
    bl_label = "Convert Animation"
    bl_options = {"REGISTER", "UNDO"}
    bl_description = "Convert animation from camera to geometry, or from geometry to camera (depending on the tracking target)"

    @classmethod
    def poll(cls, context):
        state = properties.PolychaseData.from_context(context)
        if not state:
            return False

        tracker = state.active_tracker
        return (
            tracker is not None and tracker.camera is not None
            and tracker.geometry is not None)

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

        # Determine source and target objects
        if tracker.tracking_target == "CAMERA":
            source_obj = geometry
            target_obj = camera
        else:
            source_obj = camera
            target_obj = geometry

        source_data_paths = [
            "location", utils.get_rotation_data_path(source_obj)
        ]
        target_data_paths = [
            "location", utils.get_rotation_data_path(target_obj)
        ]

        # Get all keyframes from source object
        source_fcurves = keyframes.get_fcurves(source_obj, source_data_paths)

        if not source_fcurves:
            self.report(
                {'WARNING'}, "No animation keyframes found on source object")
            return {"CANCELLED"}

        # Collect all keyframe times
        keyframe_times = {}
        for fcurve in source_fcurves:
            for keyframe in fcurve.keyframe_points:
                if keyframe not in keyframe_times:
                    keyframe_times[int(keyframe.co[0])] = keyframe.type

        if len(keyframe_times) == 0:
            self.report({'WARNING'}, "No keyframes found")
            return {"CANCELLED"}

        # Store current frame to restore later
        current_frame = context.scene.frame_current

        Rm0 = utils.get_rotation_quat(geometry)
        tm0 = geometry.location.copy()

        # Blender's camera matrix_world/matrix_local is the inverse of the view matrix
        Rv0 = utils.get_rotation_quat(camera).inverted()
        tv0 = -(Rv0 @ camera.location)

        def compose(
            r1: mathutils.Quaternion,
            r2: mathutils.Quaternion,
            t1: mathutils.Vector,
            t2: mathutils.Vector,
        ) -> tuple[mathutils.Quaternion, mathutils.Vector]:
            return r2 @ r1, r2 @ t1 + t2

        # Convert animation for each keyframe
        result = []
        for frame in keyframe_times.keys():
            context.scene.frame_set(frame)

            Rm = utils.get_rotation_quat(geometry)
            tm = geometry.location.copy()

            Rv = utils.get_rotation_quat(camera).inverted()
            tv = -(Rv @ camera.location)

            Rmv, tmv = compose(Rm, Rv, tm, tv)

            if tracker.tracking_target == "CAMERA":
                R = Rmv @ Rm0.inverted()
                t = tmv - R @ tm0

                # Blender's camera matrix_world/matrix_local is the inverse of the view matrix
                R = R.inverted()
                t = -(R @ t)

                result.append((R, t))
            else:
                assert tracker.tracking_target == "GEOMETRY"
                Rv0_inv = Rv0.inverted()
                R = Rv0_inv @ Rmv
                t = Rv0_inv @ (tmv - tv0)
                result.append((R, t))

        for (frame, keytype), (R, t) in zip(keyframe_times.items(), result):
            context.scene.frame_set(frame)

            utils.set_rotation_quat(target_obj, R)
            target_obj.location = t

            keyframes.insert_keyframe(
                obj=target_obj,
                frame=frame,
                data_paths=target_data_paths,
                keytype=keytype,
            )

        keyframes.clear_keyframes(source_obj, lambda _: True)

        # Restore initial transformation of the source object
        if tracker.tracking_target == "GEOMETRY":
            utils.set_rotation_quat(camera, Rv0.inverted())
            camera.location = -(Rv0.inverted() @ tv0)
        else:
            utils.set_rotation_quat(geometry, Rm0)
            geometry.location = tm0

        # Restore original frame
        context.scene.frame_set(current_frame)

        return {"FINISHED"}

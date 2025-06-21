import typing

import bpy
import bpy.types
import mathutils

from .. import keyframes, properties, utils


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

        model_matrix = geometry.matrix_world
        view_matrix = utils.get_camera_view_matrix(camera)
        view_matrix_inv = view_matrix.inverted()
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
        geometry.matrix_world.translation += offset

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

        # Get all keyframes from source object
        source_fcurves = keyframes.get_fcurves(source_obj, ["location"])

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

        tm0, Rm0, _ = utils.get_object_model_matrix_loc_rot_scale(geometry)
        tv0, Rv0 = utils.get_camera_view_matrix_loc_rot(camera)

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

            tm, Rm, _ = utils.get_object_model_matrix_loc_rot_scale(geometry)
            tv, Rv = utils.get_camera_view_matrix_loc_rot(camera)

            Rmv, tmv = compose(Rm, Rv, tm, tv)

            if tracker.tracking_target == "CAMERA":
                R = Rmv @ Rm0.inverted()
                t = tmv - R @ tm0
                result.append((R, t))
            else:
                assert tracker.tracking_target == "GEOMETRY"
                Rv0_inv = Rv0.inverted()
                R = Rv0_inv @ Rmv
                t = Rv0_inv @ (tmv - tv0)
                result.append((R, t))

        data_paths = ["location", utils.get_rotation_data_path(target_obj)]

        for (frame, keytype), (R, t) in zip(keyframe_times.items(), result):
            context.scene.frame_set(frame)

            if tracker.tracking_target == "GEOMETRY":
                utils.set_object_model_matrix(geometry, t, R)
            else:
                utils.set_camera_view_matrix(camera, t, R)

            keyframes.insert_keyframe(
                obj=target_obj,
                frame=frame,
                data_paths=data_paths,
                keytype=keytype,
            )

        keyframes.clear_keyframes(source_obj, lambda _: True)

        # Restore initial transformation of the source object
        if tracker.tracking_target == "GEOMETRY":
            utils.set_camera_view_matrix(camera, tv0, Rv0)
        else:
            utils.set_object_model_matrix(geometry, tm0, Rm0)

        # Restore original frame
        context.scene.frame_set(current_frame)

        return {"FINISHED"}


_disable_update = True
_original_geom_mat_world: mathutils.Matrix = mathutils.Matrix()
_original_cam_mat_world: mathutils.Matrix = mathutils.Matrix()


def transform_scene_on_coords_changed(
        operator: bpy.types.bpy_struct, context: bpy.types.Context):
    global _disable_update

    if _disable_update:
        return

    operator = typing.cast(PC_OT_TransformScene, operator)

    state = properties.PolychaseData.from_context(context)
    if not state:
        return

    tracker = state.active_tracker
    if not tracker or not tracker.geometry or not tracker.camera:
        return

    ref_obj = tracker.geometry if operator.reference == "GEOMETRY" else tracker.camera

    if operator.coords == "WORLD":
        loc, rot, _ = ref_obj.matrix_world.decompose()
    else:
        loc, rot, _ = ref_obj.matrix_local.decompose()

    _disable_update = True
    operator.location = loc
    operator.rotation = rot.to_euler()
    _disable_update = False


def transform_scene_update_reference(
        tracker: properties.PolychaseTracker,
        reference: typing.Literal["GEOMETRY", "CAMERA"]):
    assert tracker.camera
    assert tracker.geometry

    global _original_geom_mat_world
    global _original_cam_mat_world

    if reference == "GEOMETRY":
        update = tracker.geometry.matrix_world @ _original_geom_mat_world.inverted(
        )
        tracker.camera.matrix_world = update @ _original_cam_mat_world
    else:
        update = tracker.camera.matrix_world @ _original_cam_mat_world.inverted(
        )
        tracker.geometry.matrix_world = update @ _original_geom_mat_world

    tracker.camera.scale = (1.0, 1.0, 1.0)


def transform_scene_on_transform_changed(
        operator: bpy.types.bpy_struct, context: bpy.types.Context):
    global _disable_update

    if _disable_update:
        return

    operator = typing.cast(PC_OT_TransformScene, operator)

    state = properties.PolychaseData.from_context(context)
    if not state:
        return

    tracker = state.active_tracker
    if not tracker or not tracker.geometry or not tracker.camera:
        return

    if operator.reference == "GEOMETRY":
        ref_obj = tracker.geometry
        scale = operator.scale * mathutils.Vector(tracker.geometry_scale)
    else:
        ref_obj = tracker.camera
        scale = None

    matrix = mathutils.Matrix.LocRotScale(
        operator.location,
        operator.rotation,
        scale,
    )
    if operator.coords == "WORLD":
        ref_obj.matrix_world = matrix
    else:
        ref_obj.matrix_world = (
            ref_obj.matrix_world @ ref_obj.matrix_local.inverted()) @ matrix

    transform_scene_update_reference(tracker, operator.reference)


class PC_OT_TransformScene(bpy.types.Operator):
    bl_idname = "polychase.transform_scene"
    bl_label = "Transform Scene"
    bl_options = {"REGISTER", "UNDO"}
    bl_description = "Transform tracked geometry or camera while maintaining correct relative poses between them"

    reference: bpy.props.EnumProperty(
        name="Reference",
        items=(
            ("GEOMETRY", "Geometry", "Transform geometry"),
            ("CAMERA", "Camera", "Transform camera")),
    )
    coords: bpy.props.EnumProperty(
        name="Coordinates",
        items=(
            ("LOCAL", "Local", "Local Coordinates"),
            ("WORLD", "World", "World Coordinates")),
        default="WORLD",
        update=transform_scene_on_coords_changed,
    )
    scale: bpy.props.FloatProperty(
        name="Scale",
        default=0.0,
        precision=3,
        min=0.0,
        update=transform_scene_on_transform_changed,
    )
    rotation: bpy.props.FloatVectorProperty(
        name="Rotation",
        size=3,
        subtype="EULER",
        precision=3,
        update=transform_scene_on_transform_changed,
    )
    location: bpy.props.FloatVectorProperty(
        name="Location",
        size=3,
        subtype="TRANSLATION",
        precision=3,
        update=transform_scene_on_transform_changed,
    )

    _geom_mat_world: mathutils.Matrix
    _cam_mat_world: mathutils.Matrix

    @classmethod
    def poll(cls, context):
        state = properties.PolychaseData.from_context(context)
        if not state:
            return False

        tracker = state.active_tracker
        return (
            tracker is not None and tracker.camera is not None
            and tracker.geometry is not None)

    def draw(self, context: bpy.types.Context):
        layout = self.layout
        assert layout

        layout.use_property_split = True
        col = layout.column()
        col.prop(self, "coords")
        col.prop(self, "location")
        col.prop(self, "rotation")
        if self.reference == "GEOMETRY":
            col.prop(self, "scale")

    def invoke(self, context: bpy.types.Context, event: bpy.types.Event) -> set:
        global _original_geom_mat_world
        global _original_cam_mat_world

        assert context.window_manager
        assert context.scene

        state = properties.PolychaseData.from_context(context)
        if not state:
            return {"CANCELLED"}

        tracker = state.active_tracker
        if not tracker or not tracker.geometry or not tracker.camera:
            return {"CANCELLED"}

        self._geom_mat_world = tracker.geometry.matrix_world.copy()
        self._cam_mat_world = tracker.camera.matrix_world.copy()

        new_scale = tracker.geometry.matrix_world.to_scale()
        if new_scale.normalized().dot(mathutils.Vector(
                tracker.geometry_scale).normalized()) < 0.99:
            self.report({'ERROR'}, f"Non uniform scale detected")
            return {"CANCELLED"}

        # Get original poses of geometry and camera as if the user hasn't
        # changed them.
        if tracker.tracking_target == "GEOMETRY":
            _original_cam_mat_world = mathutils.Matrix.LocRotScale(
                tracker.camera_loc,
                tracker.camera_rot,
                None,
            )
            # Re-evaluate the frame to get the original matrix_world
            context.scene.frame_set(context.scene.frame_current)
            _original_geom_mat_world = tracker.geometry.matrix_world.copy()

            # Restore
            tracker.geometry.matrix_world = self._geom_mat_world
        else:
            _original_geom_mat_world = mathutils.Matrix.LocRotScale(
                tracker.geometry_loc,
                tracker.geometry_rot,
                tracker.geometry_scale,
            )

            # Re-evaluate the frame to get the original matrix_world
            context.scene.frame_set(context.scene.frame_current)
            _original_cam_mat_world = tracker.camera.matrix_world.copy()
            # Restore
            tracker.camera.matrix_world = self._cam_mat_world

        global _disable_update
        _disable_update = True

        self.scale = new_scale.magnitude / mathutils.Vector(
            tracker.geometry_scale).magnitude

        _disable_update = False
        transform_scene_on_coords_changed(self, context)
        transform_scene_update_reference(tracker, self.reference)

        return context.window_manager.invoke_props_dialog(self)

    def execute(self, context: bpy.types.Context) -> set:
        assert context.scene

        state = properties.PolychaseData.from_context(context)
        if not state:
            return {"CANCELLED"}

        tracker = state.active_tracker
        if not tracker or not tracker.geometry or not tracker.camera:
            return {"CANCELLED"}

        geometry = tracker.geometry
        camera = tracker.camera

        # Determine source and target objects
        if tracker.tracking_target == "CAMERA":
            fcurves = keyframes.get_fcurves(camera, ["location"])
        else:
            fcurves = keyframes.get_fcurves(geometry, ["location"])

        if not fcurves:
            self.report(
                {'WARNING'}, "No animation keyframes found on source object")
            return {"CANCELLED"}

        # Collect all keyframe times
        keyframe_times = set()
        for fcurve in fcurves:
            for keyframe in fcurve.keyframe_points:
                if keyframe not in keyframe_times:
                    keyframe_times.add(int(keyframe.co[0]))

        if len(keyframe_times) == 0:
            self.report({'WARNING'}, "No keyframes found")
            return {"CANCELLED"}

        # Store current frame to restore later
        current_frame = context.scene.frame_current

        geom_data_paths = ["location", utils.get_rotation_data_path(geometry)]
        cam_data_paths = ["location", utils.get_rotation_data_path(camera)]

        if self.reference == "GEOMETRY":
            update = geometry.matrix_world @ _original_geom_mat_world.inverted()
        else:
            update = camera.matrix_world @ _original_cam_mat_world.inverted()

        # Update animation for each keyframe
        if tracker.tracking_target == "GEOMETRY":
            for frame in keyframe_times:
                context.scene.frame_set(frame)
                mat_world = geometry.matrix_world.copy()
                geometry.matrix_world = update @ mat_world
                keyframes.insert_keyframe(
                    obj=geometry,
                    frame=frame,
                    data_paths=geom_data_paths,
                )
                # YUCK: Restore mat_world to restore the scale!
                geometry.matrix_world = mat_world

            camera.matrix_world = update @ _original_cam_mat_world
            camera.scale = (1.0, 1.0, 1.0)

        else:
            for frame in keyframe_times:
                context.scene.frame_set(frame)
                camera.matrix_world = update @ camera.matrix_world
                camera.scale = (1.0, 1.0, 1.0)
                keyframes.insert_keyframe(
                    obj=camera,
                    frame=frame,
                    data_paths=cam_data_paths,
                )

            geometry.matrix_world = update @ _original_geom_mat_world

        # Restore original frame
        context.scene.frame_set(current_frame)
        tracker.store_geom_cam_transform()

        return {"FINISHED"}

    def cancel(self, context: bpy.types.Context):
        state = properties.PolychaseData.from_context(context)
        if not state:
            return

        tracker = state.active_tracker
        if not tracker:
            return

        if tracker.geometry:
            tracker.geometry.matrix_world = self._geom_mat_world

        if tracker.camera:
            tracker.camera.matrix_world = self._cam_mat_world

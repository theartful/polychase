import bpy
import bpy.types
import typing

TRANSFORMATION_DATAPATHS = [
    "location",
    "rotation_quaternion",
    "rotation_euler",
    "rotation_axis_angle",
]

CAMERA_DATAPATHS = [
    "lens",
    "shift_x",
    "shift_y",
]

Animatable = bpy.types.Object | bpy.types.Camera


def clear_keyframes(
    obj: Animatable,
    predicate: typing.Callable[[bpy.types.Keyframe], bool],
    data_paths: list[str] = TRANSFORMATION_DATAPATHS,
) -> int:
    if not obj.animation_data or not obj.animation_data.action:
        return 0

    num_deleted = 0
    for fcurve in reversed(obj.animation_data.action.fcurves):
        if fcurve.data_path not in data_paths:
            continue

        fcurve.keyframe_points.sort()

        for keyframe in reversed(fcurve.keyframe_points):
            if predicate(keyframe):
                fcurve.keyframe_points.remove(keyframe)
                num_deleted += 1

        if len(fcurve.keyframe_points) == 0:
            obj.animation_data.action.fcurves.remove(fcurve)

    if len(obj.animation_data.action.fcurves) == 0:
        obj.animation_data.action = None

    return num_deleted


def clear_keyframes_in_range(
    obj: Animatable,
    frame_start: int,
    frame_end_inclusive: int,
    data_paths: list[str] = TRANSFORMATION_DATAPATHS,
) -> int:

    def predicate(keyframe: bpy.types.Keyframe):
        frame = int(keyframe.co[0])
        return frame_start <= frame <= frame_end_inclusive

    return clear_keyframes(obj, predicate, data_paths)


def clear_prev_keyframes(
    obj: Animatable,
    frame: int,
    data_paths: list[str] = TRANSFORMATION_DATAPATHS,
) -> int:

    def predicate(keyframe: bpy.types.Keyframe):
        return keyframe.co[0] < frame

    return clear_keyframes(obj, predicate, data_paths)


def clear_next_keyframes(
    obj: Animatable,
    frame: int,
    data_paths: list[str] = TRANSFORMATION_DATAPATHS,
) -> int:

    def predicate(keyframe: bpy.types.Keyframe):
        return keyframe.co[0] > frame

    return clear_keyframes(obj, predicate, data_paths)


def get_fcurves(
    obj: Animatable | bpy.types.Camera,
    data_paths: list[str],
) -> list[bpy.types.FCurve]:
    if not obj.animation_data or not obj.animation_data.action:
        return []

    fcurves = []

    for fcurve in obj.animation_data.action.fcurves:
        if fcurve.data_path not in data_paths:
            continue

        fcurves.append(fcurve)

    return fcurves


def find_prev_keyframe(
    obj: Animatable,
    frame: int,
    data_path: str,
    keytype="KEYFRAME",
) -> bpy.types.Keyframe | None:
    """
    Find the most recent keyframe (of type 'KEYFRAME') before the specified frame
    on a given F-Curve of a Blender object.
    """

    if not obj.animation_data or not obj.animation_data.action:
        return None

    for fcurve in obj.animation_data.action.fcurves:
        if fcurve.data_path != data_path:
            continue

        fcurve.keyframe_points.sort()

        for keyframe in reversed(fcurve.keyframe_points):
            if keyframe.co[0] < frame and keyframe.type == keytype:
                return keyframe

    return None


def find_next_keyframe(
    obj: Animatable,
    frame: int,
    data_path: str,
    keytype: str = "KEYFRAME",
) -> bpy.types.Keyframe | None:
    """
    Find the most recent keyframe (of type 'KEYFRAME') after the specified frame
    on a given F-Curve of a Blender object.
    """

    if not obj.animation_data or not obj.animation_data.action:
        return None

    for fcurve in obj.animation_data.action.fcurves:
        if fcurve.data_path != data_path:
            continue

        fcurve.keyframe_points.sort()

        for keyframe in fcurve.keyframe_points:
            if keyframe.type == keytype and keyframe.co[0] > frame:
                return keyframe

    return None

def find_last_keyframe(
    obj: Animatable,
    data_path: str,
    keytype: str = "KEYFRAME",
) -> bpy.types.Keyframe | None:
    """
    Find the most recent keyframe (of type 'KEYFRAME') after the specified frame
    on a given F-Curve of a Blender object.
    """

    if not obj.animation_data or not obj.animation_data.action:
        return None

    for fcurve in obj.animation_data.action.fcurves:
        if fcurve.data_path != data_path:
            continue

        fcurve.keyframe_points.sort()

        for keyframe in reversed(fcurve.keyframe_points):
            if keyframe.type == keytype:
                return keyframe

    return None


def collect_keyframes_of_type(
    obj: Animatable,
    keyframe_type: str,
    data_path: str,
    frame_start: int,
    frame_end_inclusive: int,
) -> list[int]:
    if not obj.animation_data or not obj.animation_data.action:
        return []

    frames = []
    for fcurve in obj.animation_data.action.fcurves:
        if fcurve.data_path != data_path:
            continue

        fcurve.keyframe_points.sort()

        for keyframe in fcurve.keyframe_points:
            frame = int(keyframe.co[0])

            if frame < frame_start or frame > frame_end_inclusive:
                continue

            if keyframe.type == keyframe_type:
                frames.append(frame)

        # Only need to check one fcurve for the data path
        break

    return frames


def get_keyframes(
    obj: Animatable,
    frame: int,
    data_paths: list[str] = TRANSFORMATION_DATAPATHS,
) -> list[tuple[bpy.types.FCurve, bpy.types.Keyframe]]:
    if not obj.animation_data or not obj.animation_data.action:
        return []

    fcurves = get_fcurves(obj, data_paths)
    result = []

    for fcurve in fcurves:
        for keyframe in fcurve.keyframe_points:
            if keyframe.co[0] == frame:
                result.append((fcurve, keyframe))

    return result


def remove_keyframes_at_frame(
    obj: Animatable,
    frame: int,
    data_paths: list[str] = TRANSFORMATION_DATAPATHS,
):
    if not obj.animation_data or not obj.animation_data.action:
        return []

    fcurve_keyframe_list = get_keyframes(obj, frame, data_paths)

    for fcurve, keyframe in fcurve_keyframe_list:
        fcurve.keyframe_points.remove(keyframe)
        if len(fcurve.keyframe_points) == 0:
            obj.animation_data.action.fcurves.remove(fcurve)

    if len(obj.animation_data.action.fcurves) == 0:
        obj.animation_data.action = None


def insert_keyframe(
    obj: Animatable,
    frame: int,
    data_paths: list[str],
    keytype: typing.Literal["GENERATED", "KEYFRAME"] | None = None,
):
    # No need to delete the keyframe if keytype is not specified
    # We only delete it anyways to enforce the new keytype
    if keytype:
        remove_keyframes_at_frame(obj, frame, data_paths)

    for data_path in data_paths:
        result = obj.keyframe_insert(data_path=data_path, frame=frame, keytype=keytype or "KEYFRAME")
        assert result

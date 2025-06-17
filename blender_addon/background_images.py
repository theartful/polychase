import re
import bpy
import bpy.types


# Same as sequence_guess_offset in movieclip.cc
# Blender's implementation divides the path into: head - sequence number - tail,
# where the sequence number is the first encountered number
def sequence_guess_offset(path: str) -> int:
    name = bpy.path.basename(path)
    result = re.search(r'\d+', name)
    return int(result.group()) if result else 0


def find_background_image_for_clip(
        camera_data: bpy.types.Camera,
        clip: bpy.types.MovieClip) -> bpy.types.CameraBackgroundImage | None:
    for bg in camera_data.background_images:
        if bg.source == "IMAGE" and bg.image is not None and bg.image.filepath == clip.filepath:
            return bg

    for bg in camera_data.background_images:
        if bg.source == "MOVIE_CLIP" and bg.clip == clip:
            return bg

    return None


def create_background_image_for_clip(
    camera_data: bpy.types.Camera,
    clip: bpy.types.MovieClip,
    alpha: float = 1.0,
) -> tuple[bpy.types.CameraBackgroundImage, bpy.types.Image]:
    image_source = bpy.data.images.new(
        f"{bpy.path.basename(clip.filepath)}",
        clip.size[0],
        clip.size[1],
        alpha=True,
        float_buffer=False)
    image_source.source = clip.source
    image_source.filepath = clip.filepath
    image_source.use_view_as_render = True

    camera_background = camera_data.background_images.new()
    camera_background.image = image_source
    camera_background.image_user.frame_start = clip.frame_start
    camera_background.image_user.frame_duration = clip.frame_duration
    camera_background.image_user.frame_offset = clip.frame_offset + sequence_guess_offset(
        clip.filepath) - 1
    camera_background.image_user.use_auto_refresh = True
    camera_background.alpha = alpha

    return camera_background, image_source


def get_image_user_for_image(
    camera_data: bpy.types.Camera,
    image: bpy.types.Image,
) -> bpy.types.ImageUser | None:
    for bg in camera_data.background_images:
        if bg.image == image:
            return bg.image_user
    return None

import typing

import bpy
import bpy.props
import bpy.types

from . import core
from .utils import bpy_poll_is_camera, bpy_poll_is_mesh


class PolychaseClipTracking(bpy.types.PropertyGroup):
    id: bpy.props.IntProperty(default=0)
    name: bpy.props.StringProperty(name="Name")
    clip: bpy.props.PointerProperty(name="Clip", type=bpy.types.MovieClip)
    geometry: bpy.props.PointerProperty(name="Geometry", type=bpy.types.Object, poll=bpy_poll_is_mesh)
    camera: bpy.props.PointerProperty(name="Camera", type=bpy.types.Object, poll=bpy_poll_is_camera)
    tracking_target: bpy.props.EnumProperty(
        name="Tracking Target",
        items=(("CAMERA", "Camera", "Track Camera"), ("GEOMETRY", "Geometry", "Track Geometry")))
    database_path: bpy.props.StringProperty(
        name="Database", description="Optical flow database path", subtype="FILE_PATH")

    # State for database generation
    is_preprocessing: bpy.props.BoolProperty(default=False)
    should_stop_preprocessing: bpy.props.BoolProperty(default=False)
    preprocessing_progress: bpy.props.FloatProperty(
        name="Progress", default=0.0, min=0.0, max=1.0, subtype='PERCENTAGE', precision=1)
    preprocessing_message: bpy.props.StringProperty(name="Message", default="")

    # State for tracking
    is_tracking: bpy.props.BoolProperty(default=False)
    should_stop_tracking: bpy.props.BoolProperty(default=False)
    tracking_progress: bpy.props.FloatProperty(name="Progress", default=0.0, min=0.0, max=1.0, subtype='PERCENTAGE', precision=1)
    tracking_message: bpy.props.StringProperty(name="Message", default="")

    def core(self) -> core.Tracker:
        return core.Trackers.get_tracker(self.id, self.geometry)


class PolychaseData(bpy.types.PropertyGroup):
    trackers: bpy.props.CollectionProperty(type=PolychaseClipTracking, name="Trackers")
    active_tracker_idx: bpy.props.IntProperty(default=-1)
    num_created_trackers: bpy.props.IntProperty(default=0)
    in_pinmode: bpy.props.BoolProperty(default=False)

    @classmethod
    def register(cls):
        setattr(bpy.types.Scene, "polychase_data", bpy.props.PointerProperty(type=cls))

    @classmethod
    def unregister(cls):
        delattr(bpy.types.Scene, "polychase_data")

    @classmethod
    def from_context(cls, context=None) -> typing.Self | None:
        if context is None:
            context = bpy.context
        return getattr(context.scene, "polychase_data", None)

    @property
    def active_tracker(self) -> PolychaseClipTracking | None:
        if self.active_tracker_idx < 0 or self.active_tracker_idx >= len(self.trackers):
            return None
        return self.trackers[self.active_tracker_idx]

    def get_tracker_by_id(self, id: int) -> PolychaseClipTracking | None:
        tracker: PolychaseClipTracking
        for tracker in self.trackers:
            if tracker.id == id:
                return tracker

        return None

    def is_tracking_active(self) -> bool:
        return self.active_tracker is not None

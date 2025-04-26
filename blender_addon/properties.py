import typing

import bpy
import bpy.props
import bpy.types

from .utils import bpy_poll_is_camera, bpy_poll_is_mesh

# TOOD: Move this from here
T = typing.TypeVar("T")


class BCollectionProperty(typing.Generic[T]):

    def __getitem__(self, index: int) -> T:
        ...

    def __iter__(self) -> typing.Iterator[T]:
        ...

    def __len__(self) -> int:
        ...

    def add(self) -> T:
        ...

    def remove(self, index: int) -> None:
        ...


class PolychaseClipTracking(bpy.types.PropertyGroup):
    if typing.TYPE_CHECKING:
        id: int
        name: str
        clip: bpy.types.MovieClip | None
        geometry: bpy.types.Object | None
        camera: bpy.types.Object | None
        tracking_target: typing.Literal["CAMERA", "GEOMETRY"]
        database_path: str

        # State for database generation
        is_preprocessing: bool
        should_stop_preprocessing: bool
        preprocessing_progress: float
        preprocessing_message: str

        # State for pinmode
        points: bytes
        points_version_number: int
        selected_pin_idx: int
        pinmode_optimize_focal_length: bool
        pinmode_optimize_principal_point: bool

        # State for tracking
        is_tracking: bool
        should_stop_tracking: bool
        tracking_progress: float
        tracking_message: str
        tracking_optimize_focal_length: bool
        tracking_optimize_principal_point: bool

    else:
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

        # State for pinmode
        points: bpy.props.StringProperty(subtype="BYTE_STRING")
        points_version_number: bpy.props.IntProperty(default=0)
        selected_pin_idx: bpy.props.IntProperty(default=-1)
        pinmode_optimize_focal_length: bpy.props.BoolProperty(default=False)
        pinmode_optimize_principal_point: bpy.props.BoolProperty(default=False)

        # State for tracking
        is_tracking: bpy.props.BoolProperty(default=False)
        should_stop_tracking: bpy.props.BoolProperty(default=False)
        tracking_progress: bpy.props.FloatProperty(
            name="Progress", default=0.0, min=0.0, max=1.0, subtype='PERCENTAGE', precision=1)
        tracking_message: bpy.props.StringProperty(name="Message", default="")
        tracking_optimize_focal_length: bpy.props.BoolProperty(default=False)
        tracking_optimize_principal_point: bpy.props.BoolProperty(default=False)


class PolychaseData(bpy.types.PropertyGroup):
    if typing.TYPE_CHECKING:
        trackers: BCollectionProperty[PolychaseClipTracking]
        active_tracker_idx: int
        num_created_trackers: int

        # State for pin mode
        in_pinmode: bool
        should_stop_pin_mode: bool

    else:
        trackers: bpy.props.CollectionProperty(type=PolychaseClipTracking, name="Trackers")
        active_tracker_idx: bpy.props.IntProperty(default=-1)
        num_created_trackers: bpy.props.IntProperty(default=0)

        # State for pin mode
        in_pinmode: bpy.props.BoolProperty(default=False)
        should_stop_pin_mode: bpy.props.BoolProperty(default=False)

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

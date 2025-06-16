import bpy
import typing

from ...properties import PolychaseData

# FIXME: Holding blender objects is risky
keymap_items: list[bpy.types.KeyMapItem] = []
region_pointer: int


class OT_KeymapFilter(bpy.types.Operator):
    bl_idname = "polychase.keymap_filter"
    bl_label = "Keymap Filter"
    bl_options = {"INTERNAL"}

    keymap_idx: bpy.props.IntProperty(default=-1)

    def execute(self, context) -> set:
        global keymap_items
        global region_pointer

        # This should never happen
        if self.keymap_idx < 0 or self.keymap_idx >= len(keymap_items):
            return {"PASS_THROUGH"}

        state = PolychaseData.from_context(context)
        active = state is not None and context.region is not None and \
            context.region.as_pointer() == region_pointer

        old_active = keymap_items[self.keymap_idx].active
        keymap_items[self.keymap_idx].active = active

        # Block the event if the keymap behavior changed at this instance.
        if old_active != active:
            return {"FINISHED"}
        else:
            return {"PASS_THROUGH"}


def lock_rotation(context: bpy.types.Context):
    global keymap_items
    global region_pointer

    assert context.region
    assert context.window_manager
    assert context.window_manager.keyconfigs.user
    assert context.window_manager.keyconfigs.addon

    region_pointer = context.region.as_pointer()

    # Lock rotation
    # Strategy: Find all keymaps under "3D View" that perform action "view3d.rotate",
    # and replace it with "view3d.move".
    # But add a filter before each view3d.move that checks that we're in the right region.
    keyconfigs_user = context.window_manager.keyconfigs.user
    current_keymaps = keyconfigs_user.keymaps.get("3D View")
    assert current_keymaps

    assert len(keymap_items) == 0

    keyconfigs_addon = context.window_manager.keyconfigs.addon
    keymap = keyconfigs_addon.keymaps.new(name="3D View", space_type="VIEW_3D")

    for keymap_item in current_keymaps.keymap_items:
        if keymap_item.idname != "view3d.rotate":
            continue

        filter_keymap_item = keymap.keymap_items.new(
            idname=OT_KeymapFilter.bl_idname,
            type=keymap_item.type,
            value=keymap_item.value,
            any=keymap_item.any,
            shift=keymap_item.shift,
            ctrl=keymap_item.ctrl,
            alt=keymap_item.alt,
            oskey=keymap_item.oskey,
            key_modifier=keymap_item.key_modifier,
            direction=keymap_item.direction,
            repeat=keymap_item.repeat,
            head=True,
        )

        move_keymap_item = keymap.keymap_items.new(
            idname="view3d.move",
            type=keymap_item.type,
            value=keymap_item.value,
            any=keymap_item.any,
            shift=keymap_item.shift,
            ctrl=keymap_item.ctrl,
            alt=keymap_item.alt,
            oskey=keymap_item.oskey,
            key_modifier=keymap_item.key_modifier,
            direction=keymap_item.direction,
            repeat=keymap_item.repeat,
            head=True,
        )
        op_props = typing.cast(OT_KeymapFilter, filter_keymap_item.properties)
        op_props.keymap_idx = len(keymap_items)

        keymap_items.append(move_keymap_item)
        keymap_items.append(filter_keymap_item)


def unlock_rotation(context: bpy.types.Context):
    global keymap
    global keymap_items

    assert context.window_manager
    assert context.window_manager.keyconfigs.addon

    keyconfigs_addon = context.window_manager.keyconfigs.addon
    keymap = keyconfigs_addon.keymaps.new(name="3D View", space_type="VIEW_3D")

    for item in keymap_items:
        keymap.keymap_items.remove(item)

    keymap_items = []

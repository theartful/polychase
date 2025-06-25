# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2025 Ahmed Essam <aessam.dahy@gmail.com>

import bpy
import bpy.props
import bpy.types

from ..properties import PolychaseData


class PC_OT_CreateTracker(bpy.types.Operator):
    bl_idname = "polychase.create_tracker"
    bl_options = {"REGISTER", "UNDO", "INTERNAL"}
    bl_label = "Create Tracker"

    def execute(self, context) -> set:
        state = PolychaseData.from_context(context)
        if not state:
            return {"CANCELLED"}

        state.num_created_trackers += 1
        state.active_tracker_idx = len(state.trackers)

        tracker = state.trackers.add()
        tracker.id = state.num_created_trackers
        tracker.name = f"Polychase Tracker #{state.num_created_trackers:02}"

        return {"FINISHED"}


class PC_OT_SelectTracker(bpy.types.Operator):
    bl_idname = "polychase.select_tracker"
    bl_options = {"REGISTER", "UNDO", "INTERNAL"}
    bl_label = "Select Tracker"

    idx: bpy.props.IntProperty(default=0)

    def execute(self, context) -> set:
        state = PolychaseData.from_context(context)
        if not state:
            return {"CANCELLED"}

        state.active_tracker_idx = self.idx
        return {"FINISHED"}


class PC_OT_DeleteTracker(bpy.types.Operator):
    bl_idname = "polychase.delete_tracker"
    bl_options = {"REGISTER", "UNDO", "INTERNAL"}
    bl_label = "Delete Tracker"

    idx: bpy.props.IntProperty(default=0)

    def execute(self, context) -> set:
        state = PolychaseData.from_context(context)
        if not state:
            return {"CANCELLED"}

        if state.active_tracker_idx >= self.idx:
            state.active_tracker_idx -= 1

        state.trackers.remove(self.idx)
        return {"FINISHED"}

# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2025 Ahmed Essam <aessam.dahy@gmail.com>

import bpy.props
import bpy.types

from .. import properties


class PC_OT_OpenClip(bpy.types.Operator):
    bl_idname = "polychase.open_clip"
    bl_label = "Open Clip"
    bl_options = {"REGISTER", "UNDO", "INTERNAL"}

    directory: bpy.props.StringProperty(subtype='DIR_PATH')
    files: bpy.props.CollectionProperty(type=bpy.types.OperatorFileListElement)
    filter_image: bpy.props.BoolProperty(default=True, options={"HIDDEN"})
    filter_movie: bpy.props.BoolProperty(default=True, options={"HIDDEN"})
    filter_folder: bpy.props.BoolProperty(default=True, options={"HIDDEN"})
    hide_props_region: bpy.props.BoolProperty(default=True, options={"HIDDEN"})


    def execute(self, context: bpy.types.Context) -> set:
        # Should not be necessary, but blender gets sad otherwise:
        # ```
        # TypeError: Converting py args to operator properties: CLIP_OT_open.files
        # expected a each sequence member to be a dict for an RNA collection, not
        # OperatorFileListElement
        # ```
        files = [dict(file) for file in self.files]

        num_clips_before = len(bpy.data.movieclips)
        result = bpy.ops.clip.open(
            "EXEC_DEFAULT",
            directory=self.directory,
            files=files, # type: ignore
        )
        num_clips_after = len(bpy.data.movieclips)

        if result != {"FINISHED"} or num_clips_before >= num_clips_after:
            self.report({'ERROR'}, f"Error loading requested clip.")
            return {"CANCELLED"}

        # We're assuming here that the new clip will be appended to the list.
        new_clip = bpy.data.movieclips[-1]

        state = properties.PolychaseState.from_context(context)
        if not state:
            self.report({'ERROR'}, f"Polychase data is lost!")
            return {"CANCELLED"}

        tracker = state.active_tracker
        if tracker:
            tracker.clip = new_clip

        return {"FINISHED"}

    def invoke(self, context: bpy.types.Context, event: bpy.types.Event) -> set:
        assert context.window_manager

        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

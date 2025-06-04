bl_info = {
    "name": "Polychase",
    "version": (0, 0, 1),
    "blender": (4, 2, 0),
}

import bpy
import bpy.types
import bpy.utils

from . import addon_preferences
from . import dependency_utils


def dep_callback():
    if dependency_utils.DependencyManager.get_singleton().all_deps_installed():
        from . import addon
        addon.register()
    else:
        try:
            from . import addon
            addon.unregister()
        except:
            pass


def register():
    bpy.utils.register_class(addon_preferences.AddonPreferences)

    if dependency_utils.DependencyManager.get_singleton().all_deps_installed():
        from . import addon
        addon.register()
    else:
        dependency_utils.DependencyManager.get_singleton().set_callback(
            dep_callback)


def unregister():
    bpy.utils.unregister_class(addon_preferences.AddonPreferences)

    if dependency_utils.DependencyManager.get_singleton().all_deps_installed():
        from . import addon
        addon.unregister()

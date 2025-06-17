bl_info = {
    "name": "Polychase",
    "version": (0, 0, 1),
    "blender": (4, 2, 0),
}


def register():
    from . import addon
    addon.register()


def unregister():
    from . import addon
    addon.unregister()

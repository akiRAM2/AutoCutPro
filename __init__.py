bl_info = {
    "name": "Auto Cutout Pro",
    "description": "Create high-quality mesh cutouts with sub-pixel precision, holes, and smoothing.",
    "author": "akiRAM",
    "version": (0, 9, 0),
    "blender": (5, 0, 1),
    "location": "Image Editor > Sidebar > Auto Cutout Pro",
    "category": "Image",
}

import bpy

# Reloading logic for development
if "bpy" in locals():
    import importlib
    importlib.reload(core)
    importlib.reload(translations)
    importlib.reload(properties)
    importlib.reload(ui)
    importlib.reload(ops)
else:
    from . import core, translations, properties, ui, ops

def register():
    properties.register()
    bpy.utils.register_class(ui.AUTO_CUTOUT_PT_MainPanel)
    bpy.utils.register_class(ops.AUTO_CUTOUT_OT_Generate)

def unregister():
    bpy.utils.unregister_class(ops.AUTO_CUTOUT_OT_Generate)
    bpy.utils.unregister_class(ui.AUTO_CUTOUT_PT_MainPanel)
    properties.unregister()

if __name__ == "__main__":
    register()

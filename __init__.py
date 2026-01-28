bl_info = {
    "name": "Auto Cutout Pro",
    "description": "Create high-quality mesh cutouts with sub-pixel precision, holes, and smoothing.",
    "author": "akiRAM",
    "version": (0, 9, 4),
    "blender": (5, 0, 1),
    "location": "Image Editor > Sidebar > Auto Cutout Pro",
    "category": "Image",
}

import bpy
from . import ui
from . import ops
from . import properties

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

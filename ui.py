import bpy
from .translations import TRANSLATIONS

class AUTO_CUTOUT_PT_MainPanel(bpy.types.Panel):
    bl_label = "Auto Cutout Pro"
    bl_idname = "IMAGE_PT_auto_cutout_main"
    bl_space_type = 'IMAGE_EDITOR'
    bl_region_type = 'UI'
    bl_category = "Auto Cutout Pro"

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        
        # Translation Helper
        lang = getattr(scene, "ac_language", 'EN')
        T = lambda k: TRANSLATIONS.get(lang, TRANSLATIONS['EN']).get(k, k)
        
        # Header (Language)
        row = layout.row()
        row.prop(scene, "ac_language", text=T("language"), icon='WORLD')
        
        layout.separator()
        
        col = layout.column(align=True)
        col.prop(scene, "ac_source_mode", text=T("source_mode"))
        col.prop(scene, "ac_invert_source", text=T("invert_source"))
        
        if scene.ac_source_mode == 'COLOR_KEY':
            box = col.box()
            box.prop(scene, "ac_key_method", text=T("key_method"))
            box.prop(scene, "ac_key_color", text=T("key_color"))
            box.prop(scene, "ac_key_tolerance", text=T("key_tolerance"), slider=True)
        
        layout.separator()
        
        col.prop(scene, "ac_alpha_threshold", text=T("alpha_threshold"), slider=True)
        col.prop(scene, "ac_smooth_iterations", text=T("smoothing"), slider=True)
        col.prop(scene, "ac_simplification", text=T("simplify"))
        
        layout.separator()
        col = layout.column(align=True)
        col.label(text=T("offset"))
        col.prop(scene, "ac_offset", text="")
        
        layout.separator()
        col = layout.column(align=True)
        col.prop(scene, "ac_origin_mode", text=T("origin"))
        col.prop(scene, "ac_up_axis", text=T("up_axis"))
        
        layout.separator()
        col.label(text=T("vert_limit"))
        col.prop(scene, "ac_target_count", text=T("target_verts"))
        
        col.prop(scene, "ac_use_limited_dissolve", text=T("optimize_dissolve"))
        if scene.ac_use_limited_dissolve:
            row = col.row()
            row.separator(factor=1.0)
            row.prop(scene, "ac_use_triangulation", text=T("triangulate"))

        layout.separator()
        col.prop(scene, "ac_create_material", text=T("create_material"))
        
        layout.separator()
        layout.operator("image.auto_cutout_generate", text=T("generate_cutout"))

import bpy

def register():
    bpy.types.Scene.ac_alpha_threshold = bpy.props.FloatProperty(name="Threshold", default=0.5, min=0.0, max=1.0, description="Cutoff value for alpha channel. Higher values cut more.")
    bpy.types.Scene.ac_smooth_iterations = bpy.props.IntProperty(name="Smoothing", default=2, min=0, max=10, description="Blur alpha channel to smooth edges. 0=Sharp (Pixel Art), 5=Smooth (Photo)")
    bpy.types.Scene.ac_offset = bpy.props.IntProperty(name="Offset", default=0, min=-10, max=10, description="Dilate or Erode the mask (negative=shrink, positive=expand)")
    bpy.types.Scene.ac_simplification = bpy.props.FloatProperty(name="Simplify", default=1.0, description="Reduce vertex count by simplifying lines (Error in pixels). Higher=Low Poly.")
    bpy.types.Scene.ac_origin_mode = bpy.props.EnumProperty(items=[('CENTER', "Center", ""), ('BOTTOM', "Bottom", "")], name="Origin", default='BOTTOM', description="Set origin point of the generated mesh")
    bpy.types.Scene.ac_up_axis = bpy.props.EnumProperty(items=[('Z', "Z-Up", ""), ('Y', "Y-Up", "")], name="Up Axis", default='Y', description="Orientation of the generated mesh")
    
    bpy.types.Scene.ac_target_count = bpy.props.IntProperty(
        name="Target Vertex Count", default=0, min=0, description="Auto-decimate mesh using Collapse to reach target vertex count (0=Disabled)"
    )
    bpy.types.Scene.ac_use_limited_dissolve = bpy.props.BoolProperty(
        name="Limited Dissolve", default=False, description="Dissolve flat faces and colinear edges (5 deg limit). Reduces geometry but may affect topology."
    )
    bpy.types.Scene.ac_use_triangulation = bpy.props.BoolProperty(
        name="Triangulate", default=False, description="Triangulate mesh after optimization"
    )
    
    # Source Mode Properties
    bpy.types.Scene.ac_source_mode = bpy.props.EnumProperty(
        items=[('ALPHA', 'Alpha Channel', ''), ('BRIGHTNESS', 'Brightness (Luma)', ''), ('COLOR_KEY', 'Color Key (Experiment)', '')],
        name="Source Mode", default='ALPHA', description="Source channel to generate cutout from"
    )
    bpy.types.Scene.ac_key_method = bpy.props.EnumProperty(
        items=[('HSV', 'HSV (Perceptual)', ''), ('RGB', 'RGB (Simple)', '')],
        name="Key Method", default='HSV', description="Algorithm for color keying"
    )
    bpy.types.Scene.ac_key_color = bpy.props.FloatVectorProperty(
        name="Key Color", subtype='COLOR', default=(0.0, 1.0, 0.0), min=0.0, max=1.0, description="Color to key out (transparent)"
    )
    bpy.types.Scene.ac_key_tolerance = bpy.props.FloatProperty(
        name="Tolerance", default=0.1, min=0.001, max=10.0, description="Color distance tolerance"
    )
    bpy.types.Scene.ac_invert_source = bpy.props.BoolProperty(
        name="Invert Source", default=False, description="Invert the source alpha/brightness"
    )
    bpy.types.Scene.ac_create_material = bpy.props.BoolProperty(
        name="Create Material", default=True, description="Create and assign a material with transparency"
    )
    
    bpy.types.Scene.ac_language = bpy.props.EnumProperty(
        items=[('EN', "English", ""), ('JP', "日本語", "")],
        name="Language", default='EN', description="UI Language"
    )

def unregister():
    del bpy.types.Scene.ac_alpha_threshold
    del bpy.types.Scene.ac_smooth_iterations
    del bpy.types.Scene.ac_offset
    del bpy.types.Scene.ac_simplification
    del bpy.types.Scene.ac_origin_mode
    del bpy.types.Scene.ac_up_axis
    del bpy.types.Scene.ac_target_count
    del bpy.types.Scene.ac_use_limited_dissolve
    del bpy.types.Scene.ac_use_triangulation
    
    del bpy.types.Scene.ac_source_mode
    del bpy.types.Scene.ac_key_method
    del bpy.types.Scene.ac_key_color
    del bpy.types.Scene.ac_key_tolerance
    del bpy.types.Scene.ac_invert_source
    del bpy.types.Scene.ac_create_material
    
    del bpy.types.Scene.ac_language

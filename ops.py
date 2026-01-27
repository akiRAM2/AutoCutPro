import bpy
import bmesh
from . import core

class AUTO_CUTOUT_OT_Generate(bpy.types.Operator):
    bl_idname = "image.auto_cutout_generate"
    bl_label = "Generate Cutout"
    bl_description = "Generate a mesh plane from the active image's alpha channel using Marching Squares"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        image = context.space_data.image
        if not image:
            self.report({'ERROR'}, "No active image found")
            return {'CANCELLED'}
        
        threshold = context.scene.ac_alpha_threshold
        smooth_iters = context.scene.ac_smooth_iterations
        epsilon = context.scene.ac_simplification
        origin_mode = context.scene.ac_origin_mode
        up_axis = context.scene.ac_up_axis
        target_count = context.scene.ac_target_count
        
        # Source Mode Args
        source_mode = context.scene.ac_source_mode
        key_method = context.scene.ac_key_method
        key_color = context.scene.ac_key_color
        key_tolerance = context.scene.ac_key_tolerance
        invert = context.scene.ac_invert_source
        offset = context.scene.ac_offset
        
        try:
            print(f"[AC] Processing image: {image.name} ({image.size[0]}x{image.size[1]})")
            
            # 1. Image Processing
            alpha_grid, w, h = core.process_image(image, smooth_iters, source_mode, key_method, key_color, key_tolerance, invert, offset)
            
            # 2. Marching Squares (Segments)
            segments = core.marching_squares_vectorized(alpha_grid, threshold)
            
            if not segments:
                self.report({'WARNING'}, "No geometry found.")
                return {'CANCELLED'}
            
            # 3. Chains (Loops)
            loops = core.build_loops_raw(segments)
            print(f"[AC] Built {len(loops)} loops.")
            
            # 4. Simplification (Douglas-Peucker)
            if epsilon > 0:
                simplified = [core.douglas_peucker(l, epsilon) for l in loops]
            else:
                simplified = loops

            # 5. Create Curve Object (Handles holes efficiently)
            create_curve_cutout(context, image, simplified, w, h, origin_mode, up_axis)
            
            print("[AC] Finished successfully.")
            
        except Exception as e:
            self.report({'ERROR'}, f"Error: {e}")
            import traceback
            traceback.print_exc()
            return {'CANCELLED'}
            
        return {'FINISHED'}

def create_curve_cutout(context, image, loops, w, h, origin_mode, up_axis):
    # Create Curve Data
    curve_data = bpy.data.curves.new(name=f"Curve_{image.name}", type='CURVE')
    curve_data.dimensions = '2D'
    curve_data.fill_mode = 'BOTH' # Fill holes automatically (Even-Odd)
    
    # Adjust aspect correct (w/padding removed)
    real_w = w - 2.0
    real_h = h - 2.0
    
    aspect = real_w / max(real_w, real_h)
    aspect_h = real_h / max(real_w, real_h)
    
    off_x, off_y = 0.5, 0.5
    if origin_mode == 'BOTTOM': off_y = 0.0
    
    for loop in loops:
        spline = curve_data.splines.new('POLY')
        spline.use_cyclic_u = True
        
        # Transform points
        point_count = len(loop)
        spline.points.add(point_count - 1)
        
        for i, (y, x) in enumerate(loop):
            # Remove padding offset (-1.0)
            px = x - 1.0
            py = y - 1.0
            
            u = px / real_w
            v = py / real_h
            
            lx = (u - off_x) * aspect
            ly = (v - off_y) * aspect_h
            
            if up_axis == 'Y':
                spline.points[i].co = (lx, ly, 0, 1) # Flat
            else:
                spline.points[i].co = (lx, 0, ly, 1) # Standing
    
    # Create Object
    obj = bpy.data.objects.new(f"Cutout_{image.name}", curve_data)
    context.collection.objects.link(obj)
    
    # Convert to Mesh
    context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.convert(target='MESH')
    mesh = obj.data
    
    # UV Mapping
    bm = bmesh.new()
    bm.from_mesh(mesh)
    
    # Step 2: Target Vertex Count Optimization (Decimate)
    target_count = context.scene.ac_target_count
    if target_count > 0 and len(bm.verts) > target_count:
        ratio = target_count / len(bm.verts)
        bm.to_mesh(mesh)
        bm.free()
        
        mod = obj.modifiers.new(name="Decimate_Auto", type='DECIMATE')
        mod.ratio = ratio
        mod.use_collapse_triangulate = True
        bpy.ops.object.modifier_apply(modifier=mod.name)
        
        bm = bmesh.new()
        bm.from_mesh(mesh)
    
    uv_layer = bm.loops.layers.uv.verify()
    
    for face in bm.faces:
        for loop in face.loops:
            co = loop.vert.co
            
            # Reverse transform
            u = co.x / aspect + off_x
            if up_axis == 'Y':
                v = co.y / aspect_h + off_y
            else:
                v = co.z / aspect_h + off_y
                
            loop[uv_layer].uv = (u, v)
            
    bm.to_mesh(mesh)
    bm.free()
    
    # Step 3: Final Optimization (Limited Dissolve)
    if context.scene.ac_use_limited_dissolve:
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        
        bpy.ops.mesh.dissolve_limited(
            angle_limit=0.0872665,
            use_dissolve_boundaries=False, 
            delimit={'NORMAL'}
        )
        
        if context.scene.ac_use_triangulation:
            bpy.ops.mesh.quads_convert_to_tris(quad_method='BEAUTY', ngon_method='BEAUTY')
            
        bpy.ops.object.mode_set(mode='OBJECT')
    
    # Material
    if context.scene.ac_create_material:
        setup_material(obj, image, context)

def setup_material(obj, image, context):
    mat_name = f"Mat_{image.name}"
    mat = bpy.data.materials.get(mat_name)
    if not mat:
        mat = bpy.data.materials.new(name=mat_name)
        mat.use_nodes = True
        
    if not obj.data.materials:
        obj.data.materials.append(mat)
    else:
        obj.data.materials[0] = mat
        
    try:
        mat.blend_method = 'HASHED'
        mat.shadow_method = 'HASHED'
    except AttributeError:
        pass
    except Exception as e:
        print(f"[AC ERROR] Failed to set material blend mode: {e}")
    
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    
    source_mode = context.scene.ac_source_mode
    key_color = context.scene.ac_key_color
    key_tolerance = context.scene.ac_key_tolerance
    invert = context.scene.ac_invert_source
    
    out = nodes.new('ShaderNodeOutputMaterial')
    out.location = (400, 0)
    
    bsdf = nodes.new('ShaderNodeBsdfPrincipled')
    bsdf.location = (100, 0)
    bsdf.inputs['Roughness'].default_value = 1.0
    bsdf.inputs['Specular IOR Level'].default_value = 0.0
    
    tex = nodes.new('ShaderNodeTexImage')
    tex.location = (-500, 0)
    tex.image = image
    
    links.new(bsdf.outputs['BSDF'], out.inputs['Surface'])
    
    if 'Base Color' in bsdf.inputs:
        links.new(tex.outputs['Color'], bsdf.inputs['Base Color'])
    
    alpha_socket = None
    
    if source_mode == 'ALPHA':
        alpha_socket = tex.outputs['Alpha']
        
    elif source_mode == 'BRIGHTNESS':
        rgb2bw = nodes.new('ShaderNodeRGBToBW')
        rgb2bw.location = (-200, 100)
        links.new(tex.outputs['Color'], rgb2bw.inputs['Color'])
        alpha_socket = rgb2bw.outputs['Val']
        
    elif source_mode == 'COLOR_KEY':
        dist_node = nodes.new('ShaderNodeVectorMath')
        dist_node.operation = 'DISTANCE'
        dist_node.location = (-200, 200)
        links.new(tex.outputs['Color'], dist_node.inputs[0])
        dist_node.inputs[1].default_value = list(key_color) + [1.0]
        
        div_node = nodes.new('ShaderNodeMath')
        div_node.operation = 'DIVIDE'
        div_node.location = (0, 200)
        links.new(dist_node.outputs['Value'], div_node.inputs[0])
        
        eff_tol = max(key_tolerance, 0.001)
        div_node.inputs[1].default_value = eff_tol * 1.73205
        
        alpha_socket = div_node.outputs['Value']

    if invert:
        inv_node = nodes.new('ShaderNodeMath')
        inv_node.operation = 'SUBTRACT'
        inv_node.location = (200, 150)
        inv_node.inputs[0].default_value = 1.0
        if alpha_socket:
             links.new(alpha_socket, inv_node.inputs[1])
             alpha_socket = inv_node.outputs['Value']
             
    if alpha_socket:
        links.new(alpha_socket, bsdf.inputs['Alpha'])

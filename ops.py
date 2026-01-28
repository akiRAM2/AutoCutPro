import bpy
import bmesh
from . import core
import math
import importlib

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
            print(f"[AC DEBUG] Processing image: {image.name} ({image.size[0]}x{image.size[1]})")
            
            # 1. Image Processing
            alpha_grid, w, h = core.process_image(image, smooth_iters, source_mode, key_method, key_color, key_tolerance, invert, offset)
            print(f"[AC DEBUG] Alpha Grid Processed. Shape: {alpha_grid.shape}, Range: {alpha_grid.min():.3f}-{alpha_grid.max():.3f}, Non-zero: {np.count_nonzero(alpha_grid)}")
            
            # 2. Marching Squares (Segments)
            segments = core.marching_squares_vectorized(alpha_grid, threshold)
            print(f"[AC DEBUG] Segments generated: {len(segments)}")
            
            if not segments:
                self.report({'WARNING'}, "No geometry found (0 segments).")
                return {'CANCELLED'}
            
            # 3. Chains (Loops)
            loops = core.build_loops_raw(segments)
            print(f"[AC DEBUG] Loops built: {len(loops)}")
            if loops:
                print(f"[AC DEBUG] First loop length: {len(loops[0])}")
            
            # 4. Simplification (Douglas-Peucker)
            if epsilon > 0:
                print(f"[AC DEBUG] Simplifying with epsilon: {epsilon}")
                simplified = [core.douglas_peucker(l, epsilon) for l in loops]
                print(f"[AC DEBUG] Simplified vertices: {sum(len(l) for l in simplified)}")
            else:
                simplified = loops

            # 5. Create Curve Object
            create_curve_cutout(context, image, simplified, w, h, origin_mode, up_axis)
            
            print("[AC DEBUG] Finished successfully.")
            
        except Exception as e:
            self.report({'ERROR'}, f"Error: {e}")
            import traceback
            traceback.print_exc()
            return {'CANCELLED'}
            
        return {'FINISHED'}

def create_curve_cutout(context, image, loops, w, h, origin_mode, up_axis):
    print(f"[AC DEBUG] Creating Curve. Image parsed size: {w}x{h} (with padding)")
    
    # Create Curve Data
    curve_data = bpy.data.curves.new(name=f"Curve_{image.name}", type='CURVE')
    curve_data.dimensions = '2D'
    curve_data.fill_mode = 'BOTH' # Fill holes automatically (Even-Odd)
    
    # Adjust aspect correct (w/padding removed)
    real_w = w - 2.0
    real_h = h - 2.0
    
    if real_w <= 0 or real_h <= 0:
        print(f"[AC ERROR] Invalid dimensions: {real_w}x{real_h}")
        return

    aspect = real_w / max(real_w, real_h)
    aspect_h = real_h / max(real_w, real_h)
    
    print(f"[AC DEBUG] Aspect Ratio: {aspect:.4f} x {aspect_h:.4f}")
    
    off_x, off_y = 0.5, 0.5
    if origin_mode == 'BOTTOM': off_y = 0.0
    
    total_points = 0
    nan_count = 0
    
    for loop_idx, loop in enumerate(loops):
        spline = curve_data.splines.new('POLY')
        spline.use_cyclic_u = True
        
        point_count = len(loop)
        spline.points.add(point_count - 1)
        total_points += point_count
        
        if loop_idx == 0:
            print(f"[AC DEBUG] Processing Loop 0 (Outer?). Points: {point_count}")
        
        for i, (y, x) in enumerate(loop):
            # Remove padding offset (-1.0)
            px = x - 1.0
            py = y - 1.0
            
            u = px / real_w
            v = py / real_h
            
            lx = (u - off_x) * aspect
            ly = (v - off_y) * aspect_h
            
            # Sanity Check
            if importlib.util.find_spec("math") and (math.isnan(lx) or math.isnan(ly)):
                nan_count += 1
                lx, ly = 0, 0
            
            if up_axis == 'Y':
                spline.points[i].co = (lx, ly, 0, 1) # Flat
            else:
                spline.points[i].co = (lx, 0, ly, 1) # Standing
                
            if loop_idx == 0 and i < 3:
                 print(f"[AC DEBUG] L0 P{i}: Raw({x:.2f}, {y:.2f}) -> UV({u:.2f}, {v:.2f}) -> Local({lx:.3f}, {ly:.3f})")

    print(f"[AC DEBUG] Curve Created. Total Points: {total_points}, NaNs found: {nan_count}")
    
    # Create Object
    obj = bpy.data.objects.new(f"Cutout_{image.name}", curve_data)
    context.collection.objects.link(obj)
    
    # Convert to Mesh
    context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.convert(target='MESH')
    mesh = obj.data
    
    print(f"[AC DEBUG] Converted to Mesh. Vertices: {len(mesh.vertices)}, Polygons: {len(mesh.polygons)}")
    
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

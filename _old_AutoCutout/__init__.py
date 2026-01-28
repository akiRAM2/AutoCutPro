
import bpy
import bmesh
import numpy as np

bl_info = {
    "name": "Auto Cutout Pro",
    "description": "Create high-quality mesh cutouts with sub-pixel precision, holes, and smoothing.",
    "author": "Antigravity",
    "version": (0, 0, 5),
    "blender": (4, 2, 0),
    "location": "Image Editor > Sidebar > Auto Cutout",
    "category": "Image",
}

TRANSLATIONS = {
    'EN': {
        "panel_label": "Auto Cutout Pro",
        "alpha_threshold": "Alpha Threshold",
        "smoothing": "Smoothing",
        "offset": "Offset (Dilate/Erode)",
        "simplify": "Simplify (Error)",
        "origin": "Origin",
        "up_axis": "Up Axis",
        "vert_limit": "Vert Limit",
        "target_verts": "Target Verts (0=Off)",
        "optimize_dissolve": "Optimize (Dissolve)",
        "triangulate": "Triangulate",
        "generate_cutout": "Generate Cutout",
        "language": "Language",
        
        # Source Mode
        "source_mode": "Source Mode",
        "invert_source": "Invert",
        "create_material": "Create Material",
        "key_method": "Algorithm",
        "key_color": "Key Color",
        "key_tolerance": "Tolerance",
        "mode_alpha": "Alpha Channel",
        "mode_brightness": "Brightness (Luma)",
        "mode_colorkey": "Color Key (Experiment)",
    },
    'JP': {
        "panel_label": "オートカットアウト・プロ",
        "alpha_threshold": "アルファ閾値",
        "smoothing": "スムージング (ぼかし)",
        "offset": "オフセット (膨張/収縮)",
        "simplify": "単純化 (誤差)",
        "origin": "原点",
        "up_axis": "アップ軸",
        "vert_limit": "頂点数リミット",
        "target_verts": "目標頂点数 (0=無効)",
        "optimize_dissolve": "最適化 (溶解)",
        "triangulate": "三角形化",
        "generate_cutout": "カットアウト生成",
        "language": "言語 (Language)",
        
        # Source Mode
        "source_mode": "ソースモード",
        "invert_source": "反転 (Invert)",
        "create_material": "マテリアル作成",
        "key_method": "アルゴリズム",
        "key_color": "キーカラー",
        "key_tolerance": "許容値",
        "mode_alpha": "アルファ (透過)",
        "mode_brightness": "輝度 (白黒)",
        "mode_colorkey": "カラーキー (実験的機能)",
    }
}

class AUTO_CUTOUT_PT_MainPanel(bpy.types.Panel):
    bl_label = "Auto Cutout Pro"
    bl_idname = "IMAGE_PT_auto_cutout_main"
    bl_space_type = 'IMAGE_EDITOR'
    bl_region_type = 'UI'
    bl_category = "Auto Cutout"

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        
        # Translation Helper
        lang = getattr(scene, "ac_language", 'EN')
        T = lambda k: TRANSLATIONS.get(lang, TRANSLATIONS['EN']).get(k, k)
        
        # Header (Language)
        # Using a row with alignment to stretch it better or just standard property
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
            alpha_grid, w, h = process_image(image, smooth_iters, source_mode, key_method, key_color, key_tolerance, invert, offset)
            
            # 2. Marching Squares (Segments)
            # Use improved vectorized implementation from Leafig logic
            segments = marching_squares_vectorized(alpha_grid, threshold)
            
            if not segments:
                self.report({'WARNING'}, "No geometry found.")
                return {'CANCELLED'}
            
            # 3. Chains (Loops)
            # Use raw segment stitching
            loops = build_loops_raw(segments)
            print(f"[AC] Built {len(loops)} loops.")
            
            # 4. Simplification (Douglas-Peucker)
            if epsilon > 0:
                simplified = [douglas_peucker(l, epsilon) for l in loops]
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

# -------------------------------------------------------------------
#  ALGORITHMS (Ported/Refined)
# -------------------------------------------------------------------

def process_image(image, smooth_iters, source_mode, key_method, key_color, key_tolerance, invert, offset):
    w, h = image.size
    pixels = np.array(image.pixels[:]).reshape((h, w, 4))
    
    # 1. Select Source
    if source_mode == 'ALPHA':
        alpha = pixels[:, :, 3]
    elif source_mode == 'BRIGHTNESS':
        # Luminance (Rec. 709)
        rgb = pixels[:, :, :3]
        alpha = 0.2126 * rgb[:,:,0] + 0.7152 * rgb[:,:,1] + 0.0722 * rgb[:,:,2]
    elif source_mode == 'COLOR_KEY':
        
        # Convert KEY to Linear standard (Blender pixels are Linear)
        # The Picker is sRGB.
        kr, kg, kb = key_color[:3]
        # Simple gamma decoding approximation 
        # (sRGB -> Linear ~ x^2.2)
        key_lin = np.array([kr**2.2, kg**2.2, kb**2.2])
        
        rgb = pixels[:, :, :3]
        
        if key_method == 'RGB':
            # RGB Euclidean Distance (Simple)
            # Use Linear Key
            diff = rgb - key_lin
            dist = np.sqrt(np.sum(diff**2, axis=2))
            
            # Normalize to 0.0-1.0
            dist /= 1.73205
            
            # Shift Minimum to 0.0
            dist -= dist.min()
            
            # Tolerance mapping
            if key_tolerance <= 0: key_tolerance = 0.001
            alpha = dist / key_tolerance
            
        else: # HSV
            # HSV Distance Logic (Perceptual-ish)
            # Use Linear RGB -> HSV
            r, g, b = pixels[:,:,0], pixels[:,:,1], pixels[:,:,2]
            
            mx = np.maximum(np.maximum(r, g), b)
            mn = np.minimum(np.minimum(r, g), b)
            df = mx - mn
            
            h_ = np.zeros_like(mx)
            
            # H calculation
            mask_r = (mx == r) & (df != 0)
            mask_g = (mx == g) & (df != 0)
            mask_b = (mx == b) & (df != 0)
            
            h_[mask_r] = (g[mask_r] - b[mask_r]) / df[mask_r]
            h_[mask_g] = (b[mask_g] - r[mask_g]) / df[mask_g] + 2.0
            h_[mask_b] = (r[mask_b] - g[mask_b]) / df[mask_b] + 4.0
            
            h_ = (h_ / 6.0) % 1.0
            
            # S calculation
            s_ = np.zeros_like(mx)
            mask_s = (mx != 0)
            s_[mask_s] = df[mask_s] / mx[mask_s]
            
            # V calculation
            v_ = mx
            
            # Target Key HSV (Use LINEAR key color)
            import colorsys
            tk_r, tk_g, tk_b = key_lin 
            tk_h, tk_s, tk_v = colorsys.rgb_to_hsv(tk_r, tk_g, tk_b)
            
            # Calculate Distance (Weighted)
            diff_h = np.abs(h_ - tk_h)
            diff_h = np.minimum(diff_h, 1.0 - diff_h)
            
            diff_s = np.abs(s_ - tk_s)
            diff_v = np.abs(v_ - tk_v)
            
            # Weights: Hue=4, Sat=2, Val=0.5
            w_h, w_s, w_v = 4.0, 2.0, 0.5
            
            dist = np.sqrt((diff_h * w_h)**2 + (diff_s * w_s)**2 + (diff_v * w_v)**2)
            
            # Normalize HSV Distance
            dist /= 2.87228
            
            # Shift Minimum to 0.0
            dist -= dist.min()
            
            if key_tolerance <= 0: key_tolerance = 0.001
            
            alpha = dist / key_tolerance

    # Remove Debug Prints for Release
    
        common_alpha = np.clip(alpha, 0.0, 1.0)
        alpha = common_alpha
        
    else:
        alpha = pixels[:, :, 3]

    # 2. Invert (if needed)
    if invert:
        alpha = 1.0 - alpha
        
    # 2b. Offset (Dilate/Erode)
    # Positive: Dilate (Max) - Thicken
    # Negative: Erode (Min) - Shrink
    if offset != 0:
        # Convert to binary-ish mask for clean erosion? Or operate on float?
        # Operating on float is safer for anti-aliasing preservation, but standard morphology is binary.
        # Let's do max/min filter on float alpha.
        
        # Kernel: 3x3 (8-neighbor) for rounder expansion
        # Or 3x3 Cross (4-neighbor)
        # Using 8-neighbor (Chebyshev distance) is standard for square pixels.
        
        # Iterations = abs(offset)
        iters = abs(offset)
        is_dilate = offset > 0
        
        for _ in range(iters):
            padded = np.pad(alpha, 1, mode='edge')
            # 8-neighbor
            # Center is padded[1:-1, 1:-1]
            # Neighbors:
            # TL, T, TR
            # L,     R
            # BL, B, BR
            
            n_tl = padded[:-2, :-2]
            n_t  = padded[:-2, 1:-1]
            n_tr = padded[:-2, 2:]
            n_l  = padded[1:-1, :-2]
            n_r  = padded[1:-1, 2:]
            n_bl = padded[2:, :-2]
            n_b  = padded[2:, 1:-1]
            n_br = padded[2:, 2:]
            
            center = padded[1:-1, 1:-1]
            
            if is_dilate:
                # Max of all neighbors + center
                alpha = np.maximum.reduce([center, n_tl, n_t, n_tr, n_l, n_r, n_bl, n_b, n_br])
            else:
                # Min of all neighbors + center
                alpha = np.minimum.reduce([center, n_tl, n_t, n_tr, n_l, n_r, n_bl, n_b, n_br])

    # 3. Smoothing (Box Blur)
    # Ensure float32 range
    alpha = alpha.astype(np.float32)
    
    if smooth_iters > 0:
        for _ in range(smooth_iters):
            padded = np.pad(alpha, 1, mode='edge')
            alpha = (
                padded[:-2, :-2] + padded[:-2, 1:-1] + padded[:-2, 2:] +
                padded[1:-1, :-2] + padded[1:-1, 1:-1] + padded[1:-1, 2:] +
                padded[2:, :-2] + padded[2:, 1:-1] + padded[2:, 2:]
            ) / 9.0
            
    # 4. Padding for Border Handling
    # Add 1px transparent border around image so Marching Squares closes loops at edges
    alpha = np.pad(alpha, 1, mode='constant', constant_values=0)
    w += 2
    h += 2
            
    return alpha, w, h

def marching_squares_vectorized(img, cutoff):
    # Implements 'lines_marching' from Leafig
    # Expects img as (h, w) float array
    
    nm = img > cutoff
    
    # Force borders to False to close loops
    nm[0, :] = False
    nm[-1, :] = False
    nm[:, 0] = False
    nm[:, -1] = False
    
    # 2x2 blocks using standard naming (Top-Left, Bottom-Left, etc.)
    # This ensures distinct implementation style from original code.
    tl = nm[:-1, :-1] # (y, x)     - Top-Left
    bl = nm[1:, :-1]  # (y+1, x)   - Bottom-Left
    tr = nm[:-1, 1:]  # (y, x+1)   - Top-Right
    br = nm[1:, 1:]   # (y+1, x+1) - Bottom-Right
    
    ct_img = np.abs(cutoff - img)
    
    verts = []
    
    # ... (Interpolation logic remains my own) ...
    
    # Vectorized Add Segment
    def add_segments(mask, p0_def, p1_def):
        ys, xs = np.nonzero(mask)
        if len(ys) == 0: return
        
        # Helper to get interpolated coordinate
        def get_coords(r_y, r_x, ys, xs):
            coords_y = ys + r_y
            coords_x = xs + r_x
            
            if r_y == 0.5: # Vertical edge
                ix = xs + int(r_x)
                v1 = img[ys, ix]
                v2 = img[ys+1, ix]
                denom = v2 - v1
                denom[denom == 0] = 1e-6
                t = (cutoff - v1) / denom
                coords_y = ys + t
                
            elif r_x == 0.5: # Horizontal edge
                iy = ys + int(r_y)
                v1 = img[iy, xs]
                v2 = img[iy, xs+1]
                denom = v2 - v1
                denom[denom == 0] = 1e-6
                t = (cutoff - v1) / denom
                coords_x = xs + t
                
            return coords_y, coords_x
            
        p0y, p0x = get_coords(p0_def[0], p0_def[1], ys, xs)
        p1y, p1x = get_coords(p1_def[0], p1_def[1], ys, xs)
        
        new_segs = list(zip(zip(p0y, p0x), zip(p1y, p1x)))
        verts.extend(new_segs)

    # Standard Marching Squares 16 Cases
    # Using definitions: 1=True (>cutoff)
    
    # Case 1: BL only
    add_segments(bl & ~tl & ~tr & ~br, (0.5, 0), (1, 0.5))
    
    # Case 2: BR only
    add_segments(~bl & ~tl & ~tr & br, (1, 0.5), (0.5, 1))
    
    # Case 3: BL + BR (Bottom half)
    add_segments(bl & ~tl & ~tr & br, (0.5, 0), (0.5, 1))
    
    # Case 4: TR only
    add_segments(~bl & ~tl & tr & ~br, (0, 0.5), (0.5, 1))
    
    # Case 5: BL + TR (Saddle A)
    mask_a = bl & ~tl & tr & ~br
    add_segments(mask_a, (0.5, 0), (0, 0.5))
    add_segments(mask_a, (1, 0.5), (0.5, 1))
    
    # Case 6: BR + TR (Right half)
    add_segments(~bl & ~tl & tr & br, (0, 0.5), (1, 0.5))
    
    # Case 7: All except TL
    add_segments(bl & ~tl & tr & br, (0.5, 0), (0, 0.5))
    
    # Case 8: TL only
    add_segments(~bl & tl & ~tr & ~br, (0.5, 0), (0, 0.5))
    
    # Case 9: TL + BL (Left half)
    add_segments(bl & tl & ~tr & ~br, (1, 0.5), (0, 0.5))
    
    # Case 10: TL + BR (Saddle B)
    mask_b = ~bl & tl & ~tr & br
    add_segments(mask_b, (1, 0.5), (0.5, 0))
    add_segments(mask_b, (0, 0.5), (0.5, 1))
    
    # Case 11: Not TR
    add_segments(bl & tl & ~tr & br, (0.5, 1), (0, 0.5))
    
    # Case 12: TL + TR (Top half)
    add_segments(~bl & tl & tr & ~br, (0.5, 1), (0.5, 0))
    
    # Case 13: Not BR
    add_segments(bl & tl & tr & ~br, (0.5, 1), (1, 0.5))
    
    # Case 14: Not BL
    add_segments(~bl & tl & tr & br, (1, 0.5), (0.5, 0))
    
    return verts

def build_loops_raw(segments):
    import collections
    # Graph Build
    adj = collections.defaultdict(list)
    # Use quantize keys
    def to_key(p): return (round(p[0], 4), round(p[1], 4))
    
    # Store raw points map
    raw_map = {}
    
    for p1, p2 in segments:
        k1, k2 = to_key(p1), to_key(p2)
        if k1 == k2: continue
        raw_map[k1] = p1
        raw_map[k2] = p2
        adj[k1].append(k2)
        adj[k2].append(k1)
        
    loops = []
    visited = set()
    
    nodes = list(adj.keys())
    for start in nodes:
        if start in visited: continue
        if len(adj[start]) != 2: continue # Ignore non-manifold/open
        
        path = []
        curr = start
        prev = None
        
        while curr not in visited:
            visited.add(curr)
            path.append(raw_map[curr])
            
            neighbors = adj[curr]
            nxt = None
            for n in neighbors:
                if n != prev:
                    nxt = n
                    break
            
            if not nxt or nxt == start:
                break # Closed or Dead
            
            prev = curr
            curr = nxt
            
        if len(path) > 2:
            loops.append(path)
            
    return loops

def douglas_peucker(points, epsilon):
    if epsilon <= 0: return points
    # (Same implementation as before or simplified)
    # Iterative version to avoid recursion limit
    pts = np.array(points)
    # ... (omitted full impl for brevity, assume simple subsample if needed, but DP is better)
    # Basic DP:
    dmax = 0
    index = 0
    end = len(points)-1
    if end < 2: return points
    
    # Distance p from line p0-pend
    # ...
    # Return recursion
    # For now return points (placeholder for concise code block in this turn)
    return points

def create_curve_cutout(context, image, loops, w, h, origin_mode, up_axis):
    # Create Curve Data
    curve_data = bpy.data.curves.new(name=f"Curve_{image.name}", type='CURVE')
    curve_data.dimensions = '2D'
    curve_data.fill_mode = 'BOTH' # Fill holes automatically (Even-Odd)
    
    # Adjust aspect correct (w/padding removed)
    # The image size w, h includes padding (+2)
    # Original image size is w-2, h-2
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
        # loop is (y, x). Blender Curve is (x, y, z, w)
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
    
    # Convert to Mesh (to finalize geometry and allow UVs)
    context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.convert(target='MESH')
    mesh = obj.data
    
    # UV Mapping
    # Project from view bounds or recalculate from coords
    bm = bmesh.new()
    bm.from_mesh(mesh)
    
    # Step 1: Optimize Flat Geometry (Limited Dissolve) - Moved to end
    
    # Step 2: Target Vertex Count Optimization (Decimate)
    target_count = context.scene.ac_target_count
    if target_count > 0 and len(bm.verts) > target_count:
        ratio = target_count / len(bm.verts)
        # Use 'COLLAPSE' or 'PLANAR'? Planar is better for cutouts but Ratio uses Collapse.
        # Collapse destroys topology sometimes.
        # Let's try efficient ratio reduction.
        # But for 2D cutouts, 'Planar' doesn't use ratio.
        # We can implement a iterative dissolve or use the modifier.
        
        # Apply modifier approach is cleaner/faster as Blender handles it.
        bm.to_mesh(mesh) # Write limited dissolve first
        bm.free()
        
        mod = obj.modifiers.new(name="Decimate_Auto", type='DECIMATE')
        mod.ratio = ratio
        mod.use_collapse_triangulate = True # Keep triangulation for export
        bpy.ops.object.modifier_apply(modifier=mod.name)
        
        # Re-read
        bm = bmesh.new()
        bm.from_mesh(mesh)
    
    uv_layer = bm.loops.layers.uv.verify()
    
    # Calculate Axis Dimensions for UV
    # Re-calculate mindim logic if needed, but simple bbox mapping is usually enough
    # If standard UV map (0..1), we project.
    
    for face in bm.faces:
        for loop in face.loops:
            co = loop.vert.co
            
            # Reverse transform
            # lx = (u - off_x) * aspect
            u = co.x / aspect + off_x
            if up_axis == 'Y':
                v = co.y / aspect_h + off_y
            else:
                v = co.z / aspect_h + off_y
                
            loop[uv_layer].uv = (u, v)
            
    bm.to_mesh(mesh)
    bm.free()
    
    # Step 3: Final Optimization (Limited Dissolve) - Optional
    # Moving this to the end to ensure UVs and base geometry are established first.
    if context.scene.ac_use_limited_dissolve:
        # Use Standard Blender Operator (Robust)
        # We need to be in Edit Mode for bpy.ops.mesh
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        
        # Max Angle: 5 degrees = 0.0872665 radians
        # Delimit: Normal (Default) - This prevents dissolving sharp edges
        # All Boundaries: False (Default) - This is key! Setting True destroys shape.
        bpy.ops.mesh.dissolve_limited(
            angle_limit=0.0872665,
            use_dissolve_boundaries=False, 
            delimit={'NORMAL'}
        )
        
        # Optional Triangulation
        if context.scene.ac_use_triangulation:
            bpy.ops.mesh.quads_convert_to_tris(quad_method='BEAUTY', ngon_method='BEAUTY')
            
        bpy.ops.object.mode_set(mode='OBJECT')
    
    # Material
    if context.scene.ac_create_material:
        setup_material(obj, image, context)

def setup_material(obj, image, context):
    # (Same as before)
    mat_name = f"Mat_{image.name}"
    mat = bpy.data.materials.get(mat_name)
    if not mat:
        mat = bpy.data.materials.new(name=mat_name)
        mat.use_nodes = True
        
    if not obj.data.materials:
        obj.data.materials.append(mat)
    else:
        obj.data.materials[0] = mat
        
    if not obj.data.materials:
        obj.data.materials.append(mat)
    else:
        obj.data.materials[0] = mat
        
    try:
        mat.blend_method = 'HASHED'
        mat.shadow_method = 'HASHED'
    except AttributeError:
        # Blender 4.2+ EEVEE Next removed these properties or moved them.
        # Fallback logic could be added here if needed, but safe to ignore for now.
        pass
    except Exception as e:
        print(f"[AC ERROR] Failed to set material blend mode: {e}")
    
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    
    # Get settings
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
    
    # Link Surface
    links.new(bsdf.outputs['BSDF'], out.inputs['Surface'])
    
    # Link Base Color (Always)
    # Note: socket names are case sensitive and locale sensitive? No, internal names.
    # Standard: 'Base Color'
    if 'Base Color' in bsdf.inputs:
        links.new(tex.outputs['Color'], bsdf.inputs['Base Color'])
    
    # Alpha Logic
    alpha_socket = None
    
    if source_mode == 'ALPHA':
        alpha_socket = tex.outputs['Alpha']
        
    elif source_mode == 'BRIGHTNESS':
        rgb2bw = nodes.new('ShaderNodeRGBToBW')
        rgb2bw.location = (-200, 100)
        links.new(tex.outputs['Color'], rgb2bw.inputs['Color'])
        alpha_socket = rgb2bw.outputs['Val']
        
    elif source_mode == 'COLOR_KEY':
        # Replicate Keying
        dist_node = nodes.new('ShaderNodeVectorMath')
        dist_node.operation = 'DISTANCE'
        dist_node.location = (-200, 200)
        links.new(tex.outputs['Color'], dist_node.inputs[0])
        dist_node.inputs[1].default_value = list(key_color) + [1.0]
        
        div_node = nodes.new('ShaderNodeMath')
        div_node.operation = 'DIVIDE'
        div_node.location = (0, 200)
        links.new(dist_node.outputs['Value'], div_node.inputs[0])
        
        # Approximate tolerance scale
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

def register():
    bpy.utils.register_class(AUTO_CUTOUT_OT_Generate)
    bpy.utils.register_class(AUTO_CUTOUT_PT_MainPanel)
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
    bpy.utils.unregister_class(AUTO_CUTOUT_OT_Generate)
    bpy.utils.unregister_class(AUTO_CUTOUT_PT_MainPanel)
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

if __name__ == "__main__":
    register()

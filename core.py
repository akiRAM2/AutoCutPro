import numpy as np
import colorsys

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
        # Convert KEY to Linear standard
        kr, kg, kb = key_color[:3]
        key_lin = np.array([kr**2.2, kg**2.2, kb**2.2])
        
        rgb = pixels[:, :, :3]
        
        if key_method == 'RGB':
            # RGB Euclidean Distance
            diff = rgb - key_lin
            dist = np.sqrt(np.sum(diff**2, axis=2))
            
            # Normalize to 0.0-1.0
            dist /= 1.73205
            
            # Tolerance mapping
            if key_tolerance <= 0: key_tolerance = 0.001
            alpha = dist / key_tolerance
            
        else: # HSV
            # HSV Distance Logic
            r, g, b = pixels[:,:,0], pixels[:,:,1], pixels[:,:,2]
            
            mx = np.maximum(np.maximum(r, g), b)
            mn = np.minimum(np.minimum(r, g), b)
            df = mx - mn
            
            # Prevent division by zero
            df[df == 0] = 1e-6
            
            h_ = np.zeros_like(mx)
            
            # H calculation
            mask_r = (mx == r)
            mask_g = (mx == g) & (~mask_r) # strict else
            mask_b = (mx == b) & (~mask_r) & (~mask_g)
            
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
            
            # Target Key HSV
            tk_r, tk_g, tk_b = key_lin 
            tk_h, tk_s, tk_v = colorsys.rgb_to_hsv(tk_r, tk_g, tk_b)
            
            # Calculate Distance (Weighted)
            diff_h = np.abs(h_ - tk_h)
            diff_h = np.minimum(diff_h, 1.0 - diff_h)
            
            diff_s = np.abs(s_ - tk_s)
            diff_v = np.abs(v_ - tk_v)
            
            w_h, w_s, w_v = 4.0, 2.0, 0.5
            dist = np.sqrt((diff_h * w_h)**2 + (diff_s * w_s)**2 + (diff_v * w_v)**2)
            dist /= 2.87228
            
            if key_tolerance <= 0: key_tolerance = 0.001
            alpha = dist / key_tolerance

        common_alpha = np.clip(alpha, 0.0, 1.0)
        alpha = common_alpha
        
    else:
        alpha = pixels[:, :, 3]

    # 2. Invert
    if invert:
        alpha = 1.0 - alpha
        
    # 2b. Offset (Dilate/Erode)
    if offset != 0:
        iters = abs(offset)
        is_dilate = offset > 0
        
        for _ in range(iters):
            padded = np.pad(alpha, 1, mode='edge')
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
                alpha = np.maximum.reduce([center, n_tl, n_t, n_tr, n_l, n_r, n_bl, n_b, n_br])
            else:
                alpha = np.minimum.reduce([center, n_tl, n_t, n_tr, n_l, n_r, n_bl, n_b, n_br])

    # 3. Smoothing (Box Blur)
    alpha = alpha.astype(np.float32)
    
    if smooth_iters > 0:
        for _ in range(smooth_iters):
            padded = np.pad(alpha, 1, mode='edge')
            alpha = (
                padded[:-2, :-2] + padded[:-2, 1:-1] + padded[:-2, 2:] +
                padded[1:-1, :-2] + padded[1:-1, 1:-1] + padded[1:-1, 2:] +
                padded[2:, :-2] + padded[2:, 1:-1] + padded[2:, 2:]
            ) / 9.0
            
    # 4. Padding
    alpha = np.pad(alpha, 1, mode='constant', constant_values=0)
    w += 2
    h += 2
            
    return alpha, w, h

def marching_squares_vectorized(img, cutoff):
    nm = img > cutoff
    
    # Force borders to False
    nm[0, :] = False
    nm[-1, :] = False
    nm[:, 0] = False
    nm[:, -1] = False
    
    tl = nm[:-1, :-1] # Top-Left
    bl = nm[1:, :-1]  # Bottom-Left
    tr = nm[:-1, 1:]  # Top-Right
    br = nm[1:, 1:]   # Bottom-Right
    
    verts = []
    
    def add_segments(mask, p0_def, p1_def):
        ys, xs = np.nonzero(mask)
        if len(ys) == 0: return
        
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

    # Marching Squares Cases
    add_segments(bl & ~tl & ~tr & ~br, (0.5, 0), (1, 0.5))
    add_segments(~bl & ~tl & ~tr & br, (1, 0.5), (0.5, 1))
    add_segments(bl & ~tl & ~tr & br, (0.5, 0), (0.5, 1))
    add_segments(~bl & ~tl & tr & ~br, (0, 0.5), (0.5, 1))
    
    mask_a = bl & ~tl & tr & ~br
    add_segments(mask_a, (0.5, 0), (0, 0.5))
    add_segments(mask_a, (1, 0.5), (0.5, 1))
    
    add_segments(~bl & ~tl & tr & br, (0, 0.5), (1, 0.5))
    add_segments(bl & ~tl & tr & br, (0.5, 0), (0, 0.5))
    add_segments(~bl & tl & ~tr & ~br, (0.5, 0), (0, 0.5))
    add_segments(bl & tl & ~tr & ~br, (1, 0.5), (0, 0.5))
    
    mask_b = ~bl & tl & ~tr & br
    add_segments(mask_b, (1, 0.5), (0.5, 0))
    add_segments(mask_b, (0, 0.5), (0.5, 1))
    
    add_segments(bl & tl & ~tr & br, (0.5, 1), (0, 0.5))
    add_segments(~bl & tl & tr & ~br, (0.5, 1), (0, 0.5))
    add_segments(bl & tl & tr & ~br, (0.5, 1), (1, 0.5))
    add_segments(~bl & tl & tr & br, (1, 0.5), (0.5, 0))
    
    return verts

def build_loops_raw(segments):
    import collections
    adj = collections.defaultdict(list)
    def to_key(p): return (round(p[0], 4), round(p[1], 4))
    
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
        if len(adj[start]) != 2: continue
        
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
                break
            
            prev = curr
            curr = nxt
            
        if len(path) > 2:
            loops.append(path)
            
    return loops

def douglas_peucker(points, epsilon):
    if epsilon <= 0: return points
    
    end = len(points) - 1
    if end < 2: return points
    
    p1 = np.array(points[0])
    p2 = np.array(points[end])
    pts = np.array(points)
    
    nom = np.abs(np.cross(p2-p1, pts-p1))
    denom = np.linalg.norm(p2-p1)
    
    if denom == 0:
        dists = np.linalg.norm(pts - p1, axis=1)
    else:
        dists = nom / denom
        
    index = np.argmax(dists)
    dmax = dists[index]
    
    if dmax > epsilon:
        res1 = douglas_peucker(points[:index+1], epsilon)
        res2 = douglas_peucker(points[index:], epsilon)
        return res1[:-1] + res2
    else:
        return [points[0], points[end]]

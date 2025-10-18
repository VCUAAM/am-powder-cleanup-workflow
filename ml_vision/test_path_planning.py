import numpy as np
import cv2
import heapq
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from collections import deque

# =====================
# PARAMETERS
# =====================
save_path = "ml_vision/testdata"
npz_file = save_path + "/rgb_xyz_capture_aligned.npz"
output_file = save_path + "/robot_path.npz"
cluster_size = 10
offset_pixels = 5
offset_meters = 0.015
visualize = True

# =====================
# LOAD DATA
# =====================
data = np.load(npz_file)
mask = data['mask'].astype(np.uint8)  # 0=obstacle, 255=free
# rgb = data['rgb']  # Uncomment if RGB is valid
xyz = data['xyz']

# =====================
# APPLY OFFSET
# =====================
kernel_size = offset_pixels*2 + 1
mask_offset = cv2.erode(mask, np.ones((kernel_size,kernel_size), np.uint8), iterations=1)

# =====================
# CLUSTER MASK
# =====================
h, w = mask_offset.shape
clusters_h = h // cluster_size
clusters_w = w // cluster_size

cluster_grid = np.zeros((clusters_h, clusters_w), dtype=np.uint8)
for i in range(clusters_h):
    for j in range(clusters_w):
        block = mask_offset[i*cluster_size:(i+1)*cluster_size, j*cluster_size:(j+1)*cluster_size]
        if np.any(block > 0):
            cluster_grid[i,j] = 1

indices = np.argwhere(cluster_grid == 1)

# Get the first occurrence
if indices.size > 0:
    first_index = indices[0]
    print(f"First index of 1: {first_index}")
else:
    print(f"1 not found in the array.")
    
# =====================
# SPIRAL HEURISTIC
# =====================
def spiral_order_indices(h, w):
    """Generate a 2D array of spiral indices from outside-in"""
    spiral = np.zeros((h, w), dtype=int)
    val = 0
    top, left = 0, 0
    bottom, right = h-1, w-1
    while top <= bottom and left <= right:
        for j in range(left, right+1):
            spiral[top,j] = val; val+=1
        top+=1
        for i in range(top, bottom+1):
            spiral[i,right] = val; val+=1
        right-=1
        for j in range(right, left-1, -1):
            spiral[bottom,j] = val; val+=1
        bottom-=1
        for i in range(bottom, top-1, -1):
            spiral[i,left] = val; val+=1
        left+=1
    return spiral

# =====================
# A* IMPLEMENTATION
# =====================
def spiral_visit_all_clusters(grid, spiral_grid, start=(0,0)):
    """
    Plan a path that visits all free clusters:
    - Follows an outside-in spiral
    - Never crosses obstacles
    - Favors long straight paths
    - Minimizes revisiting clusters
    """
    h, w = grid.shape
    visited = set()
    path_clusters = []

    # Directions: up, right, down, left (clockwise)
    directions = [(-1,0),(0,1),(1,0),(0,-1)]

    # Current cluster and movement direction
    current = start
    prev_dir = None

    # Precompute spiral order
    spiral_coords = sorted([(i,j) for i in range(h) for j in range(w) if grid[i,j]==1],
                           key=lambda x: spiral_grid[x])

    # BFS-based movement toward nearest unvisited cluster along spiral order
    while len(visited) < len(spiral_coords):
        # Visit current cluster if not already visited
        if current not in visited and grid[current]:
            path_clusters.append(current)
            visited.add(current)

        # Find nearest unvisited cluster in spiral order
        unvisited = [c for c in spiral_coords if c not in visited]
        if not unvisited:
            break
        target = unvisited[0]  # always take next spiral cluster

        # Determine candidate neighbors (4-connectivity)
        ci, cj = current
        candidates = []
        for di,dj in directions:
            ni, nj = ci+di, cj+dj
            if 0 <= ni < h and 0 <= nj < w and grid[ni,nj]==1:
                candidates.append((ni,nj,(di,dj)))

        if not candidates:
            # No free neighbors: jump to target via BFS
            path_to_target = bfs_to_target(grid, current, target)
            for c in path_to_target[1:]:  # skip current since already added
                if c not in visited:
                    path_clusters.append(c)
                    visited.add(c)
            current = target
            prev_dir = None
        else:
            # Choose neighbor that minimizes manhattan distance to target
            best_neighbor = min(candidates, key=lambda x: abs(x[0]-target[0])+abs(x[1]-target[1]))
            neighbor_coord, move_dir = best_neighbor[:2], best_neighbor[2]
            current = neighbor_coord
            prev_dir = move_dir

    return path_clusters

# ---------------------------
# Simple BFS to reach target if blocked
# ---------------------------
def bfs_to_target(grid, start, target):
    """
    BFS to find a valid path between two clusters without crossing obstacles.
    Returns list of clusters from start to target.
    """
    from collections import deque
    h, w = grid.shape
    queue = deque()
    queue.append((start, [start]))
    visited = set()
    visited.add(start)

    while queue:
        current, path = queue.popleft()
        if current == target:
            return path
        ci, cj = current
        for di,dj in [(-1,0),(1,0),(0,-1),(0,1)]:
            ni, nj = ci+di, cj+dj
            if 0 <= ni < h and 0 <= nj < w and grid[ni,nj]==1 and (ni,nj) not in visited:
                visited.add((ni,nj))
                queue.append(((ni,nj), path+[ (ni,nj) ]))
    return [start]  # fallback if target unreachable


spiral_grid = spiral_order_indices(clusters_h, clusters_w)
path_pixels = np.asarray(spiral_visit_all_clusters(cluster_grid, spiral_grid, start=(15,23)))
# =====================
# MAP PIXELS TO XYZ
# =====================
#path_xyz = np.zeros((len(path_pixels),3), dtype=np.float32)
#for idx,(y,x) in enumerate(path_pixels):
    #path_xyz[idx] = xyz[y,x]

# =====================
# SAVE PATH
# =====================
#np.savez_compressed(output_file, path=path_xyz)
#print(f"Saved robot path with {len(path_xyz)} points to {output_file}")

# =====================
# OPTIONAL VISUALIZATION
# =====================

fig,ax = plt.subplots()
plt.plot(path_pixels[:,0],path_pixels[:,1],)
#rect = Rectangle((x1,y1),x2 - x1,y2 - y1,linewidth=1, edgecolor='black', facecolor='none')
#ax.add_patch(rect)
#ax.axis('off')
plt.savefig(save_path + '/path_overlay.png')
#plt.show()
#mask_vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
#for y,x in path_pixels:
    #mask_vis[y,x] = [0,0,255]
#plt.savefig(save_path + '/path_overlay.png')
#plt.imshow(mask_vis)
#plt.title("A* Spiral Path Overlay")
#plt.show()

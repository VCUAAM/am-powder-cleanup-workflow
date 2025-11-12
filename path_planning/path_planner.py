import numpy as np
import cv2
import matplotlib as mpl
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

mpl.use('TkAgg')

class PathPlanner:
    def __init__(self,save_path):
        if not save_path:
            print('Path planner needs to be initialized with a save path')
            quit()
        self.cluster_size = 30
        self.obstacle = None
        self.save_path = save_path
        self.smooth = True # If true, will smooth polygon offsets
        self.epsilon = .01 # Epsilon ratio for polygon smoothing function

    def load_npz(self,npz):
        data = np.load(npz)
        self.mask = data['mask']
        self.rgb = data["color"]
        self.xyz = data["xyz"]

    # --------- VISUALIZER ----------
    def visualizer(self, path):
        fig, ax = plt.subplots()

        # Draw polygon if present
        if self.obstacle is not None:
            obstacle = np.squeeze(self.obstacle)
            smooth_poly = cv2.approxPolyDP(obstacle, 2.5, True).reshape(-1, 2)
            patch = Polygon(smooth_poly, closed=True, facecolor='black', edgecolor='black')
            ax.add_patch(patch)

        h,w = self.mask.shape

        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(0, w)
        ax.set_ylim(0, h)

        # ------- RED → GREEN gradient path -------
        # Build line segments from successive points
        points = path.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # Normalize color from 0 → 1 along the path
        num = len(segments)
        colors = plt.get_cmap('plasma')(np.linspace(0, 1, num))

        lc = LineCollection(segments, colors=colors, linewidths=3)
        ax.add_collection(lc)

        plt.savefig(self.save_path + 'data/path_overlay.png', bbox_inches='tight', pad_inches=.1)
        print('Saved path overlay')

    # --------- OFFSET SINGLE POLYGON ----------
    def offset_polygon(self, polygon):
        poly = np.squeeze(polygon).astype(np.float32)
        centroid = np.mean(poly, axis=0)

        directions = poly - centroid
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        unit_dirs = directions / norms
        offset_poly = poly + unit_dirs * self.cluster_size

        diffs = offset_poly - poly
        offset_poly_rounded = np.where(diffs > 0, np.ceil(offset_poly), np.floor(offset_poly)).astype(int)

        if self.smooth:
            x_min, y_min = offset_poly_rounded[:, 0].min(), offset_poly_rounded[:, 1].min()
            x_max, y_max = offset_poly_rounded[:, 0].max(), offset_poly_rounded[:, 1].max()

            mask = np.zeros((y_max - y_min + 20, x_max - x_min + 20), dtype=np.uint8)
            shifted = offset_poly_rounded - [x_min - 10, y_min - 10]
            cv2.fillPoly(mask, [shifted], 255)

            _, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            largest = max(contours, key=cv2.contourArea)

            peri = cv2.arcLength(largest, True)
            approx = cv2.approxPolyDP(largest, self.epsilon * peri, True)
            offset_poly_rounded = approx + [x_min - 10, y_min - 10]

        return np.squeeze(offset_poly_rounded, axis=1)

    # --------- GENERATE CONCENTRIC OFFSETS ----------
    def generate_offset_polygons(self):
        x, y, w, h = cv2.boundingRect(self.obstacle)
        bbox = (x, y, x + w, y + h)

        polygons = []
        current_poly = np.squeeze(self.obstacle, axis=1)

        while True:
            offset_poly = self.offset_polygon(current_poly)

            xs, ys = offset_poly[:, 0], offset_poly[:, 1]
            inside = (xs >= bbox[0]) & (xs <= bbox[2]) & (ys >= bbox[1]) & (ys <= bbox[3])

            if not np.any(inside):
                break

            polygons.append(offset_poly)
            current_poly = offset_poly.copy()

        return polygons

    # --------- GRID COMPUTATION ----------
    def compute_grid(self):
        h, w = self.mask.shape
        
        clusters_h = h // self.cluster_size
        clusters_w = w // self.cluster_size

        inverted = cv2.bitwise_not(self.mask)
        mask_contour, _ = cv2.findContours(inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bounded = self.mask.copy()

        if mask_contour:
            self.obstacle = max(mask_contour, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(self.obstacle)
            bounded[y:y + h, x:x + w] = 0
        
        grid = np.zeros((clusters_h, clusters_w), dtype=np.uint8)

        for i in range(clusters_h):
            for j in range(clusters_w):
                block = bounded[i * self.cluster_size:(i + 1) * self.cluster_size,
                                j * self.cluster_size:(j + 1) * self.cluster_size]
                if np.any(block > 0):
                    grid[i, j] = 1

        return grid

    def compute_average_height(self):
        z = self.xyz[:,:,2]
        mean = np.mean(z)
        std = np.std(z)

        print(f'Mean: {mean}\nStandard Deviation: {std}')

    # --------- WRAPPING RASTER SCAN ----------
    def path_planner(self, h, w, boundaries):
        path = []
        bottom, top, left, right = boundaries

        for j in range(bottom):
            if j % 2 == 0:
                path.append((0, j))
                path.append((w, j))
            else:
                path.append((w, j))
                path.append((0, j))

        if abs(path[-1][0] - 0) > abs(path[-1][0] - w):
            upp = w
            low = right
            dir_adj = 1
        else:
            upp = left
            low = 0
            dir_adj = 0

        for j in range(path[-1][1] + 1, top + 1):
            if j % 2 == 0:
                path.append((low + dir_adj, j))
                path.append((upp - 1 + dir_adj, j))
            else:
                path.append((upp - 1 + dir_adj, j))
                path.append((low + dir_adj, j))

        if path[-1][0] != 0 and abs(path[-1][0] - w) > abs(path[-1][0] - 0):
            path.append((0, path[-1][1]))
        elif path[-1][0] != w and abs(path[-1][0] - 0) > abs(path[-1][0] - w):
            path.append((w, path[-1][1]))

        path.append((path[-1][0], h))

        if path[-1][0] == w:
            dir_adj = 1
        else:
            dir_adj = 0

        upp = w - dir_adj
        low = 1 - dir_adj

        for j in range(h, top, -1):
            if (h - j) % 2 == 1 - dir_adj:
                path.append((upp, j))
                path.append((low, j))
            else:
                path.append((low, j))
                path.append((upp, j))

        if abs(path[-1][0] - 0) == 1:
            path.append((w, path[-1][1]))
        if abs(path[-1][0] - w) == 1:
            path.append((0, path[-1][1]))

        if abs(path[-1][0] - 0) > abs(path[-1][0] - w):
            upp = w
            low = right
            dir_adj = 1
        else:
            upp = left - 1
            low = 0
            dir_adj = 0

        for j in range(top, bottom - 1, -1):
            if (j - top) % 2 == dir_adj:
                path.append((low + dir_adj, j))
                path.append((upp, j))
            else:
                path.append((upp, j))
                path.append((low + dir_adj, j))

        return path

    # --------- REORDER POLYGON START ----------
    def reorder_polygon_start(self, poly, start):
        dists = np.linalg.norm(poly - np.array(start), axis=1)
        idx = np.argmin(dists)

        poly_reordered = np.roll(poly, -idx, axis=0)

        return np.vstack([poly_reordered, poly_reordered[0]])

    # --------- MAIN ENTRY POINT ----------
    def compute_path(self):
        grid = self.compute_grid()
        h, w = grid.shape
        
        ys, xs = np.where(grid == 0)

        if len(xs) == 0 or len(ys) == 0:
            path = []
            for i in range(h):
                row = [(i, j) for j in (range(w) if i % 2 == 0 else range(w - 1, -1, -1))]
                path.extend(row)
        else:
            boundaries = [ys.min(), ys.max(), xs.min(), xs.max()]
            path = self.path_planner(h, w, boundaries)

        path_clean = [path[0],path[-1]]

        for i in range(1,len(path) - 1):
            if abs(path[i - 1][0] - path[i][0]) == 0 and abs(path[i][0] - path[i + 1][0]) == 0:
                continue
            elif abs(path[i - 1][1] - path[i][1]) == 0 and abs(path[i][1] - path[i + 1][1]) == 0:
                continue
            else:
                path_clean.insert(-1,path[i])

        path_px = []
        
        # Compute leftover pixels and center the grid
        pad_y = (h - h // self.cluster_size * self.cluster_size)
        pad_x = (w - w // self.cluster_size * self.cluster_size)

        for (i, j) in path_clean:
            cy = i * self.cluster_size + pad_y + self.cluster_size // 2 
            cx = j * self.cluster_size + pad_x + self.cluster_size // 2 
            path_px.append((cy, cx))

        if self.obstacle:
            offset_polygons = self.generate_offset_polygons()

            for layer in offset_polygons:
                poly = self.reorder_polygon_start(layer, path_px[-1])
                for [i, j] in poly:
                    path_px.append((i, j))
                    path.append((i / self.cluster_size, j / self.cluster_size))
        
        robot_path = [self.xyz[i, j] for (i, j) in path_px]
        print([(int(1000*i),int(1000*j),int(1000*k)) for (i,j,k) in robot_path])
        self.visualizer(np.asarray(path_px))
        
        np.savez_compressed(self.save_path + "data/robot_path.npz",path=robot_path)
        print('Saved robot path NPZ')

# ---------- Example main ----------
if __name__ == "__main__":
    pln = PathPlanner(save_path="path_planning/")
    pln.load_npz('ml_vision/data/rgb_xyz_aligned.npz')
    pln.compute_path()
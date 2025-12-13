import numpy as np
import cv2
import matplotlib as mpl
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon,Rectangle
from matplotlib.ticker import FuncFormatter,MaxNLocator,FixedLocator

mpl.use('TkAgg')

class PathPlanner:
    def __init__(self):
        self.cluster_size = 20
        self.obstacle = False
        self.save_path = "path_planning/data/" # Where to save debugging files 
        self.smooth = True # If true, will smooth polygon offsets
        self.epsilon = .01 # Epsilon ratio for polygon smoothing function
        self.debugging = False
        self.border = 10 #How much to cut off around the edges to compensate for hose size
        self.z_offset = 0.001 # Amount to increase robot path
        self.avg_z = None

    def load_npz(self,name):
        data = np.load('data/' + name)
        self.mask = data['mask']
        self.mask_clip = self.mask.copy()[self.border:-self.border,self.border:-self.border]
        if (self.mask_clip[self.cluster_size:-self.cluster_size,self.cluster_size:-self.cluster_size] == 0).all():
            raise ValueError('Imported mask is all zeros')
        self.xyz = data["xyz"]
    
    '''
    Visualizer for the planned path. Takes the following inputs:
    ticks (number of ticks on axes)                     Dtype: int     Default: 6
    mm (whether to display in millimeters or pixels)    Dtype: bool    Default: False
    grid (whether to display as obstacle or grid box)   Dtype: bool    Default: False
    '''
    def visualizer(self, path,ticks=6,mm=False,grid=False):
        fig, ax = plt.subplots()
        h,w = self.mask.shape

        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(0, w)
        ax.set_ylim(0, h)

        # Setting the ticks such that they stop at the maximum value of the mask size
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

        xticks = np.linspace(0, w, ticks).astype(int)
        ax.xaxis.set_major_locator(FixedLocator(xticks))
        yticks = np.linspace(0, h, ticks).astype(int)
        ax.yaxis.set_major_locator(FixedLocator(yticks))

        # Scaling to millimeters if desired
        if mm:
            # Finding height and width of cylinder
            xyz = np.nan_to_num(self.xyz)

            left_mean  = np.nanmean(xyz[:, 0, 0]) 
            right_mean = np.nanmean(xyz[:, -1, 0])

            top_mean    = np.nanmean(xyz[0, :, 1])
            bottom_mean = np.nanmean(xyz[-1, :, 1])

            width  = int(abs(right_mean - left_mean)*1000)
            height = int(abs(bottom_mean - top_mean)*1000)

            # Configuring plot appropriately
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: int(x * width/w)))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: int(y * height/h)))
            ax.set_xlabel('mm')
            ax.set_ylabel('mm')
        
        # Draw polygon if present, as obstacle if not grid or as box if grid
        if self.obstacle is not False:
            if grid:
                x, y, w_, h_ = cv2.boundingRect(self.obstacle)
                patch = Rectangle((y - self.border,x - self.border),w_ + 2*self.border,h_ + 2*self.border,linewidth=1,edgecolor='black',facecolor='black')
                ax.add_patch(patch)
            else:
                obstacle = np.squeeze(self.obstacle)
                smooth_poly = cv2.approxPolyDP(obstacle, 2.5, True).reshape(-1, 2)
                patch = Polygon(smooth_poly, closed=True, facecolor='black', edgecolor='black')
                ax.add_patch(patch)

        # Normalizing color across path to show gradient and direction
        points = path.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        colors = plt.get_cmap('plasma')(np.linspace(0, 1, len(segments)))
        lc = LineCollection(segments, colors=colors, linewidths=3)
        ax.add_collection(lc)

        # Save path if in debugging mode
        if self.debugging:
            plt.savefig(self.save_path + 'path_overlay.png', bbox_inches='tight', pad_inches=.1)
            print('Saved path overlay')

    # Finds offset contour to input polygon, by a measure of 30 pixels (approximate size of vacuum hose)
    def offset_polygon(self, polygon):
        # Converting to float and finding direction to offset points
        poly = np.squeeze(polygon).astype(np.float32)
        centroid = np.mean(poly, axis=0)
        directions = poly - centroid
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        unit_dirs = directions / norms

        # Offsetting points and converting to integer
        offset_poly = poly + unit_dirs * self.cluster_size
        diffs = offset_poly - poly
        offset_poly_rounded = np.where(diffs > 0, np.ceil(offset_poly), np.floor(offset_poly)).astype(int)

        # Smoothing polygon contour
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

    # generate offset polygons until area exceeds bounding box
    def generate_offset_polygons(self):
        x, y, w, h = cv2.boundingRect(self.obstacle)
        bbox = (x - self.cluster_size*1.5, y - self.cluster_size*1.5, x + w + self.cluster_size*1.5, y + h + self.cluster_size*1.5)

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

    # Computing grid for use in navigation
    def compute_grid(self):
        # Computing size of grid
        h, w = self.mask_clip.shape
        clusters_h = h // self.cluster_size
        clusters_w = w // self.cluster_size
        
        # Compute leftover pixels and center the grid
        self.pad_y = (h - clusters_h * self.cluster_size)
        self.pad_x = (w - clusters_w * self.cluster_size)

        # Checking to see if there is an obstacle in the grid
        inverted = cv2.bitwise_not(self.mask_clip)
        mask_contour, _ = cv2.findContours(inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bounded = self.mask_clip.copy()
        
        if mask_contour:
            self.obstacle = max(mask_contour, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(self.obstacle)
            bounded[y - self.cluster_size:y + h + self.cluster_size, x - self.cluster_size:x + w + self.cluster_size] = 0
            
            # Computing average height of smaller inset to equalize path heights
            xyz = self.xyz.copy()[self.border:-self.border,self.border:-self.border]
            xyz[y - self.border:y + h + self.border, x - self.border:x + w + self.border] = np.nan
            z = xyz[:,:,2]
            z_mean = np.nanmean(z)
            z_std = np.nanstd(z)

            # Rounding off floating decimal points (no idea why np.round wasn't working)
            self.avg_z = float(int(100000*z_mean))/100000

            if self.debugging:
                print(f'Mean Height: {float(int(100000*z_mean))/100} mm\nStandard Deviation: {float(int(100000*z_std))/100} mm')

        # Defining grid (for path planning) and grid_lookup (to remember mask indices to correlate to xyz)
        grid = np.ones((clusters_h, clusters_w), dtype=np.uint8)
        self.grid_lookup = np.zeros((clusters_h,clusters_w,2),dtype=int)

        # Mapping grid and grid lookup arrays
        for i in range(clusters_h):
            for j in range(clusters_w):
                    block = bounded[i * self.cluster_size + self.pad_y:(i + 1) * self.cluster_size + self.pad_y,
                                    j * self.cluster_size + self.pad_x:(j + 1) * self.cluster_size + self.pad_x]

                    if np.count_nonzero(block) < 0.3*block.size:
                        grid[i, j] = 0
                    self.grid_lookup[i,j] = [max((i + 1/2)*self.cluster_size + self.pad_y - self.border,0),max((j + 1/2)*self.cluster_size + self.pad_x + 1.5*self.border,0)]

        return grid

    # Complex path planning function, given the shape of the grid and the boundaries of the obstacle
    def path_planner(self, h, w, boundaries):
        path = []
        bottom, top, left, right = boundaries

        # Rasterizing across bottom portion of grid
        for j in range(bottom):
            if j % 2 == 0:
                path.append((0, j))
                path.append((w, j))
            else:
                path.append((w, j))
                path.append((0, j))

        # Checking to see which side the bottom raster finished on and setting appropriate bounds
        if abs(path[-1][0] - 0) > abs(path[-1][0] - w):
            upp = w
            low = right
            dir_adj = 1
        else:
            upp = left
            low = 0
            dir_adj = 0
       
        # Rasterizing first side
        for j in range(path[-1][1] + 1, top + 1):
            if j % 2 == 0:
                path.append((low + dir_adj, j))
                path.append((upp - 1 + dir_adj, j))
            else:
                path.append((upp - 1 + dir_adj, j))
                path.append((low + dir_adj, j))
        
        # Checking to see if current location is already on edge of grid, and moving to edge if not
        if path[-1][0] != 0 and abs(path[-1][0] - w) > abs(path[-1][0] - 0):
            path.append((0, path[-1][1]))
        elif path[-1][0] != w and abs(path[-1][0] - 0) > abs(path[-1][0] - w):
            path.append((w, path[-1][1]))
        
        # Moving to top of grid
        path.append((path[-1][0], h))

        # Setting bounds for top region
        if path[-1][0] == w:
            dir_adj = 1
        else:
            dir_adj = 0

        upp = w - dir_adj
        low = 1 - dir_adj

        # Rasterizing top region
        for j in range(h, top, -1):
            if (h - j) % 2 == 1 - dir_adj:
                path.append((upp, j))
                path.append((low, j))
            else:
                path.append((low, j))
                path.append((upp, j))

        # Checking to see if current location is already on edge of grid, and moving to edge if not
        if abs(path[-1][0] - 0) == 1:
            path.append((w, path[-1][1]))
        if abs(path[-1][0] - w) == 1:
            path.append((0, path[-1][1]))

        # Setting bounds for other side
        if abs(path[-1][0] - 0) > abs(path[-1][0] - w):
            upp = w
            low = right
            dir_adj = 1
        else:
            upp = left - 1
            low = 0
            dir_adj = 0

        # Rasterizing other side
        for j in range(top, bottom - 1, -1):
            if (j - top) % 2 == dir_adj:
                path.append((low + dir_adj, j))
                path.append((upp, j))
            else:
                path.append((upp, j))
                path.append((low + dir_adj, j))

        return path

    # Reordering the offset polygons such that the closest point in the polygon to the path is the start
    def reorder_polygon_start(self, poly, start):
        dists = np.linalg.norm(poly - np.array(start), axis=1)
        idx = np.argmin(dists)

        poly_reordered = np.roll(poly, -idx, axis=0)

        return np.vstack([poly_reordered, poly_reordered[0]])

    # General path planning function and organizer
    def compute_path(self):
        grid = self.compute_grid()
        h, w = grid.shape

        # Detecting it here is an obstacle 
        ys, xs = np.where(grid == 0)

        # Choosing path planning regime based on if obstacle is detected or not
        if len(xs) == 0 or len(ys) == 0:
            path = []
            for i in range(h - 1):
                row = [(i, j) for j in (range(w) if i % 2 == 0 else range(w - 1, -1, -1))]
                path.extend(row)
        else:
            boundaries = [ys.min(), ys.max(), xs.min(), xs.max()]
            path = self.path_planner(h - 1, w - 1, boundaries)
        
        # Cleaning the path to remove unncessary travel points along straight lines
        path_clean = [path[0],path[-1]]

        for i in range(1,len(path) - 1):
            if abs(path[i - 1][0] - path[i][0]) == 0 and abs(path[i][0] - path[i + 1][0]) == 0 and path[i - 1] != path[i + 1]:
                continue
            elif abs(path[i - 1][1] - path[i][1]) == 0 and abs(path[i][1] - path[i + 1][1]) == 0 and path[i - 1] != path[i + 1]:
                continue
            else:
                path_clean.insert(-1,path[i])

        # Translating the grid path points to mask (pixel) points, compensating for path planning adjustments
        path_px = [tuple(self.grid_lookup[i,j]) for [j,i] in path_clean]

        # Adding offset path around obstacle if one is detected
        if self.obstacle is not False:
            pass
            offset_polygons = self.generate_offset_polygons()

            for layer in reversed(offset_polygons):
                poly = self.reorder_polygon_start(layer, path_px[-1])
                
                for [i, j] in poly:
                    path_px.append((i, j))

        # Visualizing path
        self.visualizer(np.asarray(path_px),grid=False)
        robot_path = np.asarray([self.xyz[i, j] for (i,j) in path_px])

        # Changing heights to average height and adding offset
        robot_path[:,2] = self.avg_z + self.z_offset

        if self.debugging:
            print([(int(1000*i),int(1000*j),int(1000*k)) for (i,j,k) in robot_path])
        
        np.savez_compressed("data/robot_path.npz",path=robot_path)
        print('Saved robot path NPZ')

# ---------- Example main ----------
if __name__ == "__main__":
    pln = PathPlanner()
    pln.debugging = True
    pln.avg_z = 0.1801
    pln.load_npz('rgb_xyz_aligned.npz')
    pln.compute_path()
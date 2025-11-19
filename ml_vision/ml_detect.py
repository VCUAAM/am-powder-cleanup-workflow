import torch 
import numpy as np
import pathlib
import platform
import cv2

# Everything related to image processing
class ImageProcessor:
    def __init__(self):
        self.save_path = 'ml_vision/data/'
        self.debugging = False
        self.visualize = False

        # Data Variables
        self.rgb = None
        self.xyz = None
        self.model = None
        self.model_dir = 'ml_vision/best.pt'
        self.model_name = 'ml_vision/yolov5'
        self.use_prev_mask = False

        # Image Processing Parameters
        self.border_exp = 1 # Amount to increase size of image to ensure complete capture of target 
        self.targ_class = None
        self.offset_px = 15 # Amount of px to constrict image to give clearance around vacuum nozzle
        self.obstacle_border = 25


        # Obstacle parameters
        self.x_lo = None
        self.x_hi = None
        self.y_lo = None
        self.y_hi = None

        # Check the operating system and set the appropriate path type
        if platform.system() == 'Windows':
            pathlib.PosixPath = pathlib.WindowsPath
        else:
            pathlib.WindowsPath = pathlib.PosixPath

        torch.serialization.safe_globals([np._core.multiarray._reconstruct])

    def real_rect_size(self,xyz,debug = False):
        """
        Given an (H, W, 3) array of real-space coordinates,
        compute the physical width (x-direction) and height (y-direction)
        by averaging the left/right and top/bottom edges respectively.
        """
        #print(xyz[:, 0, 1])
        # Drop invalid points (NaN or 0) if needed
        xyz = np.nan_to_num(xyz)
        #rint(xyz[:, 0, 0])
        # Compute mean of left and right edges (averaging over rows)
        left_mean  = np.nanmean(xyz[:, 0, 0])   # average X of left column
        right_mean = np.nanmean(xyz[:, -1, 0])  # average X of right column

        # Compute mean of top and bottom edges (averaging over columns)
        top_mean    = np.nanmean(xyz[0, :, 1])  # average Y of top row
        bottom_mean = np.nanmean(xyz[-1, :, 1]) # average Y of bottom row

        # Compute mean of surface
        surface_mean = int(np.nanmean(xyz[:,:,2])*1000)

        # Differences in world coordinates
        width  = int(abs(right_mean - left_mean)*1000)
        height = int(abs(bottom_mean - top_mean)*1000)
        
        # Calculating center of box
        cx = int((right_mean + left_mean)*500)
        cy = int((top_mean + bottom_mean)*500)
        
        # Converting edges into mm
        top = int(top_mean*1000)
        bottom = int(bottom_mean*1000)
        left = int(left_mean*1000)
        right = int(right_mean*1000)

        if self.debugging or debug:
            # Printing out values for debugging
            print(f'Size of box: {(width,height)} mm')
            print(f'Center of box: {(cx,cy)} mm')
            print(f'Top, Bottom: {top,bottom} mm')
            print(f'Left, Right: {left,right} mm')
            print(f'Surface of box: {surface_mean} mm')
        
        rot = (right + 5000) - (left + 5000) < 0

        return rot

    def shrink_box(self,rect):
        """
        Shrink (offset < 0) or expand (offset > 0) a rotated rectangle
        returned by cv2.minAreaRect by a fixed amount in all directions.
        """

        (cx, cy), (w, h), angle = rect

        # Reduce or increase width and height
        new_w = max(w - 2 * self.offset_px, 1)  # prevent negative or zero
        new_h = max(h - 2 * self.offset_px, 1)

        # Create new, smaller rectangle
        new_rect = ((cx, cy), (new_w, new_h), angle)

        box = cv2.boxPoints(new_rect)
        box = np.int32(box).reshape((-1,1,2))
        
        return box

    def load_npz(self,name,mask=False):
        data = np.load('data/' + name)
        self.rgb = data["color"]
        self.xyz = data["xyz"]
        self.depth = data["depth"]

        if mask:
            self.mask = data["mask"]

    # Runs YOLOv5 model and extracts bounding box
    def extract_model_bbox(self):
        RGB_img = cv2.cvtColor(self.rgb, cv2.COLOR_RGB2BGR)
        results = self.model(RGB_img)
        
        if self.debugging:
            cv2.imwrite(self.save_path + 'yolo.png',results.render()[0])

        # Extracting data out of the results
        class_names = results.names
        detections = results.xyxy[0].cpu().numpy()

        # Adding all boxes belonging to target class 
        boxes = []
        for x1, y1, x2, y2, conf, cls_id in detections:
            label = class_names[int(cls_id)]
            
            # Filter if specific class given
            if label not in self.targ_class:
                continue

            off_x = max(0, (y2 - y1 - (x2 - x1)) // 2)
            bbox = [
                int(x1 - off_x - self.border_exp),
                int(y1 - self.border_exp),
                int(x2 + off_x + self.border_exp),
                int(y2 + self.border_exp/2)
            ]

            boxes.append([label, bbox, conf])

        # Sorting so that only highest confidence with matching label is returned
        self.x_lo,self.y_lo,self.x_hi,self.y_hi = max(boxes, key=lambda b: float(b[2]))[1]

    # Uses YOLO box to threshold and identify true oriented bounding box
    def get_mask(self,bounded=None):
        if not bounded:
            self.model = torch.hub.load(self.model_name,'custom',
                                    path=self.model_dir,
                                    force_reload=True,source='local')
        try:
            if not bounded:
                self.extract_model_bbox()
                # Grabbing coordinates out of model bounding box and clipping image
                bounded = self.rgb[self.y_lo + self.border_exp:self.y_hi - self.border_exp,self.x_lo + self.border_exp:self.x_hi - self.border_exp]
            
            # Thresholding image to make difference between sections more distinct
            gray = cv2.cvtColor(bounded, cv2.COLOR_BGR2GRAY)
            gray_f = gray.astype(np.float32)
            scale = 255.0 / (220 - 160) #hi - lo
            thresh = (gray_f - 160) * scale #gray_f - lo
            thresh = np.clip(thresh, 0, 255).astype(np.uint8)
            
            # After thresholding
            gray_f = gray.astype(np.float32)
            scale = 255.0 / (220 - 160)
            thresh = (gray_f - 160) * scale
            thresh = np.clip(thresh, 0, 255).astype(np.uint8)

            # Optional: Corner enhancement before Canny
            corner_resp = cv2.preCornerDetect(thresh, ksize=5)
            corner_resp = cv2.convertScaleAbs(corner_resp)

            # Combine with threshold image to reinforce corners
            enhanced = cv2.addWeighted(thresh, 0.75, corner_resp, 0.25, 0)

            # Then continue with Canny
            blur = cv2.GaussianBlur(enhanced, (11,11), 0)
            edges = cv2.Canny(blur, 1, 80)
            
            # Detecting Hough lines from Canny edges
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=20,
                                    minLineLength=275, maxLineGap=200)
            
            # Draw Hough lines
            vis_hough = bounded.copy()
            for (x1,y1,x2,y2) in lines[:,0]:
                cv2.line(vis_hough, (x1,y1), (x2,y2), (0,255,0), 2)
            
            # Adding Hough lines to points array
            pts = []
            for (x1,y1,x2,y2) in lines[:,0]:
                if edges[y1, x1] == 0 or edges[y2, x2] == 0:
                    # Skip lines whose endpoints aren't on real edges
                    continue
                pts.append([x1,y1])
                pts.append([x2,y2])
            
            # Drawing bounding box based off of Hough lines
            pts = np.array(pts, dtype=np.float32)
            rect = cv2.minAreaRect(pts)        # center, (w,h), angle
            box = cv2.boxPoints(rect)          # 4 rotated corners
            box = np.int32(box).reshape((-1,1,2))
            
            # --- shift box into full image coordinates ---
            box_full = box + np.array([[self.x_lo, self.y_lo]])
            
            shrink_box = self.shrink_box(rect) + np.array([[self.x_lo, self.y_lo]])
            
            # visualize on full RGB image
            vis_rect = self.rgb.copy()
            cv2.drawContours(vis_rect, [box_full], 0, (255, 0, 0), 1)
            cv2.drawContours(vis_rect, [shrink_box], 0, (0, 0, 255), 1)

            # ------ Correct way to build mask (avoids clipped box) ------
            full_mask = np.zeros(self.rgb.shape[:2], dtype=np.uint8)
            cv2.fillPoly(full_mask, [box_full], 255)

        except Exception:
            self.debugging = True
            
        finally:
            try:
                # Saving images if visualization
                if self.debugging:
                    i = 0
                    i += 1
                    #cv2.imwrite(self.save_path + 'edges.png',edges)
                    i += 1
                    #cv2.imwrite(self.save_path + 'thresh.png',thresh)
                    i += 1
                    #cv2.imwrite(self.save_path + 'hough.png',vis_hough)
                    i += 1
                    #cv2.imwrite(self.save_path + 'bbox.png',vis_rect)
                    i += 1
                    print('All image processing debugging images saved')
                #cv2.imwrite(self.save_path + 'bounded_mask.png',full_mask)
                print('Saved mask')
            except UnboundLocalError:
                match i:
                    case 0:
                        img = 'None of them'
                    case 1:
                        img = 'clipped.png'
                    case 2:
                        img = 'edges.png'
                    case 3:
                        img = 'thresh.png'
                    case 4:
                        img = 'hough.png'
                    case 5:
                        img = 'bbox.png'
                print('Something broke, check test images to see why')
                print(f'Last image that worked was {img}')
                quit()
            
            return full_mask

    def align_mask_to_img(self,mask):
        try:
            for i in range(10):
                cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                c = max(cnts, key=cv2.contourArea)
                rect = cv2.minAreaRect(c)
                angle = rect[2]

                #if rect[1][0] < rect[1][1]:
                    #angle += 90
                if angle > 45:
                    angle += -abs(angle)/angle*90

                # --- build rotation matrix about the object centroid ---
                M = cv2.getRotationMatrix2D((mask.shape[1]/2,mask.shape[0]/2), angle, 1.0)

                def warp(im, interp=cv2.INTER_LINEAR):
                    return cv2.warpAffine(im, M, (im.shape[1], im.shape[0]),
                                        flags=interp)#, borderMode=cv2.BORDER_CONSTANT)

                # --- rotate all aligned arrays ---
                rgb_rot  = np.dstack([warp(self.rgb[...,c]) for c in range(3)])
                mask_rot  = warp(mask, interp=cv2.INTER_NEAREST)
                depth_rot  = warp(self.depth, interp=cv2.INTER_NEAREST)
                xyz_rot   = np.dstack([warp(self.xyz[...,c],interp=cv2.INTER_NEAREST) for c in range(3)])

                x_,y_,w_,h_ = cv2.boundingRect(mask_rot)
                self.offsets = (x_,y_)
                mask_offset = mask_rot[y_ + self.offset_px:y_ + h_ - self.offset_px,x_ + self.offset_px:x_ + w_ - self.offset_px]
                rgb_offset = rgb_rot[y_ + self.offset_px:y_ + h_ - self.offset_px,x_ + self.offset_px:x_ + w_ - self.offset_px]
                xyz_offset = xyz_rot[y_ + self.offset_px:y_ + h_ - self.offset_px,x_ + self.offset_px:x_ + w_ - self.offset_px]
                depth_offset = depth_rot[y_ + self.offset_px:y_ + h_ - self.offset_px,x_ + self.offset_px:x_ + w_ - self.offset_px]
                rot = self.real_rect_size(xyz_offset)
                
                if rot:
                    xyz_offset = np.fliplr(xyz_offset)
                    print('Rotated')
                    self.real_rect_size(xyz_offset,debug=True)
                
                if abs(mask_offset.shape[0] - mask_offset.shape[1])/max(mask_offset.shape[:2]) > .1:
                    if i == 9:
                        print('Bounding box wrong size')
                        continue
                    continue
                
                break
        
        except Exception as e:
            print(f'Exception: {e}')

        finally:
            #cv2.imwrite(self.save_path + "aligned_mask.png", mask_offset)
            if i > 8:
                print('Alignment did not succeed within 10 iterations')
                quit()
            np.savez_compressed('data/rgb_xyz_aligned_test.npz',
                            color=rgb_offset,
                            xyz=xyz_offset,
                            depth=depth_offset,
                            mask=mask_offset)
            print('Aligned NPZ Saved')

    def pointcloud(self):
        import open3d as o3d
        origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100, origin=[0,0,0])
        coords = np.asarray(list(zip(*np.nonzero(self.xyz))))
        points = [self.xyz[i,j] for i,j in coords[:,:2]]
        scalar = 1
        points_f = [(float(i*scalar),float(j*scalar),float(k*scalar)) for i,j,k in points]
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points_f)
        
        o3d.visualization.draw_geometries([cloud])
    
    def find_obstacle(self,img):
        h,w = img.shape
        bounded = img[self.obstacle_border:h - self.obstacle_border,self.obstacle_border:w - self.obstacle_border]
        #print(bounded)
        # Thresholding image to make difference between sections more distinct
        #gray = cv2.cvtColor(bounded, cv2.COLOR_BGR2GRAY)
        depth = bounded.astype(np.float32)
        print(img)
        #percentage = .1
        
        #z_clipped = np.clip(depth, 0.0, 0.255)

        #z = xyz[..., 2].astype(np.float32)

        z_max = np.max(depth)
        thresh = z_max - 0.002  # 1mm below the highest surface point

        binary = (depth >= thresh).astype(np.uint8) * 255

        # Scale: 1mm => +1 grayscale intensity
        #norm = (z_clipped * 1000.0).astype(np.uint8)
        #MIN_DEPTH = np.percentile(depth, percentage)
        #MAX_DEPTH = np.percentile(depth, 100 - percentage)
        #norm = cv2.normalize(z_clipped, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        #norme = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
        #norm = np.clip((norm - MIN_DEPTH) * 255.0 / (MAX_DEPTH - MIN_DEPTH), 0, 255).astype(np.uint8)
        #norm = norme.astype(np.uint8)
        #bounded = bounded.astype(np.uint8)
        
        # Then continue with Canny
        #blur = cv2.GaussianBlur(norm, (11,11), 0)
        #edges = cv2.Canny(norm, 40, 150)
        
        #kernel = np.ones((8,8), np.uint8)
        #edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        #inverted = cv2.bitwise_not(edges)
        kernel = np.ones((8,8), np.uint8)
        edges = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # Find contours directly from edge map
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            raise ValueError("No contours detected")

        # Select the largest contour (likely your cylinder rim)
        #cnt = max(contours, key=cv2.contourArea)

        center = np.array([w/2, h/2])

        def contour_center(c):
            M = cv2.moments(c)
            if M['m00'] == 0: return np.inf
            return np.array([M['m10']/M['m00'], M['m01']/M['m00']])

        cnt = min(contours, key=lambda c: np.linalg.norm(contour_center(c)-center))

        # Approximate to polygon (Douglas-Peucker)
        epsilon = 0.02 * cv2.arcLength(cnt, True)  # adjust epsilon if needed
        poly = cv2.approxPolyDP(cnt, epsilon, True)
        poly_pts = poly.reshape((-1,1,2)).astype(np.int32)

        poly_pts = poly_pts + np.array([[self.obstacle_border, self.obstacle_border]])

        # visualize on full RGB image
        vis_rect = self.mask.copy()

        cv2.fillPoly(vis_rect, [poly_pts], 0)
        #cv2.drawContours(vis_rect, [shrink_box], 0, (0, 0, 255), 1)
        print(vis_rect)
        # ------ Correct way to build mask (avoids clipped box) ------
        #full_mask = np.zeros(self.rgb.shape[:2], dtype=np.uint8)
        #cv2.fillPoly(full_mask, [box], 255)
        
        cv2.imshow('thresh',edges)
        #cv2.imshow('edges',depth_edges)
        #cv2.imshow('hough',vis_hough)
        #cv2.imshow('shadow_removed',shadow_free)
        cv2.imshow('bbox',vis_rect)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

        np.savez_compressed('data/rgb_xyz_aligned_test_test.npz',
                            color=self.rgb,
                            xyz=self.xyz,
                            depth=self.depth,
                            mask=vis_rect)


    def process_image(self):
        if not self.use_prev_mask:
            mask = self.get_mask()
            self.use_prev_mask = True
        else:
            mask = cv2.imread('ml_vision/data/bounded_mask.png',cv2.IMREAD_GRAYSCALE)
        self.align_mask_to_img(mask)
        self.load_npz('rgb_xyz_aligned_test.npz',mask=True)
        self.find_obstacle(self.xyz[...,2])

if __name__ == '__main__':
    img = ImageProcessor()
    img.debugging = True
    img.targ_class='build_cylinder'
    img.use_prev_mask = True
    img.load_npz('rgb_xyz_base.npz')
    img.process_image()
    #mask = img.get_oriented_bbox()
    #mask = cv2.imread('ml_vision/data/bounded_mask.png',cv2.IMREAD_GRAYSCALE)
    #img.align_mask(mask)
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
        self.depth = None
        self.mask = None
        self.model = None
        self.model_dir = 'ml_vision/v2_1213.pt'
        self.model_name = 'ml_vision/yolov5'
        self.use_prev_mask = False
        self.base_height = 0.1801 # Height of testbed base in m
        self.find_height = False

        # Image Processing Parameters
        self.border_exp = 1 # Amount to increase size of image to ensure complete capture of target 
        self.targ_class = None
        self.offset_px = 15 # Amount of px to constrict image to give clearance around vacuum nozzle

        # Obstacle detection parameters
        self.obstacle_border = 25 # Amount of px to constrict during obstacle detection
        self.obstacle = False # Checking if there is an obstacle detected in the image
        self.bottomedOut = False # Checking to see if the height differential exceeds the limits
        self.obstacleThreshold = 0.003 # height to check to see if there is an obstacle (m)
        self.bottomedOutThreshold = 0.080

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
        # Drop invalid points (NaN or 0) if needed
        xyz = np.nan_to_num(xyz)

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

    def load_npz(self,name):
        data = np.load('data/' + name)
        self.rgb = data["color"]
        self.xyz = data["xyz"]
        self.depth = data["depth"]

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
            hough = bounded.copy()
            for (x1,y1,x2,y2) in lines[:,0]:
                cv2.line(hough, (x1,y1), (x2,y2), (0,255,0), 2)
            
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
            mask_vis = self.rgb.copy()
            cv2.drawContours(mask_vis, [box_full], 0, (255, 0, 0), 1)
            cv2.drawContours(mask_vis, [shrink_box], 0, (0, 0, 255), 1)

            # ------ Correct way to build mask (avoids clipped box) ------
            full_mask = np.zeros(self.rgb.shape[:2], dtype=np.uint8)
            cv2.fillPoly(full_mask, [box_full], 255)

        except Exception:
            self.debugging = True
            
        finally:
            try:
                # Saving images for debugging if needed
                if self.debugging:
                    i = 0
                    cv2.imwrite(self.save_path + 'edges.png',edges)
                    i += 1
                    cv2.imwrite(self.save_path + 'thresh.png',thresh)
                    i += 1
                    cv2.imwrite(self.save_path + 'hough.png',hough)
                    i += 1
                    cv2.imwrite(self.save_path + 'bbox.png',mask_vis)
                    i += 1
                    print('All image processing debugging images saved')
                cv2.imwrite(self.save_path + 'bounded_mask.png',full_mask)
                print('Saved mask')
            except UnboundLocalError:
                match i:
                    case 0:
                        img = 'None of them'
                    case 1:
                        img = 'edges.png'
                    case 2:
                        img = 'thresh.png'
                    case 3:
                        img = 'hough.png'
                    case 4:
                        img = 'bbox.png'
                    case 5:
                        img = 'bounded_mask.png'
                    case _:
                        img = 'None of them'
                print('Something broke, check test images to see why')
                print(f'Last image that worked was {img}')
                quit()
            
            self.find_height = True

            return full_mask

    # Finds angle of provided mask, then rotates all image layers such that the mask is flat
    def align_mask_to_img(self,mask):
        try:
            for i in range(10):
                # Finding amgle of mask 
                cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                c = max(cnts, key=cv2.contourArea)
                angle = cv2.minAreaRect(c)[2]

                if angle > 45:
                    angle += -abs(angle)/angle*90

                # Find rotation matrix for given mask
                M = cv2.getRotationMatrix2D((mask.shape[1]/2,mask.shape[0]/2), angle, 1.0)

                def warp(im, interp=cv2.INTER_LINEAR):
                    return cv2.warpAffine(im, M, (im.shape[1], im.shape[0]),
                                        flags=interp)

                # Rotating all arrays based on rotation matrix
                rgb_rot  = np.dstack([warp(self.rgb[...,c]) for c in range(3)])
                xyz_rot   = np.dstack([warp(self.xyz[...,c],interp=cv2.INTER_NEAREST) for c in range(3)])
                depth_rot  = warp(self.depth, interp=cv2.INTER_NEAREST)
                mask_rot  = warp(mask, interp=cv2.INTER_NEAREST)
                
                # Finding bounding rectangle around mask to determine how much to cut off
                x_,y_,w_,h_ = cv2.boundingRect(mask_rot)
                self.offsets = (x_,y_)
                
                # Slicing images around the detected build cylinder, plus the defined offset
                rgb_offset = rgb_rot[y_ + self.offset_px:y_ + h_ - self.offset_px,x_ + self.offset_px:x_ + w_ - self.offset_px]
                xyz_offset = xyz_rot[y_ + self.offset_px:y_ + h_ - self.offset_px,x_ + self.offset_px:x_ + w_ - self.offset_px]
                depth_offset = depth_rot[y_ + self.offset_px:y_ + h_ - self.offset_px,x_ + self.offset_px:x_ + w_ - self.offset_px]
                mask_offset = mask_rot[y_ + self.offset_px:y_ + h_ - self.offset_px,x_ + self.offset_px:x_ + w_ - self.offset_px]

                # Determining if array is backwards or not and fixing
                rot = self.real_rect_size(xyz_offset)
                
                if rot:
                    xyz_offset = np.fliplr(xyz_offset)
                    if self.debugging:
                        print('XYZ flipped horizontally')
                
                # Checking to make sure size is correct
                if abs(mask_offset.shape[0] - mask_offset.shape[1])/max(mask_offset.shape[:2]) > .1:
                    if i == 9:
                        print('Bounding box wrong size')
                        continue
                    continue
                
                break
        
        except Exception as e:
            print(f'Exception: {e}')

        finally:
            if i > 8:
                print('Alignment did not succeed within 10 iterations')
                quit()
            cv2.imwrite(self.save_path + "aligned_mask.png", mask_offset)

            # Checking for obstacle and bottomed threshold on restricted section
            check_obstacle = xyz_offset[self.obstacle_border:- self.obstacle_border,self.obstacle_border:- self.obstacle_border]
            if not self.obstacle and np.max(check_obstacle[:,:,2]) - np.min(check_obstacle[:,:,2]) > self.obstacleThreshold:
                self.obstacle = True
                print('Obstacle detected')
            if not self.bottomedOut and np.max(check_obstacle[:,:,2]) - np.min(check_obstacle[:,:,2]) > self.bottomedOutThreshold:
                self.bottomedOut = True
                print('Reached bottom of cylinder')
            
            # Find height of base if requested
            if self.find_height:
                z = xyz_offset.copy()[:,:,2]
                z[z == 0] = np.nan
                self.base_height = np.nanmean(z)
                self.find_height = False
                print(f'Testbed surface height: {int(self.base_height*10000)/10} mm')

            # Saving aligned mask
            np.savez_compressed('data/rgb_xyz_aligned.npz',
                            color=rgb_offset,
                            xyz=xyz_offset,
                            depth=depth_offset,
                            mask=mask_offset)
            
            # Updating image parameters for use in obstacle detection, or signifying end of module
            if self.obstacle:
                self.rgb = rgb_offset
                self.xyz = xyz_offset
                self.depth = depth_offset
                self.mask = mask_offset
            else:
                print('Aligned NPZ Saved')
    
    # Ultimately, the best processing for a non-lit obstacle was the xyz map, but a lit map would probably show better results with the rgb
    def find_obstacle(self,img):
        h,w = img.shape
        bounded = img[self.obstacle_border:h - self.obstacle_border,self.obstacle_border:w - self.obstacle_border]
        depth = bounded.astype(np.float32)
        z_max = np.max(depth)
        thresh = z_max - 0.004 # 2mm below the highest surface point

        # Finding obstacle boundaries based on height map
        binary = (depth >= thresh).astype(np.uint8) * 255

        # Smoothing out edges of contours
        kernel = np.ones((8,8), np.uint8)
        edges = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # Find contours directly from binary
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Finding the smallest contour near the center of the image (I don't think this matters for the xyz threshold approach, but it works and I don't want to risk breaking it)
        center = np.array([w/2, h/2])

        def contour_center(c):
            M = cv2.moments(c)
            if M['m00'] == 0: return np.inf
            return np.array([M['m10']/M['m00'], M['m01']/M['m00']])

        cnt = min(contours, key=lambda c: np.linalg.norm(contour_center(c)-center))

        # Approximate identified contour to polygon
        epsilon = 0.02 * cv2.arcLength(cnt, True)  # epislon of 0.02 was working well, tried a few values around it but it made it far worse
        poly = cv2.approxPolyDP(cnt, epsilon, True)

        # Adjusting polygon values to line up with mask, and drawing over mask
        poly_pts = poly.reshape((-1,1,2)).astype(np.int32)
        poly_pts = poly_pts + np.array([[self.obstacle_border//2, self.obstacle_border//2]])
        _,_,w,h = cv2.boundingRect(poly_pts)
        hb,wb = bounded.shape
        obstacle_mask = self.mask.copy()

        # Ensuring that a full obstacle is not mistakenly used
        if abs(wb - w) < 2 and abs(hb - h) < 2:
            print('Mistaken no obstacle detected')
        else:
            obstacle_mask = self.mask.copy()
            cv2.fillPoly(obstacle_mask, [poly_pts], 0)

        # Saving altered mask for path planning algorithm 
        np.savez_compressed('data/rgb_xyz_aligned.npz',
                            color=self.rgb,
                            xyz=self.xyz,
                            depth=self.depth,
                            mask=obstacle_mask)
        cv2.imwrite(self.save_path + 'obstacle_mask.png',obstacle_mask)
        print('Aligned NPZ with Obstacle Saved')
    
    # Pipeline to process an image, and sets appropriate inputs to be handled (use_prev_mask will use the previously identified mask rather than using model to find new one)
    def process_image(self):
        if not self.use_prev_mask:
            mask = self.get_mask()
            self.use_prev_mask = True
        else:
            mask = self.mask if self.mask is not None else cv2.imread('ml_vision/data/bounded_mask.png',cv2.IMREAD_GRAYSCALE)
        
        self.align_mask_to_img(mask)
        
        if self.obstacle:
            self.find_obstacle(self.xyz[...,2])

if __name__ == '__main__':
    img = ImageProcessor()
    img.debugging = False
    img.targ_class='build_cylinder'
    img.use_prev_mask = True
    img.load_npz('rgb_xyz_base.npz')
    img.process_image()
import torch 
import numpy as np
import pathlib
import platform
import cv2
import pyrealsense2 as rs
import time
from scipy.ndimage import distance_transform_edt
class RealSenseCamera:
    def __init__(self,save_path = None):
        if not save_path:
            print('Camera needs to be initialized with a save path')
            quit()
        self.save_path = save_path
        self.debugging = False

        # Camera parameters
        self.auto_exposure = True
        self.exposure = None
        self.auto_wb = True
        self.wb = None
        self.set_emitter = True
        self.h = 480
        self.w = 848

        # Depth processing parameters
        self.max_distance = 2 #m

        # Decimate filter
        self.decimate = rs.decimation_filter()
        self.decimate.set_option(rs.option.filter_magnitude, 2)
        
        # Threshold filter
        self.threshold = rs.threshold_filter()
        self.threshold.set_option(rs.option.max_distance, self.max_distance)

        # Disparity domain
        self.depth_to_disparity = rs.disparity_transform(True)
        
        # Spatial filter
        self.spatial = rs.spatial_filter()
        self.spatial.set_option(rs.option.filter_magnitude, 2)
        self.spatial.set_option(rs.option.filter_smooth_alpha, .5)
        self.spatial.set_option(rs.option.filter_smooth_delta, 20)
        self.spatial.set_option(rs.option.holes_fill, 2)
        
        # Temporal Filter
        self.temporal = rs.temporal_filter()
        self.temporal_length = 15 #frames

        # Depth domain
        self.disparity_to_depth = rs.disparity_transform(False)
        
        # Hole filling filter
        self.hole_filling = rs.hole_filling_filter(1)

        # Build pipeline and config
        self.pipeline = rs.pipeline()
        config = rs.config()

        # Configuring image streams
        config.enable_stream(rs.stream.depth, self.w, self.h, rs.format.z16, 60)
        config.enable_stream(rs.stream.color, self.w, self.h, rs.format.bgr8, 60)

        # Start pipeline
        profile = self.pipeline.start(config)

        # Configure camera exposure + white balance
        self.depth_sensor,self.color_sensor,_ = profile.get_device().query_sensors()
        
        # Alignment + pointcloud objects moved into class
        self.align = rs.align(rs.stream.color)
        self.pc = rs.pointcloud()

    def configure_image(self):
        self.depth_sensor.set_option(rs.option.emitter_enabled, self.set_emitter)
        self.color_sensor.set_option(rs.option.enable_auto_exposure, self.auto_exposure)
        
        if not self.auto_exposure and self.exposure:
            self.color_sensor.set_option(rs.option.exposure, self.exposure)
        
        self.color_sensor.set_option(rs.option.enable_auto_white_balance, self.auto_wb)

        if self.debugging and self.color_sensor.get_option(rs.option.exposure) != self.exposure:
            print(f'Exposure level: {self.color_sensor.get_option(rs.option.exposure)}')
            print(f'Desired exposure level: {self.exposure}')

        if not self.auto_wb and self.wb:
            self.color_sensor.set_option(rs.option.white_balance, self.wb)
    
    # Applying colormap to depth image
    def normalize_cmap(self,img):
        valid = img[img > 0]
        img_min = valid.min()
        img_max = valid.max()
        img_norm = np.zeros_like(img, dtype=np.uint8)
        img_norm[img > 0] = (((img[img > 0] - img_min) /
                              (img_max - img_min)) * 255).astype(np.uint8)
        
        return cv2.applyColorMap(img_norm, cv2.COLORMAP_JET)

    def convert_to_xyz(self,points):
        v, t = points.get_vertices(), points.get_texture_coordinates()
        xyz = np.asanyarray(v).view(np.float32).reshape(-1, 3)
        uv = np.asanyarray(t).view(np.float32).reshape(-1, 2)
        
        # Extract XYZ and UV
        X = xyz[:,0]
        Y = xyz[:,1]
        Z = xyz[:,2]
        u = uv[:,0]
        v = uv[:,1]

        H = int(self.h / 2)
        W = int(self.w / 2)
        
        # Convert normalized UV → pixel indices
        cols = (u * (W - 1)).astype(np.int32)
        rows = (v * (H - 1)).astype(np.int32)

        # Allocate output XYZ image
        xyz_image = np.zeros((H, W, 3), dtype=np.float32)
        xyz_image[rows, cols] = np.column_stack((X, Y, Z))

        # Resize to proper size
        xyz_image = cv2.resize(xyz_image, (self.w, self.h), interpolation=cv2.INTER_NEAREST)

        return xyz_image
    
    def write_ply(self,xyz, rgb):
        filename = self.save_path + 'data/debugging/xyz.ply'
        points = xyz.reshape(-1,3)
        colors = rgb.reshape(-1,3)
        valid = points[:,2] > 0
        pts = np.hstack((points[valid], colors[valid])).astype(np.float32)

        with open(filename, 'w') as f:
            f.write("ply\nformat ascii 1.0\n")
            f.write(f"element vertex {pts.shape[0]}\n")
            f.write("property float x\nproperty float y\nproperty float z\n")
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
            f.write("end_header\n")
            for x,y,z,r,g,b in pts:
                f.write(f"{x} {y} {z} {int(r)} {int(g)} {int(b)}\n")

    def capture(self):
        while True:
            start = time.time()
            try:
                self.configure_image()
                # Let camera settle
                for _ in range(5):
                    _ = self.pipeline.wait_for_frames()
                
                for _ in range(self.temporal_length):
                    # Capture frames
                    frames = self.pipeline.wait_for_frames()
                    aligned_frames = self.align.process(frames)

                    depth_frame = aligned_frames.get_depth_frame()
                    color_frame = aligned_frames.get_color_frame()

                    if not depth_frame or not color_frame:
                        raise RuntimeError("Could not acquire both color and depth frames.")

                    depth_frame = self.decimate.process(depth_frame)
                    depth_frame = self.threshold.process(depth_frame)
                    depth_frame = self.depth_to_disparity.process(depth_frame)
                    depth_frame = self.spatial.process(depth_frame)
                    depth_frame = self.temporal.process(depth_frame)
                    depth_frame = self.disparity_to_depth.process(depth_frame)
                    depth_frame = self.hole_filling.process(depth_frame)
                    
                # Generate pointcloud
                self.pc.map_to(color_frame)
                points = self.pc.calculate(depth_frame)
                xyz_image = self.convert_to_xyz(points)
                
                # Convert to numpy
                color_image = np.asanyarray(color_frame.get_data())
                depth_image = np.asanyarray(depth_frame.get_data())
                
                depth_colormap = self.normalize_cmap(depth_image)
                
                if self.debugging:
                    cv2.imwrite(self.save_path + 'data/debugging/raw_rgb.png', color_image)
                    cv2.imwrite(self.save_path + 'data/debugging/raw_depth.png', depth_colormap)
                    print('Raw RGB and Depth image saved')
                    end = time.time()
                    time_elapsed = end - start
                    self.write_ply(xyz_image,color_image)
                    print(f'Took {time_elapsed*1000} ms')

                np.savez_compressed(self.save_path + "data/rgb_xyz_capture.npz",
                                    color=color_image,
                                    xyz=xyz_image)
                print('Capture NPZ saved')
                break

            except RuntimeError:
                print("Device was reset due to a runtime error. Retrying capture...")
                self.reset()
                continue
            finally:
                self.shutdown()
    
    def reset(self):
        ctx = rs.context()
        list = ctx.query_devices()
        for dev in list:
            dev.hardware_reset()
        time.sleep(1)

    def shutdown(self):
        self.pipeline.stop()
        print('Camera pipeline ended')

# Everything related to image processing
class ImageProcessor:
    def __init__(self,save_path = None):
        if not save_path:
            print('Image processor needs to be initialized with a save path')
            quit()
        self.save_path = save_path
        self.debugging = False
        self.visualize = False

        # Data Variables
        self.rgb = None
        self.xyz = None
        self.model = None
        self.model_name = 'best.pt'

        # Image Processing Parameters
        self.border_exp = 5 # Amount to increase size of image to ensure complete capture of target 
        self.targ_class = None
        self.offset_px = 10 # Amount of px to constrict image to give clearance around vacuum nozzle

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

    def load_npz(self,name):
        data = np.load(self.save_path + 'data/' + name)
        self.rgb = data["color"]
        self.xyz = data["xyz"]

    # Runs YOLOv5 model and extracts bounding box
    def extract_model_bbox(self):
        RGB_img = cv2.cvtColor(self.rgb, cv2.COLOR_BGR2RGB)
        results = self.model(RGB_img)
        
        if self.debugging:
            cv2.imwrite(self.save_path + 'data/debugging/yolo.png',results.render()[0])

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
    def get_oriented_bbox(self):
        self.model = torch.hub.load(self.save_path + 'yolov5','custom',
                                   path=self.save_path + self.model_name,
                                   force_reload=True,source='local')
        try:
            self.extract_model_bbox()
            # Grabbing coordinates out of model bounding box and clipping image
            bounded = self.rgb[self.y_lo:self.y_hi,self.x_lo:self.x_hi]
            
            # Thresholding image to make difference between sections more distinct
            gray = cv2.cvtColor(bounded, cv2.COLOR_BGR2GRAY)
            gray_f = gray.astype(np.float32)
            scale = 255.0 / (220 - 160) #hi - lo
            thresh = (gray_f - 160) * scale #gray_f - lo
            thresh = np.clip(thresh, 0, 255).astype(np.uint8)
            
            # Canny edge detection
            blur = cv2.GaussianBlur(thresh, (11,11), 0)
            edges = cv2.Canny(blur, 1, 70)
            
            # Detecting Hough lines from Canny edges
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=11,
                                    minLineLength=150, maxLineGap=50)
            
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

            # visualize on full RGB image
            vis_rect = self.rgb.copy()
            cv2.drawContours(vis_rect, [box_full], 0, (255, 0, 0), 2)

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
                    cv2.imwrite(self.save_path + 'data/debugging/clipped.png',bounded)
                    i += 1
                    cv2.imwrite(self.save_path + 'data/debugging/edges.png',edges)
                    i += 1
                    cv2.imwrite(self.save_path + 'data/debugging/thresh.png',thresh)
                    i += 1
                    cv2.imwrite(self.save_path + 'data/debugging/bbox.png',vis_rect)
                    i += 1
                    cv2.imwrite(self.save_path + 'data/debugging/hough.png',vis_hough)
                    i += 1
                    cv2.imwrite(self.save_path + 'data/debugging/bounded_mask.png',full_mask)
                    print('All image processing debugging images saved')

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
                        img = 'bbox.png'
                    case 5:
                        img = 'Vis_Hough'
                print('Something broke, check test images to see why')
                print(f'Last image that worked was {img}')
                quit()
            
            return full_mask

    def align_mask(self,mask):
        try:
            for i in range(50):
                cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                c = max(cnts, key=cv2.contourArea)
                rect = cv2.minAreaRect(c)
                angle = rect[2]

                if rect[1][0] < rect[1][1]:
                    angle += 90

                angle += -abs(angle)/angle*90

                # --- build rotation matrix about the object centroid ---
                M = cv2.getRotationMatrix2D((mask.shape[1]/2,mask.shape[0]/2), angle, 1.0)

                def warp(im, interp=cv2.INTER_LINEAR):
                    return cv2.warpAffine(im, M, (im.shape[1], im.shape[0]),
                                        flags=interp)#, borderMode=cv2.BORDER_CONSTANT)

                # --- rotate all aligned arrays ---
                rgb_rot  = np.dstack([warp(self.rgb[...,c]) for c in range(3)])
                mask_rot  = warp(mask, interp=cv2.INTER_NEAREST)
                xyz_rot   = np.dstack([warp(self.xyz[...,c],interp=cv2.INTER_NEAREST) for c in range(3)])

                x_,y_,w_,h_ = cv2.boundingRect(mask_rot)

                mask_clip = mask_rot[y_:y_ + h_,x_:x_ + w_]
                rgb_clip = rgb_rot[y_:y_ + h_,x_:x_ + w_]
                xyz_clip = xyz_rot[y_:y_ + h_,x_:x_ + w_]

                # Clip all the frames to the size of the mask for ease in path planning
                h,w = mask_clip.shape
                mask_offset = mask_clip[self.offset_px:h - self.offset_px, self.offset_px:w - self.offset_px]
                rgb_offset = rgb_clip[self.offset_px:h - self.offset_px, self.offset_px:w - self.offset_px]
                xyz_offset = xyz_clip[self.offset_px:h - self.offset_px, self.offset_px:w - self.offset_px]

                if np.all(mask_offset == 255):
                    break

        except Exception as e:
            print(f'Exception: {e}')

        finally:
            cv2.imwrite(self.save_path + "data/debugging/aligned_mask.png", rgb_offset)
            if i > 48:
                print('Alignment did not succeed within 50 iterations')
                quit()
            np.savez_compressed(self.save_path + 'data/rgb_xyz_aligned.npz',
                            color=rgb_offset,
                            xyz=xyz_offset,
                            mask=mask_offset)
            print('Aligned NPZ Saved')

    def pointcloud(self):
        import open3d as o3d
        origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100, origin=[0,0,0])
        coords = np.asarray(list(zip(*np.nonzero(self.xyz))))
        points = [self.xyz[i,j] for i,j in coords[:,:2]]
        scalar = 1
        points_f = [(float(i*scalar),float(j*scalar),float(k*scalar)) for i,j,k in points]
        #print(points_f)
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points_f)
        
        o3d.visualization.draw_geometries([cloud])

if __name__ == '__main__':
    dir = 'ml_vision/'
    model_name = 'best.pt'
    camera = RealSenseCamera(save_path=dir)
    camera.auto_exposure = False
    camera.exposure = 200
    camera.auto_wb = False
    camera.wb = 3000
    camera.debugging = True
    #camera.capture()
    #quit()
    img = ImageProcessor(save_path=dir)
    img.debugging = True
    img.model = torch.hub.load(dir + 'yolov5','custom',path=dir + model_name,force_reload=True,source='local')
    img.load_npz('rgb_xyz_capture.npz')
    
    img.targ_class='build_cylinder'
    #img.pointcloud()
    mask = img.get_oriented_bbox()
    #mask = cv2.imread('ml_vision/data/debugging/bounded_mask.png',cv2.IMREAD_GRAYSCALE)
    img.align_mask(mask)
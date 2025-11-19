import numpy as np
import cv2
import pyrealsense2 as rs
import time
from tqdm import tqdm

class RealSenseCamera:
    def __init__(self):
        self.save_path = 'am_vision/data/'
        self.debugging = False
        
        # Camera parameters
        self.auto_exposure = True # Importantly, enables custom auto exposure algorithm, does not use the built in one
        self.exposure = 280
        self.auto_wb = True
        self.brightness_goal = 175 # If you adjust this, you will likely need to adjust the Canny edge detection in the ml_detect package
        self.wb = None
        self.set_emitter = True
        self.h = 480 
        self.w = 848

        # Cutoff distance for depth camera
        self.max_distance = 2 #m

        # Decimate filter parameters
        self.decimate = rs.decimation_filter()
        
        # Threshold filter parameters
        self.threshold = rs.threshold_filter()
        self.threshold.set_option(rs.option.max_distance, self.max_distance)

        # Disparity domain
        self.depth_to_disparity = rs.disparity_transform(True)
        
        # Spatial filter parameters
        self.spatial = rs.spatial_filter()
        self.spatial.set_option(rs.option.holes_fill, 1)
        self.spatial.set_option(rs.option.filter_magnitude,1)
        self.spatial.set_option(rs.option.filter_smooth_alpha,.5)
        self.spatial.set_option(rs.option.filter_smooth_delta,50)
        # Temporal Filter
        self.temporal = rs.temporal_filter()
        self.temporal_length = 50 # This is how many frames that will prime the filters before taking an actual image. 15 is the minimum I found that worked

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

        # Define sensors for use in the configuration function
        self.depth_sensor,self.color_sensor,_ = profile.get_device().query_sensors()
        
        # Alignment + pointcloud objects
        self.align = rs.align(rs.stream.color)
        self.pc = rs.pointcloud()

    # This runs before capturing an image, primarily calibrating exposure and white balance
    def configure_image(self):
        #self.depth_sensor.set_option(rs.option.emitter_enabled, self.set_emitter)
        self.color_sensor.set_option(rs.option.enable_auto_exposure, False)
        self.color_sensor.set_option(rs.option.exposure, self.exposure)

        if self.auto_exposure:
            self.calibrate_exposure()
        
        self.color_sensor.set_option(rs.option.enable_auto_white_balance, self.auto_wb)

        if not self.auto_wb and self.wb:
            self.color_sensor.set_option(rs.option.white_balance, self.wb)
    
    # Helper function used for calibrating exposure that retursn the grayscale intensity
    def get_bright_diff(self,exposure):
        self.color_sensor.set_option(rs.option.exposure, exposure)
        avg = []

        for _ in range(5):
            _ = self.pipeline.wait_for_frames()

        for _ in range(50):
            frame = self.pipeline.wait_for_frames()
            rgb_f = frame.get_color_frame()
            rgb_np = np.asanyarray(rgb_f.get_data())
            gray = cv2.cvtColor(rgb_np, cv2.COLOR_BGR2GRAY)
            avg.append(np.mean(gray))

        # Cuts out any values 2 standard deviations away. Almost never relevant, but it doesn't hurt
        avg = np.asarray(avg)
        mean = np.mean(avg)
        std = np.std(avg)

        mask = abs(avg - mean) < 2*std
        avg = np.mean(avg[mask])

        return avg - self.brightness_goal
    
    # Exposure calibration routine
    def calibrate_exposure(self):
        exposure = self.exposure
        max_iterations = 50
        failed = False

        if abs(self.get_bright_diff(exposure)) < .5: # If it's already correct, don't bother calibrating
            return

        # Progress bar to view max time left for calibration
        print('Calibrating exposure')
        
        pbar = tqdm(range(max_iterations), leave=False)
        for i in pbar:
            diff = self.get_bright_diff(exposure)
            
            # I was getting some exposure runoff occasionally, unsure why exactly and I haven't bothered to fix it yet but I will at least include error handling
            if exposure != self.color_sensor.get_option(rs.option.exposure):
                failed = True
                break # I've noticed that it will still usually be around where it's supposed to, but you can change to quit() if you'd prefer to figure it out

            # Changing the exposure as needed. .3 is probably more than necessary, but I was getting good results with it so why not
            if abs(diff) < .3:
                pbar.update(50 - i) # Necessary to set final value of pbar so that it doesn't get left at
                if self.debugging:
                    print(f'Exposure Calibration Successful')
                    print(f'Exposure level set to {exposure} from {self.exposure}')
                    print(f'Brightness: {diff + self.brightness_goal}')
                break
            elif abs(diff) > 1:
                exposure -= int(diff)
            elif -1 < diff < -.3:
                exposure += 1
            else:
                exposure -= 1

            exposure = round(exposure,0)
            if i == 49:
                failed = True
                break
        
        if failed:
            print('Exposure Calibration Failed')
            print(f'Final exposure level of {exposure} from {self.exposure}')
            print(f'Brightness: {diff + self.brightness_goal}')

    # Applying colormap to depth image
    def normalize_cmap(self,img):
        valid = img[img > 0]
        img_min = valid.min()
        img_max = valid.max()
        img_norm = np.zeros_like(img, dtype=np.uint8)
        img_norm[img > 0] = (((img[img > 0] - img_min) /
                              (img_max - img_min)) * 255).astype(np.uint8)
        
        return cv2.applyColorMap(img_norm, cv2.COLORMAP_JET)

    # Converting an rs pointcloud to xyz coordinates
    def convert_to_xyz(self,pc):
        v, t = pc.get_vertices(), pc.get_texture_coordinates()
        xyz = np.asanyarray(v).view(np.float32).reshape(-1, 3) # XYZ values
        uv = np.asanyarray(t).view(np.float32).reshape(-1, 2) # Corresponding (u,v) index of aligned RGB image
        
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

        # Arrange xyz image out of components
        xyz_image = np.zeros((H, W, 3), dtype=np.float32)
        xyz_image[rows, cols] = np.column_stack((X, Y, Z))

        # Resize to rgb, upscaling with a nearest neighbor interpolation to "undo" decimation, but smoother
        xyz_image = cv2.resize(xyz_image, (self.w, self.h), interpolation=cv2.INTER_NEAREST)

        return xyz_image

    # Primarily for debugging or visualization if desired, but will output a .ply for a matching xyz, rgb image set
    def write_ply(self,xyz, rgb):
        filename = self.save_path + 'xyz.ply'
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
    
    # Main image capture routine
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
                    
                # Generate pointcloud and convert to xyz image
                self.pc.map_to(color_frame)
                points = self.pc.calculate(depth_frame)
                xyz_image = self.convert_to_xyz(points)
                
                # Convert to Numpy arrays to save as an image
                color_image = np.asanyarray(color_frame.get_data())
                depth_image = np.asanyarray(depth_frame.get_data())
                
                #depth_image = cv2.resize(depth_image, (self.w, self.h), interpolation=cv2.INTER_NEAREST)
                depth_colorized = self.normalize_cmap(depth_image)
                
                if self.debugging:
                    cv2.imwrite(self.save_path + 'raw_rgb.png', color_image)
                    cv2.imwrite(self.save_path + 'raw_depth.png', depth_colorized)
                    print('Raw RGB and Depth image saved')
                    end = time.time()
                    time_elapsed = end - start
                    self.write_ply(xyz_image,color_image)
                    print(f'Took {time_elapsed*1000} ms')

                np.savez_compressed("data/rgb_xyz_capture.npz",
                                    color=color_image,
                                    depth=depth_image,
                                    xyz=xyz_image)
                print('Capture NPZ saved')
                break
            
            # For whatever reason, certain USB ports on the NUC just don't like the camera. This will catch it and reset the camera if needed
            except RuntimeError:
                self.reset()
                continue
            finally:
                self.pipeline.stop()
                print('Camera pipeline ended')
    
    # Helper function to quickly reset camera in case there were issues
    def reset(self):
        context = rs.context()
        list = context.query_devices()
        for device in list:
            device.hardware_reset()
        print("Device was reset due to a runtime error. Retrying capture...")
        time.sleep(1) # Sleep is necessary, otherwise it will try to connect too quickly and fail

if __name__ == '__main__':
    camera = RealSenseCamera()
    camera.auto_exposure = False
    camera.debugging = True
    camera.capture()
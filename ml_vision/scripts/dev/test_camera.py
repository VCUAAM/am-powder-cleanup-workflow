import pyrealsense2 as rs
import numpy as np
import cv2
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

    def start_stream(self):
        # Build pipeline and config
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        # Start pipeline
        profile = self.pipeline.start(config)

        # Configure camera exposure + white balance
        color_sensor = profile.get_device().query_sensors()[1]
        color_sensor.set_option(rs.option.enable_auto_exposure, self.auto_exposure)

        if not self.auto_exposure and self.exposure:
            color_sensor.set_option(rs.option.exposure, self.exposure)
        
        color_sensor.set_option(rs.option.enable_auto_white_balance, self.auto_wb)

        if self.debugging and color_sensor.get_option(rs.option.exposure) != self.exposure:
            print(f'Exposure level: {color_sensor.get_option(rs.option.exposure)}')
            print(f'Desired exposure level: {self.exposure}')

        if not self.auto_wb and self.wb:
            color_sensor.set_option(rs.option.white_balance, self.wb)

        # Alignment + pointcloud objects moved into class
        self.align = rs.align(rs.stream.color)
        self.pc = rs.pointcloud()

    def capture(self):
        try:
            self.start_stream()
            # Let auto-exposure settle (unchanged)
            for _ in range(5):
                _ = self.pipeline.wait_for_frames()

            # Capture frames
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)

            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not depth_frame or not color_frame:
                raise RuntimeError("Could not acquire both color and depth frames.")

            # Generate pointcloud
            points = self.pc.calculate(depth_frame)
            self.pc.map_to(color_frame)

            # Convert to numpy
            color_image = np.asanyarray(color_frame.get_data())
            vtx = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)

            # reshape xyz to same resolution as rgb
            h, w = color_image.shape[:2]
            xyz_image = vtx.reshape(h, w, 3)

            # Save outputs exactly the same

            if self.debugging:
                cv2.imwrite(self.save_path + '/debugging/raw_rgb.png', color_image)
                print('Raw RGB image saved')

            np.savez_compressed(self.save_path + "rgb_xyz_capture.npz",
                                color=color_image,
                                xyz=xyz_image)
            print('Capture NPZ saved')
    
        finally:
            self.shutdown()

    def shutdown(self):
        self.pipeline.stop()

if __name__ == "__main__":
    save_path = "ml_vision/data/"
    camera = RealSenseCamera(save_path)
    camera.debugging = True
    camera.auto_exposure = False
    camera.exposure = 225
    camera.auto_wb = False
    camera.wb = 3000
    camera.capture()
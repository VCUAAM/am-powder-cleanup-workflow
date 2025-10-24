import pyrealsense2 as rs
import numpy as np
import cv2
    
def save_capture(color_image,xyz_image,save_path,shape):

    # 5. Save RGB + XYZ to a compressed .npz
    cv2.imwrite(save_path + shape + '.png',color_image)
    print('Saved:' + shape + '.png')

def camera_capture():
    while True:
        # 1. Start RealSense pipeline
        pipeline = rs.pipeline()
        config = rs.config()
        context = rs.context()
        devices = context.query_devices()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        profile = pipeline.start(config)
        color_sensor = profile.get_device().query_sensors()[1]
        color_sensor.set_option(rs.option.enable_auto_exposure, True)
        color_sensor.set_option(rs.option.enable_auto_white_balance, False)
        color_sensor.set_option(rs.option.white_balance,3000)
        # 2. Create alignment and pointcloud objects
        align = rs.align(rs.stream.color)
        pc = rs.pointcloud()

        try:
            # Let auto-exposure settle
            for _ in range(5):
                frames = pipeline.wait_for_frames()

            # 3. Capture and align frames
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not depth_frame or not color_frame:
                raise RuntimeError("Could not acquire both color and depth frames.")

            # 4. Generate 3D point cloud
            points = pc.calculate(depth_frame)
            pc.map_to(color_frame)  # ensures XYZ matches RGB pixels

            # Convert to numpy
            color_image = np.asanyarray(color_frame.get_data())
            vtx = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)
            
            # Reshape XYZ to image shape
            h, w = color_image.shape[:2]
            xyz_image = vtx.reshape(h, w, 3)

            return color_image, xyz_image

        except RuntimeError:
            for device in devices:
                device.hardware_reset()
            print("Device was reset due to a runtime error. Retrying capture...")
            continue

        finally:
            pipeline.stop()
            

def main():
    save_path = "/home/rosnuc/GitHub/vcu_am_post_processing/ml_vision/"
    shape = 'test'

    color_image,xyz_image = camera_capture()
    save_capture(color_image,xyz_image,save_path,shape)
    

if __name__ == "__main__":
    main()
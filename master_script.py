from robot_control.robot import UR5Robot
from ml_vision.camera import RealSenseCamera, ImageProcessor
from path_planning.path_planner import PathPlanner

if __name__ == "__main__":
    try:
        vision = "ml_vision/"

        rob = UR5Robot()
        rob.testing = True #DISABLE IF NOT TESTING, WILL ELEVATE ENTIRE PATH PLANNING AND CAUSE COLLISIONS
    
        camera = RealSenseCamera(save_path=vision)
        camera.debugging = True 
        camera.auto_exposure = False
        camera.exposure = 200
        camera.auto_wb = False
        camera.wb = 3000

        img = ImageProcessor(save_path=vision)
        img.debugging = True
        img.targ_class='build_cylinder'
        img.model_name = 'best.pt'

        pln = PathPlanner(save_path='path_planning/')
        pln.cluster_size = 30

        p_home = [-1.57, -1.308, -2.268, -1.136, 1.571, 0.001] # absolute
        p_scan_0 = [-1.26, -1.898, -2.473, 1.223, 1.255, -0.002] # absolute
        p_scan = [-1.463, -2.488, -1.191, 0.549, 1.447, -0.012] # absolute 
        p_prevac = [-1.396, -2.371, -1.15, -1.19, 1.588, 0.029] #absolute

        # Actual logic
        rob.set_tcp('base')
        rob.moveJ(p_home)
        rob.set_tcp('camera')
        rob.moveJ(p_scan_0)
        rob.moveJ(p_scan)
        camera.capture()
        rob.convert_to_base(vision + 'data/')
        img.load_npz('rgb_xyz_base.npz')
        mask = img.get_oriented_bbox()
        img.align_mask(mask)
        #rob.check_testing()
        rob.moveJ(p_prevac)
        pln.load_npz(vision + 'data/rgb_xyz_aligned.npz')
        pln.compute_path()
        rob.move_path('path_planning/data/robot_path.npz')
        rob.moveJ(p_home) 

    except KeyboardInterrupt:
        print("KeyboardInterrupt detected. Disconnecting...")
    finally:
        rob.shutdown()

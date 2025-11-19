from robot_control.robot import UR5Robot
from am_vision.camera import RealSenseCamera
from ml_vision.ml_detect import ImageProcessor
from path_planning.path_planner import PathPlanner

if __name__ == "__main__":
    try:
        rob = UR5Robot()
        rob.testing = False #DISABLE IF NOT TESTING, WILL ELEVATE ENTIRE PATH PLANNING AND CAUSE COLLISIONS
        
        camera = RealSenseCamera()
        camera.debugging = True 
        camera.auto_exposure = True
        camera.exposure = 205
        camera.auto_wb = False
        camera.wb = 3000

        img = ImageProcessor()
        img.debugging = True
        img.targ_class='build_cylinder'

        pln = PathPlanner()
        pln.cluster_size = 32

        p_home = [-1.57, -1.308, -2.268, -1.136, 1.571, 0.001] # absolute
        p_scan_0 = [-1.26, -1.898, -2.473, 1.223, 1.255, -0.002] # absolute
        p_scan = [-1.463, -2.488, -1.191, 0.549, 1.447, -0.012] # absolute 
        #p_scan = [-1.463, -2.475, -1.187, 0.532, 1.447, -0.012] # absolute 
        #p_preprevac = [-1.462, -2.551, -1.202, 0.623, 1.446, -0.012] # absolute
        p_preprevac = [-1.462, -2.568, -1.203, 0.64, 1.446, -0.012] #absolute
        p_prevac = [-1.396, -2.382, -1.161, -1.167, 1.588, 0.029] #absolute

        rescan = True
        # Actual logic
        if False:
            pass
            rob.set_tcp('base')
            rob.home()
            rob.set_tcp('camera')
            rob.moveJ(p_scan_0)
            rob.moveJ(p_scan)
            camera.capture()
            #rob.moveJ(p_scan_0)
            #rob.moveJ(p_scan)
        rob.convert_to_base('rgb_xyz_capture.npz')
        quit()
        #img.use_prev_mask = True
        img.load_npz('rgb_xyz_base.npz')

        mask = img.get_oriented_bbox()
        img.align_mask(mask)
        pln.load_npz('rgb_xyz_aligned_test.npz')
        pln.compute_path()
        quit()
        rob.check_testing()
        rob.moveJ(p_preprevac)
        rob.moveJ(p_prevac)
        rob.vac_on()
        rob.move_path('robot_path.npz')
        rob.vac_off()
        rob.moveJ(p_prevac)
        rob.moveJ(p_scan_0)
        rob.moveJ(p_home) 


    except KeyboardInterrupt:
        print("KeyboardInterrupt detected. Disconnecting...")
    finally:
        rob.shutdown()

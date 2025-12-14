from robot_control.robot import UR5Robot
from am_vision.camera import RealSenseCamera
from ml_vision.ml_detect import ImageProcessor
from path_planning.path_planner import PathPlanner

p_home = [-1.571, -0.838, -2.055, -1.819, 1.571, 0.0] # absolute
p_approach = [-1.362, -0.748, -2.411, -0.009, 1.35, 0.023] # absolute
p_scan = [-1.535, -1.675, -1.754, 0.264, 1.534, 0.02] # absolute 
p_vac_approach = [-1.529, -1.648, -1.976, 0.459, 1.527, 0.02] #absolute
p_vac = [-1.547, -1.656, -1.601, -1.457, 1.586, 0.007] #absolute


def testingRoutine():
    #rob.set_tcp('base')
    #rob.moveHome()
    #rob.moveJ(p_approach)
    #rob.moveJ(p_vac_approach)
    #rob.moveJ(p_scan)

    while True:
        rob.set_tcp('camera')
        camera.capture()
        rob.convert_to_base('rgb_xyz_capture.npz')
        img.load_npz('rgb_xyz_base.npz')
        img.process_image()
        quit()
        rob.set_tcp('base')
        if img.bottomedOut:
            break
        pln.load_npz('rgb_xyz_aligned.npz')
        pln.compute_path()
        rob.moveJ(p_vac_approach)
        rob.moveJ(p_vac)
        rob.move_path('robot_path.npz')
        rob.moveJ(p_vac)
        rob.moveJ(p_vac_approach)
        rob.moveJ(p_scan)

    print('Completed scanning')
    rob.moveHome()

if __name__ == "__main__":
    try:
        rob = UR5Robot() 
        rob.testing = False #DISABLE IF NOT TESTING, WILL ELEVATE ENTIRE PATH PLANNING AND CAUSE COLLISIONS
        rob.vel = 0.75
        rob.acc = 0.25

        camera = RealSenseCamera()
        camera.debugging = True 
        camera.auto_exposure = True
        camera.exposure = 205
        camera.auto_wb = False
        camera.wb = 3000

        img = ImageProcessor()
        img.debugging = False
        img.targ_class='build_cylinder'
        img.use_prev_mask = True # Comment out or set to false if starting fresh scan
        rob.base_height = img.base_height
        
        pln = PathPlanner()
        pln.avg_z = img.base_height

        if False:
            pass
            rob.moveHome()
            rob.moveJ(p_approach)
            rob.moveJ(p_vac_approach)
            rob.moveJ(p_scan)
            rob.set_tcp('camera')
            camera.capture()
            pln.load_npz('rgb_xyz_aligned.npz')
            pln.compute_path()
        testingRoutine()

    except KeyboardInterrupt:
        print("KeyboardInterrupt detected. Disconnecting...")
    except Exception as e:
        print(e)
    finally:
        #rob.moveHome()
        rob.shutdown()
        camera.stop()

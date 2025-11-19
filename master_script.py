from robot_control.robot import UR5Robot
from am_vision.camera import RealSenseCamera
from ml_vision.ml_detect import ImageProcessor
from path_planning.path_planner import PathPlanner

p_home = [-1.57, -1.308, -2.268, -1.136, 1.571, 0.001] # absolute
p_approach = [-1.26, -1.898, -2.473, 1.223, 1.255, -0.002] # absolute
p_scan = [-1.463, -2.488, -1.191, 0.549, 1.447, -0.012] # absolute 
p_vac_approach = [-1.462, -2.568, -1.203, 0.64, 1.446, -0.012] #absolute
p_vac = [-1.396, -2.382, -1.161, -1.167, 1.588, 0.029] #absolute


def testingRoutine():
    rob.set_tcp('base')
    rob.moveHome()
    rob.moveJ(p_approach)
    rob.moveJ(p_scan)

    while True:
        rob.set_tcp('camera')
        camera.capture()
        rob.convert_to_base('rgb_xyz_capture.npz')
        img.load_npz('rgb_xyz_base.npz')
        img.process_image()
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
        camera.debugging = False 
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

        if False:
            pass

        testingRoutine()

    except KeyboardInterrupt:
        print("KeyboardInterrupt detected. Disconnecting...")
    except Exception as e:
        print('ssd')
        print(e)
    finally:
        rob.shutdown()
        camera.stop()

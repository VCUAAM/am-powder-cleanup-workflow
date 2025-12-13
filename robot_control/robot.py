import rtde_control, rtde_receive, rtde_io 
from math import *
import numpy as np
import socket, time
import tkinter as tk
from tkinter import messagebox # It feels redundant, but for whatever reason tk.messagebox doesn't work? 
from scipy.spatial.transform import Rotation as R

class UR5Robot:
    def __init__(self):
        self.ip = "192.168.1.102" # Default IP. Change if needed
        self.tcp = None
        self.testing = False
        self.debugging = False
        self.vel = 0.1 # velocity (m/s)
        self.acc = 0.25 # acceleration (m/s^2)
        self.blend = 0.002 # blending radius for path moves
        self.socket_check()
        self.home_pos = [-1.571, -0.838, -2.055, -1.819, 1.571, 0.0]
        self.base_height = False
        self.z = None
        
        # Extrinsic rotation and translation matrix from depth to color
        # Collected using `rs-enumerate-devices -c`
        self.d2c_R = np.asarray([[0.999954,    0.00596813, 0.00749689],    
                                [-0.00595368, 0.99998,   -0.00194843],
                                [-0.00750837, 0.0019037,  0.99997]])
        self.d2c_T = np.asarray([0.0147571573033929,  -2.052240051853e-05,  0.000543851230759174])

        while True:
            try:
                self.rtde_c = rtde_control.RTDEControlInterface(self.ip)
                self.rtde_r = rtde_receive.RTDEReceiveInterface(self.ip)
                self.rtde_io = rtde_io.RTDEIOInterface(self.ip)
            except:
                # Incredibly useful, but make sure that X11 forwarding is turned on, otherwise it won't work
                root = tk.Tk()
                root.withdraw()
                messagebox.showerror("Error", "Turn remote control mode on and then press ok")
                root.destroy()
                continue
            break

        print("Connected. Press Ctrl+C to stop.")
    
    # Testing mode, current programs will knock off the test bed lid otherwise
    def check_testing(self):
        if self.testing:
            root = tk.Tk()
            root.withdraw()
            messagebox.showwarning("Warning: Testing Mode", "Testing is enabled. Ensure that the top of the test bed is removed before pressing ok")
            root.destroy()

    # Ensuring that robot is available, otherwise it's hard to parse connection errors
    def socket_check(self):
        while True:
            try:
                socket.create_connection((self.ip,30004), timeout=1)
            except OSError:
                root = tk.Tk()
                root.withdraw()
                messagebox.showerror("Error", "Make sure robot is on and computer is connected to VCU_UR5e network")
                continue
            break
    
    # For asynchronous moves, this is required to keep the robot busy, otherwise it will not work
    def steady_check(self):
        while not self.rtde_c.isSteady():
            time.sleep(0.1)

    def get_path(self, xyz_path, ori):
        try:
            # Slow down if testing, and also override tcp position in case it's in the wrong spot
            if self.testing:
                #self.vel = 0.05
                pass

            ori = [0, pi, 0]

            # is it terrible? yeah. does it work? yeah. will I change it? nah
            def get_point(pos,ori,z=self.z,vel=None,acc=None,blend=None):
                return np.round(np.hstack([[pos[0],pos[1],z] if z else [i for i in pos],ori,vel if vel else self.vel,acc if acc else self.acc,blend if blend else self.blend]).astype(float),decimals=4)
            
            rob_path = [get_point(i,ori) for i in xyz_path]
            rob_path.insert(0,get_point(xyz_path[0],ori,z = self.base_height + 0.02))
            rob_path.append(get_point(xyz_path[-1],ori,z = self.base_height + 0.02))
            rob_path = np.asarray(rob_path)
            if self.testing:
                p = np.zeros_like(rob_path[0])
                p[2] = 0.03
                rob_path += p
            if self.debugging:
                # Prints out path in mm for debugging
                print([(int(1000*i),int(1000*j),int(1000*k)) for i,j,k in rob_path[:,:3]])

            return rob_path
        
        except Exception as e:
            print(e)
            quit()

    # Used as a shorthand to extract pose in joint positions to save
    def print_pos(self):
        print([round(i,3) for i in self.rtde_r.getActualQ()])

    # Function to move along a given path
    def move_path(self,npz):
        self.check_testing()
        # Very important to make sure that tcp is set correctly. It's usually redundant on either side, but better safe than collision
        self.set_tcp('base')
        data = np.load('data/' + npz)
        path_raw = data['path']
        ori = self.get_tcp()[3:]

        # Will execute each point along the path and release the thread when it's stopped moving
        path = self.get_path(path_raw,ori)

        if not self.testing:
            self.vac_on()
        self.rtde_c.moveL(path,asynchronous=True)
        self.steady_check()
        self.vac_off()

    # Helper function to quickly get the current TCP position, with terminal output for debugging
    def get_tcp(self):
        tcp = self.rtde_r.getActualTCPPose()
        if self.debugging:
            print(f'Current pose: {tcp}')
        return tcp
    
    def moveHome(self):
        self.set_tcp('base')
        current_pos = np.asarray(self.get_tcp())

        def mag(arr1,arr2):
            return np.sqrt((arr1[0] - arr2[0])**2 + (arr1[1] - arr2[1])**2)
        
        if np.allclose(np.asarray(current_pos), np.asarray(self.home_pos), atol=.1):
            return
        if mag(current_pos[:2],[-0.11468123625683228, -0.4880530357461711]) < mag(current_pos[:2],[-0.13259367960293592, -0.19587178394715143]):
            self.moveJ([-1.529, -1.648, -1.976, 0.459, 1.527, 0.02]) #vac approach
            self.moveJ([-1.362, -0.748, -2.411, -0.009, 1.35, 0.023]) # approach
            self.moveJ([-1.362, -0.748, -2.411, -0.009, 1.35, 0.023]) #scan0
        self.moveJ(self.home_pos)
            
    # Executes a MoveJ command to the given position. abs_j dictates if the pose is in joint positions (true) or a cartesian pose (false), and will run IK if needed
    def moveJ(self, pose, abs_j=True):
        try:
            if not abs_j:
                pose = self.rtde_c.getInverseKinematics(pose)
            self.rtde_c.moveJ(pose)
            self.steady_check()
        except Exception as e:
            print('ff',e)

    # Executes a MoveL command. abs_j dictates if the pose is in joint positions (true) or a cartesian pose (false), and will run IK if needed
    def moveL(self, pose, abs_j=True):
        if not abs_j:
            pose = self.rtde_c.getInverseKinematics(pose)
        self.rtde_c.moveL(pose)
        self.steady_check()

    # Initializes shutdown protocol
    def shutdown(self,reset=True):
        print("Shutting down RTDE connection.")
        self.vac_off()
        # Just some extra configuration. Probably unnecessary, but can't hurt
        if reset:
            self.set_tcp('base')
        try:
            if self.rtde_c.isConnected():
                self.rtde_c.stopScript()
                self.rtde_c.disconnect()
            if self.rtde_r.isConnected():
                self.rtde_r.disconnect()
            if self.rtde_io.isConnected():
                self.rtde_io.disconnect()
        except:
            pass
        finally:
            quit()

    # Helper function to set the TCP and store all the arrays 
    def set_tcp(self, frame='base'):
        tcp_base = [0, 0, 0, 0, 0, 0]
        tcp_cam = [-.0175, 0.03125, 0.00979, -1.57, 0, 0]
        tcp_vac = [0.06814, 0, 0.12692, 0, 0, 0]

        match frame:
            case 'base':
                tcp = tcp_base
            case 'camera':
                tcp = tcp_cam
            case 'vacuum':
                tcp = tcp_vac
            case _:
                print('Invalid TCP name')
                quit()
        self.tcp = tcp
        self.rtde_c.setTcp(tcp)
        print(f'Set tcp to {frame}')

    # Converts provided npz file from camera frame to robot base frame
    def convert_to_base(self,name):
        try:
            data = np.load('data/' + name)
            rgb = data["color"]
            xyz = data["xyz"]
            depth = data["depth"]

            self.set_tcp('camera')
            tcp = self.get_tcp()
            pos = tcp[:3]
            ori = tcp[3:]
            rpy = R.from_rotvec(ori)

            # Constructing rotation matrix from rpy (roll-pitch-yaw) of current TCP pose
            # See Modern Robotics by Kevin Lynch and Frank Park for mathematical basis
            rx,ry,rz = rpy.as_euler('xyz')
            cX,sX,cY,sY,cZ,sZ = cos(rx),sin(rx),cos(ry),sin(ry),cos(rz),sin(rz)
            rX = np.array([[1,0,0],[0,cX,-sX],[0,sX,cX]])
            rY = np.array([[cY,0,sY],[0,1,0],[-sY,0,cY]])
            rZ = np.array([[cZ,-sZ,0],[sZ,cZ,0],[0,0,1]])

            r_rot = np.matmul(rZ,np.matmul(rY,rX))

            self.set_tcp('vacuum')
            offset = np.asarray(self.rtde_c.getTCPOffset()[:3])

            # Adjusting ofset as needed, with extrinsic matrix of camera, safety offset, and a 50mm offset because I CANNOT figure out why the z-val is wrong
            # My testing showed that it is consistently wrong by 50mm, so this should work, but if you can figure it out be my guest
            offset = offset + self.d2c_T + [0,0,0.002] + [0,0,0.05]
            xyz_base = np.zeros_like(xyz)

            # Iterating through entire pointcloud
            for i in range(xyz.shape[0]):
                for j in range(xyz.shape[1]):
                    r_tot = np.matmul(self.d2c_R,r_rot)
                    p = np.matmul(r_tot,xyz[i,j])
                    p = p + pos + offset
                    xyz_base[i,j] = p

            np.savez_compressed('data/rgb_xyz_base.npz',color=rgb,xyz=xyz_base,depth=depth)
            print(f"Saved NPZ with robot base frame coordinates")
        except Exception as e:
            print(e)
 
    def vac_on(self):
        self.rtde_io.setConfigurableDigitalOut(0,1)

    def vac_off(self):
        self.rtde_io.setConfigurableDigitalOut(0,0)

# Throw whatever garbage you want in here for testing 
if __name__ == "__main__":
    try:
        rob = UR5Robot()
        rob.testing = False
        rob.debugging = False
        rob.base_height = .1801
        #rob.move_path('robot_path.npz')
        rob.moveHome()
        #rob.print_pos()
        #print(rob.get_tcp())
        #rob.moveJ([-1.462, -2.551, -1.202, 0.623, 1.446, -0.012])

    except KeyboardInterrupt:
        print("KeyboardInterrupt detected. Disconnecting...")
    finally:
        rob.shutdown()

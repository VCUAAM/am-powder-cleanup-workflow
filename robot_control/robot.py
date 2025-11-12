import rtde_control, rtde_receive
from math import *
import numpy as np
import socket
import tkinter as tk
from tkinter import messagebox
import time
from scipy.spatial.transform import Rotation as R
class UR5Robot:
    def __init__(self, ip="192.168.1.102"):
        self.ip = ip
        self.tcp = None
        self.testing = False
        self.debugging = False
        self.path_tolerance = 0.004 #m
        self.socket_check()

        # Connect loop (unchanged)
        while True:
            try:
                self.rtde_c = rtde_control.RTDEControlInterface(self.ip)
                self.rtde_r = rtde_receive.RTDEReceiveInterface(self.ip)
            except:
                root = tk.Tk()
                root.withdraw()
                messagebox.showerror("Error", "Turn remote control mode on and then press ok")
                root.destroy()
                continue
            break
        self.d2c_R = np.asarray([[0.999954,    0.00596813, 0.00749689],    
                                [-0.00595368, 0.99998,   -0.00194843],
                                [-0.00750837, 0.0019037,  0.99997]])

        self.d2c_T = np.asarray([0.0147571573033929,  -2.052240051853e-05,  0.000543851230759174])

        print("Connected. Press Ctrl+C to stop.")
    
    def check_testing(self):
        if self.testing:
            root = tk.Tk()
            root.withdraw()
            messagebox.showwarning("Warning: Testing Mode", "Testing is enabled. Ensure that the top of the test bed is removed before pressing ok")
            root.destroy()

    def socket_check(self):
        while True:
            try:
                socket.create_connection(("192.168.1.102",30004), timeout=1)
            except OSError:
                root = tk.Tk()
                root.withdraw()
                messagebox.showerror("Error", "Make sure robot is on and computer is connected to VCU_UR5e network")
                continue
            
            break

    def steady_check(self):
        while not self.rtde_c.isSteady():
            time.sleep(0.1)

    # ---------- Converted Functions (same functionality) ----------
    def get_path(self, cart_path, tcp, vel=0.25, acc=0.1, blend=0.002):
        if self.testing:
            vel = 0.05
            tcp = [0, pi, 0]

        path = []
        for p in cart_path:
            # Conditionally offset z if testing
            z_val = p[2] + 0.055 if self.testing else p[2]

            point = [p[0], p[1], z_val, tcp[0], tcp[1], tcp[2], vel, acc, blend]
            point = [round(float(i), 4) for i in point]
            path.append(point)

        if self.testing:
            print(print([(int(1000*i),int(1000*j),int(1000*k)) for (i,j,k) in cart_path]))

        return path

    def move_path(self,npz):
        self.set_tcp('base')
        data = np.load(npz)
        path_raw = data['path']
        tcp = self.get_tcp()[3:]
        path = self.get_path(path_raw,tcp)
        
        self.rtde_c.moveL(path,asynchronous=True)
        while True:
            time.sleep(.1)
            if self.rtde_c.isSteady():
                self.rtde_c.stopL()
                break


    def set_tcp(self, frame='base'):
        tcp_base = [0, 0, 0, 0, 0, 0]
        tcp_cam = [-.0175, -0.03125, 0.00979, -1.57, 0, 0]
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

    def convert_to_base(self,save_path):
        data = np.load(save_path + 'rgb_xyz_capture.npz')
        xyz = data["xyz"]
        rgb = data['color']
        self.set_tcp('camera')
        tcp = self.get_tcp()
        pos = tcp[:3]
        rot = tcp[3:]
        rpy = R.from_rotvec(rot)

        rx,ry,rz = rpy.as_euler('xyz')
        cX,sX,cY,sY,cZ,sZ = cos(rx),sin(rx),cos(ry),sin(ry),cos(rz),sin(rz)
        rX = np.array([[1,0,0],[0,cX,-sX],[0,sX,cX]])
        rY = np.array([[cY,0,sY],[0,1,0],[-sY,0,cY]])
        rZ = np.array([[cZ,-sZ,0],[sZ,cZ,0],[0,0,1]])

        r_rot = np.matmul(rZ,np.matmul(rY,rX))

        self.set_tcp('vacuum')
        offset = np.asarray(self.rtde_c.getTCPOffset()[:3])
        offset = offset + self.d2c_T
        xyz_base = np.zeros_like(xyz)

        for i in range(xyz.shape[0]):
            for j in range(xyz.shape[1]):
                #r_tot = np.matmul(r_rot,self.d2c_R)
                r_tot = np.matmul(self.d2c_R,r_rot)
                p = np.matmul(r_tot,xyz[i,j])
                if p[0] != 0:
                    p = p + pos + offset
                xyz_base[i,j] = p

        np.savez_compressed(save_path + 'rgb_xyz_base.npz',color=rgb,xyz=xyz_base)
        print(f"Saved NPZ with robot {'testing' if self.testing else 'base frame'} coordinates")

    def get_tcp(self):
        tcp = self.rtde_r.getActualTCPPose()
        if self.debugging:
            print(f'Current pose: {tcp}')
        return tcp
    
    def moveJ(self, pos, abs_j=True):
        if not abs_j:
            pos = self.rtde_c.getInverseKinematics(pos)
        self.rtde_c.moveJ(pos)
        self.steady_check()

    def moveL(self, pos, abs_j=True):
        if not abs_j:
            pos = self.rtde_c.getInverseKinematics(pos)
        self.rtde_c.moveL(pos)
        self.steady_check()

    def shutdown(self,reset=True):
        print("Shutting down RTDE connection.")
        if reset:
            self.set_tcp('base')
        try:
            if self.rtde_c.isConnected():
                self.rtde_c.stopScript()
                self.rtde_c.disconnect()
            if self.rtde_r.isConnected():
                self.rtde_r.disconnect()
        except:
            pass
        print("Connection closed.")

if __name__ == "__main__":
    try:
        rob = UR5Robot()
        rob.testing = True
        rob.debugging = True
        time.sleep(2)
        rob.convert_to_base("ml_vision/" + 'data/')
        #rob.move_path('path_planning/data/robot_path.npz')
        #test_pos = [[0.0612, -0.700, 0.2304,0,pi,0, 0.05, 0.5, 0.05], [-0.0683, -0.700, 0.2298, 0,pi,0, 0.05, 0.5, 0.05]]
        #rob.rtde_c.moveL(test_pos)
        #rob.get_tcp()
        #rob.moveL(test_pos,abs_j=False)

        #rob.convert_to_base('ml_vision/data/')
    except KeyboardInterrupt:
        print("KeyboardInterrupt detected. Disconnecting...")
    finally:
        rob.shutdown()

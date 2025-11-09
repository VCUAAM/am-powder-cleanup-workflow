import rtde_control, rtde_receive
import pyrealsense2
import robot_control.robot as hf
import tkinter as tk
from math import *
from tkinter import messagebox, simpledialog
import numpy as np
import time

class UR5Robot:
    def __init__(self, ip="192.168.1.102"):
        self.ip = ip

        hf.socket_check()

        # Connect loop (unchanged)
        while True:
            try:
                self.rtde_c = rtde_control.RTDEControlInterface(self.ip)
                self.rtde_r = rtde_receive.RTDEReceiveInterface(self.ip)
            except:
                root = tk.Tk()
                root.withdraw()
                messagebox.showerror("Error", "Turn remote control mode on and then press ok")
                continue
            break

        print("Connected. Press Ctrl+C to stop.")
    
    def steady_check(self):
        while not self.rtde_c.isSteady():
            time.sleep(0.1)

    # ---------- Converted Functions (same functionality) ----------
    def get_tcp_path(self, cart_path, tcp, vel=0.25, acc=0.5, blend=0.02):
        path = []
        for p in cart_path:
            point = [p[0], p[1], p[2], tcp[0], tcp[1], tcp[2], vel, acc, blend]
            path.append(point)
        return path

    def set_tcp(self, frame='base'):
        tcp_base = [0, 0, 0, 0, 0, 0]
        tcp_cam = [-.0175, -0.01399, 0.03125, -1.57, 0, 0]
        tcp_vac = [0.06814, 0, 0.17782, 0, 0, 0]

        match frame:
            case 'base':
                tcp = tcp_base
            case 'camera':
                tcp = tcp_cam
            case 'vacuum':
                tcp = tcp_vac
            case _:
                print('Done fucked up yo')
                quit()

        self.rtde_c.setTcp(tcp)
        print(f'Set tcp to {frame}')

    def scan(self):
        self.convert_to_base()
        pass

    def convert_to_base(self):
        pass
    
    def tcp(self):
        return self.rtde_r.getActualTCPPose()
    
    def moveJ(self, pos):
        pos_j = self.rtde_c.getInverseKinematics(pos)
        self.rtde_c.moveJ(pos_j)
        self.steady_check()

    def shutdown(self,reset=True):
        print("Shutting down RTDE connection.")
        if reset:
            rob.set_tcp('base')
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
        p_home = [-0.133, -0.346, 0.31, 0.001, -3.127, 0.001]
        p_scan_0 = [-0.044, -0.397, 0.296, 0.017, -2.215, 2.214]
        p_scan = [-0.062, -0.784, 0.323, -0.012, -2.176, 2.206]

        rob = UR5Robot()
        # path = robot.scan()
        rob.set_tcp('base')
        rob.moveJ(p_home)
        
        print(rob.rtde_c.getTCPOffset())
        rob.set_tcp('camera')
        rob.moveJ(p_home)
        tcp = rob.tcp()[3:]
        rob.moveJ(p_scan_0)
        rob.moveJ(p_scan)

    
    except KeyboardInterrupt:
        print("KeyboardInterrupt detected. Disconnecting...")
    finally:
        rob.shutdown()
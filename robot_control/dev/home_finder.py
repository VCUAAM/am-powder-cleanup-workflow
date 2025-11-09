from robot_control.robot import UR5Robot

if __name__ == "__main__":
    try:
        rob = UR5Robot()
        print([round(i,3) for i in rob.rtde_r.getActualQ()])

    except KeyboardInterrupt:
        print("KeyboardInterrupt detected. Disconnecting...")
    finally:
        rob.shutdown()
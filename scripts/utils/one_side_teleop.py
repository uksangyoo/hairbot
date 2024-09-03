import math
import time

from xarm.wrapper import XArmAPI


def prep_robots(master_bot, puppet_bot):
    master_bot.motion_enable(enable=True)
    master_bot.set_mode(0)
    master_bot.set_state(0)
    master_bot.reset(wait=True)

    puppet_bot.motion_enable(enable=True)
    puppet_bot.set_mode(0)
    puppet_bot.set_state(0)
    puppet_bot.reset(wait=True)

    master_bot.move_gohome()
    puppet_bot.move_gohome()

    # 2: joint teaching mode
    master_bot.set_mode(2)
    master_bot.set_state(0)

    # set mode: joint online trajectory planning mode
    # the running command will be interrupted when the next command is received
    puppet_bot.set_mode(6)
    puppet_bot.set_state(0)


def teleop():
    DT = 0.01
    speed = 4.0  # math.radians(220)
    mvacc = 20.0  # math.radians(1145)
    # max joint speeds

    # DT = 0.02
    # speed = math.radians(210)
    # mvacc = math.radians(800)

    master_ip = "192.168.2.157"
    puppet_ip = "192.168.1.168"

    # --- Teleoperation

    master_bot = XArmAPI(port=master_ip, report_type="real", is_radian=True)
    puppet_bot = XArmAPI(port=puppet_ip, report_type="real", is_radian=True)

    prep_robots(master_bot, puppet_bot)

    print("joint_speed_limit: ", puppet_bot.joint_speed_limit)
    print("joint_acc_limit: ", puppet_bot.joint_acc_limit)

    # # only available when report_type="rich"
    # print("collision_sensitivity: ", puppet_bot.collision_sensitivity)
    # print("teach_sensitivity: ", master_bot.teach_sensitivity)

    # set initial speed and mvacc
    zero_joints = [0.0] * 6
    code = master_bot.set_servo_angle(angle=zero_joints, speed=speed, mvacc=mvacc, is_radian=True, wait=False)
    code = puppet_bot.set_servo_angle(angle=zero_joints, speed=speed, mvacc=mvacc, is_radian=True, wait=False)

    try:
        while True:
            # sync joint positions
            _, master_state_joints = master_bot.get_servo_angle(is_radian=True)
            code = puppet_bot.set_servo_angle(angle=master_state_joints, speed=speed, mvacc=mvacc, is_radian=True, wait=False)
            # sleep DT
            time.sleep(DT)
    except (KeyboardInterrupt, Exception):
        pass

    # set_mode: position mode
    master_bot.set_mode(0)
    master_bot.set_state(0)
    master_bot.reset(wait=True)
    master_bot.disconnect()

    # set_mode: position mode
    puppet_bot.set_mode(0)
    puppet_bot.set_state(0)
    puppet_bot.reset(wait=True)
    puppet_bot.disconnect()


if __name__ == "__main__":
    teleop()

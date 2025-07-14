import time
from dynamixel_sdk import *  # Uses Dynamixel SDK library
import numpy as np 
TORQUE_PORT = 64
class MOE :
    def __init__(self, port_name='/dev/ttyUSB0', baudrate=57600, protocol_version=2.0, servo_ids=[0,1,2,3], override_initial_positions=[-4.74609375, -0.8349609375, 2.900390625, 7.998046875]):
        self.portHandler = PortHandler(port_name)
        self.packetHandler = PacketHandler(protocol_version)
        self.groupSyncWrite = GroupSyncWrite(self.portHandler, self.packetHandler, 116, 4)

        if self.portHandler.openPort():
            print("Succeeded to open the port")
        else:
            print("Failed to open the port")
            quit()

        if self.portHandler.setBaudRate(baudrate):
            print("Succeeded to change the baudrate")
        else:
            print("Failed to change the baudrate")
            quit()

        self.DXL_IDs = servo_ids # Dynamixel ID list
        self.TORQUE_ENABLE = 1
        self.TORQUE_DISABLE = 0
        self.MIN_POSITION = int((-20 / 180.0) * 4096) + 2048
        self.MAX_POSITION = int((20 / 180.0) * 4096) + 2048

        for dxl_id in self.DXL_IDs:
            self.packetHandler.write1ByteTxRx(self.portHandler, dxl_id, TORQUE_PORT, self.TORQUE_ENABLE)
        # Store the initial positions
        self.initial_positions = self.get_current_positions()
        print(f"Initial positions: {self.initial_positions}")
        if override_initial_positions is not None:
            self.initial_positions = override_initial_positions
        self.motion_timeout = 2.0

    def restart_torque(self):
        for dxl_id in self.DXL_IDs:
            self.packetHandler.write1ByteTxRx(self.portHandler, dxl_id, TORQUE_PORT, self.TORQUE_DISABLE)
        for dxl_id in self.DXL_IDs:
            self.packetHandler.write1ByteTxRx(self.portHandler, dxl_id, TORQUE_PORT, self.TORQUE_ENABLE)

    def get_current_positions(self):
        positions = []
        for dxl_id in self.DXL_IDs:
            position, _, _ = self.packetHandler.read4ByteTxRx(self.portHandler, dxl_id, 132)
            # Convert position from Dynamixel units to degrees
            position_in_degrees = (position - 2048) * 180.0 / 4096
            positions.append(position_in_degrees)
        return positions
    
    def position_to_angle(self, position):
        return position * 180. / 2048.
    
    def angle_to_position(self, angle):
        # angle = p * 180 / 2048
        return int(np.round(angle * 2048. / 180., 0))

    def move_servo_to_absolute_position(self, servo_id, degrees):
        position = self.angle_to_position(degrees)
        dxl_comm_result, dxl_error = self.packetHandler.write4ByteTxRx(self.portHandler, servo_id, 116, position)
        if dxl_comm_result != COMM_SUCCESS:
            print(f"{self.packetHandler.getTxRxResult(dxl_comm_result)}")
        elif dxl_error != 0:
            print(f"{self.packetHandler.getRxPacketError(dxl_error)}")
        else:
            print(f"Goal position set to {position} degrees")

        # Wait for the movement to complete
        while True:
            # Read present position
            dxl_present_position, dxl_comm_result, dxl_error = self.packetHandler.read4ByteTxRx(self.portHandler, servo_id, 132)
            if dxl_comm_result != COMM_SUCCESS:
                print(f"{self.packetHandler.getTxRxResult(dxl_comm_result)}")
            elif dxl_error != 0:
                print(f"{self.packetHandler.getRxPacketError(dxl_error)}")

            print(f"[ID:{servo_id}] Present Position : {dxl_present_position}")

            if abs(dxl_present_position - position) <= 10:
                break

            time.sleep(0.1)

    def get_servo_currents(self):
        currents = []
        for dxl_id in self.DXL_IDs:
            current, dxl_comm_result, dxl_error = self.packetHandler.read2ByteTxRx(self.portHandler, dxl_id, 126)
            if dxl_comm_result != COMM_SUCCESS:
                print(f"{self.packetHandler.getTxRxResult(dxl_comm_result)}")
            elif dxl_error != 0:
                print(f"{self.packetHandler.getRxPacketError(dxl_error)}")
            else:
                if current >60000:
                    current = current - 65535
                currents.append(current)
        return currents

    def get_current_absolute_positions(self):
        abs_positions = []
        # Read present position
        for dxl_id in self.DXL_IDs:
            dxl_present_position, dxl_comm_result, dxl_error = self.packetHandler.read4ByteTxRx(self.portHandler, dxl_id, 132)
            # convert to signed int 
            if dxl_present_position > 2147483647:
                dxl_present_position = dxl_present_position - 4294967296
            

            # Convert to degrees
            absolute_angle = self.position_to_angle(dxl_present_position)
            abs_positions.append(absolute_angle)
            print(f"Servo {dxl_id} has position {absolute_angle} degrees, converted back to {self.angle_to_position(absolute_angle)}, ground truth is {dxl_present_position}")
        return abs_positions
    

    def move_servos_to_positions(self, positions):
        """
        Move each servo to the corresponding position provided in the positions list.
        :param positions: List of 6 positions in degrees, each between -20 and 20.
        """
        assert len(positions) == len(self.DXL_IDs), "Positions must be a list of 6 values."

        goal_positions = [
            int((position / 180.0) * 4096) + 2048 for position in positions
        ]
        
        for i, dxl_id in enumerate(self.DXL_IDs):
            param_goal_position = [DXL_LOBYTE(DXL_LOWORD(goal_positions[i])),
                                   DXL_HIBYTE(DXL_LOWORD(goal_positions[i])),
                                   DXL_LOBYTE(DXL_HIWORD(goal_positions[i])),
                                   DXL_HIBYTE(DXL_HIWORD(goal_positions[i]))]
            self.groupSyncWrite.addParam(dxl_id, param_goal_position)

        self.groupSyncWrite.txPacket()
        self.groupSyncWrite.clearParam()

    def smooth_move_to_positions(self, target_positions, steps=100, delay=0.00000001, threshold=-1):
        """
        Smoothly move the servos to the desired target positions.
        :param target_positions: List of 6 positions in degrees, each between -20 and 20.
        :param steps: Number of intermediate steps for the smooth movement.
        :param delay: Time to wait between steps in seconds.
        """
        current_positions = self.get_current_positions()

        for step in range(steps + 1):
            intermediate_positions = [
                current + (target - current) * (step / steps)
                for current, target in zip(current_positions, target_positions)
            ]
            self.move_servos_to_positions(intermediate_positions)
            if delay > 0:
                time.sleep(delay)

        # Ensure the servos reach the final target positions
        self.move_servos_to_positions(target_positions)
        start_time = time.time()
        if threshold >= 0:
            while True:
                current_positions = self.get_current_positions()
                if all(abs(current - target) < threshold for current, target in zip(current_positions, target_positions)):
                    break
                if time.time() - start_time > self.motion_timeout:
                    print("Motion timeout reached.")
                    break
                time.sleep(0.1)
    def close_fingers(self):
        target_positions = [-20, 70, 75, -20]
        self.smooth_move_to_positions(target_positions, delay=0.1, threshold=20)

    def return_to_initial_positions(self):
        """
        Return the servos to their initial positions when the RobotHand was initialized.
        """
        self.smooth_move_to_positions(self.initial_positions, delay=-1, threshold=20)

    def disable_torque(self):
        for dxl_id in self.DXL_IDs:
            self.packetHandler.write1ByteTxRx(self.portHandler, dxl_id, TORQUE_PORT, self.TORQUE_DISABLE)
        self.portHandler.closePort()

# Example usage
if __name__ == "__main__":
    hand = MOE()
    hand.return_to_initial_positions()
    try:
        target_positions = [50, 50, 50, 50]
        hand.smooth_move_to_positions(target_positions, delay=0.1, threshold=20)
        time.sleep(3)
        hand.return_to_initial_positions()
        target_positions = [-50, 50, 50, -50]
        hand.smooth_move_to_positions(target_positions, delay=0.1, threshold=20)
        time.sleep(3)
        hand.return_to_initial_positions() 
        target_positions = [50, -50, -50, 50]
        hand.smooth_move_to_positions(target_positions, delay=0.1, threshold=20)
        time.sleep(3)
        hand.return_to_initial_positions()       
        target_positions = [-50, -50, -50, -50]
        hand.smooth_move_to_positions(target_positions, delay=0.1, threshold=20)
        time.sleep(3)
        hand.return_to_initial_positions()

    except KeyboardInterrupt:
        print("Returning to initial positions...")
        hand.return_to_initial_positions()
        hand.disable_torque()
        print("Servos returned to initial positions and torque disabled.")

import struct
import socket

class FTSensor:
    def __init__(self, ip_addr='192.168.0.100', port=2001):
        self.ip_addr = ip_addr
        self.port = port
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.settimeout(2.0)
        self.connect()
        
    def connect(self):
        """Establish a connection to the sensor."""
        try:
            self.s.connect((self.ip_addr, self.port))
            # Start data transmission
            self.send_command('03' + '07' + '01')  # CMD_TYPE_SENSOR_TRANSMIT + SENSOR_TRANSMIT_TYPE_START
            self.recv_msg()
        except Exception as e:
            print(f"Failed to connect: {e}")
    
    def send_command(self, hex_command):
        """Send a command to the sensor."""
        send_data = bytearray.fromhex(hex_command)
        self.s.send(send_data)

    def recv_msg(self):
        """Receive a message from the sensor."""
        recv_data = bytearray(self.s.recv(2))

        while len(recv_data) < recv_data[0]:
            recv_data += bytearray(self.s.recv(recv_data[0] - len(recv_data)))

        self.print_msg(recv_data)
        return recv_data

    def print_msg(self, msg):
        """Print a received message in a readable format."""
        # print(f"Msg len: {msg[0]} Msg type: {msg[1]}")
        data_str = "DATA: " + " ".join(str(msg[i + 2]) for i in range(msg[0] - 2))
        # print(data_str)
    
    def get_ft(self):
        """Get the force and torque values from the sensor."""
        recv_data = self.recv_msg()
        Fx = struct.unpack('!d', recv_data[2:10])[0]
        Fy = struct.unpack('!d', recv_data[10:18])[0]
        Fz = struct.unpack('!d', recv_data[18:26])[0]
        Tx = struct.unpack('!d', recv_data[26:34])[0]
        Ty = struct.unpack('!d', recv_data[34:42])[0]
        Tz = struct.unpack('!d', recv_data[42:50])[0]
        return Fx, Fy, Fz, Tx, Ty, Tz

    def stop(self):
        """Stop data transmission and close the connection."""
        self.send_command('03' + '07' + '00')  # CMD_TYPE_SENSOR_TRANSMIT + SENSOR_TRANSMIT_TYPE_STOP
        while True:
            recv_data = self.recv_msg()
            if recv_data[0] == 3 and recv_data[1] == int('07', 16):  # CMD_TYPE_SENSOR_TRANSMIT
                break
        self.s.close()

if __name__ == "__main__":
    sensor = FTSensor()
    try:
        for _ in range(10):  # Example: Get 10 readings
            force, torque = sensor.get_ft()[:3], sensor.get_ft()[3:]
            print(f"Force: {force}, Torque: {torque}")
    finally:
        sensor.stop()

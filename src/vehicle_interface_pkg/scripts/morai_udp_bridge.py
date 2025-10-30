#!/usr/bin/env python3
"""MORAI UDP <-> ROS 브릿지."""

import math
import socket
import struct
from typing import Optional

import rospy
from rospy.timer import TimerEvent
from morai_msgs.msg import CtrlCmd, EgoVehicleStatus
from std_msgs.msg import Header

INFO_PREFIX = b"#MoraiInfo$"
CMD_PREFIX = b"#MoraiCtrlCmd$"
TAIL_BYTES = b"\r\n"
INFO_AUX_LEN = 12
CMD_AUX_LEN = 12
INFO_PAYLOAD_LEN = 152  # 50 bytes + 102 bytes
CMD_PAYLOAD_LEN = 23  # 3 bytes + 20 bytes

KPH_TO_MPS = 1.0 / 3.6
DEG_TO_RAD = math.pi / 180.0
RAD_TO_DEG = 180.0 / math.pi


class MoraiUDPBridge:
    """EgoVehicleStatus 수신 및 CtrlCmd 송신을 담당."""

    def __init__(self) -> None:
        self.recv_ip = rospy.get_param("~status_ip", "0.0.0.0")
        self.recv_port = rospy.get_param("~status_port", 9091)
        self.send_ip = rospy.get_param("~cmd_ip", "127.0.0.1")
        self.send_port = rospy.get_param("~cmd_port", 9092)

        self.ctrl_mode = int(rospy.get_param("~ctrl_mode", 2))  # AutoMode
        self.gear_cmd = int(rospy.get_param("~gear_cmd", 4))  # Drive
        self.long_cmd_type = int(rospy.get_param("~long_cmd_type", 1))
        self.max_steer_deg = float(rospy.get_param("~max_steer_deg", 36.25))

        self.recv_sock = self._create_socket(self.recv_ip, self.recv_port, bind=True)
        self.send_sock = self._create_socket(self.send_ip, self.send_port, bind=False)

        self.status_pub = rospy.Publisher("/morai/status", EgoVehicleStatus, queue_size=1)
        self.cmd_sub = rospy.Subscriber("/ctrl_cmd", CtrlCmd, self.cmd_cb, queue_size=1)
        rospy.Timer(rospy.Duration(0.01), self.poll_udp)

        rospy.loginfo(
            "[interface] UDP bridge listening %s:%d, sending -> %s:%d",
            self.recv_ip,
            self.recv_port,
            self.send_ip,
            self.send_port,
        )

    @staticmethod
    def _create_socket(ip: str, port: int, bind: bool) -> socket.socket:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        if bind:
            sock.bind((ip, port))
        return sock

    def poll_udp(self, _: TimerEvent) -> None:
        """UDP 수신 데이터를 ROS 메시지로 변환."""
        self.recv_sock.settimeout(0.0)
        while not rospy.is_shutdown():
            try:
                data, _ = self.recv_sock.recvfrom(512)
            except BlockingIOError:
                return

            status = self.parse_status(data)
            if status is not None:
                self.status_pub.publish(status)

    def parse_status(self, data: bytes) -> Optional[EgoVehicleStatus]:
        """UDP 패킷 -> EgoVehicleStatus 변환."""
        if len(data) < len(INFO_PREFIX) + INFO_AUX_LEN + INFO_PAYLOAD_LEN + len(TAIL_BYTES):
            rospy.logdebug("[interface] status packet too short len=%d", len(data))
            return None
        if not data.startswith(INFO_PREFIX):
            rospy.logdebug("[interface] invalid status header")
            return None
        if not data.endswith(TAIL_BYTES):
            rospy.logdebug("[interface] invalid status tail")
            return None

        offset = len(INFO_PREFIX)
        data_len = struct.unpack_from("<I", data, offset)[0]
        offset += 4
        offset += INFO_AUX_LEN

        payload = data[offset : offset + data_len]
        if len(payload) < INFO_PAYLOAD_LEN:
            rospy.logdebug("[interface] payload length mismatch: %d", len(payload))
            return None

        idx = 0
        ts_sec, ts_nsec = struct.unpack_from("<II", payload, idx)
        idx += 8

        ctrl_mode = payload[idx]
        idx += 1
        gear = payload[idx]
        idx += 1

        signed_velocity_kph = struct.unpack_from("<f", payload, idx)[0]
        idx += 4
        map_id = struct.unpack_from("<i", payload, idx)[0]
        idx += 4

        accel_cmd, brake_cmd = struct.unpack_from("<ff", payload, idx)
        idx += 8

        size_x, size_y, size_z = struct.unpack_from("<fff", payload, idx)
        idx += 12
        overhang, wheel_base, rear_overhang = struct.unpack_from("<fff", payload, idx)
        idx += 12

        pos_x, pos_y, pos_z = struct.unpack_from("<fff", payload, idx)
        idx += 12
        roll_deg, pitch_deg, yaw_deg = struct.unpack_from("<fff", payload, idx)
        idx += 12

        vel_x_kph, vel_y_kph, vel_z_kph = struct.unpack_from("<fff", payload, idx)
        idx += 12
        ang_vel_x_deg, ang_vel_y_deg, ang_vel_z_deg = struct.unpack_from("<fff", payload, idx)
        idx += 12
        accel_x, accel_y, accel_z = struct.unpack_from("<fff", payload, idx)
        idx += 12

        steer_deg = struct.unpack_from("<f", payload, idx)[0]
        idx += 4

        link_id_raw = payload[idx : idx + 38]
        link_id = link_id_raw.split(b"\x00", 1)[0].decode("utf-8", errors="ignore")

        status = EgoVehicleStatus()
        stamp = rospy.Time.from_sec(ts_sec) + rospy.Duration(nsecs=ts_nsec)
        status.header = Header(frame_id="map", stamp=stamp)
        status.unique_id = map_id

        status.position.x = pos_x
        status.position.y = pos_y
        status.position.z = pos_z

        status.acceleration.x = accel_x
        status.acceleration.y = accel_y
        status.acceleration.z = accel_z

        status.velocity.x = signed_velocity_kph * KPH_TO_MPS
        status.velocity.y = 0.0
        status.velocity.z = 0.0

        status.heading = math.radians(yaw_deg)
        status.accel = float(accel_cmd)
        status.brake = float(brake_cmd)
        status.wheel_angle = math.radians(steer_deg)

        rospy.logdebug(
            "[interface] status ctrl_mode=%d gear=%d vel=%.2f kph link=%s",
            ctrl_mode,
            gear,
            signed_velocity_kph,
            link_id,
        )
        return status

    def cmd_cb(self, msg: CtrlCmd) -> None:
        """ROS CtrlCmd를 UDP 패킷으로 직렬화."""
        packet = self.serialize_cmd(msg)
        self.send_sock.sendto(packet, (self.send_ip, self.send_port))

    def serialize_cmd(self, msg: CtrlCmd) -> bytes:
        """CtrlCmd -> UDP 바이너리."""
        long_cmd = self.long_cmd_type if msg.longlCmdType == 0 else int(msg.longlCmdType)

        velocity_kph = float(msg.velocity) * 3.6
        acceleration_ms2 = float(msg.acceleration)

        accel_cmd = float(max(0.0, min(1.0, msg.accel)))
        brake_cmd = float(max(0.0, min(1.0, msg.brake)))

        steer_ratio = 0.0
        if self.max_steer_deg > 1e-6:
            steer_ratio = math.degrees(msg.steering) / self.max_steer_deg
        steer_ratio = float(max(-1.0, min(1.0, steer_ratio)))

        payload = struct.pack(
            "<BBBfffff",
            int(self.ctrl_mode),
            int(self.gear_cmd),
            long_cmd,
            velocity_kph,
            acceleration_ms2,
            accel_cmd,
            brake_cmd,
            steer_ratio,
        )

        packet = bytearray()
        packet.extend(CMD_PREFIX)
        packet.extend(struct.pack("<I", len(payload)))
        packet.extend(b"\x00" * CMD_AUX_LEN)
        packet.extend(payload)
        packet.extend(TAIL_BYTES)

        rospy.logdebug(
            "[interface] send cmd accel=%.2f brake=%.2f steer_ratio=%.2f vel=%.2f km/h",
            accel_cmd,
            brake_cmd,
            steer_ratio,
            velocity_kph,
        )
        return bytes(packet)

    def close(self) -> None:
        self.recv_sock.close()
        self.send_sock.close()


def main() -> None:
    rospy.init_node("morai_udp_bridge")
    bridge = MoraiUDPBridge()
    rospy.on_shutdown(bridge.close)
    rospy.spin()


if __name__ == "__main__":
    main()

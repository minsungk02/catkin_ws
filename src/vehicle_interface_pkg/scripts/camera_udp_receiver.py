#!/usr/bin/env python3
"""Receive MORAI camera frames over UDP and publish as CompressedImage."""

from __future__ import annotations

import socket
from typing import Optional, Tuple

import rospy
from sensor_msgs.msg import CompressedImage


class MoraiCameraUDPReceiver:
    """Listen for UDP camera packets and republish them as ROS messages."""

    def __init__(self) -> None:
        self.bind_ip = rospy.get_param("~ip", "0.0.0.0")
        self.bind_port = int(rospy.get_param("~port", 15006))
        self.topic = rospy.get_param("~topic", "/camera/image/compressed")
        self.frame_id = rospy.get_param("~frame_id", "morai_camera")
        self.timeout = float(rospy.get_param("~timeout", 0.5))

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(self.timeout)
        self.sock.bind((self.bind_ip, self.bind_port))

        self.pub = rospy.Publisher(self.topic, CompressedImage, queue_size=1)

        rospy.loginfo(
            "[camera_udp] listening on %s:%d -> publishing %s",
            self.bind_ip,
            self.bind_port,
            self.topic,
        )

        self.run()

    def run(self) -> None:
        while not rospy.is_shutdown():
            try:
                data, addr = self.sock.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError as exc:  # socket closed
                rospy.logwarn("[camera_udp] socket error: %s", exc)
                break

            image_bytes = self.extract_jpeg(data)
            if image_bytes is None:
                rospy.logwarn_throttle(5.0, "[camera_udp] failed to detect JPEG markers.")
                continue

            msg = CompressedImage()
            msg.header.stamp = rospy.Time.now()
            msg.header.frame_id = self.frame_id
            msg.format = "jpeg"
            msg.data = image_bytes

            self.pub.publish(msg)
            rospy.logdebug(
                "[camera_udp] received %d bytes from %s:%d", len(image_bytes), addr[0], addr[1]
            )

    @staticmethod
    def extract_jpeg(payload: bytes) -> Optional[bytes]:
        """Return JPEG slice if markers exist; otherwise None."""
        start = payload.find(b"\xff\xd8")
        end = payload.rfind(b"\xff\xd9")
        if start == -1 or end == -1:
            return None
        end += 2
        return payload[start:end]


def main() -> None:
    rospy.init_node("morai_camera_udp_receiver")
    try:
        MoraiCameraUDPReceiver()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()

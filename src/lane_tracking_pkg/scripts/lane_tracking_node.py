#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""차선 인지 및 스티어링 산출 노드."""

import os
import sys
from collections import deque
from typing import Deque

import rospy
import numpy as np
import cv2
from std_msgs.msg import Float32
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge

# MORAI 제어 메시지 (선택적 사용)
try:
    from morai_msgs.msg import CtrlCmd  # type: ignore
    HAVE_MORAI = True
except ImportError:  # pragma: no cover - 패키지 미존재 환경 대비
    HAVE_MORAI = False

# vision 모듈 경로 추가
pkg_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if pkg_path not in sys.path:
    sys.path.append(pkg_path)

from vision.preprocess import preprocess_image  # type: ignore
from vision.detector import detect_lane_center  # type: ignore


class LaneTrackingNode:
    """주요 라인트래킹 로직 클래스."""

    def __init__(self) -> None:
        self.bridge = CvBridge()

        # 파라미터 로드
        self.camera_topic = rospy.get_param("~camera_topic", "/image_jpeg/compressed")
        self.kp = rospy.get_param("~kp", 0.6)
        self.debug = rospy.get_param("~debug", True)
        self.roi_y_ratio = rospy.get_param("~roi_y_ratio", 0.55)
        self.canny_low = rospy.get_param("~canny_low", 50)
        self.canny_high = rospy.get_param("~canny_high", 150)
        self.gaussian_kernel = rospy.get_param("~gaussian_kernel", 5)
        self.avg_window = rospy.get_param("~avg_window", 5)
        self.use_ctrl_cmd = rospy.get_param("~use_ctrl_cmd", False)

        # 퍼블리셔
        self.offset_pub = rospy.Publisher("/lane/center_offset", Float32, queue_size=1)
        self.steer_pub = rospy.Publisher("/lane/steering_angle", Float32, queue_size=1)
        self.overlay_pub = rospy.Publisher("/lane_overlay", Image, queue_size=1)

        if self.use_ctrl_cmd and HAVE_MORAI:
            self.ctrl_pub = rospy.Publisher("/ctrl_cmd", CtrlCmd, queue_size=1)
        else:
            self.ctrl_pub = None
            if self.use_ctrl_cmd and not HAVE_MORAI:
                rospy.logwarn("morai_msgs 패키지를 찾을 수 없어 CtrlCmd 퍼블리시를 비활성화합니다.")

        self.offset_buffer: Deque[float] = deque(maxlen=self.avg_window)

        # 메시지 타입 자동 감지를 위한 AnyMsg 구독
        self.sub = rospy.Subscriber(self.camera_topic, rospy.AnyMsg, self.image_cb, queue_size=1)
        rospy.loginfo("[lane_tracking] subscribe: %s", self.camera_topic)

    def update_params(self) -> None:
        """파라미터 서버 값 갱신."""
        self.kp = rospy.get_param("~kp", self.kp)
        self.roi_y_ratio = rospy.get_param("~roi_y_ratio", self.roi_y_ratio)

    def image_cb(self, msg: rospy.AnyMsg) -> None:
        """이미지 콜백: 압축/Raw 자동 처리."""
        try:
            mtype = msg._connection_header.get('type', '')
            if mtype == 'sensor_msgs/CompressedImage':
                cm = CompressedImage()
                cm.deserialize(msg._buff)
                np_arr = np.frombuffer(cm.data, np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            elif mtype == 'sensor_msgs/Image':
                im = Image()
                im.deserialize(msg._buff)
                frame = self.bridge.imgmsg_to_cv2(im, "bgr8")
            else:
                rospy.logwarn("지원하지 않는 메시지 타입: %s", mtype)
                return
        except Exception as exc:  # 디코드 실패 시
            rospy.logwarn("이미지 디코드 실패: %s", exc)
            return

        self.update_params()

        edges, roi_color = preprocess_image(frame, self.roi_y_ratio,
                                            self.canny_low, self.canny_high,
                                            self.gaussian_kernel)

        lane_center, overlay = detect_lane_center(edges, roi_color, self.debug)
        img_center = roi_color.shape[1] / 2.0
        offset = lane_center - img_center
        offset_norm = float(np.clip(offset / img_center, -1.0, 1.0))
        self.offset_buffer.append(offset_norm)
        smooth_offset = float(np.mean(self.offset_buffer))

        # 좌(+)/우(-) 기준. 차선이 오른쪽에 있을수록 offset_norm > 0 이므로 우회전(음수) 필요.
        steering = float(np.clip(self.kp * (-smooth_offset), -1.0, 1.0))

        self.offset_pub.publish(Float32(data=smooth_offset))
        self.steer_pub.publish(Float32(data=steering))

        if self.debug:
            try:
                overlay_msg = self.bridge.cv2_to_imgmsg(overlay, encoding="bgr8")
                self.overlay_pub.publish(overlay_msg)
            except Exception as exc:
                rospy.logwarn("오버레이 퍼블리시 실패: %s", exc)

        if self.ctrl_pub is not None:
            cmd = CtrlCmd()
            cmd.steering = steering
            self.ctrl_pub.publish(cmd)

    def spin(self) -> None:
        rospy.spin()


def main() -> None:
    rospy.init_node("lane_tracking_node")
    node = LaneTrackingNode()
    node.spin()


if __name__ == "__main__":
    main()

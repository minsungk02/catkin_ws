#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import numpy as np
import cv2
from std_msgs.msg import Float32
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class LaneTracker:
    def __init__(self):
        self.bridge = CvBridge()
        self.kp = rospy.get_param("~kp", 0.6)
        self.debug = rospy.get_param("~debug", True)
        topic = rospy.get_param("~camera_topic", "/camera/image_raw")

        self.sub = rospy.Subscriber(topic, Image, self.image_cb, queue_size=1)
        self.pub = rospy.Publisher("/lane/steering_angle", Float32, queue_size=1)

        rospy.loginfo("[lane_tracking] subscribe: %s", topic)

    def image_cb(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            rospy.logwarn("cv_bridge error: %s", e)
            return

        h, w = frame.shape[:2]
        roi = frame[int(h*0.6):h, :]    # 하단 40%만 사용
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        edges = cv2.Canny(blur, 50, 150)

        # 허프 변환으로 직선 검출
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50,
                                minLineLength=40, maxLineGap=50)

        lane_pts_x = []
        if lines is not None:
            for l in lines:
                x1, y1, x2, y2 = l[0]
                # 너무 수직인 라인은 제외(노이즈 완화)
                if x2 == x1:
                    continue
                slope = float(y2 - y1) / float(x2 - x1)
                if abs(slope) < 0.3:   # 거의 수평은 제외
                    continue
                lane_pts_x += [x1, x2]
                if self.debug:
                    cv2.line(roi, (x1,y1), (x2,y2), (0,255,0), 2)

        # 차선 중앙 추정 (가장 단순한 버전)
        if lane_pts_x:
            lane_center = np.mean(lane_pts_x)
        else:
            lane_center = w/2.0  # 못 찾으면 정중앙 가정

        img_center = w/2.0
        error = (lane_center - img_center) / (w/2.0)  # -1 ~ 1 범위로 정규화
        steer = - self.kp * error                      # 좌/우 보정

        # 현재는 Float32로 퍼블리시(후에 MORAI 제어 토픽으로 매핑)
        self.pub.publish(Float32(data=float(steer)))

        if self.debug:
            out = roi.copy()
            cv2.circle(out, (int(lane_center), int(roi.shape[0]*0.9)), 5, (255,0,0), -1)
            cv2.line(out, (int(img_center),0), (int(img_center),out.shape[0]), (0,0,255), 1)
            cv2.putText(out, f"steer:{steer:.3f}", (10,25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.imshow("lane_debug", out)
            cv2.waitKey(1)

def main():
    rospy.init_node("lane_tracking_node")
    LaneTracker()
    rospy.spin()

if __name__ == "__main__":
    main()

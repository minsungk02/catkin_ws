"""차선 중심 탐지 모듈."""
from typing import Tuple
import cv2
import numpy as np


def detect_lane_center(edges: np.ndarray, roi_color: np.ndarray,
                        debug: bool = False) -> Tuple[int, np.ndarray]:
    """허프 변환을 이용해 차선 중심을 추정.

    Args:
        edges: Canny 엣지 이미지.
        roi_color: ROI BGR 이미지.
        debug: True일 때 검출 선 오버레이.
    Returns:
        center_x: 추정된 차선 중심 x좌표.
        overlay: 디버그용 BGR 이미지.
    """
    h, w = edges.shape[:2]
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50,
                            minLineLength=40, maxLineGap=50)
    lane_pts_x = []
    overlay = roi_color.copy() if debug else roi_color

    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            if x2 == x1:
                continue
            slope = float(y2 - y1) / float(x2 - x1)
            if abs(slope) < 0.3:
                continue
            lane_pts_x += [x1, x2]
            if debug:
                cv2.line(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)

    if lane_pts_x:
        center_x = int(np.mean(lane_pts_x))
    else:
        center_x = w // 2

    if debug:
        cv2.circle(overlay, (center_x, int(h * 0.9)), 5, (255, 0, 0), -1)
    return center_x, overlay

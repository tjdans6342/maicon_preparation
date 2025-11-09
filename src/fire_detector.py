#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage


class FireDetector:
    """
    ✅ 화재 감지 모듈 (Fire Detection Module)
    - 외부 카메라 토픽에서 영상을 받아 불꽃/화염 영역을 감지
    - ROI 기반 색상 분류 (HSV 공간)
    - 건물 번호 또는 (x, y) 위치 반환
    """

    def __init__(self, topic_name="/fire_cam/image_raw/compressed", visualize=False):
        """
        Parameters
        ----------
        topic_name : str
            화재 카메라 이미지 토픽 이름
        visualize : bool
            True이면 감지 결과를 윈도우에 시각화
        """
        self.bridge = CvBridge()
        self.visualize = visualize
        self.fire_detected = False
        self.last_fire_center = None
        self.last_fire_intensity = 0.0

        # 이미지 구독
        self.sub = rospy.Subscriber(
            topic_name,
            CompressedImage,
            self._callback,
            queue_size=1,
            tcp_nodelay=True
        )

        rospy.loginfo(f"🔥 FireDetector initialized — listening to {topic_name}")

    # -------------------------------------------------------
    #  이미지 콜백
    # -------------------------------------------------------
    def _callback(self, msg):
        try:
            frame = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            rospy.logerr(f"[FireDetector] image conversion failed: {e}")
            return

        self._process_frame(frame)

    # -------------------------------------------------------
    #  화재 감지 로직
    # -------------------------------------------------------
    def _process_frame(self, frame):
        """
        HSV 색상 기반 간단 화염 감지
        - 주로 빨강~노랑 영역의 픽셀 비율 기반
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 불꽃 색상 범위 (노랑 ~ 빨강)
        lower_fire1 = np.array([0, 120, 200])
        upper_fire1 = np.array([20, 255, 255])

        lower_fire2 = np.array([160, 120, 200])
        upper_fire2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv, lower_fire1, upper_fire1)
        mask2 = cv2.inRange(hsv, lower_fire2, upper_fire2)
        mask = cv2.bitwise_or(mask1, mask2)

        # 노이즈 제거
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)

        # 화염 영역 비율 및 중심 계산
        fire_pixels = cv2.countNonZero(mask)
        total_pixels = frame.shape[0] * frame.shape[1]
        fire_ratio = fire_pixels / float(total_pixels)

        self.fire_detected = fire_ratio > 0.01  # 1% 이상이면 화재 감지
        self.last_fire_intensity = fire_ratio

        if self.fire_detected:
            moments = cv2.moments(mask)
            if moments["m00"] != 0:
                cx = int(moments["m10"] / moments["m00"])
                cy = int(moments["m01"] / moments["m00"])
                self.last_fire_center = (cx, cy)
            else:
                self.last_fire_center = None
        else:
            self.last_fire_center = None

        if self.visualize:
            vis = frame.copy()
            if self.fire_detected and self.last_fire_center:
                cv2.circle(vis, self.last_fire_center, 10, (0, 0, 255), -1)
                cv2.putText(vis, "FIRE!", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            cv2.imshow("Fire Detection", vis)
            cv2.waitKey(1)

    # -------------------------------------------------------
    #  상태 조회
    # -------------------------------------------------------
    def get_fire_status(self):
        """
        Returns
        -------
        dict
            {
                "detected": bool,
                "center": (x, y) or None,
                "intensity": float (0~1 비율)
            }
        """
        return {
            "detected": self.fire_detected,
            "center": self.last_fire_center,
            "intensity": self.last_fire_intensity
        }

    # -------------------------------------------------------
    #  특정 구역 판단 (9개 건물 중 어느 구역인지 등)
    # -------------------------------------------------------
    def get_fire_region(self, grid_shape=(3, 3), frame_size=(640, 480)):
        """
        예: 3x3 구역 중 어느 건물(번호)에 화재 발생했는지 반환

        Returns
        -------
        int or None
            1~9 건물 번호 (왼쪽 위부터 오른쪽 아래 순서)
        """
        if not self.fire_detected or self.last_fire_center is None:
            return None

        cols, rows = grid_shape
        fw, fh = frame_size
        gx, gy = self.last_fire_center

        col = int((gx / fw) * cols)
        row = int((gy / fh) * rows)
        col = min(max(col, 0), cols - 1)
        row = min(max(row, 0), rows - 1)

        region_num = row * cols + col + 1
        return region_num

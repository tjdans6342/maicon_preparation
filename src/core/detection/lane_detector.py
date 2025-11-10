#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage


class LaneDetector:
    def __init__(self, image_topic="/usb_cam/image_raw/compressed"):
        """
        LaneDetector 클래스
        - 이미지 구독 및 차선 인식 (BEV 변환 + Hough + Sliding Window)
        - detect() 호출 시 heading, offset 반환
        """
        self.bridge = CvBridge()
        self.image = None

        # ROS 구독자 등록
        rospy.Subscriber(
            image_topic,
            CompressedImage,
            self._camera_callback,
            queue_size=1,
            tcp_nodelay=True
        )

        rospy.loginfo("📷 LaneDetector subscribed to {}".format(image_topic))

    # -------------------------------------------------------
    #  이미지 콜백
    # -------------------------------------------------------
    def _camera_callback(self, msg):
        self.image = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")

    # -------------------------------------------------------
    #  BEV 변환 (Bird’s Eye View)
    # -------------------------------------------------------
    def _warp_bev(self, image):
        src = np.float32([[40, 100], [0, 480], [600, 100], [640, 480]])
        dst = np.float32([[0, 0], [0, 480], [480, 0], [480, 480]])
        M = cv2.getPerspectiveTransform(src, dst)
        bev = cv2.warpPerspective(image, M, (480, 480))
        return bev

    # -------------------------------------------------------
    #  색상 필터 (HLS 기반 흰색/노란색 강조)
    # -------------------------------------------------------
    def _color_filter(self, image):
        hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
        black_lower = np.array([0, 0, 0])
        black_upper = np.array([180, 140, 200])
        mask = cv2.inRange(hls, black_lower, black_upper)
        return cv2.bitwise_and(image, image, mask=mask)

    # -------------------------------------------------------
    #  Hough 변환 (차선 후보 추출)
    # -------------------------------------------------------
    def _hough_transform(self, binary):
        lines = cv2.HoughLines(binary, 1, np.pi / 180, 80)
        h, w = binary.shape
        out = np.zeros((h, w), dtype=np.uint8)

        if lines is not None:
            for line in lines:
                rho, theta = line[0]
                a, b = np.cos(theta), np.sin(theta)
                x0, y0 = a * rho, b * rho
                x1, y1 = int(x0 + 1000 * (-b)), int(y0 + 1000 * a)
                x2, y2 = int(x0 - 1000 * (-b)), int(y0 - 1000 * a)

                slope_deg = 90 - np.degrees(np.arctan2(b, a))
                if abs(slope_deg) < 10 or abs(slope_deg - 180) < 10:
                    cv2.line(out, (x1, y1), (x2, y2), 100, 5)
                else:
                    cv2.line(out, (x1, y1), (x2, y2), 255, 2)
        return out

    # -------------------------------------------------------
    #  Sliding Window로 중심선 탐지
    # -------------------------------------------------------
    def _sliding_window_center(self, binary):
        h, w = binary.shape
        histogram = np.sum(binary[h // 2:, :], axis=0)
        midx = np.argmax(histogram)
        nwindows = 15
        margin = 150
        minpix = 15

        window_height = h // nwindows
        nz = binary.nonzero()
        mid_lane_inds = []
        x_list, y_list = [], []

        for window in range(nwindows - 4):
            y_low = h - (window + 1) * window_height
            y_high = h - window * window_height
            x_low = midx - margin
            x_high = midx + margin

            good_inds = (
                (nz[0] >= y_low)
                & (nz[0] < y_high)
                & (nz[1] >= x_low)
                & (nz[1] < x_high)
            ).nonzero()[0]
            mid_lane_inds.append(good_inds)

            if len(good_inds) > minpix:
                midx = int(np.mean(nz[1][good_inds]))

            x_list.append(midx)
            y_list.append((y_low + y_high) / 2)

        if len(x_list) < 3:
            return None

        fit = np.polyfit(y_list, x_list, 2)
        center_x_bottom = np.polyval(fit, h)
        distance = (w / 2) - center_x_bottom # 왼쪽이 +, 오른쪽이 -
        offset = distance / (w / 2) # 0.0 ~ ±1.0 로 정규화
        heading = np.arctan(fit[1])  # 기울기 근사
        return {"heading": heading, "offset": offset}

    # -------------------------------------------------------
    #  최종 detect() — 외부에서 호출되는 메인 함수
    # -------------------------------------------------------
    def detect(self, image=None):
        """
        입력 이미지(BGR)를 받아 차선을 인식하고 heading과 offset 반환.
        Robot 클래스에서 주기적으로 호출됨.
        """
        if image is None:
            image = self.image
        if image is None:
            return None

        # 1️⃣ 전처리: BEV + Blur + 색상 필터
        bev = self._warp_bev(image)
        blur = cv2.GaussianBlur(bev, (7, 7), 5)
        filtered = self._color_filter(blur)

        # 2️⃣ 이진화 + canny + Hough
        gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
        canny = cv2.Canny(binary, 10, 100)
        hough_img = self._hough_transform(canny)


        cv2.namedWindow('Original')
        cv2.moveWindow('Original', 0, 0)
        cv2.imshow('Original', image)

        cv2.namedWindow('BEV')
        cv2.moveWindow('BEV', 800, 0)
        cv2.imshow('BEV', bev)
        
        cv2.namedWindow('Blurred')
        cv2.moveWindow('Blurred', 1300, 0)
        cv2.imshow('Blurred', blur)
        
        cv2.namedWindow('Color filter')
        cv2.moveWindow('Color filter', 0, 500)
        # cv2.circle(filtered, (240,240), 2, (255,255,255), thickness=-1)
        cv2.imshow('Color filter', filtered)
        
        cv2.namedWindow('binary')
        cv2.moveWindow('binary', 500, 500)
        cv2.imshow('binary', binary)

        cv2.namedWindow('Canny')
        cv2.moveWindow('Canny', 1000, 500)
        cv2.imshow('Canny', canny)

        cv2.namedWindow('Hough')
        cv2.moveWindow('Hough', 1500, 500)
        cv2.imshow('Hough', hough_img)

        # cv2.namedWindow('Sliding Window')
        # cv2.moveWindow('Sliding Window', 1400, 0)
        # cv2.imshow("Sliding Window", out_img)

        cv2.waitKey(1)

        # 3️⃣ Sliding window로 중심선 계산
        result = self._sliding_window_center(hough_img)
        return result


# -------------------------------------------------------
#  단독 테스트용 (rosrun lane_detector.py 실행 시)
# -------------------------------------------------------
if __name__ == "__main__":
    rospy.init_node("lane_detector_test")
    detector = LaneDetector()

    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        if detector.image is not None:
            res = detector.detect()
            if res:
                rospy.loginfo("[Lane] heading={:.3f}, offset={:.1f}".format(res['heading'], res['offset']))
        rate.sleep()

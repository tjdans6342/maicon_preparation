#!/usr/bin/env python
# -*- coding: utf-8 -*-


import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage

from src.configs.lane_config import LaneConfig
from src.utils.image_utils import to_roi, to_bev, color_filter, get_hough_image

from collections import deque

class LaneDetector:
    def __init__(self, image_topic="/usb_cam/image_raw/compressed", config=None, error_queue=None):
        """
        LaneDetector 클래스
        - 이미지 구독 및 차선 인식 (BEV 변환 + Hough + Sliding Window)
        - detect() 호출 시 heading, offset 반환
        """
        self.bridge = CvBridge()
        self.image = None
        # Config 로드 (yaml_path 없으면 기본값 사용)
        self.cfg = config or LaneConfig()  # 없으면 기본 config 로드

        # ROS 구독자 등록
        rospy.Subscriber(
            image_topic,
            CompressedImage,
            self._camera_callback,
            queue_size=1,
            tcp_nodelay=True
        )

        if error_queue == None:
            self.error_queue = {
                'heading': deque([0] * 20),
                'lat': deque([0] * 20),
            }
        else:
            self.error_queue = error_queue
            
        self.image_dict = {
            "Original": None,
            "BEV": None,
            "Filtered": None,
            "gray": None,
            "Blurred": None,
            "binary": None,
            "Canny": None,
            "Hough": None,
            "Lane Detection": None
        }

        rospy.loginfo("📷 LaneDetector subscribed to {}".format(image_topic))

    # -------------------------------------------------------
    #  이미지 콜백
    # -------------------------------------------------------
    def _camera_callback(self, msg):
        self.image = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")

    # -------------------------------------------------------
    #  Sliding Window로 중심선 탐지
    # -------------------------------------------------------
    def _lane_detection(self, hough, nwindows, width, minpix):
        h, w = hough.shape
        histogram = np.sum(hough[h // 2:, :], axis=0)
        midx = np.argmax(histogram)

        if width < 1.0:
            width = np.int(w * width)
        margin = width // 2

        window_height = h // nwindows
        nz = hough.nonzero()
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
        offset = distance / (w / 2) # 0.0 ~ ±1.0 으로 정규화
        heading = np.arctan(fit[1])  # 기울기 근사

        if center_x_bottom == 0:
            offset = 0
        
        # print("center_x_bottom:", center_x_bottom, "offset:", offset)

        self.error_queue['heading'].popleft()
        self.error_queue['lat'].popleft()

        self.error_queue['heading'].append(heading)
        self.error_queue['lat'].append(offset)

        return {
            "heading": heading, "offset": offset,
            "fit": fit, "x": x_list, "y": y_list,
            "mid_avg": np.mean(x_list)
        }

    # -------------------------------------------------------
    #  시각화 함수
    # -------------------------------------------------------
    def _visualize_lane_detection(self, hough_img, x, y, fit, mid_avg, nwindows):
        """
        시각화 + 설명 출력 함수
        - _lane_detection()의 결과를 이용해 차선 검출 과정을 시각화.
        """
        vis = cv2.cvtColor(hough_img, cv2.COLOR_GRAY2BGR)
        h, w = hough_img.shape[:2]

        nwindows = self.cfg.nwindows
        margin = self.cfg.width // 2
        window_height = int(h / nwindows)

        # ---------- (1) 슬라이딩 윈도우 박스 시각화 ----------
        for cx, cy in zip(x, y):
            win_yl = int(cy - window_height / 2)
            win_yh = int(cy + window_height / 2)
            win_xl = int(cx - margin)
            win_xh = int(cx + margin)
            cv2.rectangle(vis, (win_xl, win_yl), (win_xh, win_yh), (0, 255, 0), 2)

        # ---------- (2) 중심점 표시 ----------
        for cx, cy in zip(x, y):
            cv2.circle(vis, (int(cx), int(cy)), 6, (255, 0, 0), -1)

        # ---------- (3) 2차 곡선 시각화 ----------
        y_plot = np.linspace(0, h - 1, h)
        x_fit = fit[0] * y_plot ** 2 + fit[1] * y_plot + fit[2]
        for i in range(1, len(y_plot)):
            cv2.line(vis,
                    (int(x_fit[i - 1]), int(y_plot[i - 1])),
                    (int(x_fit[i]), int(y_plot[i])),
                    (0, 255, 255), 3)

        # ---------- (4) 평균 중심선 시각화 ----------
        cv2.line(vis, (int(mid_avg), 0), (int(mid_avg), h), (255, 100, 255), 2)

        return vis

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

        """
            Pipeline: 
                Original 
                → (ROI) → BEV 
                → color_filter() 
                → Gray Scale: cv2.cvtColor()
                → cv2.GaussianBlur() 
                → cv2.thresholds() 
                → cv2.Canny() 
                → get_hough_image()
        """
        bev_img, _ = to_bev(
            image,
            top=self.cfg.roi_top,
            bottom=self.cfg.roi_bottom,
            margin=self.cfg.roi_margin,
            normalized=self.cfg.bev_normalized
        )
        filtered_img = color_filter(bev_img, hls_range=self.cfg.hls)
        gray_img = cv2.cvtColor(filtered_img, cv2.COLOR_BGR2GRAY)
        blur_img = cv2.GaussianBlur(gray_img, (7, 7), 5)
        _, binary_img = cv2.threshold(blur_img, self.cfg.binary_threshold[0], self.cfg.binary_threshold[1], cv2.THRESH_BINARY)
        canny_img = cv2.Canny(binary_img, 10, 100)
        hough_img = get_hough_image(
            canny_img,
            slope_threshold=self.cfg.slope_threshold, 
            min_votes=self.cfg.min_votes
        )

        # _lane_detection()으로 중심선 계산
        result = self._lane_detection(
            hough_img, 
            nwindows=self.cfg.nwindows, 
            width=self.cfg.width, 
            minpix=self.cfg.minpix
        )

        self.image_dict = {
            "Original": image,
            "BEV": bev_img,
            "Filtered": filtered_img,
            "gray": gray_img,
            "Blurred": blur_img,
            "binary": binary_img,
            "Canny": canny_img,
            "Hough": hough_img,
            # "Lane Detection": lane_detected_img
        }


        if self.cfg.display_mode:
            lane_detected_img = self._visualize_lane_detection(
                hough_img,
                x=result["x"] if result else [],
                y=result["y"] if result else [],
                fit=result["fit"] if result else [0,0,0],
                mid_avg=result["mid_avg"] if result else 0,
                nwindows=self.cfg.nwindows
            )

            self.image_dict["Lane Detection"] = lane_detected_img

            window_pos = [
                (0, 0), (600, 0), (1200, 0),
                (0, 600), (600, 600), (1200, 600),
                (0, 0), (600, 0), (1200, 0),
                (0, 600), (600, 600), (1200, 600)
            ]

            display_names = self.cfg.image_names

            # print(display_names)

            for i, name in enumerate(display_names):
                cv2.namedWindow(name)
                cv2.moveWindow(name, window_pos[i][0], window_pos[i][1])
                cv2.imshow(name, self.image_dict[name])

            cv2.waitKey(1)
        
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

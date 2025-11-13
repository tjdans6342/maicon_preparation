#!/usr/bin/env python
# -*- coding: utf-8 -*-

DARK_HLS = [[0, 0, 0], [180, 140, 200]] # 기존에 했던 값

# WHITE_HLS = [(0, 160, 0), (180, 255, 255)] # white line 1213_area1
# WHITE_HLS = [(0, 150, 0), (180, 255, 255)] # white line night_area1
# WHITE_HLS = [(0, 120, 0), (180, 255, 255)] # white line 2139_area1


# WHITE_HLS = [(0, 140, 0), (180, 255, 255)] # white line 1213_area2
WHITE_HLS = [(0, 140, 0), (180, 255, 255)] # white line 1653_area2
# WHITE_HLS = [(0, 180, 0), (180, 255, 255)] # white line 1020_area2 (not~~~)

YELLOW_HLS = [(20, 70, 12), (40, 130, 110)] # yellow line



import rospy
import time
import numpy as np

from src.core.detection.lane_detector import LaneDetector
from src.core.detection.fire_detector import FireDetector
# from src.core.detection.aruco_trigger import ArucoTrigger
from src.core.detection.aruco_trigger_capture_yolo import ArucoTrigger
from src.core.control.pid_controller import PIDController
from src.core.control.controller import Controller


from src.configs.lane_config import LaneConfig



#### video recoding

from src.configs.video_config import VideoConfig
from src.core.recording.video_recorder import VideoRecorder

########



    
from collections import deque
    
    
class Robot:
    """
    ✅ Robot Main Controller
    - 모든 센서 및 모듈을 통합 관리
    - 모드 전환 (LANE_FOLLOW / FIRE_DETECT / ARUCO)
    - 실시간 제어 루프 실행
    """

    def __init__(self):
        rospy.init_node("robot_main_node", anonymous=False)
        rospy.loginfo("🤖 Robot system initializing...")

        # LaneDetector 설정값
        cfg = LaneConfig()
        cfg.update( # LaneDetector 설정값 오버라이드
            # bev_normalized = False,
            roi_top = 0.75,
            roi_bottom = 0.0,
            roi_margin = 0.1,

            hls=[WHITE_HLS],
            binary_threshold=(20, 255),

            nwindows=15,
            width=150,
            minpix=15,

            slope_threshold=20,
            min_votes=50, #60,

            display_mode=True,
            image_names=["Original", "BEV", "Filtered", "Canny"] #, "Hough", "Lane Detection"]
            # "Original", "BEV", "Filtered":, "gray", "Blurred", "binary", "Canny", "Hough", "Lane Detection"
        )
        
        # Control 설정값
        self.control_configs = {
            'default-setting': [0.05, 1.2, 1.0],

            # linear
            'linear0.10': [0.1 * 1.0, 0.3 * 1.0, 0.3 * 1.0], 
            'linear0.15': [0.1 * 1.5, 0.3 * 1.5, 0.3 * 1.5],
            'linear0.20': [0.1 * 2.0, 0.3 * 2.0, 0.3 * 2.0],
            'linear0.30': [0.1 * 3.0, 0.3 * 3.0, 0.3 * 3.0],
            'linear0.50': [0.1 * 5.0, 0.3 * 3.0, 0.3 * 3.0],

            # curved
            'curved0.10': [0.1 * 1.0, 0.7 * 1.0, 0.7 * 1.0], 
            'curved0.15': [0.1 * 1.5, 0.7 * 1.5, 0.7 * 1.5],
            'curved0.20': [0.1 * 2.0, 0.7 * 2.0, 0.7 * 2.0],
            'curved0.30': [0.1 * 3.0, 0.7 * 3.0, 0.7 * 3.0],
        }

        best_combinations = [
            ['curved0.10', 'curved0.10', 1],
            ['curved0.30', 'curved0.10', 5], # (abs(he) < 0.5)
            ['linear0.50', 'curved0.10', 20], # (abs(he) < 0.5) and (abs(le) < 1.0)
        ]

        self.error_queue_size = 20 # can be tuned
        self.error_queue = {
            'heading': deque([0] * self.error_queue_size),
            'lat': deque([0] * self.error_queue_size),
        }
        self.linear_option = self.control_configs['linear0.15'] # can be tuned
        self.curved_option = self.control_configs['curved0.15'] # can be tuned

        self.base_speed, self.lat_weight, self.heading_weight = self.linear_option

        self.lane = LaneDetector(image_topic="/usb_cam/image_raw/compressed", config=cfg, error_queue=self.error_queue)

        # ArucoTrigger 초기화 & Controller 초기화 & PIDController 초기화
        self.aruco = ArucoTrigger(cmd_topic="/cmd_vel")
        self.controller = Controller("/cmd_vel")
        self.pid = PIDController(kp=0.65, ki=0.001, kd=0.01, integral_limit=2.0)
        # self.fire = FireDetector(topic_name="/fire_cam/image_raw/compressed")


        self.mode = "LANE_FOLLOW"  # 초기 모드
        # 가능한 모드: LANE_FOLLOW, ARUCO, POTHOLE_AVOID


        self.last_switch_time = rospy.get_time()

        rospy.loginfo("Starting main control loop...")


        ### video recoding

            # Video Recording Module

        # video_cfg = VideoConfig()
        # self.video_recorder = VideoRecorder(config=video_cfg)
        
        # # Start recording
        # self.video_recorder.start_recording()
        
        # rospy.loginfo("✅ All subsystems initialized.")



    # -------------------------------------------------------
    #  차선 기반 주행 모드
    # -------------------------------------------------------
    def _lane_follow(self):
        lane_info = self.lane.detect()
        if lane_info is None or (lane_info["heading"] == 0.0 and lane_info["offset"] == 0.0):
            rospy.logwarn_throttle(1.0, "[Lane] No lane detected.")
            self.controller.stop()
            return

        heading_err = lane_info["heading"]
        lateral_err = lane_info["offset"]

        # change mode (linear ↔ curved)
        is_linear = True
        for he, le in zip(self.error_queue['heading'], self.error_queue['lat']):
            if not (abs(he) < 0.3):
                is_linear = False
                break
        
        if is_linear: # linear mode
            self.base_speed, self.lat_weight, self.heading_weight= self.linear_option 
        else: # curved mode
            # print("Passing Curved line!!")
            self.base_speed, self.lat_weight, self.heading_weight = self.curved_option
                       

        # print("total_heading:", self.heading_weight * heading_err, "total_later:", self.lat_weight * lateral_err)

        # 종합 오차
        combined_err = (self.lat_weight * lateral_err) + (self.heading_weight * heading_err)

        # PID 계산
        control = self.pid.update(combined_err, rospy.get_time())
        control = np.clip(control, -1.5, 1.5) # -1.5 ~ 1.5 제한

        # 주행 명령 퍼블리시
        self.controller.publish(linear=self.base_speed, angular=control)

        # print("heading_err: ", heading_err, "lateral_err: ", lateral_err)
        # print("cmd_ang: ", control)

        self.aruco.step()  # 아루코 액션 중이면 계속 실행 (이거 사실 필요 없을 거 같은데..)


            # Add frame to video recorder

        # if self.lane.image is not None:
        #     self.video_recorder.add_frame(self.lane.image)



            # Add frame to video recorder

        # if self.lane.image is not None:
        #     self.video_recorder.add_frame(self.lane.image)


    # -------------------------------------------------------
    #  화재 감지 모드
    # -------------------------------------------------------
    def _fire_mode(self):
        pass
        # fire_status = self.fire.get_fire_status()
        # if fire_status["detected"]:
        #     region = self.fire.get_fire_region()
        #     rospy.loginfo_throttle(2.0, f"🔥 Fire detected! region={region}, intensity={fire_status['intensity']:.3f}")
        #     # 여기서 controller를 통해 로봇을 멈추거나 특정 위치로 이동 가능
        #     self.controller.stop()
        # else:
        #     rospy.loginfo_throttle(2.0, "🚫 No fire detected.")
        #     self.controller.publish(linear=0.02, angular=0.0)

    # -------------------------------------------------------
    #  모드 전환 로직
    # -------------------------------------------------------
    def _check_mode_transition(self):
        # --- 아루코 감지 먼저 실행 ---
        if self.mode == "LANE_FOLLOW":
            frame = self.lane.image
            # if frame is not None:
            #     self.aruco.observe_and_maybe_trigger(frame)

            # --- 포트홀 감지 (임시 로직, 추후 YOLO로 교체 가능) ---
            image_name = "binary"
            if self.mode == "LANE_FOLLOW" and self.lane.image_dict[image_name] is not None:
                # pothole_detected = self.aruco.observe_pothole(self.lane.image_dict[image_name])
                # if pothole_detected:
                #     rospy.loginfo("[Robot] 🕳️ Pothole detected! Triggering avoidance.")
                #     self.aruco.pending_actions = list(self.aruco.rules["pothole"][1])
                #     self.aruco.mode = "EXECUTE_ACTION"
                #     self.mode = "ARUCO"
                #     return
                
                nth = self.aruco.observe_pothole(self.lane.image_dict[image_name])

                if nth:
                    rospy.loginfo("[Robot] 🕳️ Pothole detected! nth={}".format(nth))

                    actions = self.aruco.rules["pothole"].get(nth)
                    if actions:
                        self.aruco.pending_actions = list(actions)
                        self.aruco.mode = "EXECUTE_ACTION"
                        self.mode = "ARUCO"
                        return


        # --- 아루코 상태 확인 ---
        if self.aruco.mode == "EXECUTE_ACTION":
            self.mode = "ARUCO"
            return

        elif self.aruco.mode == "LANE_FOLLOW":
            # (선택) 화재 감지 병렬 확인
            # fire_status = self.fire.get_fire_status()
            # if fire_status["detected"]:
            #     self.mode = "FIRE_DETECT"
            # else:
            #     self.mode = "LANE_FOLLOW"
            self.mode = "LANE_FOLLOW"


    # -------------------------------------------------------
    #  메인 루프
    # -------------------------------------------------------
    def run(self):
        rate = rospy.Rate(20)

        #Register cleanup callback
        
        # rospy.on_shutdown(self._cleanup)

        while not rospy.is_shutdown():
            self._check_mode_transition()

            if self.mode == "LANE_FOLLOW":
                self._lane_follow()

            elif self.mode == "ARUCO":
                # ArucoTrigger 내부에서 step()이 액션 실행 중임
                self.aruco.step()


                # send video 

                # if self.lane.image is not None:
                #     self.video_recorder.add_frame(self.lane.image)

                # 모두 끝나면 ArucoTrigger가 자동으로 LANE_FOLLOW 복귀
                if self.aruco.mode == "LANE_FOLLOW":
                    self.mode = "LANE_FOLLOW"
                    self.pid.reset()

            # elif self.mode == "FIRE_DETECT":
            #     self._fire_mode()

            rate.sleep()

    def _cleanup(self):
        """
        Cleanup resources on ROS shutdown
        - Stop video recording
        - Stop robot movement
        """
        rospy.loginfo("🛑 Robot shutting down...")
        
        # Stop video recording properly
        if hasattr(self, 'video_recorder') and self.video_recorder.is_recording():
            rospy.loginfo("[Cleanup] Stopping video recorder...")
            self.video_recorder.stop_recording()
        
        # Stop robot movement
        if hasattr(self, 'controller'):
            rospy.loginfo("[Cleanup] Stopping robot...")
            self.controller.stop()
        
        rospy.loginfo("✅ Cleanup complete")


# -----------------------------------------------------------
#  Entry Point
# -----------------------------------------------------------
if __name__ == "__main__":
    try:
        robot = Robot()
        robot.run()
    except rospy.ROSInterruptException:
        pass

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
import time
import math
from geometry_msgs.msg import Twist


# --- ArUco 기본 설정 ---
try:
    ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
except AttributeError:
    ARUCO_DICT = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)

try:
    ARUCO_PARAMS = cv2.aruco.DetectorParameters_create()
except AttributeError:
    ARUCO_PARAMS = cv2.aruco.DetectorParameters()


class ArucoTrigger:
    """
    ✅ ArUco 마커 감지 → 행동 트리거 모듈
    Robot 클래스에서 다음처럼 사용됨:
        self.aruco = ArucoTrigger()
        if self.aruco.observe_and_maybe_trigger(self.image):
            self.mode = "ARUCO"
        ...
        if self.mode == "ARUCO":
            finished = self.aruco.step()
            if finished: self.mode = "LANE"
    """

    def __init__(self, cmd_topic="/cmd_vel"):
        self.drive_pub = rospy.Publisher(cmd_topic, Twist, queue_size=1)

        # 감지 파라미터
        self.required_consecutive = 3 # 연속 감지 프레임 수
        self.min_area = 80.0 # 최소 마커 면적
        self.min_y, self.max_y = 60.0, 460.0 # 유효 Y좌표 범위

        # 상태 변수
        self.mode = "LANE_FOLLOW" # 현재 모드
        self._consec = {} # 연속 감지 프레임 카운트 / consec is "consecutive"
        self.last_trigger_times = {} # 마커별 마지막 트리거 시간
        self.seen_counts = {} # 마커별 등장 횟수
        self.pending_actions = [] # 수행 대기 중인 행동 목록

        # 기본 쿨다운 시간(마커별로 다르게 적용 가능)
        self.cooldown_default = 5.0 # 초
        self.cooldown_per_id = { # 마커 ID별 쿨다운 시간
            0: 16.5,
            2: 11.0,
            3: 14.0,
            4: 11.0,
        }

        # 행동 규칙 정의
        # ex) 0번 마커 첫 감지 → 전진 1.3초 → 우회전 → 좌회전
        self.rules = {
            0: {1: [("forward", 1.3), ("right", 90), ("left", 90)]},
            2: {1: ("right", 90)},
            3: {1: [("forward", 2.8), ("right", 90)], 2: ("right", 90)},
            4: {1: [("forward", 4.0), ("right", 90)]},
        }

    # -------------------------------------------------------
    #  ArUco 감지
    # -------------------------------------------------------
    def _detect_markers(self, bgr_img): # input: BGR 이미지, output: 감지된 마커 리스트
        gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, ARUCO_DICT, parameters=ARUCO_PARAMS)
        results = []
        if ids is not None:
            ids = ids.flatten()
            for c, i in zip(corners, ids):
                pts = c.reshape(-1, 2) 
                # 이것이 하는 것은 마커의 코너 좌표를 2D 배열로 변환하는 것입니다. 예를 들어, pts는 [[x1, y1], [x2, y2], [x3, y3], [x4, y4]] 형태가 됩니다.
                # 기존에 c의 형태는 (1, 4, 2)였지만, reshape(-1, 2)를 통해 (4, 2) 형태로 바뀌게 됩니다.
                # (1, 4, 2)에서 1이 의미하는 건 배열이 하나라는 뜻이고, 4는 네 개의 코너 포인트, 2는 각 포인트의 (x, y) 좌표를 의미합니다.  
                cx, cy = float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1]))
                w, h = float(np.max(pts[:, 0]) - np.min(pts[:, 0])), float(np.max(pts[:, 1]) - np.min(pts[:, 1]))
                area = abs(w * h)
                results.append({"id": int(i), "center": (cx, cy), "area": area})
        return results # ex) [{"id": 2, "center": (320.0, 240.0), "area": 1500.0}, ...]

    # -------------------------------------------------------
    #  유효 감지 필터
    # -------------------------------------------------------
    def _gate(self, det): # input: 단일 감지 결과 dict, output: 유효 여부 bool
        area_ok = det["area"] >= self.min_area # 일정 면적 이상
        y = det["center"][1]
        y_ok = self.min_y <= y <= self.max_y
        return area_ok and y_ok

    # observe_and_maybe_trigger()와 step()는 Robot 클래에서 모두 호출해야 함

    # observe_and_maybe_trigger(): LANE_FOLLOW 모드에서 ArUco 관찰 → self.mode를 EXECUTE_ACTION로 전환
    # step(): EXECUTE_ACTION 라면 pending_actions 수행 → 모두 끝나면 self.mode를 LANE_FOLLOW로 전환

    # -------------------------------------------------------
    #  마커 관찰 → 트리거 발생 여부 판단
    # -------------------------------------------------------
    def observe_and_maybe_trigger(self, bgr_img): # input: BGR 이미지, output: 트리거 발생 여부
        """
        ArUco 마커를 관찰하고, 새로 트리거할 상황이면 True 반환
        (Robot이 mode를 ARUCO로 전환하게 됨)
        """
        if self.mode != "LANE_FOLLOW": # 이미 ARUCO 모드인 경우
            return False

        now = time.time()
        detections = [d for d in self._detect_markers(bgr_img) if self._gate(d)] # d is 'marker detection dict'
        if not detections:
            self._consec = {}
            return False

        det = max(detections, key=lambda x: x["area"]) # 가장 큰 마커 선택
        # max(detections, key=lambda x: x["area"])는 detections 리스트에서 'area' 값이 가장 큰 항목을 반환합니다.
        # 예를 들어, detections가 [{"id": 1, "area": 500}, {"id": 2, "area": 1500}]라면,
        # 이 표현식은 {"id": 2, "area": 1500}를 반환합니다.
        mid = det["id"] # mid is "marker ID"

        # 연속 감지 프레임 카운트
        self._consec[mid] = self._consec.get(mid, 0) + 1
        # self.consec.get(mid, 0)에서 get 메서드는 딕셔너리 self.consec에서 키 mid에 해당하는 값을 반환합니다.
        # 만약 mid 키가 존재하지 않으면 기본값으로 0을 반환합니다
        for k in list(self._consec.keys()): # 다른 마커들은 카운트 초기화
        # 이전에 발견한 마커 ID들을 반복하면서, 현재 감지된 마커 ID(mid)와 다른 ID(k)에 대해
        # self._consec 딕셔너리에서 해당 ID의 카운트를 0으로 초기화합니다.
            if k != mid:
                self._consec[k] = 0

        if self._consec[mid] < self.required_consecutive: # 아직 연속 감지 기준 미달
            return False

        # 마커별 쿨다운 체크
        last = self.last_trigger_times.get(mid, 0.0)
        cooldown = self.cooldown_per_id.get(mid, self.cooldown_default)
        if (now - last) < cooldown: # 쿨다운 중
            return False

        # 등장 횟수 기반 행동 매칭
        nth = self.seen_counts.get(mid, 0) + 1
        self.seen_counts[mid] = nth

        if mid in self.rules and nth in self.rules[mid]:
            actions = self.rules[mid][nth] # ex) action = [("forward", 1.3), ("right", 90), ("left", 90)]
            if isinstance(actions, tuple): # action이 단일 액션인 경우 튜플로 정의됨
                actions = [actions]
            self.pending_actions = list(actions)
            self.mode = "EXECUTE_ACTION"
            self.last_trigger_times[mid] = now
            self._consec = {}
            rospy.loginfo("🔸 ArUco ID={} triggered | sequence={}".format(mid, self.pending_actions))
            return True

        return False

    # -------------------------------------------------------
    #  단일 액션 수행 (forward / right / left / turn)
    # -------------------------------------------------------
    def _execute_action(self, action):
        kind = action[0].lower()

        if kind == "forward":
            self._move_forward(seconds=float(action[1]))
        elif kind in ("right", "left", "turn"):
            deg = float(action[1]) if len(action) > 1 else 90
            self._rotate_in_place(kind, deg)

    # -------------------------------------------------------
    #  전진 (시간 기반)
    # -------------------------------------------------------
    def _move_forward(self, seconds=1.0, lin_speed=0.05):
        msg = Twist()
        msg.linear.x = abs(lin_speed)
        msg.angular.z = 0.0
        rate = rospy.Rate(20)

        t0 = rospy.Time.now().to_sec()
        while (rospy.Time.now().to_sec() - t0) < seconds and not rospy.is_shutdown():
            self.drive_pub.publish(msg)
            rate.sleep()

        self.drive_pub.publish(Twist())

    # -------------------------------------------------------
    #  제자리 회전 (좌/우)
    # -------------------------------------------------------
    def _rotate_in_place(self, direction, degrees=90, ang_speed=1.0):
        msg = Twist()
        msg.linear.x = 0.0

        if direction == "right":
            msg.angular.z = -abs(ang_speed)
        elif direction == "left":
            msg.angular.z = abs(ang_speed)
        elif direction == "turn":
            msg.angular.z = abs(ang_speed)
        else:
            return

        duration = abs(degrees) * math.pi / 180.0 / abs(ang_speed)
        rate = rospy.Rate(20)

        t0 = rospy.Time.now().to_sec()
        while (rospy.Time.now().to_sec() - t0) < duration and not rospy.is_shutdown():
            self.drive_pub.publish(msg)
            rate.sleep()

        self.drive_pub.publish(Twist())

    # -------------------------------------------------------
    #  step(): Robot이 매 프레임마다 호출함
    # -------------------------------------------------------
    def step(self):
        """
        pending_actions를 순서대로 수행.
        모두 끝나면 True 반환 (Robot이 LANE 모드로 복귀함)
        """

        if self.mode != "EXECUTE_ACTION": # self.mode가 LANE_FOLLOW인 경우
            return True  

        if not self.pending_actions: # pendig_action를 모두 수행한 경우
            self.mode = "LANE_FOLLOW"
            return True

        # 안전 정지
        self.drive_pub.publish(Twist())
        rospy.sleep(0.1)

        # 맨 앞 액션 수행
        action = self.pending_actions.pop(0)
        self._execute_action(action)

        if not self.pending_actions:
            self.mode = "LANE_FOLLOW"
            rospy.loginfo("✅ ArUco action sequence finished.")
            return True

        return False


# -------------------------------------------------------
#  단독 실행 테스트 (rosrun aruco_trigger.py)
# -------------------------------------------------------
if __name__ == "__main__":
    rospy.init_node("aruco_trigger_test")
    aruco = ArucoTrigger()

    cap = cv2.VideoCapture(0)
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        ret, frame = cap.read()
        if not ret:
            continue

        triggered = aruco.observe_and_maybe_trigger(frame)
        if triggered:
            rospy.loginfo("Triggered! Executing sequence...")
            while not aruco.step():
                rate.sleep()

        cv2.imshow("aruco_view", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

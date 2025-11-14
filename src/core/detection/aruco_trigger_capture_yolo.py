# -*- coding: utf-8 -*-
#!/usr/bin/env python

import rospy, time, math, os
import cv2
import numpy as np
from cv_bridge import CvBridge
# from pyzbar import pyzbar
from geometry_msgs.msg import Twist
import sys

# 프로젝트 루트 경로 추가
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
from src.utils.marker_utils import detect_aruco_markers
from src.utils.image_utils import save_image_with_counter
from src.utils.pothole_utils import check_binary_image_pothole
from src.utils.motion_utils import compute_turn_duration, compute_drive_duration, compute_circle_duration, compute_circle_angular_speed
from src.utils.aruco_utils import check_cooldown, check_consecutive_frames, get_marker_action, normalize_action
from src.configs.aruco_rules import ARUCO_TRIGGER_RULES

# ArUco 마커를 검출해주는 전용 헬퍼 클래스 정의 (기존 호환성 유지)
class ArucoDetector(object):
    # 생성자: CvBridge를 초기화하여 ROS 이미지 <-> OpenCV 이미지를 변환 가능하게 함
    def __init__(self):
        self.bridge = CvBridge()

    # BGR 이미지(bgr_img)를 입력받아 마커 정보 리스트 반환
    # 리팩토링: 유틸 함수 사용
    def detect_ids(self, bgr_img):
        """ bgr_img에서 (id, center(x,y), 면적) 리스트 반환 """
        return detect_aruco_markers(bgr_img)

# ArucoTrigger: ArUco 마커를 감지해서 로봇의 행동을 트리거하는 핵심 클래스
class ArucoTrigger(object):
    """
    - LANE_FOLLOW 상태에서만 마커를 감지해 트리거.
    - 새 ID 등장(혹은 동일 ID의 n번째 등장) + 쿨다운 충족 시 pending_actions 세팅.
    - step()에서 리스트의 액션들을 순차 실행 후 다시 LANE_FOLLOW 복귀.
    """
    def __init__(self, cmd_topic="/cmd_vel", motor_interface=None, rules=None):
        """
        Parameters
        ----------
        cmd_topic : str, default="/cmd_vel"
            ROS 토픽 이름 (motor_interface가 None일 때만 사용)
        motor_interface : MotorInterface, optional
            모터 제어 인터페이스. None이면 자동으로 ROSMotorController 생성
        rules : dict, optional
            마커 트리거 규칙. None이면 기본 규칙 사용
        """
        # 리팩토링: 규칙을 설정 파일에서 로드
        self.rules = rules if rules is not None else ARUCO_TRIGGER_RULES

        # ArUco 마커 검출 헬퍼 객체 생성
        self.detector = ArucoDetector()
        
        # 리팩토링: MotorInterface 사용 (기존 호환성 유지)
        if motor_interface is None:
            from platform.ros.ros_motor_controller import ROSMotorController
            self.motor = ROSMotorController(topic_name=cmd_topic)
            # 기존 코드 호환성을 위해 drive_pub 유지 (향후 제거 가능)
            self.drive_pub = rospy.Publisher(cmd_topic, Twist, queue_size=1)
        else:
            self.motor = motor_interface
            # motor_interface 사용 시 drive_pub은 None
            self.drive_pub = None

        # 현재 모드 초기화: 기본은 차선 따라가기 모드(LANE_FOLLOW)
        self.mode = "LANE_FOLLOW"
        # 실행 대기 중인 액션 리스트 (큐로 사용)
        self.pending_actions = []
        # 마커 ID별 감지 횟수 기록용 딕셔너리 {id: count}
        self.seen_counts = {}

        # 마커 트리거 쿨다운 설정 (동일 마커가 연속해서 트리거되지 않도록)
        self.cooldown_default = 5.0  # 기본 쿨다운 시간 (초)
        self.cooldown_per_id = {
            0: 6.5,
            2: 1.0,
            4: 8.0,
            5: 5.0,
            10: 10.0
        }  # ID별 개별 쿨다운 시간
        self.last_trigger_times = {}  # 마커 ID별 마지막 트리거 시각 기록

        # 화재 건물(일반) 이미지 저장 디렉토리 설정
        self.save_dir = os.path.expanduser("~/catkin_ws/src/ROKAF_Autonomous_Car_2025/images")
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            rospy.loginfo("[ArucoTrigger] Created directory: %s", self.save_dir)

        # YOLO 추론용 이미지 저장 디렉토리 설정
        self.yolo_save_dir = os.path.expanduser("~/catkin_ws/src/ROKAF_Autonomous_Car_2025/yolo_images")
        if not os.path.exists(self.yolo_save_dir):
            os.makedirs(self.yolo_save_dir)
            rospy.loginfo("[ArucoTrigger] Created YOLO directory: %s", self.yolo_save_dir)

        # 화재 건물 캡처 이미지 파일 번호 카운터
        self.capture_count = {}   # {marker_id: count}
        # YOLO 추론용 캡처 이미지 파일 번호 카운터
        self.yolo_capture_count = {}  # {marker_id: count}

        # 마지막으로 본 프레임과 마커 ID 저장 (캡처 시 사용)
        self._last_bgr_img = None
        self._last_marker_id = None

        # 노이즈 제거를 위한 연속 프레임 요구 횟수
        self.required_consecutive = 1
        # 각 마커ID별 현재 연속 만족 프레임 횟수
        self._consec = {}

        # 마커 유효성 필터 기준
        self.min_area = 80.0    # 최소 면적 (픽셀)
        self.min_y = 60.0       # 검출 관심 영역의 최소 y좌표
        self.max_y = 460.0      # 검출 관심 영역의 최대 y좌표

        # QR 코드 감지기 설정 (pyzbar 이용)
        self.qr_detector = True
        self._qr_last_logged = {}
        self.qr_log_cooldown = 1.0
        try:
            import pyzbar  # noqa: F401 (for checking availability)
        except ImportError as exc:
            rospy.logwarn("[ArucoTrigger] pyzbar not available: %s", exc)
            self.qr_detector = False

        # 포트홀 감지 관련 변수 초기화
        self.pothole_seen_count = 0        # 몇 번째 포트홀인지
        self.pothole_last_trigger = 0.0    # 최근 포트홀 트리거 시각
        self.pothole_cooldown = 5.0        # 포트홀 감지 쿨다운(원하는 만큼)


    # _gate: 하나의 검출 결과 det(dict)에 대해 유효한 마커로 간주할지 결정
    def _gate(self, det):
        # 면적 조건 확인
        area_ok = det["area"] >= self.min_area
        # 세로 위치 조건 확인 (min_y <= center_y <= max_y)
        y = det["center"][1]
        y_ok = (y >= self.min_y) and (y <= self.max_y)
        # 두 조건을 모두 만족해야 True 반환
        return area_ok and y_ok

    # _capture_image: 마지막 저장된 프레임과 마지막 마커 ID를 사용해 이미지를 저장 (일반 캡처)
    # 리팩토링: 유틸 함수 사용
    def _capture_image(self):
        if self._last_bgr_img is None or self._last_marker_id is None:
            rospy.logwarn("[ArucoTrigger] Cannot capture image: last frame or ID is missing.")
            return
        mid = self._last_marker_id
        filepath = save_image_with_counter(
            img=self._last_bgr_img,
            save_dir=self.save_dir,
            marker_id=mid,
            counter_dict=self.capture_count,
            prefix="triggered_object",
            file_extension=".jpg"
        )
        if filepath:
            rospy.loginfo("[ArucoTrigger] Triggered image saved: {}".format(filepath))

    # _capture_yolo_image: 마지막 프레임을 별도 디렉토리에 YOLO 추론용 이미지로 저장
    # 리팩토링: 유틸 함수 사용
    def _capture_yolo_image(self):
        if self._last_bgr_img is None or self._last_marker_id is None:
            rospy.logwarn("[ArucoTrigger] Cannot YOLO-capture image: last frame or ID is missing.")
            return
        mid = self._last_marker_id
        filepath = save_image_with_counter(
            img=self._last_bgr_img,
            save_dir=self.yolo_save_dir,
            marker_id=mid,
            counter_dict=self.yolo_capture_count,
            prefix="yolo_",
            file_extension=".jpg"
        )
        if filepath:
            rospy.loginfo("[ArucoTrigger] YOLO image saved: %s", filepath)

    # observe_and_maybe_trigger: LANE_FOLLOW 모드에서 매 프레임 호출.
    # 아루코 마커를 감지하고 조건 충족 시 pending_actions에 액션을 넣고 모드를 EXECUTE_ACTION으로 전환.
    def observe_and_maybe_trigger(self, bgr_img):
        # 최신 프레임 저장 (캡처 액션에서 사용하기 위함)
        self._last_bgr_img = bgr_img
        # QR 코드 처리 (로그 용도)
        self._process_qr_codes(bgr_img)

        # LANE_FOLLOW 모드가 아니라면 트리거 처리 없이 종료
        if self.mode != "LANE_FOLLOW":
            return False

        now = time.time()
        # 현재 프레임에서 ArUco 마커 검출
        dets = self.detector.detect_ids(bgr_img)
        if not dets:
            # 마커 없으면 연속 감지 카운트 초기화
            self._consec = {}
            self._last_marker_id = None
            return False

        # 유효성 조건(_gate)에 맞는 마커만 필터링
        dets = [d for d in dets if self._gate(d)]
        if not dets:
            # 조건 만족 마커 없으면 초기화 후 종료
            self._consec = {}
            self._last_marker_id = None
            return False

        # 가장 큰 마커 하나 선택 (면적이 가장 큰 것)
        det = max(dets, key=lambda x: x["area"])
        mid = det["id"]
        # 마지막 감지 마커 ID 업데이트 (캡처 파일 이름 등에 사용)
        self._last_marker_id = mid

        # 연속 프레임 만족 횟수 업데이트
        self._consec[mid] = self._consec.get(mid, 0) + 1
        # 다른 마커들의 연속 횟수는 리셋
        for k in list(self._consec.keys()):
            if k != mid:
                self._consec[k] = 0

        # 리팩토링: 유틸 함수 사용
        if not check_consecutive_frames(mid, self.required_consecutive, self._consec):
            return False

        # 리팩토링: 유틸 함수 사용
        if not check_cooldown(mid, self.last_trigger_times, self.cooldown_per_id, self.cooldown_default):
            return False

        # 등장 횟수(nth) 갱신
        nth = self.seen_counts.get(mid, 0) + 1
        self.seen_counts[mid] = nth

        # 리팩토링: 유틸 함수 사용
        actions = get_marker_action(self.rules, mid, nth)
        if actions:
            print("actoions info:", mid, nth)
            # 리팩토링: 유틸 함수 사용
            actions = normalize_action(actions)
            # pending_actions 리스트에 액션들 저장하고 모드 전환
            self.pending_actions = list(actions)
            self.mode = "EXECUTE_ACTION"
            # 마지막 트리거 시각 업데이트
            self.last_trigger_times[mid] = now
            # 연속 카운트 리셋 (다음 트리거 대비)
            self._consec = {}
            return True

        return False

    def observe_pothole(self, binary_img):
        """
        포트홀을 감지합니다.
        리팩토링: 유틸 함수 사용
        """
        if binary_img is None or not isinstance(binary_img, np.ndarray):
            return False

        now = time.time()

        # --- 쿨다운 체크 ---
        if (now - self.pothole_last_trigger) < self.pothole_cooldown:
            return False

        # 리팩토링: 유틸 함수 사용
        is_pothole = check_binary_image_pothole(binary_img, white_threshold=0.1)

        # 버퍼 관리 (연속 프레임 확인)
        buffer_size = 3
        if not hasattr(self, "_pothole_buffer"):
            self._pothole_buffer = [False] * buffer_size

        self._pothole_buffer.pop(0)
        self._pothole_buffer.append(is_pothole)

        detected = all(self._pothole_buffer)

        if not detected:
            return False

        # ---------------------------
        #   포트홀 nth 업데이트
        # ---------------------------
        self.pothole_seen_count += 1
        nth = self.pothole_seen_count

        rospy.loginfo("[ArucoTrigger] Pothole detected! nth={}".format(nth))

        self.pothole_last_trigger = now
        del self._pothole_buffer

        # ---------------------------
        #  규칙이 있는지 반환
        # ---------------------------
        if "pothole" in self.rules and nth in self.rules["pothole"]:
            return nth  # nth 반환
        return False

    # _rotate_in_place: 주어진 방향과 각도로 로봇을 제자리 회전
    # 리팩토링: MotorInterface 사용
    def _rotate_in_place(self, direction, degrees, ang_speed=1.0):
        # 회전 각도 결정
        if direction == "right":
            angular_speed = -abs(ang_speed)  # rad/s
            angle = abs(degrees)
        elif direction == "left":
            angular_speed = abs(ang_speed)  # rad/s
            angle = abs(degrees)
        elif direction == "turn":
            angular_speed = abs(ang_speed)  # rad/s
            angle = 120.0  # 고정 각도
        elif direction == "turn1":
            angular_speed = -abs(ang_speed)  # rad/s
            angle = abs(degrees)
        else:
            # 정의되지 않은 방향 명령은 무시
            return

        # 리팩토링: 유틸 함수로 회전 시간 계산
        duration = compute_turn_duration(angle, ang_speed)

        # 리팩토링: MotorInterface 사용
        if self.motor:
            self.motor.set_linear_angular(0.0, angular_speed)
            rospy.sleep(duration)
            self.motor.stop()
        else:
            # 기존 방식 (호환성 유지)
            msg = Twist()
            msg.linear.x = 0.0
            msg.angular.z = angular_speed
            rate = rospy.Rate(20)
            t0 = rospy.Time.now().to_sec()
            while (rospy.Time.now().to_sec() - t0) < duration and (not rospy.is_shutdown()):
                self.drive_pub.publish(msg)
                rate.sleep()
            self.drive_pub.publish(Twist())

    def _drive_distance(self, distance=1.0, speed=0.1):
        """
            지정한 거리(m)만큼 지정 속도(m/s)로 직진 주행하는 함수.
            리팩토링: MotorInterface 사용

            Parameters
            ----------
            distance : float
                이동할 거리 (단위: m)
            speed : float
                주행 속도 (단위 : m/s)
        """
        # 리팩토링: 유틸 함수로 주행 시간 계산
        duration = compute_drive_duration(distance, speed)

        # 리팩토링: MotorInterface 사용
        if self.motor:
            self.motor.set_linear_angular(speed, 0.0)
            rospy.sleep(duration)
            self.motor.stop()
        else:
            # 기존 방식 (호환성 유지)
            msg = Twist()
            msg.linear.x = speed
            msg.angular.z = 0.0
            rate = rospy.Rate(20)
            start_time = rospy.Time.now().to_sec()
            while (rospy.Time.now().to_sec() - start_time) < duration and not rospy.is_shutdown():
                self.drive_pub.publish(msg)
                rate.sleep()
            self.drive_pub.publish(Twist())
        
        rospy.loginfo("[ArucoTrigger] Drove {:.2f}m at {:.2f}m/s".format(distance, speed))

    def _drive_circle(self, diameter=0.3, curvature=1.0, speed=0.1, arrow="left"):
        """
        포트홀 회피용 반원형 주행 동작 (곡률 기반 기하학적 계산)
        리팩토링: MotorInterface 사용
        
        Parameters
        ----------
        diameter : float
            회피 경로의 지름 (m)
        curvature : float
            곡률 (1.0 → 정확한 반원, 0.5 → 완만한 곡선)
        speed : float
            주행 속도 (m/s)
        arrow : str
            회전 방향 ("left" or "right")
        """
        # 리팩토링: 유틸 함수 사용
        angular_speed, R = compute_circle_angular_speed(diameter, curvature, speed, arrow)
        duration = compute_circle_duration(diameter, speed)

        rospy.loginfo(
            "[ArucoTrigger] Circular avoidance start: dir={}, diam={}, curv={}, R={:.3f}m, duration={:.2f}s".format(arrow, diameter, curvature, R, duration)
        )

        # 리팩토링: MotorInterface 사용
        if self.motor:
            self.motor.set_linear_angular(speed, angular_speed)
            rospy.sleep(duration)
            self.motor.stop()
            rospy.sleep(0.1)  # 짧은 완충 시간
        else:
            # 기존 방식 (호환성 유지)
            msg = Twist()
            msg.linear.x = speed
            msg.angular.z = angular_speed
            rate = rospy.Rate(20)
            t_start = rospy.Time.now().to_sec()
            while (rospy.Time.now().to_sec() - t_start) < duration and not rospy.is_shutdown():
                self.drive_pub.publish(msg)
                rate.sleep()
            self.drive_pub.publish(Twist())
            rospy.sleep(0.1)
        
        rospy.loginfo("[ArucoTrigger] Finished circular avoidance ({}), curvature={}".format(arrow, curvature))

    # step: EXECUTE_ACTION 모드에서 pending_actions의 액션들을 순차적으로 실행
    def step(self):
        if self.mode == "EXECUTE_ACTION" and self.pending_actions:
            # 우선 로봇 정지(속도 0) 명령으로 잠깐 멈춤
            if self.motor:
                self.motor.stop()
            elif self.drive_pub:
                self.drive_pub.publish(Twist())
            rospy.sleep(0.15)

            # 실행할 액션 하나 꺼내기
            action = self.pending_actions.pop(0)

            # --- 인자 언패킹 (길이에 따라 자동 처리) ---
            direction = action[0]
            args = action[1:]
            # ---------------------------------------
            if direction == "capture":
                self._capture_image()
            elif direction == "yolo_capture":
                self._capture_yolo_image()
            elif direction == "drive":
                # drive_distance(distance, speed)
                distance = args[0] if len(args) >= 1 else 1.0
                speed = args[1] if len(args) >= 2 else 0.1
                self._drive_distance(distance=distance, speed=speed)
            elif direction == "circle":
                # drive_circle(diameter, curvature, speed, arrow)
                diameter = args[0] if len(args) >= 1 else 0.3
                curvature = args[1] if len(args) >= 2 else 1.0
                speed = args[2] if len(args) >= 3 else 0.1
                arrow = args[3] if len(args) >= 4 else "left"
                self._drive_circle(diameter=diameter, curvature=curvature, speed=speed, arrow=arrow)
            else:
                # 회전 관련 명령 (오른쪽, 왼쪽, turn 등)
                degrees = args[0] if len(args) >= 1 else 90.0
                self._rotate_in_place(direction, degrees, ang_speed=1.0)

            # 모든 액션을 끝마치면 모드를 LANE_FOLLOW로 복귀
            if not self.pending_actions:
                self.mode = "LANE_FOLLOW"

    def _process_qr_codes(self, frame):
        """
        QR 코드를 감지하고 새로운 텍스트가 확인되면 주기적으로 (cooldown 적용) 로그 출력.
        """
        if self.qr_detector is None:
            return
        if not self.qr_detector:
            return

        decoded_payloads = []
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            results = pyzbar.decode(gray)
        except Exception as exc:
            rospy.logwarn_throttle(5.0, "[ArucoTrigger] QR detection error: %s", exc)
            return

        for obj in results:
            data = obj.data.decode('utf-8', errors='ignore').strip() if obj.data else ''
            if data:
                decoded_payloads.append(data)

        if not decoded_payloads:
            return

        now = time.time()
        for text in decoded_payloads:
            last = self._qr_last_logged.get(text, 0.0)
            if (now - last) >= self.qr_log_cooldown:
                rospy.loginfo("[QR] Detected payload: %s", text)
                self._qr_last_logged[text] = now

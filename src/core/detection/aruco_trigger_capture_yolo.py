# -*- coding: utf-8 -*-
#!/usr/bin/env python

import rospy, time, math, os
import cv2
import numpy as np
from cv_bridge import CvBridge
# from pyzbar import pyzbar
from geometry_msgs.msg import Twist

# OpenCV ArUco dictionary and parameters (handle version compatibility)
try:
    ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
except AttributeError:
    ARUCO_DICT = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)

try:
    ARUCO_PARAMS = cv2.aruco.DetectorParameters_create()
except AttributeError:
    ARUCO_PARAMS = cv2.aruco.DetectorParameters()

# ArUco 마커를 검출해주는 전용 헬퍼 클래스 정의
class ArucoDetector(object):
    # 생성자: CvBridge를 초기화하여 ROS 이미지 <-> OpenCV 이미지를 변환 가능하게 함
    # -> ROS에서 받아온 sensor_msgs/CompressedImage를 OpenCV가 처리할 수 있는 numpy 배열로 바꿔야만 마커를 찾을 수 있기 때문
    def __init__(self):
        self.bridge = CvBridge()

    # BGR 이미지(bgr_img)를 입력받아
    #  - id: 마커 ID
    #  - center: (x, y) 중심 좌표
    #  - area: 마커의 대략적인 면적(픽셀 단위)
    # 를 담은 딕셔너리들의 리스트를 반환
    def detect_ids(self, bgr_img):
        """ bgr_img에서 (id, center(x,y), 면적) 리스트 반환 """
        # 컬러 이미지를 그레이스케일로 변환 (마커 검출은 흑백 영상에서 수행)
        gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
        # ArUco 마커 검출 수행
        # corners: 검출된 각 마커의 꼭짓점 좌표들
        # ids: 검출된 마커들의 ID 배열
        # _ : (unused) rejected candidates etc.
        corners, ids, _ = cv2.aruco.detectMarkers(gray, ARUCO_DICT, parameters=ARUCO_PARAMS)
        results = []
        if ids is not None:
            ids = ids.flatten()
            # 각 검출된 마커마다 정보 추출
            for c, i in zip(corners, ids):
                pts = c.reshape(-1, 2)  # (4,2) shape array of corner points
                # 마커 중심 좌표 계산 (x, y 평균)
                cx = float(np.mean(pts[:, 0]))
                cy = float(np.mean(pts[:, 1]))
                # 마커의 폭과 높이 계산 (픽셀 단위)
                w = float(np.max(pts[:, 0]) - np.min(pts[:, 0]))
                h = float(np.max(pts[:, 1]) - np.min(pts[:, 1]))
                # 면적을 사각형으로 근사 (w * h)
                area = abs(w * h)
                # id, 중심, 면적 정보를 결과 리스트에 추가
                results.append({"id": int(i), "center": (cx, cy), "area": area})
        return results

# ArucoTrigger: ArUco 마커를 감지해서 로봇의 행동을 트리거하는 핵심 클래스
class ArucoTrigger(object):
    """
    - LANE_FOLLOW 상태에서만 마커를 감지해 트리거.
    - 새 ID 등장(혹은 동일 ID의 n번째 등장) + 쿨다운 충족 시 pending_actions 세팅.
    - step()에서 리스트의 액션들을 순차 실행 후 다시 LANE_FOLLOW 복귀.
    """
    def __init__(self, cmd_topic="/cmd_vel"):
        # self.rules: 마커 ID와 등장 횟수(nth)에 따라 실행할 액션을 정의하는 규칙 테이블
        # 형식: { marker_id: { nth: action 또는 [action들] } }
        # 예시로 캡처 액션이 포함된 규칙들 정의
        self.rules = {
            # id=0 마커: 1번째 등장 -> 오른쪽 90도 회전, 2번째 등장 -> 오른쪽 20도 회전 후 YOLO 이미지 캡처
            # 0: {1: [("right", 90), ("left", 90)], 2: [("right", 20), ("yolo_capture", 0)]},
            0: {
                1: [("drive", 0.25, 0.2111), ("right", 90), ("left", 90)],
            },
            2: {
                1: [("drive", 0.3, 0.2111), ("right", 90)],
                2: [("left", 20), ("drive", 0.5, 0.2111), ("left", 90)],
            },
            # # id=2 마커: 1번째 등장 -> 오른쪽 90도 회전
            # 2: {1: ("right", 90)},
            # # id=3 마커: 1번째 등장 -> 왼쪽 90도 회전 후 일반 이미지 캡처, 2번째 등장 -> 오른쪽 90도 회전
            # 3: {1: [("left", 90), ("capture", 0)], 2: ("right", 90)},
            3: {
                1: [("drive", 0.35, 0.2111), ("left", 90)], 
                2: [("left", 0)],
            },
            # # id=4 마커: 2번째 등장 -> 왼쪽 90도 회전
            # 4: {
                # 1: [("drive", 0.3, 0.2111), ("right", 90)], 
                # 2: [("drive", 0.2, 0.2111), ("left", 90)],
            # },
            5: {
                1: [("drive", 0.3, 0.2111), ("right", 90)], 
                2: [("drive", 0.3, 0.2111), ("left", 90)],
            },
            10: {
                1: [("drive", 0.25, 0.2111), ("right", 90)],
            },

        }

        # ArUco 마커 검출 헬퍼 객체 생성
        self.detector = ArucoDetector()
        # 로봇 속도 명령을 publish하는 ROS Publisher 생성
        self.drive_pub = rospy.Publisher(cmd_topic, Twist, queue_size=1)

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
    def _capture_image(self):
        if self._last_bgr_img is None or self._last_marker_id is None:
            rospy.logwarn("[ArucoTrigger] Cannot capture image: last frame or ID is missing.")
            return
        mid = self._last_marker_id
        if mid not in self.capture_count:
            self.capture_count[mid] = 0
        self.capture_count[mid] += 1
        filename = os.path.join(self.save_dir, "triggered_object{}_{}.jpg".format(mid, self.capture_count[mid]))
        cv2.imwrite(filename, self._last_bgr_img)
        rospy.loginfo("[ArucoTrigger] Triggered image saved: {}".format(filename))

    # _capture_yolo_image: 마지막 프레임을 별도 디렉토리에 YOLO 추론용 이미지로 저장
    def _capture_yolo_image(self):
        if self._last_bgr_img is None or self._last_marker_id is None:
            rospy.logwarn("[ArucoTrigger] Cannot YOLO-capture image: last frame or ID is missing.")
            return
        mid = self._last_marker_id
        if mid not in self.yolo_capture_count:
            self.yolo_capture_count[mid] = 0
        self.yolo_capture_count[mid] += 1
        filename = os.path.join(self.yolo_save_dir, "yolo_{}_{}.jpg".format(mid, self.yolo_capture_count[mid]))
        cv2.imwrite(filename, self._last_bgr_img)
        rospy.loginfo("[ArucoTrigger] YOLO image saved: %s", filename)

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

        # 설정한 연속 프레임 횟수 미만이면 트리거하지 않고 대기
        if self._consec[mid] < self.required_consecutive:
            return False

        # 쿨다운 체크: 마지막 트리거 시각과 비교
        last = self.last_trigger_times.get(mid, 0.0)
        cooldown = self.cooldown_per_id.get(mid, self.cooldown_default)
        if (now - last) < cooldown:
            return False

        # 등장 횟수(nth) 갱신
        nth = self.seen_counts.get(mid, 0) + 1
        self.seen_counts[mid] = nth

        # 규칙에 해당 (marker id와 nth 조합)하는 액션이 정의되어 있다면 실행
        if mid in self.rules and nth in self.rules[mid]:
            actions = self.rules[mid][nth]
            print("actoions info:", mid, nth)
            if isinstance(actions, tuple):
                actions = [actions]
            # pending_actions 리스트에 액션들 저장하고 모드 전환
            self.pending_actions = list(actions)
            self.mode = "EXECUTE_ACTION"
            # 마지막 트리거 시각 업데이트
            self.last_trigger_times[mid] = now
            # 연속 카운트 리셋 (다음 트리거 대비)
            self._consec = {}
            return True

        return False

    # _rotate_in_place: 주어진 방향과 각도로 로봇을 제자리 회전
    def _rotate_in_place(self, direction, degrees, ang_speed=1.0):
        msg = Twist()
        msg.linear.x = 0.0  # 회전만 할 것이므로 직진속도 0
        if direction == "right":
            # 오른쪽 회전 (z축 음수 각속도)
            msg.angular.z = -abs(ang_speed)
            # 회전 지속 시간 = 회전 각도(라디안) / 각속도(라디안/초)
            duration = abs(degrees) * math.pi/180.0 / abs(ang_speed)
        elif direction == "left":
            # 왼쪽 회전 (z축 양수 각속도)
            msg.angular.z = abs(ang_speed)
            duration = abs(degrees) * math.pi/180.0 / abs(ang_speed)
        elif direction == "turn":
            # "turn": 고정 각도 (120도) 왼쪽 회전
            msg.angular.z = abs(ang_speed)
            duration = 120.0 * math.pi/180.0 / abs(ang_speed)
        elif direction == "turn1":
            # "turn1": 오른쪽 회전을 degrees만큼 수행
            msg.angular.z = -abs(ang_speed)
            duration = abs(degrees) * math.pi/180.0 / abs(ang_speed)
        else:
            # 정의되지 않은 방향 명령은 무시
            return

        rate = rospy.Rate(20)  # 20 Hz로 명령 퍼블리시
        t0 = rospy.Time.now().to_sec()
        # duration 동안 회전 명령을 계속 퍼블리시
        while (rospy.Time.now().to_sec() - t0) < duration and (not rospy.is_shutdown()):
            self.drive_pub.publish(msg)
            rate.sleep()
        # 회전 종료 후 정지 명령 한 번 퍼블리시
        self.drive_pub.publish(Twist())

    def _drive_distance(self, distance=1.0, speed=0.1):
        """
            지정한 거리(m)만큼 지정 속도(m/s)로 직진 주행하는 함수.

            Parameters
            ----------
            distance : float
                이동할 거리 (단위: m)
            speed : float
                주행 속도 (단위 : m/s)
        """
        msg = Twist()
        msg.linear.x = speed
        msg.angular.z = 0.0

        rate = rospy.Rate(20)  # 20Hz 퍼블리시
        start_time = rospy.Time.now().to_sec()
        duration = abs(distance / speed)  # 이동에 필요한 시간

        # 지정된 시간 동안 속도 명령 퍼블리시
        while (rospy.Time.now().to_sec() - start_time) < duration and not rospy.is_shutdown():
            self.drive_pub.publish(msg)
            rate.sleep()

        # 정지 명령
        self.drive_pub.publish(Twist())
        rospy.loginfo("[ArucoTrigger] Drove {:.2f}m at {:.2f}m/s".format(distance, speed))


    # step: EXECUTE_ACTION 모드에서 pending_actions의 액션들을 순차적으로 실행
    def step(self):
        if self.mode == "EXECUTE_ACTION" and self.pending_actions:
            # 우선 로봇 정지(속도 0) 명령으로 잠깐 멈춤
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

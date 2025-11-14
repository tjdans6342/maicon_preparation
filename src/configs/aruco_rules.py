# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""
ArUco 마커 트리거 규칙 설정
마커 ID와 등장 횟수(nth)에 따라 실행할 액션을 정의
"""

# 규칙 형식: { marker_id: { nth: action 또는 [action들] } }
# action 형식: ("direction", arg1, arg2, ...)
#   - ("right", degrees): 오른쪽 회전
#   - ("left", degrees): 왼쪽 회전
#   - ("drive", distance, speed): 직진 주행
#   - ("circle", diameter, curvature, speed, arrow): 원형 주행
#   - ("capture", 0): 일반 이미지 캡처
#   - ("yolo_capture", 0): YOLO 추론용 이미지 캡처

ARUCO_TRIGGER_RULES = {
    0: {
        1: [("drive", 0.25, 0.2111), ("right", 90), ("left", 90)],
    },
    2: {
        1: [("drive", 0.3, 0.2111), ("right", 90)],
        2: [("left", 0), ("drive", 0.5, 0.2111), ("left", 90)],
    },
    3: {
        1: [("drive", 0.4, 0.2111), ("left", 90)], 
        2: [("left", 0)], 
        3: [("drive", 0.45, 0.2111), ("right", 90)],
    },
    # 4: {
    #     1: [("drive", 0.3, 0.2111), ("right", 90)], 
    #     2: [("drive", 0.2, 0.2111), ("left", 90)],
    # },
    5: {
        # 1: [("drive", 0.4, 0.2111), ("right", 90)], 
        # 2: [("drive", 0.4, 0.2111), ("left", 90)],
        1: [("drive", 0.35, 0.2111), ("right", 90)], 
        2: [("drive", 0.35, 0.2111), ("left", 90)],
    },
    # 10: {
    #     1: [("drive", 0.25, 0.2111), ("right", 90)],
    # },
    "pothole": {  # 포트홀 감지 시 트리거할 규칙
        # 1: [("circle", 0.3, 1.0, 0.1, "left"), ("drive", 0.2, 0.15)]
        1: [("drive", 0.00, 0.2111), ("right", 90), ("circle", 0.30, 1.0, 0.2, "left"), ("right", 90)],
        2: [("drive", 0.00, 0.2111), ("right", 90), ("circle", 0.30, 1.0, 0.2, "left"), ("right", 90)],
        3: [("drive", 0.00, 0.2111), ("left", 90), ("circle", 0.30, 1.0, 0.2, "right"), ("left", 90)],
    }
}


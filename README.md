
# MAICON Preparation

자율주행 로봇 시스템 개발을 위한 Vision · Control 통합 프로젝트입니다.  
**Lane / Fire / ArUco 인식과 PID 기반 주행 제어**를 포함한 로봇 제어 파이프라인을 구축합니다.  

<br>


## 📁 프로젝트 구조

> 기능 개발하면서 구조 추가하셔도 됩니다!!

```
maicon_preparation/
├── interface/                             # 플랫폼 독립 I/O 추상 계층
│   ├── camera_interface.py
│   ├── display_interface.py
│   ├── imu_interface.py
│   ├── led_interface.py
│   └── motor_interface.py
│
├── platform/                              # 실제 플랫폼 구현부
│   ├── ros/
│   │   ├── ros_camera.py
│   │   ├── ros_display.py
│   │   ├── ros_imu.py
│   │   ├── ros_led.py
│   │   └── ros_motor_controller.py
│   │
│   └── tiki/
│       └── __init__.py                    # (아직 구현 미완료)
│
├── fire/                                  
│
├── 화재 탐지/                             # 테스트 이미지 폴더(임시)
│   └── fire_image_test/
│       └── image_test/
│
├── src/                                   # 전체 로봇 로직 (Main application)
│   ├── configs/                           # 환경 설정 (Configuration)
│   │   ├── lane_config.py                 # 차선 인식 관련 설정
│   │   ├── aruco_rules.py                 # ArUco 규칙 / 임계값 설정
│   │   └── video_config.py                # 영상 관련 설정 (FPS, size 등)
│   │
│   ├── core/                              # 핵심 로직 (Core Logic)
│   │   ├── ai/                            # AI 모델 관련 기능
│   │   │   └── image_yolo.py              # YOLO 기반 이미지 감지/분류 래퍼
│   │   │
│   │   ├── control/                       # 제어(Control) 계층
│   │   │   ├── controller.py              # 전체 모드 관리 + 로직 통합
│   │   │   └── pid_controller.py          # PID 제어 알고리즘
│   │   │
│   │   ├── detection/                     # 감지(Detection) 기능
│   │   │   ├── lane_detector.py           # 차선 인식
│   │   │   ├── fire_detector.py           # 화재 감지
│   │   │   └── aruco_trigger_capture_yolo.py 
│   │   │       # YOLO 기반 ArUco trigger 모듈(캡처 활용)
│   │   │
│   │   └── recording/                     # 녹화/로깅 기능
│   │       ├── lane_analysis_recorder.py  # 차선 분석 레코더
│   │       ├── video_recorder.py          # 동영상 녹화기
│   │       └── video_to_images.py         # 영상 → 이미지 변환기
│   │
│   ├── main/                              # 메인 실행 계층
│   │   └── robot.py                       # 전체 시스템 실행 진입점
│   │
│   └── utils/                             # 공통 유틸리티 모듈
│       ├── image_utils.py                 # 이미지 전처리(HLS, BEV 등)
│       ├── aruco_utils.py                 # ArUco 관련 헬퍼
│       ├── marker_utils.py                # 마커 좌표/정합 유틸
│       ├── motion_utils.py                # 헤딩/회전/모션 유틸
│       └── pothole_utils.py               # 요철/포트홀 분석 유틸
│
└── README.md

```

<br>

이 프로젝트는 **Lane / Fire / ArUco 감지 기반 자율주행 로봇 시스템**으로,  
`core/detection`과 `core/control` 두 계층의 모듈이 상호작용하며 동작합니다.  
각 기능은 독립적으로 개발 가능하며, `main/robot.py`에서 전체 시스템의 통합 실행이 이루어집니다.


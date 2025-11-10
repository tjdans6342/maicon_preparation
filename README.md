
# MAICON Preparation

자율주행 로봇 시스템 개발을 위한 Vision · Control 통합 프로젝트입니다.  
**Lane / Fire / ArUco 인식과 PID 기반 주행 제어**를 포함한 로봇 제어 파이프라인을 구축합니다.  

<br>


## 📁 프로젝트 구조

> 기능 개발하면서 구조 추가하셔도 됩니다!!

```
src/
├── configs/                       # 환경 설정(Configuration) 관련 파일
│   └── lane_config.py             # 차선 인식 관련 파라미터 및 설정 값
│
├── core/                          # 핵심 로직(Core Logic) 계층
│   ├── detection/                 # 감지(Detection) 계열 모듈
│   │   ├── lane_detector.py       # 차선 인식 모듈
│   │   ├── fire_detector.py       # 화재 감지 모듈
│   │   └── aruco_trigger.py       # ArUco 마커 감지 및 이벤트 트리거
│   │
│   └── control/                   # 제어(Control) 계열 모듈
│       ├── controller.py          # 전체 로봇 제어 흐름 및 상태 관리
│       └── pid_controller.py      # PID 제어 알고리즘
│
├── main/                          # 시스템 실행 진입점 (Main Entry)
│   └── robot.py                   # 전체 로봇 시스템을 통합 실행하는 메인 루프
│
├── utils/                         # 공통 유틸리티 모듈
│   └── image_utils.py             # 이미지 전처리 및 보조 함수
│
├── .gitignore                     # Git 추적 제외 규칙
└── README.md                      # 프로젝트 설명 문서

```

<br>

이 프로젝트는 **Lane / Fire / ArUco 감지 기반 자율주행 로봇 시스템**으로,  
`core/detection`과 `core/control` 두 계층의 모듈이 상호작용하며 동작합니다.  
각 기능은 독립적으로 개발 가능하며, `main/robot.py`에서 전체 시스템의 통합 실행이 이루어집니다.


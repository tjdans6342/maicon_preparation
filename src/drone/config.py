"""
드론 탐지 시스템 설정 파일
"""

from pathlib import Path

# ========================================
# 🎯 탐지 설정
# ========================================

# 필수 탐지 개수 (이 개수와 일치하지 않으면 "검출 안됨" 처리)
REQUIRED_DETECTION_COUNT = 2  # 예: 정확히 1개의 클래스만 탐지되어야 함

# 탐지 개수 허용 범위 설정 (선택적)
MIN_DETECTION_COUNT = 2  # 최소 탐지 개수
MAX_DETECTION_COUNT = 2  # `최대 탐지 개수

# 신뢰도 임계값
CONFIDENCE_THRESHOLD = 0.5  # 0.0 ~ 1.0

# NMS (Non-Maximum Suppression) 임계값
IOU_THRESHOLD = 0.45  # 0.0 ~ 1.0

# ========================================
# 🤖 모델 설정
# ========================================

# 학습된 모델 경로
MODEL_PATH = 'runs/detect/drone_yolov8s/weights/best.pt'

# 백업 모델 경로 (best.pt가 없을 경우)
BACKUP_MODEL_PATH = 'yolov8s.pt'

# 디바이스 설정
DEVICE = 0  # 0: GPU, 'cpu': CPU

# 이미지 크기
IMG_SIZE = 640

# ========================================
# 📊 분석 설정
# ========================================

# 이미지 영역 분할 (위치 분석용)
# 이미지를 3x3 그리드로 나눔
GRID_ROWS = 3
GRID_COLS = 3

# 위치 라벨
POSITION_LABELS = [
    ['좌상단', '상단', '우상단'],
    ['좌측', '중앙', '우측'],
    ['좌하단', '하단', '우하단']
]

# 영어 버전
POSITION_LABELS_EN = [
    ['top-left', 'top-center', 'top-right'],
    ['middle-left', 'center', 'middle-right'],
    ['bottom-left', 'bottom-center', 'bottom-right']
]

# ========================================
# 🎨 시각화 설정
# ========================================

# 바운딩 박스 색상 (BGR)
BBOX_COLOR = (0, 255, 0)  # 초록색

# 바운딩 박스 두께
BBOX_THICKNESS = 2

# 텍스트 색상
TEXT_COLOR = (255, 255, 255)  # 흰색

# 텍스트 배경 색상
TEXT_BG_COLOR = (0, 255, 0)  # 초록색

# 폰트 크기
FONT_SCALE = 0.6

# 폰트 두께
FONT_THICKNESS = 2

# ========================================
# 💾 출력 설정
# ========================================

# 결과 저장 경로
OUTPUT_DIR = Path('outputs/detections')

# 로그 저장 경로
LOG_DIR = Path('outputs/logs')

# 결과 이미지 저장 여부
SAVE_IMAGES = True

# 결과 JSON 저장 여부
SAVE_JSON = True

# ========================================
# 🔧 기타 설정
# ========================================

# 클래스 이름
CLASS_NAMES = ['drone']

# 디버그 모드
DEBUG = True

# 로깅 레벨
LOG_LEVEL = 'INFO'  # DEBUG, INFO, WARNING, ERROR

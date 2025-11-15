"""
드론 탐지 및 분석 분류기
"""

import os
import sys
import json
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import logging

# YOLO 임포트
from ultralytics import YOLO
import torch

# 설정 파일 임포트
try:
    from config import *
except ImportError:
    print("⚠️ config.py를 찾을 수 없습니다. 기본값을 사용합니다.")
    # 기본값 설정
    REQUIRED_DETECTION_COUNT = 2
    MIN_DETECTION_COUNT = 2
    MAX_DETECTION_COUNT = 2
    CONFIDENCE_THRESHOLD = 0.5
    IOU_THRESHOLD = 0.45
    MODEL_PATH = 'runs/detect/drone_yolov8s/weights/best.pt'
    DEVICE = 0
    IMG_SIZE = 640
    GRID_ROWS = 3
    GRID_COLS = 3
    BBOX_COLOR = (0, 255, 0)
    BBOX_THICKNESS = 2
    OUTPUT_DIR = Path('outputs/detections')
    LOG_DIR = Path('outputs/logs')
    SAVE_IMAGES = True
    SAVE_JSON = True
    CLASS_NAMES = ['drone']
    DEBUG = True


# ========================================
# 📦 데이터 클래스
# ========================================

@dataclass
class BoundingBox:
    """바운딩 박스 정보"""
    x1: float  # 좌상단 x
    y1: float  # 좌상단 y
    x2: float  # 우하단 x
    y2: float  # 우하단 y
    width: float  # 너비
    height: float  # 높이
    center_x: float  # 중심 x
    center_y: float  # 중심 y
    area: float  # 면적
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class Detection:
    """단일 탐지 결과"""
    class_id: int
    class_name: str
    confidence: float
    bbox: BoundingBox
    position_grid: Tuple[int, int]  # (row, col)
    position_label: str  # "중앙", "좌상단" 등
    position_label_en: str  # "center", "top-left" 등
    
    def to_dict(self) -> Dict:
        return {
            'class_id': self.class_id,
            'class_name': self.class_name,
            'confidence': float(self.confidence),
            'bbox': self.bbox.to_dict(),
            'position_grid': self.position_grid,
            'position_label': self.position_label,
            'position_label_en': self.position_label_en
        }


@dataclass
class DetectionResult:
    """전체 탐지 결과"""
    success: bool  # 탐지 성공 여부
    message: str  # 상태 메시지
    detection_count: int  # 탐지된 객체 수
    required_count: int  # 요구되는 탐지 수
    detections: List[Detection]  # 탐지 목록
    image_shape: Tuple[int, int, int]  # (height, width, channels)
    timestamp: str  # 탐지 시간
    inference_time: float  # 추론 시간 (ms)
    
    def to_dict(self) -> Dict:
        return {
            'success': self.success,
            'message': self.message,
            'detection_count': self.detection_count,
            'required_count': self.required_count,
            'detections': [d.to_dict() for d in self.detections],
            'image_shape': self.image_shape,
            'timestamp': self.timestamp,
            'inference_time': float(self.inference_time)
        }
    
    def get_positions_array(self) -> np.ndarray:
        """
        탐지된 객체들의 중심 좌표를 numpy array로 반환
        
        Returns:
            np.ndarray: shape (N, 2), [[x1, y1], [x2, y2], ...]
        """
        if not self.detections:
            return np.array([]).reshape(0, 2)
        
        positions = np.array([
            [det.bbox.center_x, det.bbox.center_y]
            for det in self.detections
        ])
        return positions
    
    def get_bboxes_array(self) -> np.ndarray:
        """
        탐지된 객체들의 바운딩 박스를 numpy array로 반환
        
        Returns:
            np.ndarray: shape (N, 4), [[x1, y1, x2, y2], ...]
        """
        if not self.detections:
            return np.array([]).reshape(0, 4)
        
        bboxes = np.array([
            [det.bbox.x1, det.bbox.y1, det.bbox.x2, det.bbox.y2]
            for det in self.detections
        ])
        return bboxes


# ========================================
# 🤖 드론 분류기 클래스
# ========================================

class DroneClassifier:
    """
    드론 탐지 및 분석 분류기
    """
    
    def __init__(
        self,
        model_path: str = MODEL_PATH,
        device: int = DEVICE,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
        iou_threshold: float = IOU_THRESHOLD,
        required_count: int = REQUIRED_DETECTION_COUNT,
        img_size: int = IMG_SIZE,
        save_outputs: bool = True
    ):
        """
        초기화
        
        Args:
            model_path: 모델 가중치 경로
            device: 디바이스 (0: GPU, 'cpu': CPU)
            confidence_threshold: 신뢰도 임계값
            iou_threshold: NMS IoU 임계값
            required_count: 필수 탐지 개수
            img_size: 입력 이미지 크기
            save_outputs: 결과 저장 여부
        """
        
        # 로깅 설정
        self._setup_logging()
        
        # 설정 저장
        self.model_path = model_path
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.required_count = required_count
        self.img_size = img_size
        self.save_outputs = save_outputs
        
        # 출력 디렉토리 생성
        if self.save_outputs:
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            LOG_DIR.mkdir(parents=True, exist_ok=True)
        
        # 모델 로드
        self.model = self._load_model()
        
        self.logger.info("✅ DroneClassifier 초기화 완료")
        self.logger.info(f"   - 모델: {model_path}")
        self.logger.info(f"   - 디바이스: {device}")
        self.logger.info(f"   - 신뢰도 임계값: {confidence_threshold}")
        self.logger.info(f"   - 필수 탐지 개수: {required_count}")
    
    def _setup_logging(self):
        """로깅 설정"""
        self.logger = logging.getLogger('DroneClassifier')
        self.logger.setLevel(logging.DEBUG if DEBUG else logging.INFO)
        
        # 콘솔 핸들러
        if not self.logger.handlers:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.DEBUG if DEBUG else logging.INFO)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
    
    def _load_model(self) -> YOLO:
        """모델 로드"""
        try:
            if not Path(self.model_path).exists():
                self.logger.warning(f"⚠️ 모델 파일을 찾을 수 없습니다: {self.model_path}")
                self.logger.info("기본 모델(yolov8s.pt)을 사용합니다.")
                self.model_path = 'yolov8s.pt'
            
            model = YOLO(self.model_path)
            
            # GPU 사용 가능 여부 확인
            if torch.cuda.is_available() and self.device != 'cpu':
                self.logger.info(f"🚀 GPU 사용: {torch.cuda.get_device_name(0)}")
            else:
                self.logger.info("💻 CPU 사용")
            
            return model
        
        except Exception as e:
            self.logger.error(f"❌ 모델 로드 실패: {e}")
            raise
    
    def _calculate_position(
        self,
        center_x: float,
        center_y: float,
        img_width: int,
        img_height: int
    ) -> Tuple[Tuple[int, int], str, str]:
        """
        객체의 중심 좌표로부터 그리드 위치 계산
        
        Args:
            center_x: 중심 x 좌표
            center_y: 중심 y 좌표
            img_width: 이미지 너비
            img_height: 이미지 높이
        
        Returns:
            ((row, col), position_label, position_label_en)
        """
        
        # 그리드 셀 크기
        cell_width = img_width / GRID_COLS
        cell_height = img_height / GRID_ROWS
        
        # 그리드 인덱스 계산
        col = min(int(center_x / cell_width), GRID_COLS - 1)
        row = min(int(center_y / cell_height), GRID_ROWS - 1)
        
        # 위치 라벨
        position_label = POSITION_LABELS[row][col]
        position_label_en = POSITION_LABELS_EN[row][col]
        
        return (row, col), position_label, position_label_en
    
    def detect(
        self,
        image: np.ndarray,
        visualize: bool = True
    ) -> DetectionResult:
        """
        이미지에서 드론 탐지
        
        Args:
            image: 입력 이미지 (BGR)
            visualize: 시각화 여부
        
        Returns:
            DetectionResult: 탐지 결과
        """
        
        start_time = cv2.getTickCount()
        
        # 이미지 검증
        if image is None or image.size == 0:
            self.logger.error("❌ 유효하지 않은 이미지입니다.")
            return DetectionResult(
                success=False,
                message="유효하지 않은 이미지",
                detection_count=0,
                required_count=self.required_count,
                detections=[],
                image_shape=(0, 0, 0),
                timestamp=datetime.now().isoformat(),
                inference_time=0.0
            )
        
        img_height, img_width = image.shape[:2]
        
        # YOLO 추론
        try:
            results = self.model.predict(
                image,
                imgsz=self.img_size,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                device=self.device,
                verbose=False
            )
        except Exception as e:
            self.logger.error(f"❌ 추론 실패: {e}")
            return DetectionResult(
                success=False,
                message=f"추론 실패: {e}",
                detection_count=0,
                required_count=self.required_count,
                detections=[],
                image_shape=image.shape,
                timestamp=datetime.now().isoformat(),
                inference_time=0.0
            )
        
        # 추론 시간 계산
        inference_time = (cv2.getTickCount() - start_time) / cv2.getTickFrequency() * 1000
        
        # 결과 파싱
        detections = []
        
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            
            for i in range(len(boxes)):
                # 바운딩 박스 좌표
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                
                # 신뢰도
                confidence = float(boxes.conf[i].cpu().numpy())
                
                # 클래스 ID
                class_id = int(boxes.cls[i].cpu().numpy())
                class_name = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f"class_{class_id}"
                
                # 바운딩 박스 정보 계산
                width = x2 - x1
                height = y2 - y1
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                area = width * height
                
                bbox = BoundingBox(
                    x1=float(x1), y1=float(y1),
                    x2=float(x2), y2=float(y2),
                    width=float(width), height=float(height),
                    center_x=float(center_x), center_y=float(center_y),
                    area=float(area)
                )
                
                # 위치 계산
                position_grid, position_label, position_label_en = self._calculate_position(
                    center_x, center_y, img_width, img_height
                )
                
                # Detection 객체 생성
                detection = Detection(
                    class_id=class_id,
                    class_name=class_name,
                    confidence=confidence,
                    bbox=bbox,
                    position_grid=position_grid,
                    position_label=position_label,
                    position_label_en=position_label_en
                )
                
                detections.append(detection)
        
        # 탐지 개수 확인
        detection_count = len(detections)
        
        # 성공 여부 판단
        if detection_count == self.required_count:
            success = True
            message = f"✅ 탐지 성공: {detection_count}개 탐지됨"
        elif detection_count == 0:
            success = False
            message = f"❌ 검출 안됨 (요구: {self.required_count}개)"
        elif detection_count < self.required_count:
            success = False
            message = f"⚠️ 탐지 부족: {detection_count}개 탐지됨 (요구: {self.required_count}개)"
        else:
            success = False
            message = f"⚠️ 과다 탐지: {detection_count}개 탐지됨 (요구: {self.required_count}개)"
        
        # 결과 생성
        result = DetectionResult(
            success=success,
            message=message,
            detection_count=detection_count,
            required_count=self.required_count,
            detections=detections,
            image_shape=image.shape,
            timestamp=datetime.now().isoformat(),
            inference_time=inference_time
        )
        
        # 로깅
        self.logger.info(f"{message} (추론 시간: {inference_time:.1f}ms)")
        
        # 시각화
        if visualize and detection_count > 0:
            self._visualize(image, result)
        
        return result
    
    
     #디버깅용 detect, 삭제 가능.
 
    '''
    def detect(
        self,
        image: np.ndarray,
        visualize: bool = True
    ) -> DetectionResult:
        """
        이미지에서 드론 탐지
        
        Args:
            image: 입력 이미지 (BGR)
            visualize: 시각화 여부
        
        Returns:
            DetectionResult: 탐지 결과
        """
        
        start_time = cv2.getTickCount()
        
        # 이미지 검증
        if image is None or image.size == 0:
            self.logger.error("❌ 유효하지 않은 이미지입니다.")
            return DetectionResult(
                success=False,
                message="유효하지 않은 이미지",
                detection_count=0,
                required_count=self.required_count,
                detections=[],
                image_shape=(0, 0, 0),
                timestamp=datetime.now().isoformat(),
                inference_time=0.0
            )
        
        # 원본 이미지 크기
        img_height, img_width = image.shape[:2]
        self.logger.info(f"원본 이미지 크기: {img_width}x{img_height}")
        
        # YOLO 추론
        try:
            results = self.model.predict(
                image,
                imgsz=self.img_size,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                device=self.device,
                verbose=False
            )
        except Exception as e:
            self.logger.error(f"❌ 추론 실패: {e}")
            return DetectionResult(
                success=False,
                message=f"추론 실패: {e}",
                detection_count=0,
                required_count=self.required_count,
                detections=[],
                image_shape=image.shape,
                timestamp=datetime.now().isoformat(),
                inference_time=0.0
            )
        
        # 추론 시간 계산
        inference_time = (cv2.getTickCount() - start_time) / cv2.getTickFrequency() * 1000
        
        # 🔍 디버그: YOLO가 사용한 이미지 크기 확인
        if len(results) > 0:
            result = results[0]
            if hasattr(result, 'orig_shape'):
                self.logger.info(f"YOLO orig_shape: {result.orig_shape}")
            if hasattr(result, 'shape'):
                self.logger.info(f"YOLO processed shape: {result.shape}")
        
        # 결과 파싱
        detections = []
        
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            
            for i in range(len(boxes)):
                # 바운딩 박스 좌표 (xyxy 형식)
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                
                # 🔍 디버그: 첫 번째 탐지 객체의 좌표 범위 확인
                if i == 0:
                    self.logger.info(f"첫 번째 탐지 좌표: ({x1:.1f}, {y1:.1f}) → ({x2:.1f}, {y2:.1f})")
                    self.logger.info(f"좌표 범위 확인:")
                    self.logger.info(f"  - x 범위: 0 ~ {img_width} (좌표: {x1:.1f} ~ {x2:.1f})")
                    self.logger.info(f"  - y 범위: 0 ~ {img_height} (좌표: {y1:.1f} ~ {y2:.1f})")
                    
                    # 좌표가 원본 크기 기준인지 640 기준인지 판단
                    if x2 <= 640 and y2 <= 640:
                        self.logger.warning("⚠️ 좌표가 640x640 기준으로 보입니다!")
                    elif x2 <= img_width and y2 <= img_height:
                        self.logger.info("✅ 좌표가 원본 크기 기준으로 보입니다.")
                
                # 신뢰도
                confidence = float(boxes.conf[i].cpu().numpy())
                
                # 클래스 ID
                class_id = int(boxes.cls[i].cpu().numpy())
                class_name = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f"class_{class_id}"
                
                # 바운딩 박스 정보 계산
                width = x2 - x1
                height = y2 - y1
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                area = width * height
                
                bbox = BoundingBox(
                    x1=float(x1), y1=float(y1),
                    x2=float(x2), y2=float(y2),
                    width=float(width), height=float(height),
                    center_x=float(center_x), center_y=float(center_y),
                    area=float(area)
                )
                
                # 위치 계산
                position_grid, position_label, position_label_en = self._calculate_position(
                    center_x, center_y, img_width, img_height
                )
                
                # Detection 객체 생성
                detection = Detection(
                    class_id=class_id,
                    class_name=class_name,
                    confidence=confidence,
                    bbox=bbox,
                    position_grid=position_grid,
                    position_label=position_label,
                    position_label_en=position_label_en
                )
                
                detections.append(detection)
        
        # 탐지 개수 확인
        detection_count = len(detections)
        
        # 성공 여부 판단
        if detection_count == self.required_count:
            success = True
            message = f"✅ 탐지 성공: {detection_count}개 탐지됨"
        elif detection_count == 0:
            success = False
            message = f"❌ 검출 안됨 (요구: {self.required_count}개)"
        elif detection_count < self.required_count:
            success = False
            message = f"⚠️ 탐지 부족: {detection_count}개 탐지됨 (요구: {self.required_count}개)"
        else:
            success = False
            message = f"⚠️ 과다 탐지: {detection_count}개 탐지됨 (요구: {self.required_count}개)"
        
        # 결과 생성
        result = DetectionResult(
            success=success,
            message=message,
            detection_count=detection_count,
            required_count=self.required_count,
            detections=detections,
            image_shape=image.shape,
            timestamp=datetime.now().isoformat(),
            inference_time=inference_time
        )
        
        # 로깅
        self.logger.info(f"{message} (추론 시간: {inference_time:.1f}ms)")
        
        # 시각화
        if visualize and detection_count > 0:
            self._visualize(image, result)
        
        return result
    '''
    def _visualize(self, image: np.ndarray, result: DetectionResult):
        """
        탐지 결과 시각화
        
        Args:
            image: 원본 이미지
            result: 탐지 결과
        """
        
        vis_image = image.copy()
        
        for det in result.detections:
            bbox = det.bbox
            
            # 바운딩 박스 그리기
            cv2.rectangle(
                vis_image,
                (int(bbox.x1), int(bbox.y1)),
                (int(bbox.x2), int(bbox.y2)),
                BBOX_COLOR,
                BBOX_THICKNESS
            )
            
            # 라벨 텍스트
            label = f"{det.class_name} {det.confidence:.2f}"
            position_text = f"{det.position_label}"
            
            # 텍스트 크기 계산
            (label_w, label_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, FONT_THICKNESS
            )
            
            # 텍스트 배경
            cv2.rectangle(
                vis_image,
                (int(bbox.x1), int(bbox.y1) - label_h - 10),
                (int(bbox.x1) + label_w, int(bbox.y1)),
                TEXT_BG_COLOR,
                -1
            )
            
            # 텍스트
            cv2.putText(
                vis_image,
                label,
                (int(bbox.x1), int(bbox.y1) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                FONT_SCALE,
                TEXT_COLOR,
                FONT_THICKNESS
            )
            
            # 위치 정보
            cv2.putText(
                vis_image,
                position_text,
                (int(bbox.center_x) - 30, int(bbox.center_y)),
                cv2.FONT_HERSHEY_SIMPLEX,
                FONT_SCALE,
                (0, 0, 255),
                FONT_THICKNESS
            )
            
            # 중심점 표시
            cv2.circle(
                vis_image,
                (int(bbox.center_x), int(bbox.center_y)),
                5,
                (0, 0, 255),
                -1
            )
        
        # 상태 메시지
        status_color = (0, 255, 0) if result.success else (0, 0, 255)
        cv2.putText(
            vis_image,
            result.message,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            status_color,
            2
        )
        
        # 결과 저장
        if self.save_outputs:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            output_path = OUTPUT_DIR / f"detection_{timestamp}.jpg"
            cv2.imwrite(str(output_path), vis_image)
            self.logger.info(f"💾 결과 이미지 저장: {output_path}")
        
        return vis_image
    
    def detect_from_file(
        self,
        image_path: str,
        visualize: bool = True,
        save_json: bool = True
    ) -> DetectionResult:
        """
        이미지 파일에서 드론 탐지
        
        Args:
            image_path: 이미지 파일 경로
            visualize: 시각화 여부
            save_json: JSON 저장 여부
        
        Returns:
            DetectionResult: 탐지 결과
        """
        
        # 이미지 로드
        image = cv2.imread(image_path)
        
        if image is None:
            self.logger.error(f"❌ 이미지를 로드할 수 없습니다: {image_path}")
            return DetectionResult(
                success=False,
                message=f"이미지 로드 실패: {image_path}",
                detection_count=0,
                required_count=self.required_count,
                detections=[],
                image_shape=(0, 0, 0),
                timestamp=datetime.now().isoformat(),
                inference_time=0.0
            )
        
        self.logger.info(f"📷 이미지 로드: {image_path}")
        
        # 탐지 수행
        result = self.detect(image, visualize=visualize)
        
        # JSON 저장
        if save_json and self.save_outputs:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            json_path = OUTPUT_DIR / f"detection_{timestamp}.json"
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"💾 결과 JSON 저장: {json_path}")
        
        return result
    
    def detect_batch(
        self,
        image_paths: List[str],
        visualize: bool = True,
        save_json: bool = True
    ) -> List[DetectionResult]:
        """
        여러 이미지에서 배치 탐지
        
        Args:
            image_paths: 이미지 파일 경로 리스트
            visualize: 시각화 여부
            save_json: JSON 저장 여부
        
        Returns:
            List[DetectionResult]: 탐지 결과 리스트
        """
        
        results = []
        
        self.logger.info(f"🔄 배치 탐지 시작: {len(image_paths)}개 이미지")
        
        for i, image_path in enumerate(image_paths, 1):
            self.logger.info(f"\n[{i}/{len(image_paths)}] {image_path}")
            result = self.detect_from_file(image_path, visualize, save_json=False)
            results.append(result)
        
        # 전체 결과 JSON 저장
        if save_json and self.save_outputs:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            json_path = OUTPUT_DIR / f"batch_detection_{timestamp}.json"
            
            batch_result = {
                'total_images': len(image_paths),
                'successful_detections': sum(1 for r in results if r.success),
                'results': [r.to_dict() for r in results]
            }
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(batch_result, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"\n💾 배치 결과 JSON 저장: {json_path}")
        
        # 요약 출력
        success_count = sum(1 for r in results if r.success)
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"📊 배치 탐지 완료")
        self.logger.info(f"{'='*60}")
        self.logger.info(f"총 이미지: {len(image_paths)}개")
        self.logger.info(f"성공: {success_count}개")
        self.logger.info(f"실패: {len(image_paths) - success_count}개")
        self.logger.info(f"성공률: {success_count/len(image_paths)*100:.1f}%")
        self.logger.info(f"{'='*60}\n")
        
        return results


# ========================================
# 🎯 사용 예시
# ========================================

def main():
    """메인 함수"""
    
    print("🚁 드론 탐지 분류기 테스트\n")
    
    # 분류기 초기화
    classifier = DroneClassifier(
        model_path='runs/detect/drone_yolov8s/weights/best.pt',
        device=0,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        required_count=REQUIRED_DETECTION_COUNT,  # 정확히 1개의 드론만 탐지되어야 함
        save_outputs=True
    )
    '''
    # ========================================
    # 테스트 1: 단일 이미지 탐지
    # ========================================
    print("\n" + "="*60)
    print("테스트 1: 단일 이미지 탐지")
    print("="*60)
    
    test_image_path = 'src/drone/dataset/test/images/example.jpg'
    
    if Path(test_image_path).exists():
        result = classifier.detect_from_file(
            test_image_path,
            visualize=True,
            save_json=True
        )
        
        # 결과 출력
        print(f"\n📊 탐지 결과:")
        print(f"  - 성공: {result.success}")
        print(f"  - 메시지: {result.message}")
        print(f"  - 탐지 개수: {result.detection_count}/{result.required_count}")
        print(f"  - 추론 시간: {result.inference_time:.1f}ms")
        
        if result.detections:
            print(f"\n📍 탐지된 객체:")
            for i, det in enumerate(result.detections, 1):
                print(f"  [{i}] {det.class_name}")
                print(f"      - 신뢰도: {det.confidence:.3f}")
                print(f"      - 위치: {det.position_label} (그리드: {det.position_grid})")
                print(f"      - 중심: ({det.bbox.center_x:.1f}, {det.bbox.center_y:.1f})")
                print(f"      - 크기: {det.bbox.width:.1f} x {det.bbox.height:.1f}")
        
        # 위치 배열 출력
        positions = result.get_positions_array()
        print(f"\n📐 위치 배열 (중심 좌표):")
        print(positions)
        
        # 바운딩 박스 배열 출력
        bboxes = result.get_bboxes_array()
        print(f"\n📦 바운딩 박스 배열:")
        print(bboxes)
    
    else:
        print(f"⚠️ 테스트 이미지를 찾을 수 없습니다: {test_image_path}")
    '''
    # ========================================
    # 테스트 2: 배치 탐지
    # ========================================
    print("\n" + "="*60)
    print("테스트 2: 배치 탐지")
    print("="*60)
    
    test_dir = Path('src/drone/dataset/test/images')
    
    if test_dir.exists():
        image_files = list(test_dir.glob('*.jpg')) + list(test_dir.glob('*.png'))
        
        if image_files:
            # 최대 5개만 테스트
            test_images = [str(p) for p in image_files[:37]]
            
            results = classifier.detect_batch(
                test_images,
                visualize=True,
                save_json=True
            )
        else:
            print(f"⚠️ 테스트 이미지가 없습니다: {test_dir}")
    else:
        print(f"⚠️ 테스트 디렉토리를 찾을 수 없습니다: {test_dir}")
    
    print("\n✅ 테스트 완료!")


if __name__ == '__main__':
    main()

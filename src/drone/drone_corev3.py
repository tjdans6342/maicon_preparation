"""
드론 이미지 분석 통합 시스템
- ROI 자르기 기능
- 이미지 분류 기능
- YOLO 객체 탐지 및 매칭 기능
- 백업 ROI 분석 메커니즘
"""
'''
단일 이미지 처리시
python drone_corev3.py path/to/image.jpg
폴더 배치 처리
python drone_corev3.py path/to/folder output_results.csv
이거 꼭 실사용전 테스트 해봐야함

'''

import os
import sys
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.models as models
from PIL import Image
import json
from tqdm import tqdm
import csv
import time
from torch.serialization import add_safe_globals

# torchvision 변환 클래스를 안전한 전역으로 추가
add_safe_globals([
    'torchvision.transforms.transforms.Compose',
    'torchvision.transforms.transforms.Resize',
    'torchvision.transforms.transforms.ToTensor',
    'torchvision.transforms.transforms.Normalize'
])

# 로컬 모듈 임포트
try:
    from target_geometry import (
        recognize_rotated_h_marker_sift,
        transform_image_from_params,
        calculate_drone_altitude_from_scale_precise,
        detect_all_building_rooftops,
        BUILDINGS_INFO,
        H_MARKER_REAL_WIDTH_M,
        TARGET_X,
        TARGET_Y
    )
    from yolo_classfier import DroneClassifier as YoloClassifier, DetectionResult
    TARGET_GEOMETRY_IMPORTED = True
except ImportError:
    print("⚠️ target_geometry 또는 yolo_classfier 모듈을 임포트할 수 없습니다. 일부 기능이 제한됩니다.")
    TARGET_GEOMETRY_IMPORTED = False

# ========================================
# ⚙️ 전역 설정
# ========================================

# ROI 설정
CROP_SIZE_PIXELS = 250  # 잘라낼 ROI의 크기
OUTPUT_BASE_DIR = "src/drone/second"  # 저장할 기본 디렉토리

# 건물 매칭 거리 임계값 (픽셀)
MATCHING_DISTANCE_THRESHOLD = 200.0  # 기본값 200px

# 원본 이미지 크기
ORIGINAL_IMAGE_WIDTH = 4000
ORIGINAL_IMAGE_HEIGHT = 3000

# YOLO 입력 크기
YOLO_INPUT_SIZE = 640

# ========================================
# 🖼️ ROI 자르기 및 저장 함수
# ========================================

def crop_rois_for_classification(
    transformed_image: np.ndarray,
    rooftop_positions: dict[int, tuple[float, float]],
    image_index: str,
    crop_size: int = CROP_SIZE_PIXELS,
    output_base_dir: str = OUTPUT_BASE_DIR
) -> Dict[int, str]:
    """
    정렬된 이미지에서 계산된 옥상 위치를 중심으로 ROI를 잘라내어 저장합니다.
    각 ROI는 'output_base_dir/건물ID/이미지인덱스_건물ID.png' 형태로 저장됩니다.
    
    Args:
        transformed_image (np.ndarray): H 마커를 기준으로 정렬된 이미지.
        rooftop_positions (dict[int, tuple[float, float]]): 
            {건물ID: (옥상 X 좌표, 옥상 Y 좌표)} 형태의 딕셔너리.
        image_index (str): 저장할 파일명에 사용될 고유 인덱스 (예: '13', '20', 'test').
        crop_size (int): 잘라낼 ROI의 한 변 길이 (픽셀).
        output_base_dir (str): ROI를 저장할 기본 디렉토리 이름.
        
    Returns:
        Dict[int, str]: {건물ID: ROI 파일 경로} 형태의 딕셔너리
    """
    
    # 이미지 크기
    h, w = transformed_image.shape[:2]
    half_size = crop_size // 2
    
    # 기본 출력 디렉토리 생성
    if not os.path.exists(output_base_dir):
        os.makedirs(output_base_dir)
        print(f"기본 디렉토리 생성: {output_base_dir}")

    # 결과 딕셔너리 (건물ID: ROI 파일 경로)
    roi_paths = {}

    # 각 건물별 ROI 자르기 및 저장
    for building_id, (center_x_float, center_y_float) in rooftop_positions.items():
        center_x = int(round(center_x_float))
        center_y = int(round(center_y_float))
        
        # 1. 자를 영역의 경계 계산
        x_min = center_x - half_size
        x_max = center_x + half_size
        y_min = center_y - half_size
        y_max = center_y + half_size
        
        # 2. 이미지 경계 처리 및 자르기
        # Crop 영역이 이미지 밖으로 나가지 않도록 조정
        crop_x_min = max(0, x_min)
        crop_y_min = max(0, y_min)
        crop_x_max = min(w, x_max)
        crop_y_max = min(h, y_max)
        
        # 이미지 자르기 (ROI 추출)
        cropped_roi = transformed_image[crop_y_min:crop_y_max, crop_x_min:crop_x_max]
        
        # 3. 저장 디렉토리 생성 (예: src/drone/second/1, ...)
        save_dir = os.path.join(output_base_dir, str(building_id))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        # 4. 파일 저장
        # 파일명에 image_index를 사용하여 고유성 확보
        save_path = os.path.join(save_dir, f"{image_index}_B{building_id}.png")
        cv2.imwrite(save_path, cropped_roi)
        print(f"건물 {building_id} ROI 저장 완료: {save_path} (크기: {cropped_roi.shape[1]}x{cropped_roi.shape[0]})")
        
        # 결과 딕셔너리에 추가
        roi_paths[building_id] = save_path
        
    return roi_paths

# ========================================
# 🤖 이미지 분류 모델 클래스
# ========================================

class DroneClassifier(nn.Module):
    def __init__(self, num_classes=2, pretrained=False):
        super(DroneClassifier, self).__init__()
        
        # EfficientNet-B0 백본 사용
        self.backbone = models.efficientnet_b0(pretrained=pretrained)
        
        # 분류기 부분 수정
        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        return self.backbone(x)

class DroneImageClassifier:
    def __init__(self, model_path, config_path=None, device=None):
        """
        드론 이미지 분류기 초기화
        
        Args:
            model_path (str): 모델 파일 경로 (.pth)
            config_path (str, optional): 설정 파일 경로 (.json)
            device (str, optional): 사용할 디바이스 ('cuda' 또는 'cpu')
        """
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"사용 디바이스: {self.device}")
        
        # 설정 파일 로드
        if config_path is None:
            config_path = model_path.replace('.pth', '_config.json')
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        else:
            # 기본 설정
            self.config = {
                'img_size': 224,
                'class_names': ['OK', 'NG'],
                'model_type': 'EfficientNet-B0'
            }
            print(f"설정 파일을 찾을 수 없습니다. 기본 설정을 사용합니다.")
        
        # 변환 정의
        self.transform = transforms.Compose([
            transforms.Resize((self.config['img_size'], self.config['img_size'])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 모델 로드
        self.model = DroneClassifier(num_classes=len(self.config['class_names']), pretrained=False)
        
        try:
            # 먼저 weights_only=True로 시도 (더 안전한 방법)
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
        except Exception as e:
            print(f"weights_only=True로 로드 실패: {e}")
            print("weights_only=False로 시도합니다. 신뢰할 수 있는 모델인 경우에만 진행하세요.")
            
            # weights_only=False로 시도 (보안 위험이 있지만 이전 버전과의 호환성을 위해)
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.model.to(self.device)
        self.model.eval()
        
        print(f"모델 로드 완료: {model_path}")
        print(f"클래스: {self.config['class_names']}")
    
    def predict_image(self, image_path):
        """
        단일 이미지 분류
        
        Args:
            image_path (str): 이미지 파일 경로
            
        Returns:
            tuple: (예측 클래스 인덱스, 예측 클래스 이름, 신뢰도 점수)
        """
        # 이미지 로드 및 전처리
        try:
            image = Image.open(image_path).convert('RGB')
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # 예측
            with torch.no_grad():
                output = self.model(input_tensor)
                probabilities = torch.nn.functional.softmax(output, dim=1)[0]
                
            # 결과 해석
            pred_idx = torch.argmax(probabilities).item()
            pred_class = self.config['class_names'][pred_idx]
            confidence = probabilities[pred_idx].item() * 100
                
            return pred_idx, pred_class, confidence
            
        except Exception as e:
            print(f"이미지 분류 오류 ({image_path}): {e}")
            return None, None, None
    
    def predict_folder(self, folder_path, output_csv='results.csv'):
        """
        폴더 내 모든 이미지 분류
        
        Args:
            folder_path (str): 이미지 폴더 경로
            output_csv (str): 결과를 저장할 CSV 파일 경로
            
        Returns:
            dict: 분류 결과 (파일명: (클래스, 신뢰도))
        """
        results = {}
        image_files = []
        
        # 이미지 파일 찾기
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    image_files.append(os.path.join(root, file))
            
        print(f"총 {len(image_files)}개 이미지 분류 시작...")
        
        # 각 이미지 분류
        for image_path in tqdm(image_files):
            pred_idx, pred_class, confidence = self.predict_image(image_path)
            if pred_class:
                results[image_path] = (pred_class, confidence)
        
        # 결과 요약
        ok_count = sum(1 for _, (cls, _) in results.items() if cls == 'OK')
        ng_count = sum(1 for _, (cls, _) in results.items() if cls == 'NG')
        
        print(f"\n분류 결과:")
        print(f"OK: {ok_count}개 ({ok_count/len(results)*100:.1f}%)")
        print(f"NG: {ng_count}개 ({ng_count/len(results)*100:.1f}%)")
        
        # CSV 저장
        with open(output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['파일명', '분류 결과', '신뢰도(%)'])
            for image_path, (pred_class, confidence) in results.items():
                writer.writerow([image_path, pred_class, f"{confidence:.2f}"])
        print(f"결과가 {output_csv}에 저장되었습니다.")
            
        return results

# ========================================
# 🏢 건물 매칭 시스템
# ========================================

class BuildingMatcher:
    """탐지된 객체와 건물 옥상 좌표 매칭"""
    
    def __init__(
        self,
        buildings_info: Dict[int, Dict] = BUILDINGS_INFO if TARGET_GEOMETRY_IMPORTED else None,
        distance_threshold: float = MATCHING_DISTANCE_THRESHOLD
    ):
        """
        초기화
        
        Args:
            buildings_info: 건물 정보 딕셔너리
            distance_threshold: 매칭 판정 거리 임계값 (픽셀)
        """
        self.buildings_info = buildings_info
        self.distance_threshold = distance_threshold
        self.logger = logging.getLogger('BuildingMatcher')
        self.logger.info(f"건물 매칭기 초기화: 거리 임계값 = {distance_threshold}px")
    
    def calculate_distance(
        self,
        point1: Tuple[float, float],
        point2: Tuple[float, float]
    ) -> float:
        """두 점 사이의 유클리드 거리 계산"""
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def match_detection_to_building(
        self,
        detection_center: Tuple[float, float],
        rooftop_positions: Dict[int, Tuple[float, float]]
    ) -> Optional[Tuple[int, float]]:
        """
        탐지된 객체를 가장 가까운 건물에 매칭
        
        Args:
            detection_center: 탐지된 객체의 중심 좌표
            rooftop_positions: 건물 옥상 좌표 딕셔너리 {building_id: (x, y)}
        
        Returns:
            (building_id, distance) 또는 None (매칭 실패)
        """
        min_distance = float('inf')
        matched_building_id = None
        
        # 모든 건물과의 거리 계산
        distances = {}
        for building_id, rooftop_pos in rooftop_positions.items():
            distance = self.calculate_distance(detection_center, rooftop_pos)
            distances[building_id] = distance
            
            if distance < min_distance:
                min_distance = distance
                matched_building_id = building_id
        
        # 거리 정보 출력 (가까운 순으로 정렬)
        sorted_distances = sorted(distances.items(), key=lambda x: x[1])
        self.logger.info(f"  📏 건물별 거리 (가까운 순):")
        for bid, dist in sorted_distances[:5]:  # 상위 5개만
            status = "✅" if dist <= self.distance_threshold else "❌"
            self.logger.info(f"    {status} 건물 {bid}: {dist:.2f}px")
        
        # 거리 임계값 확인
        if min_distance <= self.distance_threshold:
            self.logger.info(
                f"  ✅ 매칭 성공: 건물 {matched_building_id} "
                f"(거리: {min_distance:.2f}px ≤ {self.distance_threshold}px)"
            )
            return (matched_building_id, min_distance)
        else:
            self.logger.warning(
                f"  ❌ 매칭 실패: 가장 가까운 건물 {matched_building_id}까지의 "
                f"거리({min_distance:.2f}px)가 임계값({self.distance_threshold}px)을 초과"
            )
            return None
    
    def match_all_detections(
        self,
        detection_result,
        rooftop_positions: Dict[int, Tuple[float, float]]
    ) -> List[Dict]:
        """
        모든 탐지 결과를 건물에 매칭 (같은 좌표계)
        
        Args:
            detection_result: YOLO 탐지 결과
            rooftop_positions: 건물 옥상 좌표 (같은 좌표계)
        
        Returns:
            매칭 결과 리스트
        """
        matches = []
        
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"🔍 총 {len(detection_result.detections)}개 탐지 객체 매칭 시작")
        self.logger.info(f"{'='*80}")
        self.logger.info(f"📍 건물 옥상 좌표 ({len(rooftop_positions)}개):")
        for bid, pos in rooftop_positions.items():
            self.logger.info(f"  건물 {bid}: ({pos[0]:.1f}, {pos[1]:.1f})")
        
        for idx, detection in enumerate(detection_result.detections, 1):
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"[{idx}/{len(detection_result.detections)}] 탐지 객체 매칭 중...")
            self.logger.info(f"  클래스: {detection.class_name}")
            self.logger.info(f"  신뢰도: {detection.confidence:.3f}")
            
            # 탐지 중심 좌표 (변환된 이미지 기준)
            center = (detection.bbox.center_x, detection.bbox.center_y)
            self.logger.info(f"  📍 탐지 중심: ({center[0]:.1f}, {center[1]:.1f})")
            
            # 건물 매칭
            match_result = self.match_detection_to_building(center, rooftop_positions)
            
            if match_result is not None:
                building_id, distance = match_result
                
                match_info = {
                    'detection': detection,
                    'building_id': building_id,
                    'distance': distance,
                    'center': center,
                    'bbox': (detection.bbox.x1, detection.bbox.y1, 
                            detection.bbox.x2, detection.bbox.y2),
                    'rooftop_position': rooftop_positions[building_id]
                }
                matches.append(match_info)
                
                self.logger.info(
                    f"  ✅ 최종 매칭: 건물 {building_id} (거리: {distance:.2f}px)"
                )
            else:
                self.logger.warning(f"  ❌ 매칭 실패: 임계값 내 건물 없음")
        
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"✅ 매칭 완료: {len(matches)}/{len(detection_result.detections)}개 성공")
        self.logger.info(f"{'='*80}\n")
        
        return matches

# ========================================
# 🚁 드론 통합 시스템
# ========================================

class DroneIntegratedSystem:
    """
    드론 이미지 분석 통합 시스템
    - ROI 자르기
    - 이미지 분류
    - YOLO 객체 탐지
    - 백업 ROI 분석
    """
    
    def __init__(
        self,
        yolo_classifier: YoloClassifier,
        roi_classifier_path: str,
        h_marker_template_path: str,
        buildings_info: Dict[int, Dict] = BUILDINGS_INFO if TARGET_GEOMETRY_IMPORTED else None,
        distance_threshold: float = MATCHING_DISTANCE_THRESHOLD,
        roi_config_path: str = None
    ):
        """
        드론 통합 시스템 초기화
        
        Args:
            yolo_classifier: YOLO 분류기 인스턴스
            roi_classifier_path: ROI 분류 모델 경로
            h_marker_template_path: H 마커 템플릿 이미지 경로
            buildings_info: 건물 정보
            distance_threshold: 건물 매칭 거리 임계값
            roi_config_path: ROI 분류기 설정 파일 경로
        """
        self.yolo_classifier = yolo_classifier
        self.buildings_info = buildings_info
        
        # H 마커 템플릿 로드
        self.h_template = cv2.imread(h_marker_template_path)
        if self.h_template is None:
            raise FileNotFoundError(f"H 마커 템플릿을 찾을 수 없습니다: {h_marker_template_path}")
        
        self.h_template_width = self.h_template.shape[1]
        
        # 건물 매칭기 초기화
        self.building_matcher = BuildingMatcher(
            buildings_info=buildings_info,
            distance_threshold=distance_threshold
        )
        
        # ROI 분류기 초기화
        self.roi_classifier = DroneImageClassifier(
            model_path=roi_classifier_path,
            config_path=roi_config_path
        )
        
        # 로깅 설정
        self.logger = logging.getLogger('DroneIntegratedSystem')
        self.logger.info("✅ 드론 통합 시스템 초기화 완료")
    
    def process_image(
        self,
        image_path: str,
        visualize: bool = True,
        save_results: bool = True,
        use_backup_roi: bool = True
    ) -> Dict:
        """
        드론 이미지 전체 처리 파이프라인
        
        Args:
            image_path: 입력 이미지 경로
            visualize: 시각화 여부
            save_results: 결과 저장 여부
            use_backup_roi: YOLO 실패 시 백업 ROI 분석 사용 여부
            
        Returns:
            처리 결과 딕셔너리
        """
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"🚁 드론 이미지 처리 시작: {image_path}")
        self.logger.info(f"{'='*80}\n")
        
        # 결과 딕셔너리 초기화
        result = {
            'success': False,
            'image_path': image_path,
            'h_marker_detected': False,
            'drone_altitude_m': None,
            'detection_result': None,
            'rooftop_positions': None,
            'matches': [],
            'roi_results': {},
            'error_message': None
        }
        
        # 임시 파일 경로
        temp_path = None
        roi_paths = {}
        
        try:
            # 1. 이미지 로드
            original_image = cv2.imread(image_path)
            if original_image is None:
                result['error_message'] = f"이미지 로드 실패: {image_path}"
                self.logger.error(result['error_message'])
                return result
            
            self.logger.info(f"📷 이미지 로드 완료: {original_image.shape}")
            
            # 2. H 마커 인식
            self.logger.info("\n🎯 H 마커 인식 중...")
            h_marker_params = recognize_rotated_h_marker_sift(
                original_image,
                self.h_template,
                min_match_count=10
            )
            
            if h_marker_params is None:
                result['error_message'] = "H 마커 인식 실패"
                self.logger.error(result['error_message'])
                return result
            
            result['h_marker_detected'] = True
            self.logger.info(f"✅ H 마커 인식 성공")
            
            # 3. 이미지 정렬 및 드론 고도 계산
            self.logger.info("\n🔄 이미지 정렬 중...")
            transformed_image, transformation_matrix = transform_image_from_params(
                original_image,
                h_marker_params
            )
            
            scale_factor = h_marker_params[3]
            drone_altitude = calculate_drone_altitude_from_scale_precise(
                scale_factor,
                H_MARKER_REAL_WIDTH_M,
                self.h_template_width,
                image_width_pixels=original_image.shape[1],
                fov_horizontal_deg=118
            )
            
            result['drone_altitude_m'] = drone_altitude
            self.logger.info(f"✅ 드론 고도: {drone_altitude:.2f}m")
            self.logger.info(f"✅ 변환된 이미지 크기: {transformed_image.shape}")
            
            # 4. 건물 옥상 좌표 계산 (변환된 이미지 기준)
            self.logger.info("\n🏢 건물 옥상 좌표 계산 중 (변환된 이미지 기준)...")
            rooftop_positions, _ = detect_all_building_rooftops(
                transformed_image,
                self.buildings_info,
                drone_altitude,
                transformation_matrix=transformation_matrix,
                original_image_shape=original_image.shape,
                visualize=False
            )
            
            result['rooftop_positions'] = rooftop_positions
            self.logger.info(f"✅ {len(rooftop_positions)}개 건물 옥상 좌표 계산 완료")
            
            # 5. ROI 자르기 (백업 분석용)
            if use_backup_roi:
                self.logger.info("\n✂️ ROI 자르기 중...")
                # 이미지 인덱스 생성 (파일명 기반)
                image_index = Path(image_path).stem
                
                # ROI 자르기
                roi_paths = crop_rois_for_classification(
                    transformed_image=transformed_image,
                    rooftop_positions=rooftop_positions,
                    image_index=image_index
                )
                self.logger.info(f"✅ {len(roi_paths)}개 ROI 생성 완료")
            
            # 6. 변환된 이미지를 임시 저장 (YOLO 분석용)
            temp_path = Path(image_path).parent / f"{Path(image_path).stem}_transformed_temp.jpg"
            cv2.imwrite(str(temp_path), transformed_image)
            self.logger.info(f"📁 변환된 이미지 임시 저장: {temp_path}")
            
            # 7. YOLO 객체 탐지 (변환된 이미지에서)
            self.logger.info("\n🤖 YOLO 객체 탐지 중 (변환된 이미지)...")
            detection_result = self.yolo_classifier.detect_from_file(
                str(temp_path),  # ← 변환된 이미지 사용!
                visualize=False,
                save_json=False
            )
            
            result['detection_result'] = detection_result
            self.logger.info(
                f"✅ 탐지 완료: {detection_result.detection_count}개 객체 "
                f"(성공: {detection_result.success})"
            )
            
            # 8. 탐지 객체와 건물 매칭 (같은 좌표계!)
            yolo_success = False
            if detection_result.success and detection_result.detection_count > 0:
                self.logger.info("\n🔗 건물 매칭 중 (같은 좌표계)...")
                matches = self.building_matcher.match_all_detections(
                    detection_result,
                    rooftop_positions  # 같은 좌표계!
                )
                
                result['matches'] = matches
                yolo_success = len(matches) > 0
                
                self.logger.info(f"✅ {len(matches)}개 객체 매칭 완료")
                
                # 매칭 결과 출력
                if matches:
                    self.logger.info("\n" + "="*80)
                    self.logger.info("📊 YOLO 매칭 결과:")
                    self.logger.info("="*80)
                    for i, match in enumerate(matches, 1):
                        self.logger.info(f"\n[{i}] 건물 {match['building_id']}")
                        self.logger.info(f"    - 클래스: {match['detection'].class_name}")
                        self.logger.info(f"    - 신뢰도: {match['detection'].confidence:.3f}")
                        self.logger.info(f"    - 거리: {match['distance']:.2f}px")
                        self.logger.info(f"    - 탐지 중심: ({match['center'][0]:.1f}, {match['center'][1]:.1f})")
                        self.logger.info(f"    - 옥상 좌표: ({match['rooftop_position'][0]:.1f}, {match['rooftop_position'][1]:.1f})")
                    self.logger.info("="*80)
            else:
                self.logger.warning("⚠️ YOLO 탐지 실패 또는 탐지된 객체가 없습니다.")
            
            # 9. 백업: ROI 분석 (YOLO 매칭 실패 시)
            if not yolo_success and use_backup_roi and roi_paths:
                self.logger.info("\n🔍 백업 ROI 분석 시작...")
                
                roi_results = {}
                for building_id, roi_path in roi_paths.items():
                    self.logger.info(f"건물 {building_id} ROI 분석 중...")
                    pred_idx, pred_class, confidence = self.roi_classifier.predict_image(roi_path)
                    
                    if pred_class:
                        roi_results[building_id] = {
                            'class': pred_class,
                            'confidence': confidence,
                            'roi_path': roi_path
                        }
                        self.logger.info(f"건물 {building_id}: {pred_class} ({confidence:.2f}%)")
                    else:
                        self.logger.warning(f"건물 {building_id} 분석 실패")
                
                result['roi_results'] = roi_results
                
                # OK/NG 개수 계산
                ok_count = sum(1 for r in roi_results.values() if r['class'] == 'OK')
                ng_count = sum(1 for r in roi_results.values() if r['class'] == 'NG')
                
                self.logger.info(f"✅ ROI 분석 결과: OK={ok_count}, NG={ng_count}")
                
                # ROI 분석 성공으로 처리
                result['success'] = len(roi_results) > 0
            else:
                # YOLO 매칭 결과로 성공 여부 결정
                result['success'] = yolo_success
            
            # 10. 시각화
            if visualize:
                self.logger.info("\n🎨 시각화 중...")
                vis_image = self._visualize_results(
                    transformed_image,
                    result,
                    transformation_matrix,
                    original_image.shape
                )
                
                if save_results:
                    output_path = Path(image_path).parent / f"{Path(image_path).stem}_result.jpg"
                    cv2.imwrite(str(output_path), vis_image)
                    self.logger.info(f"💾 결과 이미지 저장: {output_path}")
            
        except Exception as e:
            result['error_message'] = f"처리 중 오류 발생: {str(e)}"
            self.logger.error(result['error_message'])
            import traceback
            traceback.print_exc()
        
        finally:
            # 임시 파일 삭제
            if temp_path is not None and temp_path.exists():
                try:
                    temp_path.unlink()
                    self.logger.info(f"🗑️ 임시 파일 삭제: {temp_path}")
                except Exception as e:
                    self.logger.warning(f"⚠️ 임시 파일 삭제 실패: {e}")
        
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"✅ 처리 완료: {'성공' if result['success'] else '실패'}")
        self.logger.info(f"{'='*80}\n")
        
        return result
    
    def _visualize_results(
        self,
        transformed_image: np.ndarray,
        result: Dict,
        transformation_matrix: np.ndarray,
        original_shape: Tuple
    ) -> np.ndarray:
        """결과 시각화"""
        vis_image = transformed_image.copy()
        
        # 드론 위치 계산 (변환된 이미지에서)
        orig_height, orig_width = original_shape[:2]
        drone_point = np.array([[orig_width/2, orig_height/2]], dtype=np.float32).reshape(-1, 1, 2)
        drone_transformed = cv2.perspectiveTransform(drone_point, transformation_matrix)
        drone_x, drone_y = drone_transformed[0][0]
        
        # 드론 위치 표시
        cv2.drawMarker(vis_image, (int(drone_x), int(drone_y)),
                      (0, 255, 0), cv2.MARKER_CROSS, 50, 3)
        cv2.putText(vis_image, "DRONE", (int(drone_x) + 20, int(drone_y) - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 건물 옥상 표시 (빨간색 원)
        for building_id, roof_pos in result['rooftop_positions'].items():
            roof_pos_int = (int(roof_pos[0]), int(roof_pos[1]))
            
            # 건물 상태 색상 결정 (ROI 결과가 있는 경우)
            color = (0, 0, 255)  # 기본 빨간색
            label_text = f"B{building_id}"
            
            if building_id in result.get('roi_results', {}):
                roi_result = result['roi_results'][building_id]
                if roi_result['class'] == 'OK':
                    color = (0, 255, 0)  # OK는 초록색
                    label_text = f"B{building_id}: OK ({roi_result['confidence']:.1f}%)"
                else:
                    color = (0, 0, 255)  # NG는 빨간색
                    label_text = f"B{building_id}: NG ({roi_result['confidence']:.1f}%)"
            
            cv2.circle(vis_image, roof_pos_int, 10, color, -1)
            cv2.putText(vis_image, label_text,
                       (roof_pos_int[0] + 15, roof_pos_int[1]),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # YOLO 매칭된 객체 표시
        for match in result.get('matches', []):
            center = match['center']
            center_int = (int(center[0]), int(center[1]))
            bbox = match['bbox']
            
            # 바운딩 박스 (노란색)
            cv2.rectangle(vis_image, 
                         (int(bbox[0]), int(bbox[1])), 
                         (int(bbox[2]), int(bbox[3])),
                         (0, 255, 255), 3)  # BGR: 노란색
            
            # 중심점 (노란색 원)
            cv2.circle(vis_image, center_int, 8, (0, 255, 255), -1)
            
            # 건물 옥상과 연결선 (초록색)
            roof_pos = match['rooftop_position']
            roof_pos_int = (int(roof_pos[0]), int(roof_pos[1]))
            cv2.line(vis_image, center_int, roof_pos_int, (0, 255, 0), 2)
            
            # 라벨
            label = f"B{match['building_id']}: {match['detection'].class_name} ({match['detection'].confidence:.2f})"
            cv2.putText(vis_image, label,
                       (int(bbox[0]), int(bbox[1]) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # 거리 정보
            distance_text = f"{match['distance']:.1f}px"
            cv2.putText(vis_image, distance_text,
                       (center_int[0] - 30, center_int[1] + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        # 정보 패널
        info_y = 50
        cv2.putText(vis_image, f"Altitude: {result['drone_altitude_m']:.2f}m",
                   (50, info_y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        
        info_y += 50
        if result.get('detection_result'):
            cv2.putText(vis_image, f"YOLO: {result['detection_result'].detection_count} detections",
                       (50, info_y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        
        info_y += 50
        matches_count = len(result.get('matches', []))
        cv2.putText(vis_image, f"Matches: {matches_count}",
                   (50, info_y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        
        info_y += 50
        roi_results = result.get('roi_results', {})
        if roi_results:
            ok_count = sum(1 for r in roi_results.values() if r['class'] == 'OK')
            ng_count = sum(1 for r in roi_results.values() if r['class'] == 'NG')
            cv2.putText(vis_image, f"ROI: OK={ok_count}, NG={ng_count}",
                       (50, info_y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        
        return vis_image
    
    def process_batch(
        self,
        folder_path: str,
        output_csv: str = 'drone_analysis_results.csv',
        visualize: bool = True,
        save_results: bool = True,
        use_backup_roi: bool = True
    ) -> List[Dict]:
        """
        폴더 내 이미지 배치 처리
        
        Args:
            folder_path: 이미지 폴더 경로
            output_csv: 결과 CSV 파일 경로
            visualize: 시각화 여부
            save_results: 결과 저장 여부
            use_backup_roi: YOLO 실패 시 백업 ROI 분석 사용 여부
            
        Returns:
            list: 처리 결과 리스트
        """
        self.logger.info(f"배치 처리 시작: {folder_path}")
        
        # 이미지 파일 찾기
        image_files = []
        for file_ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            image_files.extend(list(Path(folder_path).glob(f"*{file_ext}")))
            image_files.extend(list(Path(folder_path).glob(f"*{file_ext.upper()}")))
        
        # _result.jpg, _transformed_temp.jpg 제외
        image_files = [f for f in image_files 
                      if not f.stem.endswith('_result') 
                      and not f.stem.endswith('_transformed_temp')]
        
        self.logger.info(f"총 {len(image_files)}개 이미지 처리 시작...")
        
        # 결과 저장
        results = []
        
        # 각 이미지 처리
        for idx, image_path in enumerate(image_files, 1):
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"[{idx}/{len(image_files)}] 처리 중: {image_path.name}")
            self.logger.info(f"{'='*60}")
            
            try:
                # 이미지 처리
                result = self.process_image(
                    str(image_path),
                    visualize=visualize,
                    save_results=save_results,
                    use_backup_roi=use_backup_roi
                )
                results.append(result)
                
                # 진행 상황 표시
                if result['success']:
                    if result.get('matches'):
                        self.logger.info(f"✅ YOLO 매칭 성공: {len(result['matches'])}개 매칭")
                    elif result.get('roi_results'):
                        ok_count = sum(1 for r in result['roi_results'].values() if r['class'] == 'OK')
                        ng_count = sum(1 for r in result['roi_results'].values() if r['class'] == 'NG')
                        self.logger.info(f"✅ ROI 분석 성공: OK={ok_count}, NG={ng_count}")
                else:
                    self.logger.error(f"❌ 처리 실패: {result.get('error_message', '알 수 없는 오류')}")
                    
            except Exception as e:
                self.logger.error(f"❌ 오류 발생: {e}")
                import traceback
                traceback.print_exc()
                results.append({
                    'success': False,
                    'image_path': str(image_path),
                    'error_message': str(e)
                })
        
        # 결과 요약
        success_count = sum(1 for r in results if r['success'])
        yolo_success = sum(1 for r in results if r.get('matches'))
        roi_success = sum(1 for r in results if not r.get('matches') and r.get('roi_results'))
        
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"📊 배치 처리 요약")
        self.logger.info(f"{'='*80}")
        self.logger.info(f"총 이미지: {len(results)}개")
        self.logger.info(f"성공: {success_count}개 ({success_count/len(results)*100:.1f}%)")
        self.logger.info(f"- YOLO 성공: {yolo_success}개")
        self.logger.info(f"- ROI 백업 성공: {roi_success}개")
        self.logger.info(f"실패: {len(results) - success_count}개")
        self.logger.info(f"{'='*80}")
        
        # CSV 저장
        self.save_results_to_csv(results, output_csv)
        
        return results
    
    def save_results_to_csv(self, results, output_csv):
        """결과를 CSV 파일로 저장"""
        with open(output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # 헤더 작성
            header = ['이미지 경로', '처리 성공', 'H마커 인식', '드론 고도(m)', 'YOLO 매칭 수']
            for i in range(1, 10):  # 건물 1~9
                header.extend([f'건물{i} YOLO클래스', f'건물{i} YOLO신뢰도', f'건물{i} ROI클래스', f'건물{i} ROI신뢰도'])
            writer.writerow(header)
            
            # 데이터 작성
            for result in results:
                # 기본 정보
                row = [
                    result['image_path'],
                    result['success'],
                    result.get('h_marker_detected', False),
                    f"{result.get('drone_altitude_m', 0):.2f}" if result.get('drone_altitude_m') else 'N/A',
                    len(result.get('matches', []))
                ]
                
                # 각 건물별 결과 추가
                for building_id in range(1, 10):
                    # YOLO 매칭 결과
                    yolo_match = next((m for m in result.get('matches', []) if m['building_id'] == building_id), None)
                    if yolo_match:
                        row.extend([
                            yolo_match['detection'].class_name,
                            f"{yolo_match['detection'].confidence:.2f}"
                        ])
                    else:
                        row.extend(['N/A', 'N/A'])
                    
                    # ROI 분석 결과
                    roi_result = result.get('roi_results', {}).get(building_id)
                    if roi_result:
                        row.extend([
                            roi_result['class'],
                            f"{roi_result['confidence']:.2f}"
                        ])
                    else:
                        row.extend(['N/A', 'N/A'])
                
                writer.writerow(row)
        
        self.logger.info(f"결과가 {output_csv}에 저장되었습니다.")

# ========================================
# 🎯 메인 함수
# ========================================

def main():
    """메인 함수"""
    
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger('main')
    logger.info("🚁 드론 통합 시스템 시작")
    
    # 설정 출력
    logger.info(f"⚙️ 설정:")
    logger.info(f"  - 원본 이미지 크기: {ORIGINAL_IMAGE_WIDTH}x{ORIGINAL_IMAGE_HEIGHT}")
    logger.info(f"  - YOLO 입력 크기: {YOLO_INPUT_SIZE}x{YOLO_INPUT_SIZE}")
    logger.info(f"  - 매칭 거리 임계값: {MATCHING_DISTANCE_THRESHOLD}px")
    logger.info(f"  - ROI 크기: {CROP_SIZE_PIXELS}x{CROP_SIZE_PIXELS}px")
    logger.info(f"  - ROI 저장 경로: {OUTPUT_BASE_DIR}")
    
    try:
        # 1. YOLO 분류기 초기화
        yolo_classifier = YoloClassifier(
            model_path='runs/detect/drone_yolov8s/weights/best.pt',
            device=0,
            confidence_threshold=0.5,
            required_count=2,
            save_outputs=False
        )
        
        # 2. 통합 시스템 초기화
        drone_system = DroneIntegratedSystem(
            yolo_classifier=yolo_classifier,
            roi_classifier_path='best_drone_model.pth',
            h_marker_template_path='src/drone/h_template.png',
            buildings_info=BUILDINGS_INFO,
            distance_threshold=MATCHING_DISTANCE_THRESHOLD
        )
        
        # 3. 명령줄 인수 확인
        if len(sys.argv) > 1:
            # 단일 이미지 처리
            if os.path.isfile(sys.argv[1]):
                image_path = sys.argv[1]
                logger.info(f"단일 이미지 처리: {image_path}")
                
                result = drone_system.process_image(
                    image_path,
                    visualize=True,
                    save_results=True,
                    use_backup_roi=True
                )
                
                # 결과 요약 출력
                if result['success']:
                    if result.get('matches'):
                        logger.info(f"✅ YOLO 매칭 성공: {len(result['matches'])}개 매칭")
                    elif result.get('roi_results'):
                        ok_count = sum(1 for r in result['roi_results'].values() if r['class'] == 'OK')
                        ng_count = sum(1 for r in result['roi_results'].values() if r['class'] == 'NG')
                        logger.info(f"✅ ROI 분석 성공: OK={ok_count}, NG={ng_count}")
                else:
                    logger.error(f"❌ 처리 실패: {result.get('error_message', '알 수 없는 오류')}")
            
            # 폴더 처리
            elif os.path.isdir(sys.argv[1]):
                folder_path = sys.argv[1]
                output_csv = sys.argv[2] if len(sys.argv) > 2 else 'drone_analysis_results.csv'
                
                logger.info(f"폴더 처리: {folder_path}")
                drone_system.process_batch(
                    folder_path,
                    output_csv=output_csv,
                    visualize=True,
                    save_results=True,
                    use_backup_roi=True
                )
            
            else:
                logger.error(f"오류: 파일 또는 폴더가 아닙니다: {sys.argv[1]}")
        
        # 기본 테스트 폴더 처리
        else:
            test_dir = 'src/drone/temp'
            logger.info(f"기본 테스트 폴더 처리: {test_dir}")
            
            if os.path.exists(test_dir):
                drone_system.process_batch(
                    test_dir,
                    output_csv='drone_analysis_results.csv',
                    visualize=True,
                    save_results=True,
                    use_backup_roi=True
                )
            else:
                logger.error(f"테스트 디렉토리를 찾을 수 없습니다: {test_dir}")
    
    except Exception as e:
        logger.error(f"오류 발생: {e}")
        import traceback
        traceback.print_exc()
    
    logger.info("✅ 드론 통합 시스템 종료")

if __name__ == '__main__':
    main()

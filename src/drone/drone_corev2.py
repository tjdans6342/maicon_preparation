"""
드론 핵심 통합 모듈 - 좌표계 일치 버전
변환된 이미지에서 YOLO 검출하여 좌표계 통일
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

# 로컬 모듈 임포트
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
from yolo_classfier import DroneClassifier, DetectionResult

# ========================================
# ⚙️ 전역 설정
# ========================================

# 건물 매칭 거리 임계값 (픽셀)
MATCHING_DISTANCE_THRESHOLD = 200.0  # 기본값 200px

# 원본 이미지 크기
ORIGINAL_IMAGE_WIDTH = 4000
ORIGINAL_IMAGE_HEIGHT = 3000

# YOLO 입력 크기
YOLO_INPUT_SIZE = 640


# ========================================
# 🏢 건물 매칭 시스템
# ========================================

class BuildingMatcher:
    """탐지된 객체와 건물 옥상 좌표 매칭"""
    
    def __init__(
        self,
        buildings_info: Dict[int, Dict] = BUILDINGS_INFO,
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
        detection_result: DetectionResult,
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
# 🚁 드론 핵심 통합 클래스
# ========================================

class DroneCore:
    """드론 이미지 분석 통합 시스템"""
    
    def __init__(
        self,
        classifier: DroneClassifier,
        h_marker_template_path: str,
        buildings_info: Dict[int, Dict] = BUILDINGS_INFO,
        distance_threshold: float = MATCHING_DISTANCE_THRESHOLD
    ):
        """
        초기화
        
        Args:
            classifier: DroneClassifier 인스턴스
            h_marker_template_path: H 마커 템플릿 이미지 경로
            buildings_info: 건물 정보
            distance_threshold: 건물 매칭 거리 임계값
        """
        self.classifier = classifier
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
        
        self.logger = logging.getLogger('DroneCore')
        self.logger.info("✅ DroneCore 초기화 완료")
    
    def process_image(
        self,
        image_path: str,
        visualize: bool = True,
        save_results: bool = True
    ) -> Dict:
        """
        드론 이미지 전체 처리 파이프라인
        
        Args:
            image_path: 입력 이미지 경로
            visualize: 시각화 여부
            save_results: 결과 저장 여부
        
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
            'error_message': None
        }
        
        # 임시 파일 경로
        temp_path = None
        
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
            
            # 5. 변환된 이미지를 임시 저장 ⭐
            temp_path = Path(image_path).parent / f"{Path(image_path).stem}_transformed_temp.jpg"
            cv2.imwrite(str(temp_path), transformed_image)
            self.logger.info(f"📁 변환된 이미지 임시 저장: {temp_path}")
            
            # 6. YOLO 객체 탐지 (변환된 이미지에서) ⭐
            self.logger.info("\n🤖 YOLO 객체 탐지 중 (변환된 이미지)...")
            detection_result = self.classifier.detect_from_file(
                str(temp_path),  # ← 변환된 이미지 사용!
                visualize=False,
                save_json=False
            )
            
            result['detection_result'] = detection_result
            self.logger.info(
                f"✅ 탐지 완료: {detection_result.detection_count}개 객체 "
                f"(성공: {detection_result.success})"
            )
            
            # 7. 탐지 객체와 건물 매칭 (같은 좌표계!) ⭐
            if detection_result.success and detection_result.detection_count > 0:
                self.logger.info("\n🔗 건물 매칭 중 (같은 좌표계)...")
                matches = self.building_matcher.match_all_detections(
                    detection_result,
                    rooftop_positions  # 같은 좌표계!
                )
                
                result['matches'] = matches
                result['success'] = len(matches) > 0
                
                self.logger.info(f"✅ {len(matches)}개 객체 매칭 완료")
                
                # 매칭 결과 출력
                if matches:
                    self.logger.info("\n" + "="*80)
                    self.logger.info("📊 매칭 결과:")
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
                self.logger.warning("⚠️ 탐지된 객체가 없어 매칭을 수행하지 않습니다.")
            
            # 8. 시각화
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
            cv2.circle(vis_image, roof_pos_int, 10, (0, 0, 255), -1)  # BGR: 빨간색
            cv2.putText(vis_image, f"B{building_id}",
                       (roof_pos_int[0] + 15, roof_pos_int[1]),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # 매칭된 객체 표시
        for match in result['matches']:
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
        cv2.putText(vis_image, f"Detections: {result['detection_result'].detection_count}",
                   (50, info_y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        
        info_y += 50
        cv2.putText(vis_image, f"Matches: {len(result['matches'])}",
                   (50, info_y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        
        return vis_image


# ========================================
# 🎯 사용 예시
# ========================================

def main():
    """메인 함수"""
    
    print("🚁 드론 통합 시스템 테스트 (좌표계 일치 버전)\n")
    print(f"⚙️ 설정:")
    print(f"  - 원본 이미지 크기: {ORIGINAL_IMAGE_WIDTH}x{ORIGINAL_IMAGE_HEIGHT}")
    print(f"  - YOLO 입력 크기: {YOLO_INPUT_SIZE}x{YOLO_INPUT_SIZE}")
    print(f"  - 매칭 거리 임계값: {MATCHING_DISTANCE_THRESHOLD}px")
    print(f"  - 처리 방식: 변환된 이미지로 YOLO 검출 ✅\n")
    
    # 1. DroneClassifier 초기화
    classifier = DroneClassifier(
        model_path='runs/detect/drone_yolov8s/weights/best.pt',
        device=0,
        confidence_threshold=0.5,
        required_count=2,
        save_outputs=False
    )
    
    # 2. DroneCore 초기화
    drone_core = DroneCore(
        classifier=classifier,
        h_marker_template_path=r'src/drone/h_template.png',
        buildings_info=BUILDINGS_INFO,
        distance_threshold=MATCHING_DISTANCE_THRESHOLD
    )
    
    # 3. 배치 처리
    print("\n" + "="*80)
    print("테스트: 배치 처리")
    print("="*80)
    
    test_dir = Path('src/drone/temp')
    if test_dir.exists():
        image_files = list(test_dir.glob('*.jpg'))
        # _result.jpg, _transformed_temp.jpg 제외
        image_files = [f for f in image_files 
                      if not f.stem.endswith('_result') 
                      and not f.stem.endswith('_transformed_temp')]
        image_files = image_files[:240]  # 최대 240개
        
        if len(image_files) == 0:
            print("⚠️ 처리할 이미지가 없습니다.")
        else:
            batch_results = []
            for img_path in image_files:
                print(f"\n{'='*60}")
                print(f"처리 중: {img_path.name}")
                print(f"{'='*60}")
                try:
                    result = drone_core.process_image(
                        str(img_path),
                        visualize=True,
                        save_results=True
                    )
                    batch_results.append(result)
                    
                    # 간단한 결과 출력
                    if result['success']:
                        print(f"✅ 성공: {len(result['matches'])}개 매칭")
                    else:
                        print(f"❌ 실패: {result.get('error_message', '알 수 없는 오류')}")
                        
                except Exception as e:
                    print(f"❌ 오류 발생: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            # 배치 요약
            if len(batch_results) > 0:
                success_count = sum(1 for r in batch_results if r['success'])
                total_matches = sum(len(r['matches']) for r in batch_results if r['success'])
                
                print("\n" + "="*80)
                print("📊 배치 처리 요약")
                print("="*80)
                print(f"총 이미지: {len(batch_results)}개")
                print(f"성공: {success_count}개")
                print(f"실패: {len(batch_results) - success_count}개")
                print(f"성공률: {success_count/len(batch_results)*100:.1f}%")
                print(f"총 매칭: {total_matches}개")
                if success_count > 0:
                    print(f"평균 매칭: {total_matches/success_count:.1f}개/이미지")
                print("="*80)
            else:
                print("⚠️ 처리된 이미지가 없습니다.")
    else:
        print(f"⚠️ 테스트 디렉토리를 찾을 수 없습니다: {test_dir}")
    
    print("\n✅ 테스트 완료!")


if __name__ == '__main__':
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,  # INFO <-> DEBUG 변경가능
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    main()

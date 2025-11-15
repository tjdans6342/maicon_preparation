import cv2
import numpy as np
import os
import sys # 명령줄 인수를 처리하기 위해 추가

# ==============================================================================
# 상수 정의
# ==============================================================================

CROP_SIZE_PIXELS = 250  # 잘라낼 ROI의 크기
OUTPUT_BASE_DIR = "src/drone/second" # 저장할 기본 디렉토리

# ==============================================================================
# ROI 자르기 및 저장 함수
# ==============================================================================

def crop_rois_for_classification(
    transformed_image: np.ndarray,
    rooftop_positions: dict[int, tuple[float, float]],
    image_index: str, # ✅ 이미지 인덱스 추가
    crop_size: int = CROP_SIZE_PIXELS,
    output_base_dir: str = OUTPUT_BASE_DIR
) -> None:
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
    """
    
    # 이미지 크기
    h, w = transformed_image.shape[:2]
    half_size = crop_size // 2
    
    # 기본 출력 디렉토리 생성
    if not os.path.exists(output_base_dir):
        os.makedirs(output_base_dir)
        print(f"기본 디렉토리 생성: {output_base_dir}")

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


# ==============================================================================
# 배치 실행을 위한 메인 블록 수정
# ==============================================================================

if __name__ == "__main__":
    # 1. 명령줄 인수 처리
    if len(sys.argv) < 3:
        # 인수가 부족하면 테스트용 하드코딩 값 사용
        print("경고: 명령줄 인수가 부족합니다. 테스트용 하드코딩 값을 사용합니다.")
        input_image_path = 'src/drone/originals.png'
        image_index = 'test_00' # 테스트용 인덱스
    else:
        # 명령줄 인수 사용: python script_name.py <이미지경로> <인덱스>
        input_image_path = sys.argv[1]
        image_index = sys.argv[2]
    
    # 2. 이미지 로드
    second_image = cv2.imread(input_image_path)
    
    if second_image is None:
        print(f"오류: 이미지를 로드할 수 없습니다. 경로를 확인하세요: {input_image_path}")
        sys.exit(1)

    # 3. 옥상 좌표 (정렬된 이미지 기준, 고정 값)
    second_rooftop_positions = {
        1: (1659.0, 955.0),
        2: (2023.0, 1038.0),
        3: (2593.0, 1048.0),
        4: (3373.0, 832.0),
        5: (2585.0, 1668.0),
        6: (1985.0, 1660.0),
        7: (829.0, 1956.0),
        8: (2021.0, 2160.0),
        9: (2935.0, 2178.0),
    }
    
    # 4. 함수 실행
    print(f"=== ROI 자르기 시작 (입력: {input_image_path}, 인덱스: {image_index}) ===")
    crop_rois_for_classification(
        transformed_image=second_image,
        rooftop_positions=second_rooftop_positions,
        image_index=image_index # ✅ 인덱스 전달
    )
    print("=== ROI 자르기 완료 ===")

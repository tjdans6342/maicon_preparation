import cv2
import numpy as np
import math

TARGET_X=1073 # H마커가 이동할 위치 지정
TARGET_Y=1105

H_MARKER_REAL_WIDTH_M = 0.39 # H 마커의 현실 사이즈(m단위)

BUILDINGS_INFO = {
            1: {'ground_pos': (1763, 1120), 'height_m': 0.728},
            2: {'ground_pos': (2035, 1059), 'height_m': 0.242},
            3: {'ground_pos': (2548, 1069), 'height_m': 0.198},
            4: {'ground_pos': (2798, 1117), 'height_m': 1.051},
            5: {'ground_pos': (2611, 1663), 'height_m': 0.0},
            6: {'ground_pos': (2011, 1660), 'height_m': 0.0},
            7: {'ground_pos': (1208, 1791), 'height_m': 0.809},
            8: {'ground_pos': (2010, 1943), 'height_m': 0.810},
            9: {'ground_pos': (2641, 1945), 'height_m': 0.792},
        } #  각 건물 바닥의 중심 좌표(pixel), 높이(m)

def recognize_rotated_h_marker_sift(scene_image: np.ndarray, template_image: np.ndarray, min_match_count: int = 10) -> np.ndarray | None:
    """
    SIFT 특징점 매칭과 호모그래피를 사용하여 회전 및 스케일이 변형된 'H' 마커를 인식하고,
    마커의 중심 좌표, 회전각, 스케일 비율을 반환합니다.

    :param scene_image: 'H' 마커가 포함된 경기장 이미지 객체 (NumPy 배열, Scene)
    :param template_image: 'H' 모양의 템플릿 이미지 객체 (NumPy 배열, Object)
    :param min_match_count: 마커로 인식하기 위해 필요한 최소 매칭 쌍 개수
    :return: [중심 X, 중심 Y, 회전각(도), 스케일 비율] 배열 또는 인식 실패 시 None
    """
    
    # 1. 이미지 유효성 검사 및 그레이스케일 변환
    # 입력 이미지가 이미 그레이스케일이 아닐 경우를 대비하여 변환합니다.
    if scene_image is None or template_image is None:
        print("오류: 입력 이미지 객체가 None입니다.")
        return None

    # 그레이스케일 변환 (이미지 객체는 BGR 또는 RGB일 수 있음)
    if len(scene_image.shape) == 3:
        img_scene = cv2.cvtColor(scene_image, cv2.COLOR_BGR2GRAY)
    else:
        img_scene = scene_image
        
    if len(template_image.shape) == 3:
        img_object = cv2.cvtColor(template_image, cv2.COLOR_BGR2GRAY)
    else:
        img_object = template_image

    # 2. SIFT 검출기 초기화 및 특징점/기술자 계산
    sift = cv2.SIFT_create()
    kp_obj, des_obj = sift.detectAndCompute(img_object, None)
    kp_scene, des_scene = sift.detectAndCompute(img_scene, None)
    
    # 특징점 추출 실패 또는 부족 검사
    if des_obj is None or des_scene is None or len(kp_obj) < min_match_count or len(kp_scene) < min_match_count:
        # print(f"특징점 부족: 템플릿 {len(kp_obj)}, 장면 {len(kp_scene)}")
        return None

    # 3. 특징점 매칭 (BFMatcher, NORM_L2 사용)
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(des_obj, des_scene)
    matches = sorted(matches, key = lambda x:x.distance)

    # 4. 호모그래피 계산 및 마커 인식
    if len(matches) >= min_match_count:
        # 매칭된 특징점 좌표 추출
        src_pts = np.float32([ kp_obj[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp_scene[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)

        # RANSAC을 사용하여 호모그래피 행렬 계산
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        if M is not None:
            # 템플릿 이미지의 네 모서리 좌표 정의
            h, w = img_object.shape
            # 템플릿의 중심을 계산하는 데 사용할 네 모서리 외에, 
            # 회전각 계산을 위해 템플릿의 중심점도 변환에 포함하는 것이 더 정확할 수 있습니다.
            # 하지만 여기서는 기존 로직대로 네 모서리만 사용합니다.
            pts = np.float32([ [0,0],[w-1,0],[w-1,h-1],[0,h-1] ]).reshape(-1,1,2)
            
            # 호모그래피 변환을 통해 장면 이미지에서의 마커 위치(네 모서리) 계산
            dst = cv2.perspectiveTransform(pts, M)

            # 5. 중심 좌표 계산
            # 변환된 네 모서리 좌표의 평균을 사용
            center_x = np.mean(dst[:, 0, 0])
            center_y = np.mean(dst[:, 0, 1])
            
            # 6. 회전각 계산
            # 상단 변 (dst[0] -> dst[1])을 사용하여 각도 계산
            # 이 방법은 마커가 심하게 원근 변형되었을 경우 정확하지 않을 수 있습니다.
            # 하지만 현재는 아핀 변환을 가정하고 있으므로 사용합니다.
            x1, y1 = dst[0, 0]
            x2, y2 = dst[1, 0]
            
            angle_rad = math.atan2(y2 - y1, x2 - x1)
            angle_deg = math.degrees(angle_rad)
            
            # 7. 스케일 비율 계산
            template_width = w 
            scene_width = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            scale_factor = scene_width / template_width
            
            # print(f"회전된 'H' 마커 인식 성공! (SIFT 사용) 중심: ({center_x:.2f}, {center_y:.2f}), 회전각: {angle_deg:.2f}도, 스케일: {scale_factor:.2f}")
            
            # [중심 X, 중심 Y, 회전각(도), 스케일 비율] 반환
            return np.array([center_x, center_y, angle_deg, scale_factor])
    
    # 인식 실패 시
    # print(f"회전된 'H' 마커 인식 실패. 매칭 개수 부족: {len(matches)}/{min_match_count}")
    return None

def transform_image_from_params(
    scene_image: np.ndarray,
    transform_params: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    이미지 경로와 [center_x, center_y, angle_deg, scale_factor] 배열을 받아 이미지를 변환합니다.
    변환된 이미지는 감지된 중심이 TARGET_X, TARGET_Y로 이동하도록 배치됩니다.
    출력 이미지 크기는 원본 이미지 크기(w, h)로 유지됩니다.
    
    Args:
        scene_image (np.ndarray): 변환할 이미지 (OpenCV 이미지 객체).
        transform_params (np.ndarray): [center_x, center_y, angle_deg, scale_factor] 형태의 NumPy 배열.

    Returns:
        tuple[np.ndarray, np.ndarray]: (변환된 이미지, 변환 행렬 M)
    """
    # 1. 이미지 로드
    img = scene_image
    if img is None:
        raise FileNotFoundError(f"이미지 파일을 로드할 수 없습니다")

    # 2. 파라미터 추출 및 유효성 검사
    if transform_params.shape != (4,):
        raise ValueError("transform_params는 4개의 요소를 가진 배열이어야 합니다.")

    center_x = transform_params[0]
    center_y = transform_params[1]
    angle_deg = transform_params[2]
    
    # 감지된 스케일의 역수를 취하여 원본 크기로 복원 (확대/축소)
    scale = 1.0 / transform_params[3] 
    
    h, w = img.shape[:2]
    
    # 3. 변환 행렬 M 계산
    
    # 목표: H 마크 중심 (center_x, center_y)을 기준으로 angle_deg 회전 및 scale 적용 후, 
    # 최종적으로 (TARGET_X, TARGET_Y)로 이동시키는 변환 행렬 M을 생성합니다.
    
    # 3-1. 회전/스케일 변환 행렬 M_rot (center_x, center_y를 기준으로 회전)
    # cv2.getRotationMatrix2D(center, angle, scale) 함수는 내부적으로 다음을 수행합니다:
    # 1. center를 원점으로 이동 (Translation)
    # 2. 원점에서 회전 및 스케일 적용 (Rotation + Scale)
    # 3. 원점을 다시 center로 이동 (Inverse Translation)
    M_rot = cv2.getRotationMatrix2D((center_x, center_y), angle_deg, scale)
    
    # 3-2. 평행 이동(Translation) 성분 계산
    # M_rot의 현재 평행 이동 성분 (tx_rot, ty_rot)은 회전 중심을 기준으로 계산되어 있습니다.
    # 우리는 H 마크의 최종 위치가 (TARGET_X, TARGET_Y)가 되도록 추가적인 평행 이동을 적용해야 합니다.
    
    # M_rot을 적용했을 때 (center_x, center_y)가 이동하는 최종 위치 (x', y')를 계산합니다.
    # [x'] = [ M_rot[0, 0]  M_rot[0, 1] ] [center_x] + [M_rot[0, 2]]
    # [y'] = [ M_rot[1, 0]  M_rot[1, 1] ] [center_y] + [M_rot[1, 2]]
    
    # OpenCV 행렬 곱셈을 이용한 계산
    # M_rot의 회전/스케일 부분: M_rot[:, :2]
    # M_rot의 평행 이동 부분: M_rot[:, 2]
    
    # H 마크 중심이 M_rot에 의해 이동하는 위치 (x_prime, y_prime)
    x_prime = M_rot[0, 0] * center_x + M_rot[0, 1] * center_y + M_rot[0, 2]
    y_prime = M_rot[1, 0] * center_x + M_rot[1, 1] * center_y + M_rot[1, 2]
    
    # 목표 위치 (TARGET_X, TARGET_Y)와 현재 위치 (x_prime, y_prime)의 차이가 최종 변위입니다.
    tx_final = TARGET_X - x_prime
    ty_final = TARGET_Y - y_prime
    
    # 3-3. M_rot 행렬에 최종 평행 이동 성분 추가
    M_rot[0, 2] += tx_final
    M_rot[1, 2] += ty_final
    
    M = M_rot # 최종 변환 행렬 (2x3)

    # 4. 아핀 변환 적용 및 흰색 배경 채우기
    white_color = (255, 255, 255) # BGR 포맷의 흰색
    transformed_img = cv2.warpAffine(
        img,
        M,
        (w, h),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=white_color
    )

    # ✅ 변환된 이미지와 변환 행렬을 함께 반환
    # cv2.perspectiveTransform을 사용하려면 3x3 행렬이 필요하므로 변환
    M_3x3 = np.vstack([M, [0, 0, 1]])  # 2x3 -> 3x3 변환
    
    return transformed_img, M_3x3

def calculate_drone_altitude_from_scale_precise(
    scale_factor: float,
    template_real_width_m: float,
    template_pixel_width: int,
    image_width_pixels: int = 4000,
    fov_horizontal_deg: float = 118
) -> float:
    """
    물리 기반 정밀 고도 계산
    """
    import math
    
    # 1. 이미지에서 H 마커가 차지하는 실제 픽셀 크기
    detected_marker_pixel_width = scale_factor * template_pixel_width
    
    # 2. 초점 거리 계산 (픽셀 단위)
    fov_rad = math.radians(fov_horizontal_deg)
    focal_length_pixels = (image_width_pixels / 2.0) / math.tan(fov_rad / 2.0)
    
    # 3. 원근 투영 공식
    altitude = (template_real_width_m * focal_length_pixels) / detected_marker_pixel_width
    
    return altitude

def calculate_building_rooftop_position(
    building_ground_pos: tuple[float, float],
    building_height_m: float,
    drone_altitude_m: float,
    target_pos: tuple[float, float] = (TARGET_X, TARGET_Y)
) -> tuple[float, float]:
    """
    정렬된 이미지에서 건물 옥상의 2D 좌표를 계산합니다.
    """
    x_ground, y_ground = building_ground_pos
    target_x, target_y = target_pos
    
    # 드론이 건물보다 낮으면 옥상을 볼 수 없음
    if drone_altitude_m <= building_height_m:
        print(f"경고: 드론 고도({drone_altitude_m:.2f}m)가 건물 높이({building_height_m:.2f}m)보다 낮거나 같습니다.")
        return None
    
    # 원근 투영 공식
    ratio = drone_altitude_m / (drone_altitude_m - building_height_m)  # ratio > 1
    
    x_roof = target_x + (x_ground - target_x) * ratio
    y_roof = target_y + (y_ground - target_y) * ratio
    
    return (x_roof, y_roof)

def detect_all_building_rooftops(
    transformed_image: np.ndarray,
    buildings_info: dict[int, dict],
    drone_altitude_m: float,
    transformation_matrix: np.ndarray,  # ✅ 변환 행렬 추가
    original_image_shape: tuple,  # ✅ 원본 이미지 크기 추가
    visualize: bool = True
) -> dict[int, tuple[float, float]]:
    """
    모든 건물의 옥상 위치를 계산하고 선택적으로 시각화합니다.
    """
    # ✅ 1. 원본 이미지의 중심 좌표 (드론 위치)
    orig_height, orig_width = original_image_shape[:2]
    drone_x_orig = orig_width / 2.0
    drone_y_orig = orig_height / 2.0
    
    # ✅ 2. 변환 행렬을 사용해 정렬된 이미지에서의 드론 위치 계산
    drone_point = np.array([[drone_x_orig, drone_y_orig]], dtype=np.float32).reshape(-1, 1, 2)
    drone_transformed = cv2.perspectiveTransform(drone_point, transformation_matrix)
    target_x, target_y = drone_transformed[0][0]
    
    print(f"원본 이미지 드론 위치: ({drone_x_orig:.2f}, {drone_y_orig:.2f})")
    print(f"변환된 이미지 드론 위치: ({target_x:.2f}, {target_y:.2f})")
    
    rooftop_positions = {}
    
    # 3. 각 건물의 옥상 위치 계산
    for building_id, info in buildings_info.items():
        rooftop_pos = calculate_building_rooftop_position(
            info['ground_pos'],
            info['height_m'],
            drone_altitude_m,
            target_pos=(target_x, target_y)  # ✅ 변환된 드론 위치 사용
        )
        
        if rooftop_pos is not None:
            rooftop_positions[building_id] = rooftop_pos
    
    # 4. 시각화
    if visualize:
        vis_image = transformed_image.copy()
        
        # 드론 위치 표시 (녹색 십자)
        cv2.drawMarker(vis_image, (int(target_x), int(target_y)), 
                      (0, 255, 0), cv2.MARKER_CROSS, 50, 3)
        cv2.putText(vis_image, "DRONE", (int(target_x) + 20, int(target_y) - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 건물 정보 표시
        for building_id, info in buildings_info.items():
            ground_pos = info['ground_pos']
            
            # 지면 위치 (빨간색)
            cv2.circle(vis_image, ground_pos, 8, (0, 0, 255), -1)
            cv2.putText(vis_image, f"B{building_id}", 
                       (ground_pos[0] + 15, ground_pos[1]),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # 옥상 위치 (파란색)
            if building_id in rooftop_positions:
                roof_pos = rooftop_positions[building_id]
                roof_pos_int = (int(roof_pos[0]), int(roof_pos[1]))
                cv2.circle(vis_image, roof_pos_int, 8, (255, 0, 0), -1)
                
                # 지면-옥상 연결선
                cv2.line(vis_image, ground_pos, roof_pos_int, (255, 255, 0), 2)
        
        # 고도 정보 표시
        cv2.putText(vis_image, f"Altitude: {drone_altitude_m:.2f}m",
                   (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        
        return rooftop_positions, vis_image
    
    return rooftop_positions, None


# ============== 사용 예시 ==============
if __name__ == "__main__":
    # 1. H 마커 인식 및 이미지 정렬
    template_image = cv2.imread(r'src\utils\h_template.png')
    input_image = cv2.imread(r'src\utils\for_sklearn\13.png')
    
    result = recognize_rotated_h_marker_sift(input_image, template_image)
    
    if result is not None:
        print(f"H 마커 인식 결과: {result}")
        
        # ✅ 이미지 정렬 및 변환 행렬 가져오기
        transformed_image, M = transform_image_from_params(input_image, result)
        cv2.imwrite(r'src\utils\output_image13_aligned.png', transformed_image)
        
        print(f"변환 행렬 M:\n{M}")
        
        # 2. 드론 고도 계산
        scale_factor = result[3]
        
        H_MARKER_TEMPLATE_PIXELS = template_image.shape[1] # H 마커의 pixel사이즈(가로)
        IMAGE_WIDTH = input_image.shape[1]
        
        drone_altitude = calculate_drone_altitude_from_scale_precise(
            scale_factor, 
            H_MARKER_REAL_WIDTH_M, 
            H_MARKER_TEMPLATE_PIXELS,
            image_width_pixels=IMAGE_WIDTH,
            fov_horizontal_deg=118
        )
        print(f"추정 드론 고도 (정밀): {drone_altitude:.2f}m")
                
        # 3. 옥상 위치 계산 및 시각화
        rooftop_positions, visualized_image = detect_all_building_rooftops(
            transformed_image,
            BUILDINGS_INFO,
            drone_altitude,
            transformation_matrix=M,  # ✅ 변환 행렬 전달
            original_image_shape=input_image.shape,  # ✅ 원본 이미지 크기 전달
            visualize=True
        )
        
        # 4. 결과 출력
        print("\n=== 건물 옥상 좌표 ===")
        for building_id, (x, y) in rooftop_positions.items():
            print(f"건물 {building_id}: 옥상 좌표 ({x:.2f}, {y:.2f})")
        
        # 5. 시각화 이미지 저장
        cv2.imwrite(r'src\utils\output_image13_rooftops.png', visualized_image)
        print("\n시각화 이미지 저장 완료!")
        
    else:
        print("H 마커 인식 실패")

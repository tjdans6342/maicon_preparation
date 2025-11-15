import os
import cv2
import numpy as np
import sys
import glob
from target_geometry import recognize_rotated_h_marker_sift, transform_image_from_params
from second_target_geomentry import crop_rois_for_classification 

# ==============================================================================
# sencond_target_geomentry(process_batch_directory) 상수 정의: 옥상 좌표
# (이 좌표는 정렬된 이미지에 대해 고정된 값입니다.)
# ==============================================================================

ROOFTOP_POSITIONS = {
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

def extract_frames_from_video(video_path):
    """
    동영상의 경로를 입력받아, 동영상 파일이 위치한 디렉토리에
    동영상 이름의 폴더를 생성하고 각 프레임을 이미지 파일로 추출하여 저장합니다.
    """
    # 0. jpeg 압축률 설정
    compression_quality = 90 
    
    # 1. 동영상 파일 열기
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"오류: 동영상 파일 '{video_path}'을(를) 열 수 없습니다.")
        return

    # 2. 출력 폴더 이름 및 경로 설정 수정
    video_dir = os.path.dirname(video_path)
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    
    # 출력 폴더 경로 설정
    output_dir = os.path.join(video_dir, base_name + "_frames")  # _frames 접미사 추가
    
    # 🔥 기존에 파일이 존재하면 삭제
    if os.path.isfile(output_dir):
        print(f"⚠️ 경고: '{output_dir}'이(가) 파일로 존재합니다. 삭제합니다.")
        os.remove(output_dir)

    # 3. 출력 폴더 생성
    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"✅ 폴더 생성 완료: {output_dir}")
        else:
            print(f"⚠️ 경고: 폴더 '{output_dir}'이(가) 이미 존재합니다. 기존 파일에 덮어쓰거나 추가됩니다.")
    except OSError as e:
        print(f"❌ 오류: 폴더 생성 실패 - {e}")
        cap.release()
        return

    # 4. 프레임 추출 및 저장
    frame_count = 0
    print("프레임 추출을 시작합니다...")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"총 프레임 수: {total_frames}")

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        # 파일 이름 형식: frame_000000.jpg
        frame_filename = os.path.join(output_dir, f"frame_{frame_count:06d}.jpg")

        # 프레임을 JPG 파일로 저장
        cv2.imwrite(frame_filename, frame, [cv2.IMWRITE_JPEG_QUALITY, compression_quality])

        frame_count += 1
        
        # 진행률 표시 (10프레임마다)
        if frame_count % 10 == 0 or frame_count == total_frames:
            progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
            print(f"진행 중: {frame_count}/{total_frames} ({progress:.1f}%)", end='\r')

    # 5. 종료 및 정리
    cap.release()
    print(f"\n{'='*60}")
    print(f"✅ 추출 완료!")
    print(f"총 {frame_count}개의 프레임이 저장되었습니다.")
    print(f"저장 위치: {output_dir}")
    print(f"{'='*60}")

def process_video_alignment(
    video_path: str,
    template_image_path: str,
    output_path: str,
    show_preview: bool = False,
    fps: int = None
) -> bool:
    """
    mp4 비디오 파일을 불러와서 각 프레임마다 H 마커를 인식하고 화면 보정을 수행합니다.
    
    Args:
        video_path (str): 입력 비디오 파일 경로
        template_image_path (str): H 마커 템플릿 이미지 경로
        output_path (str): 출력 비디오 파일 경로
        show_preview (bool): 실시간 미리보기 표시 여부
        fps (int): 출력 비디오 FPS (None이면 원본과 동일)
    
    Returns:
        bool: 처리 성공 여부
    """
    # 1. 템플릿 이미지 로드
    template_image = cv2.imread(template_image_path)
    if template_image is None:
        print(f"오류: 템플릿 이미지를 로드할 수 없습니다: {template_image_path}")
        return False
    
    # 2. 비디오 캡처 객체 생성
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"오류: 비디오 파일을 열 수 없습니다: {video_path}")
        return False
    
    # 3. 비디오 속성 가져오기
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    output_fps = fps if fps is not None else original_fps
    
    print(f"비디오 정보:")
    print(f"  - 해상도: {frame_width}x{frame_height}")
    print(f"  - FPS: {original_fps:.2f}")
    print(f"  - 총 프레임 수: {total_frames}")
    print(f"  - 출력 FPS: {output_fps:.2f}")
    
    # 4. 비디오 라이터 객체 생성
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, output_fps, (frame_width, frame_height))
    
    if not out.isOpened():
        print(f"오류: 출력 비디오 파일을 생성할 수 없습니다: {output_path}")
        cap.release()
        return False
    
    # 5. 프레임별 처리
    frame_count = 0
    success_count = 0
    fail_count = 0
    
    print("\n비디오 처리 중...")
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        frame_count += 1
        
        # H 마커 인식
        result = recognize_rotated_h_marker_sift(frame, template_image)
        
        if result is not None:
            # 화면 보정 수행
            transformed_frame, _ = transform_image_from_params(frame, result)
            out.write(transformed_frame)
            success_count += 1
            
            # 실시간 미리보기
            if show_preview:
                preview_frame = cv2.resize(transformed_frame, (960, 540))
                cv2.putText(preview_frame, f"Frame: {frame_count}/{total_frames}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow('Aligned Video Preview', preview_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n사용자에 의해 처리가 중단되었습니다.")
                    break
        else:
            # H 마커 인식 실패 시 원본 프레임 사용
            out.write(frame)
            fail_count += 1
            
            if show_preview:
                preview_frame = cv2.resize(frame, (960, 540))
                cv2.putText(preview_frame, f"Frame: {frame_count}/{total_frames} (FAILED)", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow('Aligned Video Preview', preview_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n사용자에 의해 처리가 중단되었습니다.")
                    break
        
        # 진행 상황 표시
        if frame_count % max(1, total_frames // 10) == 0:
            progress = (frame_count / total_frames) * 100
            print(f"진행률: {progress:.1f}% ({frame_count}/{total_frames} 프레임)")
    
    # 6. 리소스 해제
    cap.release()
    out.release()
    
    if show_preview:
        cv2.destroyAllWindows()
    
    # 7. 결과 출력
    print("\n=== 처리 완료 ===")
    print(f"총 처리 프레임: {frame_count}")
    print(f"성공: {success_count} ({success_count/frame_count*100:.1f}%)")
    print(f"실패: {fail_count} ({fail_count/frame_count*100:.1f}%)")
    print(f"출력 파일: {output_path}")
    
    return True

def process_video_alignment_with_stabilization(
    video_path: str,
    template_image_path: str,
    output_path: str,
    show_preview: bool = False,
    fps: int = None,
    use_last_valid_transform: bool = True
) -> bool:
    """
    mp4 비디오 파일을 불러와서 각 프레임마다 H 마커를 인식하고 화면 보정을 수행합니다.
    H 마커 인식 실패 시 이전 프레임의 변환 파라미터를 재사용하여 안정적인 영상을 생성합니다.
    
    Args:
        video_path (str): 입력 비디오 파일 경로
        template_image_path (str): H 마커 템플릿 이미지 경로
        output_path (str): 출력 비디오 파일 경로
        show_preview (bool): 실시간 미리보기 표시 여부
        fps (int): 출력 비디오 FPS (None이면 원본과 동일)
        use_last_valid_transform (bool): 인식 실패 시 마지막 유효 변환 사용 여부
    
    Returns:
        bool: 처리 성공 여부
    """
    # 1. 템플릿 이미지 로드
    template_image = cv2.imread(template_image_path)
    if template_image is None:
        print(f"오류: 템플릿 이미지를 로드할 수 없습니다: {template_image_path}")
        return False
    
    # 2. 비디오 캡처 객체 생성
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"오류: 비디오 파일을 열 수 없습니다: {video_path}")
        return False
    
    # 3. 비디오 속성 가져오기
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    output_fps = fps if fps is not None else original_fps
    
    print(f"비디오 정보:")
    print(f"  - 해상도: {frame_width}x{frame_height}")
    print(f"  - FPS: {original_fps:.2f}")
    print(f"  - 총 프레임 수: {total_frames}")
    print(f"  - 출력 FPS: {output_fps:.2f}")
    
    # 4. 비디오 라이터 객체 생성
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, output_fps, (frame_width, frame_height))
    
    if not out.isOpened():
        print(f"오류: 출력 비디오 파일을 생성할 수 없습니다: {output_path}")
        cap.release()
        return False
    
    # 5. 프레임별 처리
    frame_count = 0
    success_count = 0
    fail_count = 0
    last_valid_params = None  # 마지막으로 성공한 변환 파라미터 저장
    
    print("\n비디오 처리 중...")
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        frame_count += 1
        
        # H 마커 인식
        result = recognize_rotated_h_marker_sift(frame, template_image)
        
        if result is not None:
            # 화면 보정 수행
            transformed_frame, _ = transform_image_from_params(frame, result)
            out.write(transformed_frame)
            success_count += 1
            last_valid_params = result  # 성공한 파라미터 저장
            status_text = "OK"
            status_color = (0, 255, 0)
            
        else:
            # H 마커 인식 실패
            fail_count += 1
            
            if use_last_valid_transform and last_valid_params is not None:
                # 이전 프레임의 변환 파라미터 사용
                transformed_frame, _ = transform_image_from_params(frame, last_valid_params)
                out.write(transformed_frame)
                status_text = "USING LAST"
                status_color = (0, 165, 255)  # 주황색
            else:
                # 원본 프레임 사용
                out.write(frame)
                transformed_frame = frame
                status_text = "FAILED"
                status_color = (0, 0, 255)  # 빨간색
        
        # 실시간 미리보기
        if show_preview:
            preview_frame = cv2.resize(transformed_frame, (960, 540))
            cv2.putText(preview_frame, f"Frame: {frame_count}/{total_frames} [{status_text}]", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
            cv2.imshow('Aligned Video Preview', preview_frame)
            
            # 'q' 키를 누르면 중단
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n사용자에 의해 처리가 중단되었습니다.")
                break
        
        # 진행 상황 표시
        if frame_count % max(1, total_frames // 10) == 0:
            progress = (frame_count / total_frames) * 100
            print(f"진행률: {progress:.1f}% ({frame_count}/{total_frames} 프레임)")
    
    # 6. 리소스 해제
    cap.release()
    out.release()
    
    if show_preview:
        cv2.destroyAllWindows()
    
    # 7. 결과 출력
    print("\n=== 처리 완료 ===")
    print(f"총 처리 프레임: {frame_count}")
    print(f"성공: {success_count} ({success_count/frame_count*100:.1f}%)")
    print(f"실패: {fail_count} ({fail_count/frame_count*100:.1f}%)")
    print(f"출력 파일: {output_path}")
    
    return True

def process_batch_directory(input_directory: str) -> None:
    """
    지정된 디렉토리 내의 모든 이미지를 순회하며 ROI를 잘라 저장합니다.
    
    Args:
        input_directory (str): 정렬된 이미지 파일들이 위치한 폴더 경로.
    """
    
    # 1. 입력 디렉토리 유효성 검사
    if not os.path.isdir(input_directory):
        print(f"오류: 유효하지 않은 디렉토리 경로입니다: {input_directory}")
        return

    # 2. 이미지 파일 목록 가져오기 (여기서는 jpg png 파일만 처리)
    search_path = os.path.join(input_directory, '*.[jp][pn]g')
    image_files = glob.glob(search_path)

    if not image_files:
        print(f"경고: {input_directory} 폴더에서 처리할 png 혹은 jpg 파일을 찾을 수 없습니다.")
        return

    print(f"--- 총 {len(image_files)}개의 이미지를 처리합니다. ---")

    # 3. 파일 순회 및 처리
    for full_path in image_files:
        # 파일명 추출 (확장자 제외) -> 이를 image_index로 사용
        filename_with_ext = os.path.basename(full_path)
        image_index = os.path.splitext(filename_with_ext)[0]
        
        print(f"\n[처리 시작] 파일: {filename_with_ext} (인덱스: {image_index})")
        
        # 이미지 로드
        image = cv2.imread(full_path)
        
        if image is None:
            print(f"오류: 이미지 로드 실패 - {full_path}. 건너뜁니다.")
            continue
            
        # crop_rois_for_classification 함수 호출
        try:
            crop_rois_for_classification(
                transformed_image=image,
                rooftop_positions=ROOFTOP_POSITIONS,
                image_index=image_index # 파일명을 인덱스로 사용
            )
            print(f"[처리 완료] 파일: {filename_with_ext}")
        except Exception as e:
            print(f"처리 중 예외 발생: {e}")


# ===== 비디오 정렬 =====

'''
process_video_alignment_with_stabilization(
    video_path='src/drone/33.mkv',
    template_image_path='src/drone/h_template.png',
    output_path='src/drone/output_video_aligned_stable4.mkv',
    show_preview=True,
    fps=None,  # 원본 FPS 유지
    use_last_valid_transform=True  # 이전 변환 파라미터 재사용
)
'''
# --- 동영상 컷팅 ---
# 사용 예시 경로를 실제 존재하는 경로로 변경해야 합니다.

video_file_path = "src/drone/33.mkv" 
extract_frames_from_video(video_file_path) 

# second_geomentry용 함수(폴더처리)
# input_folder = 'src/drone/b_frames'
# process_batch_directory(input_folder)
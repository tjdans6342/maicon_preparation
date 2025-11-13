import time
import os
import cv2
import numpy as np
from tiki.mini import TikiMini
import keyboard

# --- 초기 설정  ---
print("로봇 초기화 중...")
tiki = TikiMini()
SPEED = 60  # 로봇 속도 (0-127) 
SAVE_DIR = "training_data" # 이미지 저장 폴더 (Thư mục lưu ảnh)
img_count = 0

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)
    print(f"폴더 생성: {SAVE_DIR}")

# --- 카메라 시작 (OpenCV)  ---
print("카메라 초기화 중 (cv2.VideoCapture)...")
cap = cv2.VideoCapture(0) # 카메라 장치 번호 (기본 0) (Số thiết bị camera)
if not cap.isOpened():
    print("오류: 카메라를 열 수 없습니다.")
    exit()

# 해상도 설정 
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
print("카메라 준비 완료.")
time.sleep(1.0) # 카메라 안정화 대기 (Đợi camera ổn định)

# --- 사용자 안내  ---
print("\n" + "="*30)
print("     로봇 제어 & 데이터 수집")
print("="*30)
print(f"- 저장 폴더: {os.path.abspath(SAVE_DIR)}")
print("- 새 창에 비디오가 표시됩니다.")
print("- 방향키 (위, 아래, 좌, 우): 로봇 이동")
print("- 'Space' (스페이스바): 이미지 캡처")
print("- 터미널에서 'q': 프로그램 종료")
print("="*30)

# 비디오 창 제목 
WINDOW_NAME = "Robot View - (터미널에서 'q'로 종료)"

try:
    while True:
        # 1. 카메라 프레임 읽기 
        ret, frame = cap.read()
        if not ret:
            print("프레임 읽기 오류, 건너뜁니다...")
            continue

        # 2. 이미지 뒤집기 (Xử lý lật ảnh)
        # 카메라가 거꾸로 설치된 경우, 아래 두 줄 중 하나를 선택하세요:
        frame_processed = cv2.flip(frame, 0)  # 상하 반전 (Lật dọc)
        # frame_processed = cv2.flip(frame, -1) # 180도 회전 (Xoay 180)

        # 3. 새 창에 비디오 표시 (Hiển thị video)
        cv2.imshow(WINDOW_NAME, frame_processed)

        # 4. 키보드 입력 확인 (Kiểm tra phím)

        # 종료 키 ('q')
        if keyboard.is_pressed('q'):
            print("종료 중...")
            break

        # 캡처 키 ('space')
        if keyboard.is_pressed('space'):
            filename = f"image_{time.strftime('%Y%m%d_%H%M%S')}_{img_count:04d}.jpg"
            filepath = os.path.join(SAVE_DIR, filename)
            
            cv2.imwrite(filepath, frame_processed)
            
            print(f"저장 완료: {filepath}")
            img_count += 1
            # 연속 캡처 방지 딜레이 (Nghỉ để tránh chụp hàng loạt)
            time.sleep(0.3) 

        # 이동 키 (Phím di chuyển)
        if keyboard.is_pressed('up'):
            tiki.forward(SPEED)
        elif keyboard.is_pressed('down'):
            tiki.backward(SPEED)
        elif keyboard.is_pressed('left'):
            tiki.counter_clockwise(SPEED)
        elif keyboard.is_pressed('right'):
            tiki.clockwise(SPEED)
        else:
            tiki.stop() # 이동 키 입력 없으면 정지 (Dừng nếu không nhấn)

        # 5. cv2.imshow()를 위한 필수 코드 (Bắt buộc cho cv2.imshow)
        # 1ms 대기 (Chờ 1ms)
        if cv2.waitKey(1) == 27: # 비디오 창에서 'ESC' 키로도 종료 가능 (Thoát bằng ESC)
             print("ESC 키로 종료.")
             break


except KeyboardInterrupt:
    print("\n프로그램 중단됨.")
except Exception as e:
    print(f"\n오류 발생: {e}")
    print("'sudo' 권한으로 실행했는지 확인하세요.")

finally:
    # --- 정리 (Dọn dẹp) ---
    print("로봇 정지, 카메라 해제, 창 닫기...")
    tiki.stop()
    cap.release()
    cv2.destroyAllWindows() # 모든 OpenCV 창 닫기 (Đóng tất cả cửa sổ)
    print("정리 완료. 종료합니다.")
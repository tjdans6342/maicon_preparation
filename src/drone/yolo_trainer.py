import os
import sys

# ========================================
# 🔧 환경 설정 (반드시 맨 위에!)
# ========================================

# matplotlib 백엔드를 non-interactive로 설정
import matplotlib
matplotlib.use('Agg')  # GUI 없는 백엔드 사용
# ✅ 2. stdout/stderr 버퍼링 비활성화
os.environ['PYTHONUNBUFFERED'] = '1'
# ✅ 3. YOLO 로깅 설정
os.environ['YOLO_VERBOSE'] = 'True'

# ========================================
# 라이브러리 임포트
# ========================================

import shutil
import random
from pathlib import Path
from PIL import Image # Pillow 라이브러리가 설치되어 있어야 합니다.
from ultralytics import YOLO
import torch
from pathlib import Path

# ========================================
# YOLO v8 설정 변수
# ========================================

# 데이터셋 설정 파일 경로
DATA_YAML = r'src/drone/dataset.yaml'

# 모델 설정
MODEL_NAME = 'yolov8s.pt'  # yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt

# 학습 하이퍼파라미터
EPOCHS = 100               # 학습 에포크 수
BATCH_SIZE = 8            # 배치 크기 (GPU 메모리에 따라 조절: 8, 16, 32 등)
IMG_SIZE = 640             # 입력 이미지 크기
DEVICE = 0                 # GPU 번호 (0, 1, 2...) 또는 'cpu'

# 학습 옵션
PATIENCE = 50              # Early stopping patience (성능 개선 없을 때 대기 에포크)
SAVE_PERIOD = 10           # 모델 저장 주기 (에포크 단위)
WORKERS = 0                # 데이터 로딩 워커 수
PROJECT = 'runs/detect'    # 결과 저장 폴더
NAME = 'drone_yolov8s'     # 실험 이름

# 추가 학습 옵션
OPTIMIZER = 'AdamW'        # 옵티마이저: SGD, Adam, AdamW
LR0 = 0.01                 # 초기 학습률
WEIGHT_DECAY = 0.0005      # 가중치 감쇠
AUGMENT = True             # 데이터 증강 활성화

# ========================================
# letterbox_and_convert_yolo 설정 변수
# ========================================

# 기본 경로
BASE_DIR = Path('src/drone/dataset')
# 원본 데이터 폴더
IMG_PRE_DIR = BASE_DIR / 'img_pre'
LABELS_PRE_DIR = BASE_DIR / 'labels_pre'
# 출력 데이터 폴더
IMG_OUT_DIR = BASE_DIR / 'img'
LABELS_OUT_DIR = BASE_DIR / 'labels'
# 목표 크기 (정사각형)
TARGET_SIZE = 640

def split_dataset_with_labels(
    img_folder: str = 'src/drone/dataset/img',
    label_folder: str = 'src/drone/dataset/labels',  # 라벨 폴더 (있는 경우)
    train_folder: str = 'src/drone/dataset/train',
    val_folder: str = 'src/drone/dataset/val',
    test_folder: str = 'src/drone/dataset/test',
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    copy_mode: bool = True
):
    """
    이미지와 라벨을 함께 분배합니다 (YOLO 형식 등).
    라벨 파일은 이미지와 같은 이름에 .txt 확장자를 가진다고 가정합니다.
    """
    
    # 비율 검증
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 0.001:
        raise ValueError(f"비율의 합이 1.0이 아닙니다: {total_ratio}")
    
    # 폴더 존재 확인
    if not os.path.exists(img_folder):
        raise FileNotFoundError(f"img 폴더를 찾을 수 없습니다: {img_folder}")
    
    has_labels = os.path.exists(label_folder)
    
    # 출력 폴더 생성 (images와 labels 서브폴더)
    for folder in [train_folder, val_folder, test_folder]:
        os.makedirs(os.path.join(folder, 'images'), exist_ok=True)
        if has_labels:
            os.makedirs(os.path.join(folder, 'labels'), exist_ok=True)
    
    # 이미지 파일 목록
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    image_files = [
        f for f in os.listdir(img_folder)
        if os.path.isfile(os.path.join(img_folder, f)) and 
        os.path.splitext(f)[1].lower() in image_extensions
    ]
    
    if len(image_files) == 0:
        print(f"경고: {img_folder}에 이미지 파일이 없습니다.")
        return
    
    print(f"총 {len(image_files)}개의 이미지 파일을 발견했습니다.")
    
    # 랜덤 셔플
    random.seed(seed)
    random.shuffle(image_files)
    
    # 분할
    total_count = len(image_files)
    train_count = int(total_count * train_ratio)
    val_count = int(total_count * val_ratio)
    
    splits = {
        'train': (image_files[:train_count], train_folder),
        'val': (image_files[train_count:train_count + val_count], val_folder),
        'test': (image_files[train_count + val_count:], test_folder)
    }
    
    print(f"\n분할 결과:")
    for split_name, (files, _) in splits.items():
        print(f"  - {split_name.capitalize()}: {len(files)}개 ({len(files)/total_count*100:.1f}%)")
    
    # 파일 처리
    print(f"\n{'복사' if copy_mode else '이동'} 작업 시작...")
    
    for split_name, (file_list, dest_folder) in splits.items():
        img_success = 0
        label_success = 0
        
        for filename in file_list:
            # 이미지 처리
            src_img = os.path.join(img_folder, filename)
            dst_img = os.path.join(dest_folder, 'images', filename)
            
            try:
                if copy_mode:
                    shutil.copy2(src_img, dst_img)
                else:
                    shutil.move(src_img, dst_img)
                img_success += 1
            except Exception as e:
                print(f"오류 (이미지 {filename}): {e}")
            
            # 라벨 처리
            if has_labels:
                label_filename = os.path.splitext(filename)[0] + '.txt'
                src_label = os.path.join(label_folder, label_filename)
                dst_label = os.path.join(dest_folder, 'labels', label_filename)
                
                if os.path.exists(src_label):
                    try:
                        if copy_mode:
                            shutil.copy2(src_label, dst_label)
                        else:
                            shutil.move(src_label, dst_label)
                        label_success += 1
                    except Exception as e:
                        print(f"오류 (라벨 {label_filename}): {e}")
        
        print(f"  - {split_name.capitalize()}: 이미지 {img_success}개, 라벨 {label_success}개 {'복사' if copy_mode else '이동'} 완료")
    
    print(f"\n=== 작업 완료 ===")

def letterbox_and_convert_yolo(
    img_pre_dir: Path,
    labels_pre_dir: Path,
    img_out_dir: Path,
    labels_out_dir: Path,
    target_size: int
):
    """
    원본 이미지를 레터박싱하여 목표 크기로 변환하고,
    이에 맞춰 YOLO 라벨 좌표를 재계산하여 저장합니다.
    """
    
    # 1. 출력 폴더 초기화
    for d in [img_out_dir, labels_out_dir]:
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)
    
    # 2. 이미지 파일 목록 가져오기 (JPG, PNG만 가정)
    image_files = [f for f in img_pre_dir.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
    
    if not image_files:
        print(f"경고: {img_pre_dir}에 이미지 파일이 없습니다.")
        return

    print(f"총 {len(image_files)}개의 이미지 파일을 발견했습니다. 변환을 시작합니다...")
    
    for i, img_path in enumerate(image_files):
        try:
            # 3. 이미지 로드 및 크기 계산
            img = Image.open(img_path).convert("RGB")
            original_width, original_height = img.size
            
            # 4. 레터박싱 스케일 팩터 계산
            scale = min(target_size / original_width, target_size / original_height)
            new_width = int(original_width * scale)
            new_height = int(original_height * scale)
            
            # 5. 이미지 리사이즈
            img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # 6. 패딩 계산
            pad_w = (target_size - new_width) // 2
            pad_h = (target_size - new_height) // 2
            
            # 7. 새 이미지 생성 및 패딩 적용 (검은색 배경)
            new_img = Image.new('RGB', (target_size, target_size), (0, 0, 0))
            new_img.paste(img_resized, (pad_w, pad_h))
            
            # 8. 새 이미지 저장
            new_img.save(img_out_dir / img_path.name)
            
            # 9. 라벨 파일 경로 설정
            label_filename = img_path.stem + '.txt'
            label_pre_path = labels_pre_dir / label_filename
            label_out_path = labels_out_dir / label_filename
            
            # 10. 라벨 파일 변환 및 저장
            if label_pre_path.exists():
                with open(label_pre_path, 'r') as f_in, open(label_out_path, 'w') as f_out:
                    for line in f_in:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            cls_id = parts[0]
                            # YOLO 라벨은 정규화된 값 (0.0 ~ 1.0)
                            x_center_norm, y_center_norm, width_norm, height_norm = map(float, parts[1:])
                            
                            # 정규화된 값을 픽셀 값으로 변환 (원본 크기 기준)
                            x_center = x_center_norm * original_width
                            y_center = y_center_norm * original_height
                            width = width_norm * original_width
                            height = height_norm * original_height
                            
                            # 레터박싱된 이미지에서의 픽셀 위치 계산
                            new_x_center = x_center * scale + pad_w
                            new_y_center = y_center * scale + pad_h
                            new_width = width * scale
                            new_height = height * scale
                            
                            # 새로운 640x640 프레임 기준으로 다시 정규화
                            new_x_center_norm = new_x_center / target_size
                            new_y_center_norm = new_y_center / target_size
                            new_width_norm = new_width / target_size
                            new_height_norm = new_height / target_size
                            
                            # 새로운 라벨 저장 (소수점 6자리까지)
                            f_out.write(
                                f"{cls_id} {new_x_center_norm:.6f} {new_y_center_norm:.6f} {new_width_norm:.6f} {new_height_norm:.6f}\n"
                            )
                        else:
                            f_out.write(line + '\n') # 형식이 맞지 않으면 그대로 복사
            
            if (i + 1) % 100 == 0:
                print(f"--- {i + 1}/{len(image_files)}개 파일 처리 완료 ---")

        except Exception as e:
            print(f"오류 발생 ({img_path.name}): {e}")
            continue

    print("\n=== 모든 파일 변환 완료 ===")

def train_yolo():
    """
    YOLO v8s 모델을 학습합니다.
    """
    
    # GPU 사용 가능 여부 확인
    print(f"PyTorch 버전: {torch.__version__}")
    print(f"CUDA 사용 가능: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU 이름: {torch.cuda.get_device_name(0)}")
        print(f"GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # 데이터셋 파일 존재 확인
    if not Path(DATA_YAML).exists():
        raise FileNotFoundError(f"데이터셋 설정 파일을 찾을 수 없습니다: {DATA_YAML}")
    
    print(f"\n{'='*60}")
    print(f"YOLO v8s 학습 시작")
    print(f"{'='*60}")
    print(f"모델: {MODEL_NAME}")
    print(f"데이터셋: {DATA_YAML}")
    print(f"에포크: {EPOCHS}")
    print(f"배치 크기: {BATCH_SIZE}")
    print(f"이미지 크기: {IMG_SIZE}")
    print(f"디바이스: {DEVICE}")
    print(f"{'='*60}\n")
    
    # 모델 로드 (사전 학습된 가중치 사용)
    model = YOLO(MODEL_NAME)
    
    # 학습 시작
    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        imgsz=IMG_SIZE,
        device=DEVICE,
        patience=PATIENCE,
        save_period=SAVE_PERIOD,
        workers=WORKERS,
        project=PROJECT,
        name=NAME,
        optimizer=OPTIMIZER,
        lr0=LR0,
        weight_decay=WEIGHT_DECAY,
        augment=AUGMENT,
        # 추가 옵션
        pretrained=True,       # 사전 학습 가중치 사용
        verbose=True,          # 상세 로그 출력
        seed=42,               # 재현성을 위한 시드
        deterministic=True,    # 재현 가능한 결과
        single_cls=False,      # 단일 클래스 모드 (False: 다중 클래스)
        rect=False,            # 직사각형 학습 (False: 정사각형)
        cos_lr=True,           # 코사인 학습률 스케줄러
        close_mosaic=10,       # 마지막 N 에포크에서 모자이크 증강 비활성화
        resume=False,          # 이전 학습 재개 (True: 중단된 학습 이어서)
        amp=True,              # Automatic Mixed Precision (메모리 절약)
        fraction=1.0,          # 학습 데이터 비율 (1.0 = 전체 사용)
        profile=False,         # 프로파일링 (속도 측정)
        # 검증 옵션
        val=True,              # 에포크마다 검증 수행
        plots=True,            # 학습 그래프 저장
        save=True,             # 최종 모델 저장
        save_json=False,       # COCO JSON 결과 저장
        # 하이퍼파라미터
        hsv_h=0.015,           # HSV-Hue 증강 (0-1)
        hsv_s=0.2,             # HSV-Saturation 증강 (0-1)
        hsv_v=0.2,             # HSV-Value 증강 (0-1)
        degrees=0.0,           # 회전 증강 (±deg)
        translate=0.1,         # 이동 증강 (±fraction)
        scale=0.1,             # 스케일 증강 (±gain)
        shear=0.0,             # 전단 증강 (±deg)
        perspective=0.0,       # 원근 증강 (±fraction)
        flipud=0.0,            # 상하 반전 확률
        fliplr=0.5,            # 좌우 반전 확률
        mosaic=1.0,            # 모자이크 증강 확률
        mixup=0.0,             # Mixup 증강 확률
        copy_paste=0.0,        # Copy-paste 증강 확률
    )
    
    print(f"\n{'='*60}")
    print(f"학습 완료!")
    print(f"{'='*60}")
    print(f"결과 저장 경로: {results.save_dir}")
    print(f"최고 모델: {Path(results.save_dir) / 'weights' / 'best.pt'}")
    print(f"마지막 모델: {Path(results.save_dir) / 'weights' / 'last.pt'}")
    print(f"{'='*60}\n")
    
    return results

def validate_model(model_path: str = None):
    """
    학습된 모델을 검증 데이터셋으로 평가합니다.
    
    Args:
        model_path: 모델 가중치 파일 경로 (None이면 최신 학습 결과 사용)
    """
    
    if model_path is None:
        # 최신 학습 결과에서 best.pt 찾기
        model_path = f"{PROJECT}/{NAME}/weights/best.pt"
    
    if not Path(model_path).exists():
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")
    
    print(f"\n{'='*60}")
    print(f"모델 검증 시작")
    print(f"{'='*60}")
    print(f"모델: {model_path}")
    print(f"{'='*60}\n")
    
    # 모델 로드
    model = YOLO(model_path)
    
    # 검증 수행
    metrics = model.val(
        data=DATA_YAML,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        plots=True,
        save_json=True,
        verbose=True
    )
    
    print(f"\n{'='*60}")
    print(f"검증 결과")
    print(f"{'='*60}")
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall: {metrics.box.mr:.4f}")
    print(f"{'='*60}\n")
    
    return metrics

if __name__ == '__main__':
    
    # 640x640 변경 메인 동작
    letterbox_and_convert_yolo(
        img_pre_dir=IMG_PRE_DIR,
        labels_pre_dir=LABELS_PRE_DIR,
        img_out_dir=IMG_OUT_DIR,
        labels_out_dir=LABELS_OUT_DIR,
        target_size=TARGET_SIZE
    )

    # ===== 이미지분배 (YOLO 형식) =====
    split_dataset_with_labels(
        img_folder='src/drone/dataset/img',
        label_folder='src/drone/dataset/labels',  # 라벨 폴더가 있는 경우
        train_folder='src/drone/dataset/train',
        val_folder='src/drone/dataset/val',
        test_folder='src/drone/dataset/test',
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42,
        copy_mode=True
        )
    
    # 학습 실행
    results = train_yolo()
        
    # 학습 완료 후 자동으로 검증 수행
    print("\n학습 완료! 최고 성능 모델로 검증을 시작합니다...\n")
    validate_model()
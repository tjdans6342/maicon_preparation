# -*- coding: utf-8 -*-
"""fire_detection_pipeline.ipynb


# Detection of Fire Building Number
### - YOLO Fire Building Inference, Building-ID Pipeline

이 노트북은 미리 학습한 `best.pt` 가중치를 불러와 `test/images` 폴더의 섹터별 샘플을 추론하고, 검출 결과를 S1~S9 건물 번호와 화재 여부로 단계적으로 매핑·시각화하는 파이프라인입니다.

## 환경 설정

필요한 라이브러리와 경로를 불러오고, 학습된 가중치(`best.pt`) 및 입력 이미지 존재 여부를 확인합니다.
"""

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO


### TODO

SOURCE_IMAGES_DIR = Path('/test/images')  # 추론 대상 이미지 폴더
WEIGHTS_PATH = Path('/weights/best.pt')  # 학습된 YOLO 가중치 경로
OUTPUT_DIR = SOURCE_IMAGES_DIR / 'yolo_inference_samples'  # 추론 결과 저장 폴더




# 데이터셋 라벨링할 때의 클래스와 동일하게 맞춤
FIRE_CLASS = 0  # fire (burning building)
NORMAL_CLASS = 1  # normal building

if not SOURCE_IMAGES_DIR.exists():
    raise FileNotFoundError(f'입력 이미지 경로를 찾을 수 없습니다: {SOURCE_IMAGES_DIR}')
if not WEIGHTS_PATH.exists():
    raise FileNotFoundError(f'best.pt 가중치를 찾을 수 없습니다: {WEIGHTS_PATH}')

IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg')
all_source_images = sorted({path for ext in IMAGE_EXTENSIONS for path in SOURCE_IMAGES_DIR.glob(f'*{ext}')})
print('총 입력 이미지 개수:', len(all_source_images))
print('가중치 경로:', WEIGHTS_PATH)

"""## 섹터별 샘플 선택

`test/images` 폴더는 섹터별로 2장씩 정리돼 있으므로, 파일명 패턴 `img_sector{번호}_...`을 이용해 해당 섹터의 이미지를 모두 불러와 그대로 사용합니다.
- 고정된 검증 세트를 활용해 파이프라인 전체를 빠르게 점검하고 디버깅합니다.

"""

SECTOR_INDICES = list(range(1, 10))

def collect_sector_images(sector: int) -> List[Path]:
    files = []
    for ext in IMAGE_EXTENSIONS:
        files.extend(SOURCE_IMAGES_DIR.glob(f'*sector{sector}_*{ext}'))
    unique_files = sorted({path for path in files}, key=lambda p: p.name)
    return unique_files

sector_samples: Dict[int, List[Path]] = {}
for sector in SECTOR_INDICES:
    candidates = collect_sector_images(sector)
    if not candidates:
        print(f'[WARN] 섹터 {sector} 이미지가 없습니다.')
        continue
    if len(candidates) != 2:
        print(f'[WARN] 섹터 {sector} 이미지가 예상과 다른 {len(candidates)}장입니다.')
    sector_samples[sector] = candidates

# 섹터별로 정리한 경로를 하나의 리스트로 묶고, 선택된 파일 정보를 콘솔에 정리해서 출력 -> 나중에 YOLO 추론 및 건물 번호 예측과 화재 건물 출력을 위해 사용
selected_image_paths = [path for paths in sector_samples.values() for path in paths]
print('선택된 총 이미지 수:', len(selected_image_paths))
for sector, paths in sorted(sector_samples.items()):
    print(f'  섹터 {sector}:', ', '.join(path.name for path in paths))

"""## YOLO 모델 로드

v05.1에서 학습한 `best.pt` 가중치를 불러와 추론용 모델을 초기화합니다.

"""

import torch

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
CONF_THRESHOLD = 0.25 # 최소 신뢰도 점수: YOLO가 박스를 예측할 때 각 박스마다 “이게 진짜일 확률”을 주는데, 그 값이 0.25보다 낮으면 노이즈로 보고 버리고, 0.25 이상이면 유효한 감지로 남기는 기준선 -> 숫자를 높이면 더 깐깐해지고, 낮추면 바인딩 박스로 잡는 게 더 많아질 수 있음.
IMG_SIZE = 640

inference_model = YOLO(str(WEIGHTS_PATH))
inference_model.to(DEVICE)
print('Using device:', DEVICE)
print('Confidence threshold:', CONF_THRESHOLD)
print('Selected image count:', len(selected_image_paths))

"""## 추론 실행 및 시각화

선택된 이미지에 대해 YOLO 추론을 수행하고, 결과 이미지를 저장(`test/images/yolo_inference_samples`)하면서 동시에 노트북에서 시각화합니다.

"""

if not selected_image_paths:
    raise RuntimeError('선택된 이미지가 없습니다. 이전 셀을 확인하세요.')

results = inference_model.predict(
    source=[str(p) for p in selected_image_paths],
    conf=CONF_THRESHOLD,
    imgsz=IMG_SIZE,
    project=str(OUTPUT_DIR), # 결과 이미지를 test/images/yolo_inference_samples/sector_samples 아래에 저장하도록 지정
    name='sector_samples',
    exist_ok=True,
    save=True,
    verbose=False, # 출력 메시지 최소화
    device=DEVICE,
)

# predict 결과를 노트북 화면에 표시하고, 각 박스의 클래스,신뢰도를 텍스트로 확인
for result, image_path in zip(results, selected_image_paths): # 방금 predict()에서 반환한 결과와 해당 이미지 경로를 한 쌍씩 꺼냄
    annotated = result.plot()  # BGR numpy array
    annotated_rgb = annotated[..., ::-1]
    plt.figure(figsize=(8, 6))
    plt.imshow(annotated_rgb)
    plt.title(f'{image_path.name} predictions')
    plt.axis('off')
    plt.show()
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        print(f'{image_path.name}: 감지된 객체가 없습니다.')
        continue
    for cls_id, conf in zip(boxes.cls.tolist(), boxes.conf.tolist()):
        label = inference_model.names[int(cls_id)]
        print(f'  - {image_path.name}: {label} (conf={conf:.3f})')

"""## 좌측 기준 건물 번호 추론

아래 셀들에서는 추론된 모든 건물 박스를 활용해 다음 과정을 순차적으로 수행합니다.
- (x축: 왼쪽이 0 -> 오른쪽으로 갈수록 증가 / y축: 위쪽이 0 -> 아래로 갈수록 증가)
1. 가장 왼쪽 박스를 섹터 7로 가정하고 시각화합니다.
2. 섹터 7 박스를 기준으로 적용할 x·y 범위를 계산해 표시합니다.
3. 섹터 6/8 탐지: 해당 범위에 있는 두 박스를 찾아, y 값이 더 작은 박스를 섹터 6, 더 큰 박스를 섹터 8로 라벨링하고 시각화합니다.
4. 섹터 5/9 탐지: 섹터 7 박스 기준으로 x 범위를 이미지 오른쪽 끝까지, y 범위는 2, 3 범위와 동일하게 잡고, 그 안에 있는 두 박스를 찾아 -> y 값이 더 작은 박스를 5로, 더 큰 박스를 9로 라벨링 및 시각화
5. 섹터 4/3/2/1 탐지: 이미 섹터 7/6/8/5/9로 지정한 박스를 제외하고 남은 박스들을 오른쪽->왼쪽으로 순차적으로 섹터 4, 3, 2, 1에 대응시키고 시각화.

### 1. 섹터 7 후보 선정

각 이미지에서 가장 왼쪽에 위치한 박스를 찾아 섹터 7로 가정하고 시각화합니다.
"""

import numpy as np
import matplotlib.patches as patches
from PIL import Image

if 'results' not in locals():
    raise RuntimeError('먼저 추론 셀을 실행해 results 변수를 생성하세요.')

image_detections = {} # Yolo 추론 결과를 저장할 딕셔너리 - 나중에 이 딕셔너리 참고해서, 박스좌표 클래스 정보 등 재사용해서, 건물 번호들 뽑아내.
sector7_refs = {} # 각 이미지에서 “가장 왼쪽 박스”를 찾았을 때 그 정보를 저장해 두는 딕셔너리 -> 얘를 S7로 둠

# 선택된 각 이미지의 YOLO 결과에서 가장 왼쪽 박스를 섹터 7 기준으로 저장해 두고, 해당 이미지를 S7 박스가 표시된 상태로 시각화해 확인
for result, image_path in zip(results, selected_image_paths):
    boxes_obj = result.boxes # 이미지에서 감지된 모든 박스 정보를 담은 객체
    if boxes_obj is None or boxes_obj.data.numel() == 0:
        print(f'{image_path.name}: 감지된 박스가 없습니다.')
        continue

    xyxy = boxes_obj.xyxy.cpu().numpy() # 박스 좌표 (x1, y1, x2, y2)
    cls_ids = boxes_obj.cls.cpu().numpy().astype(int) # 클래스 ID
    confs = boxes_obj.conf.cpu().numpy() # 신뢰도 점수(confidence scores)

    # 이미지별로 박스 정보들을 저장
    image_detections[image_path.name] = {
        'path': image_path,
        'xyxy': xyxy,
        'cls_ids': cls_ids,
        'confs': confs,
    }

    # 가장 왼쪽 박스(즉, x1 좌표가 가장 작은 박스)를 섹터 7로 저장 -> 이 S7이 기준점이 돼서 나머지 건물 번호들 결정
    ref_idx = int(np.argmin(xyxy[:, 0]))
    sector7_refs[image_path.name] = {
        'index': ref_idx,
        'box': xyxy[ref_idx],
    }

    img = Image.open(image_path).convert('RGB') # 시각화를 일관된 RGB 포맷으로 하기 위해
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(img)

    # 좌표 (x1, y1, x2, y2)를 꺼낸 뒤 patches.Rectangle로 노란색 테두리를 그리고(draw) ax.text로 S7 라벨을 붙임
    x1, y1, x2, y2 = xyxy[ref_idx]
    rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='yellow', facecolor='none')
    ax.add_patch(rect)
    ax.text(x1 + 2, y1 - 6, 'S7', color='yellow', fontsize=10,
            bbox=dict(facecolor='black', alpha=0.4, pad=2))

    # 실제 시각화 확인
    ax.set_title(f'{image_path.name} – S7 후보')
    ax.axis('off')
    plt.show()

print('S7 후보를 찾은 이미지 수:', len(sector7_refs)) # 섹터 7 후보를 찾은 이미지 총 수 = 내경우엔 18개 나오면 다 찾은거

"""### 2. 섹터 7 기준 범위 계산 및 표시

섹터 7 박스의 가로 길이 4배, 세로 범위 3배 기준으로 영역을 계산하고 해당 영역을 시각화합니다. 박스 중심이 이 영역 안에 들어오는지를 이후 단계에서 활용합니다.

"""

if not sector7_refs:
    raise RuntimeError('섹터 7 정보가 없습니다. 이전 셀을 먼저 실행하세요.')

selection_regions = {}

for image_name, ref_info in sector7_refs.items():
    data = image_detections[image_name]
    image_path = data['path']
    xyxy = data['xyxy']
    ref_box = ref_info['box']
    # 기준 박스(=S7) 크기 계산
    ref_width = ref_box[2] - ref_box[0]
    ref_height = ref_box[3] - ref_box[1]
    if ref_width <= 0 or ref_height <= 0:
        print(f'{image_name}: 기준 박스 크기가 0입니다.')
        continue

    x_limit = ref_box[0] + ref_width * 4.0 # x범위: ref_box[0] (왼쪽 테두리)에서 가로 4배 (ref_width * 4.0)까지
    # y범위: 박스 중심에서 위/아래 1.5배 (ref_height * 1.5)만큼 확장한 값을 사용해 영역을 잡음
    y_center = 0.5 * (ref_box[1] + ref_box[3])
    y_half = ref_height * 1.5
    y_min = max(0.0, y_center - y_half)
    y_max = y_center + y_half

    # 조사할 영역 범위 저장
    selection_regions[image_name] = {
        'x_min': ref_box[0],
        'x_max': x_limit,
        'y_min': y_min,
        'y_max': y_max,
    }
    # 이미지를 읽어 노란색 S7 박스와 함께 조사할 영역범위(청록색)를 시각화
    img = Image.open(image_path).convert('RGB')
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(img)

    x1, y1, x2, y2 = ref_box
    rect_ref = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='yellow', facecolor='none')
    ax.add_patch(rect_ref)
    ax.text(x1 + 2, y1 - 6, 'S7', color='yellow', fontsize=10,
            bbox=dict(facecolor='black', alpha=0.4, pad=2))

    region_rect = patches.Rectangle(
        (ref_box[0], y_min),
        x_limit - ref_box[0],
        y_max - y_min,
        linewidth=1,
        edgecolor='cyan',
        facecolor='cyan',
        alpha=0.2
    )
    ax.add_patch(region_rect)

    ax.set_title(f'{image_name} – S7 기준 영역 (4x 폭, 중심 포함)')
    ax.axis('off')
    plt.show()

print('선택 영역을 계산한 이미지 수:', len(selection_regions))

"""### 3. 섹터 6/8 할당 및 시각화

기준 영역 안에 박스 **중심**이 포함되는 후보 가운데 y 값이 작은(이미지 상의 위쪽) 박스를 섹터 6, 큰(이미지 상의 아래쪽) 박스를 섹터 8로 표시합니다.

"""

if not selection_regions:
    raise RuntimeError('선택 영역 정보가 없습니다. 이전 셀을 먼저 실행하세요.')

assignment_results = {}

for image_name, region in selection_regions.items():
    data = image_detections[image_name]
    xyxy = data['xyxy']
    ref_idx = sector7_refs[image_name]['index']

    # 기준 박스(S7) 제외하고, 선택 영역 내에 중심이 포함되는 박스 인덱스 수집
    candidates = []
    for idx, (x1, y1, x2, y2) in enumerate(xyxy):
        if idx == ref_idx:
            continue
        center_x = 0.5 * (x1 + x2)
        center_y = 0.5 * (y1 + y2)
        if center_x < region['x_min'] or center_x > region['x_max']:
            continue
        if center_y < region['y_min'] or center_y > region['y_max']:
            continue
        candidates.append(idx)

    if len(candidates) < 2:
        print(f'{image_name}: 조건을 만족하는 박스가 {len(candidates)}개입니다.')
        continue

    candidates.sort(key=lambda idx: xyxy[idx, 0])
    selected = candidates[:2]

    # S6, S8 할당: y중심 기준으로 가장 위에 있는 박스가 S6, 가장 아래에 있는 박스가 S8
    centers = {idx: (xyxy[idx, 1] + xyxy[idx, 3]) / 2.0 for idx in selected} # 각 박스의 y중심 좌표 계산
    idx_six = min(selected, key=lambda idx: centers[idx])
    idx_eight = max(selected, key=lambda idx: centers[idx])

    assignment_results[image_name] = {
        'sector7': int(ref_idx),
        'sector6': int(idx_six),
        'sector8': int(idx_eight),
    }

    # 시각화: S7(노란색), S6(청록색), S8(진한핑크색) 박스 그리기, 나머지 박스들은 흰색 점선으로 표시
    image_path = data['path']
    img = Image.open(image_path).convert('RGB')
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(img)

    def draw_box(idx, color, label):
        x1, y1, x2, y2 = xyxy[idx]
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        ax.text(
            x1 + 2,
            y1 - 6,
            label,
            color=color,
            fontsize=10,
            bbox=dict(facecolor='black', alpha=0.4, pad=2)
        )

    draw_box(ref_idx, 'yellow', 'S7')
    draw_box(idx_six, 'cyan', 'S6')
    draw_box(idx_eight, 'magenta', 'S8')

    remaining = [idx for idx in range(xyxy.shape[0]) if idx not in {ref_idx, idx_six, idx_eight}]
    for idx in remaining:
        x1, y1, x2, y2 = xyxy[idx]
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                 linewidth=1, edgecolor='white', facecolor='none', linestyle='--')
        ax.add_patch(rect)

    ax.set_title(f'{image_name} – S7/S6/S8 최종 매핑 (중심 기반)')
    ax.axis('off')
    plt.show()

print('총 매핑 성공 이미지 수:', len(assignment_results))

"""### 4. 섹터 5/9 후보 탐지

조사 영역 설정: 섹터 7 박스를 기준으로 x 축은 이미지 오른쪽 끝까지 확장하고, y 범위는 이전 단계와 동일하게 유지해 후보(조사) 영역을 표시합니다.

"""

import matplotlib.patches as patches
from PIL import Image

if not selection_regions:
    raise RuntimeError('섹터 7 기준 정보가 없습니다. 이전 단계를 실행하세요.')

sector59_regions = {}

for image_name, region in selection_regions.items():
    data = image_detections[image_name]
    image_path = data['path']
    ref_box = sector7_refs[image_name]['box']

    img = Image.open(image_path).convert('RGB')
    img_width, img_height = img.size

    # 조사 후보 영역: x_min은 S7 왼쪽 테두리, x_max는 이미지 전체 너비, y_min/y_max는 이전에 계산한 값 사용
    x_min = ref_box[0]
    x_max = float(img_width)
    y_min = region['y_min']
    y_max = region['y_max']

    sector59_regions[image_name] = {
        'x_min': x_min,
        'x_max': x_max,
        'y_min': y_min,
        'y_max': y_max,
    }

    # 후보(조사) 영역 시각화
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(img)

    x1, y1, x2, y2 = ref_box
    rect_ref = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='yellow', facecolor='none')
    ax.add_patch(rect_ref)
    ax.text(x1 + 2, y1 - 6, 'S7', color='yellow', fontsize=10,
            bbox=dict(facecolor='black', alpha=0.4, pad=2))

    region_rect = patches.Rectangle(
        (x_min, y_min),
        x_max - x_min,
        y_max - y_min,
        linewidth=1,
        edgecolor='orange',
        facecolor='orange',
        alpha=0.2
    )
    ax.add_patch(region_rect)

    ax.set_title(f'{image_name} – S5/S9 후보 영역 (전체 너비)')
    ax.axis('off')
    plt.show()

print('섹터 5/9 후보 영역 계산 수:', len(sector59_regions))

"""#### 섹터 5/9 할당 및 시각화

후보 영역 안에서 박스 **중심**이 포함되는 후보 두 개를 찾아 y 값이 작은(이미지상의 위쪽) 박스를 섹터 5, 큰(이미지상의 아래쪽) 박스를 섹터 9로 표시합니다.

"""

import matplotlib.patches as patches
from PIL import Image

if not sector59_regions:
    raise RuntimeError('섹터 5/9 후보 영역 정보가 없습니다. 이전 셀을 실행하세요.')

sector59_results = {}

# 각 이미지에 대해 S7 박스, S6/S8 박스(이미 할당됐다면) 인덱스를 used_indices로 모아 두고, "후보 박스"는 중심점이 S5/S9 영역 안에 들어오는 박스 중에서 아직 사용하지 않은 것만 고릅니다.
for image_name, region in sector59_regions.items():
    data = image_detections[image_name]
    xyxy = data['xyxy']
    ref_idx = sector7_refs[image_name]['index']

    used_indices = {ref_idx}
    prev_assign = assignment_results.get(image_name, {})
    for key in ('sector6', 'sector8'):
        idx_val = prev_assign.get(key)
        if idx_val is not None:
            used_indices.add(idx_val)

    candidates = []
    for idx, (x1, y1, x2, y2) in enumerate(xyxy):
        if idx in used_indices:
            continue
        center_x = 0.5 * (x1 + x2)
        center_y = 0.5 * (y1 + y2)
        if center_x < region['x_min'] or center_x > region['x_max']:
            continue
        if center_y < region['y_min'] or center_y > region['y_max']:
            continue
        candidates.append((idx, center_y)) # candidates 리스트: (박스 인덱스, y중심) 튜플

    if len(candidates) < 2:
        print(f'{image_name}: 섹터 5/9 후보가 {len(candidates)}개입니다.') # 예외처리
        continue

    candidates.sort(key=lambda item: item[1])  # y중심을 기준으로 정렬: center_y ascending (top -> bottom)
    idx_five = candidates[0][0] # 이미지 위쪽(작은 y중심) 박스가 S5
    idx_nine = candidates[-1][0] # 이미지 아래쪽(큰 y중심) 박스가 S9

    sector59_results[image_name] = {
        'sector5': int(idx_five),
        'sector9': int(idx_nine),
    }

    if image_name not in assignment_results:
        assignment_results[image_name] = {}
    assignment_results[image_name].update({
        'sector5': int(idx_five),
        'sector9': int(idx_nine),
    })

    # 시각화: S7(노란색), S6(청록색), S8(진한핑크색), S5(주황색), S9(보라색) 박스 그리기, 나머지 박스들은 흰색 점선으로 표시
    img = Image.open(data['path']).convert('RGB')
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(img)

    def draw_box(idx, color, label):
        x1, y1, x2, y2 = xyxy[idx]
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        ax.text(x1 + 2, y1 - 6, label, color=color, fontsize=10,
                bbox=dict(facecolor='black', alpha=0.4, pad=2))

    draw_box(ref_idx, 'yellow', 'S7')
    if prev_assign:
        for key, color in (('sector6', 'cyan'), ('sector8', 'magenta')):
            idx_val = prev_assign.get(key)
            if idx_val is not None:
                draw_box(idx_val, color, key.replace('sector', 'S'))
    draw_box(idx_five, 'orange', 'S5')
    draw_box(idx_nine, 'purple', 'S9')

    remaining = [idx for idx in range(xyxy.shape[0]) if idx not in (used_indices | {idx_five, idx_nine})]
    for idx in remaining:
        x1, y1, x2, y2 = xyxy[idx]
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                 linewidth=1, edgecolor='white', facecolor='none', linestyle='--')
        ax.add_patch(rect)

    ax.set_title(f'{image_name} – S5/S9 매핑 (중심 기반)')
    ax.axis('off')
    plt.show()

print('섹터 5/9 매핑 성공 이미지 수:', len(sector59_results))

"""### 5. 남은 박스로 섹터 4/3/2/1 할당

이미 섹터 7/6/8/5/9로 지정된 박스를 제외한 나머지 박스 중심을 오른쪽부터 왼쪽 순으로 정렬해 S4, S3, S2, S1에 대응시키고 시각화합니다.
"""

import matplotlib.patches as patches
from PIL import Image

import matplotlib.patches as patches
from PIL import Image

final_assignments = {}

for image_name, data in image_detections.items():
    xyxy = data['xyxy']
    num_boxes = xyxy.shape[0]
    if num_boxes == 0:
        continue

    used_indices = set()
    refs = sector7_refs.get(image_name)
    if refs:
        used_indices.add(refs['index'])

    prev_assign = assignment_results.get(image_name, {})
    for key in ('sector6', 'sector8', 'sector5', 'sector9'):
        idx_val = prev_assign.get(key)
        if idx_val is not None:
            used_indices.add(idx_val)

    # "이미 사용한 인덱스(used_indices): (S7, S6, S8, S5, S9)"를 제외하고, 남은 박스를 center_x(박스 중심 x좌표) 기준으로 오른쪽→왼쪽 순으로 정렬해 순서대로 S4, S3, S2, S1에 할당합니다.
    candidate_indices = [idx for idx in range(num_boxes) if idx not in used_indices]
    if len(candidate_indices) < 4:
        print(f'{image_name}: 남은 박스가 {len(candidate_indices)}개입니다. (S4~S1 할당 불가)')
        continue

    centers = [(idx, 0.5 * (xyxy[idx, 0] + xyxy[idx, 2])) for idx in candidate_indices]
    centers.sort(key=lambda item: item[1], reverse=True)

    s4_idx, s3_idx, s2_idx, s1_idx = [idx for idx, _ in centers[:4]]

    if image_name not in assignment_results:
        assignment_results[image_name] = {}
    assignment_results[image_name].update({
        'sector4': int(s4_idx),
        'sector3': int(s3_idx),
        'sector2': int(s2_idx),
        'sector1': int(s1_idx),
    })

    final_assignments[image_name] = {
        'sector4': int(s4_idx),
        'sector3': int(s3_idx),
        'sector2': int(s2_idx),
        'sector1': int(s1_idx),
    }

    # 시각화: S7(노란색), S6(청록색), S8(진한핑크색), S5(주황색), S9(보라색), S4(빨간색), S3(라임색), S2(파란색), S1(흰색) 박스 그리기, 나머지 박스들은 회색 점선으로 표시
    img = Image.open(data['path']).convert('RGB')
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(img)

    def draw_box(idx, color, label):
        x1, y1, x2, y2 = xyxy[idx]
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        ax.text(x1 + 2, y1 - 6, label, color=color, fontsize=10,
                bbox=dict(facecolor='black', alpha=0.4, pad=2))

    ref_idx = sector7_refs.get(image_name, {}).get('index')
    if ref_idx is not None:
        draw_box(ref_idx, 'yellow', 'S7')

    for key, color in (('sector6', 'cyan'), ('sector8', 'magenta'), ('sector5', 'orange'), ('sector9', 'purple')):
        idx_val = assignment_results[image_name].get(key)
        if idx_val is not None:
            draw_box(idx_val, color, key.replace('sector', 'S'))

    draw_box(s4_idx, 'red', 'S4')
    draw_box(s3_idx, 'lime', 'S3')
    draw_box(s2_idx, 'blue', 'S2')
    draw_box(s1_idx, 'white', 'S1')

    remaining = [idx for idx in range(num_boxes) if idx not in assignment_results[image_name].values()]
    for idx in remaining:
        x1, y1, x2, y2 = xyxy[idx]
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                 linewidth=1, edgecolor='gray', facecolor='none', linestyle='--')
        ax.add_patch(rect)

    ax.set_title(f'{image_name} – S4/S3/S2/S1 매핑')
    ax.axis('off')
    plt.show()

print('섹터 4/3/2/1 매핑 완료 이미지 수:', len(final_assignments))

"""## 화재 건물 번호 탐지

추론된 결과에서 불(클래스 0)로 탐지된 박스를 찾아, 매핑된 섹터 번호를 이용해 `불이 난 곳은 건물 X번입니다.` 형식으로 요약합니다.

"""

import matplotlib.patches as patches
from PIL import Image

try:
    fire_class_id = FIRE_CLASS
except NameError:
    fire_class_id = 0
    if 'inference_model' in locals():
        for cls_id, name in getattr(inference_model, 'names', {}).items():
            if isinstance(name, str) and name.lower().startswith('fire'):
                fire_class_id = int(cls_id)
                break

if not image_detections:
    raise RuntimeError('추론 및 매핑 셀을 먼저 실행하세요.')

fire_summary = {}
fires_visualized = 0

for image_name, data in image_detections.items():
    # 각 이미지에 대해 감지된 클래스 ID 배열(cls_ids)에서 fire_class_id와 일치하는 인덱스(화재 박스)를 골라 냅니다. 화재 박스가 없다면 그 이미지는 건너뜁니다.
    cls_ids = data['cls_ids']
    fire_indices = [idx for idx, cls in enumerate(cls_ids) if cls == fire_class_id]
    if not fire_indices:
        continue

    #  이미 앞 단계에서 매핑해 둔 섹터 번호 정보를 가져와, 박스 인덱스를 섹터 번호(건물번호)로 대응
    assignments = assignment_results.get(image_name, {})
    idx_to_sector = {}
    for key, idx in assignments.items():
        if str(key).startswith('sector'):
            try:
                sector_num = int(str(key).replace('sector', ''))
            except ValueError:
                continue
            idx_to_sector[int(idx)] = sector_num

    matched_sectors = [] # 화재 박스 인덱스를 섹터 번호로 변환
    unmatched = [] # 화재 박스 인덱스 중에서 섹터 매핑이 안 된 것들
    for idx in fire_indices:
        sector = idx_to_sector.get(idx)
        if sector is not None:
            matched_sectors.append(sector)
        else:
            unmatched.append(idx)

    fire_summary[image_name] = {
        'matched': sorted(set(matched_sectors)),
        'unmatched': unmatched,
    }

    # 화재 건물 번호 콘솔에 출력
    if matched_sectors:
        sectors_text = ', '.join(f'{sector}번' for sector in sorted(set(matched_sectors)))
        print(f"{image_name}: 불이 난 곳은 건물 {sectors_text}입니다.")
    if unmatched:
        print(f"  [주의] {image_name}: 섹터 매핑되지 않은 화재 박스 {unmatched}")

"""#### 화재 박스 시각화"""

print('=== 화재 박스 시각화 ===')

# 앞서 만든 fire_summary 딕셔너리를 돌면서 화재 섹터가 있는 이미지들만 골라 처리
# 각 이미지에 대해 "불이 난 곳은 건물 X번입니다." 형태로 콘솔에 출력 + 시각화
# - (시각화) 기본적으로 기존 assignment_results에 들어있는 섹터 박스를 하얀색으로, 화재가 감지된 섹터는 빨간색+“(fire)” 라벨로 표시. 만약 화재로 감지됐지만 섹터 매핑이 안 된 박스가 있다면 주황색(fire(?))으로 표기
for image_name, summary in fire_summary.items():
    matched = summary['matched']
    if not matched:
        continue
    print(f'{image_name}: 불이 난 곳은 건물 ' + ', '.join(f'{sector}번' for sector in matched) + '입니다.')

    data = image_detections[image_name]
    xyxy = data['xyxy']
    img = Image.open(data['path']).convert('RGB')
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(img)

    def draw_box(idx, color, label):
        x1, y1, x2, y2 = xyxy[idx]
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        ax.text(x1 + 2, y1 - 6, label, color=color, fontsize=10,
                bbox=dict(facecolor='black', alpha=0.4, pad=2))

    seen = set()
    assignments = assignment_results.get(image_name, {})
    for key, idx in assignments.items():
        if not str(key).startswith('sector'):
            continue
        try:
            sector_num = int(str(key).replace('sector', ''))
        except ValueError:
            continue
        label = f'S{sector_num}'
        color = 'white'
        if sector_num in matched:
            color = 'red'
            label += ' (fire)'
        draw_box(idx, color, label)
        seen.add(idx)

    for idx in summary['unmatched']:
        if idx in seen:
            continue
        draw_box(idx, 'orange', 'fire(?)')

    ax.set_title(f'{image_name} – 화재 섹터: ' + ', '.join(f'S{sector}' for sector in matched))
    ax.axis('off')
    plt.show()
    fires_visualized += 1

# 마지막엔 화재 요약이 비었는지 여부를 한번 더 확인해서 “불로 감지된 박스가 없습니다.” 또는 “전체 화재 건물 번호: X” 같은 총괄 메시지를 출력하고, 시각화한 이미지 수까지 콘솔에 보여
if not fire_summary:
    print('불로 감지된 박스가 없습니다.')
else:
    all_sectors = sorted({sector for summary in fire_summary.values() for sector in summary['matched']})
    if all_sectors:
        print('전체 화재 건물 번호:', ', '.join(f'{sector}번' for sector in all_sectors))
    else:
        print('화재로 매핑된 건물 번호가 없습니다.')

print('시각화된 이미지 수:', fires_visualized)
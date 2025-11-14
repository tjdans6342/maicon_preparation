#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import math

def to_roi(
    image, #: np.ndarray,
    vertices, # np.ndarray,
    normalized=True, # bool = True
): # -> np.ndarray:
    """
    Extract and return only the polygonal region of interest (ROI) defined by given vertices.
    (Supports both absolute pixel coordinates and normalized ratio coordinates.)

    Parameters
    ----------
    image : np.ndarray
        입력 이미지 (BGR 또는 Grayscale 형식)
    vertices : np.ndarray
        ROI를 정의하는 다각형의 꼭짓점 좌표 배열.
        normalized=True일 경우, (x, y)는 [0.0~1.0] 비율로 표현됨.
        예: np.array([[(0.1, 0.7), (0.9, 0.7), (1.0, 1.0), (0.0, 1.0)]])
    normalized : bool, default=True
        True → vertices가 비율 단위로 입력됨 (자동으로 픽셀 좌표로 변환)
        False → vertices가 픽셀 단위로 직접 입력됨

    Returns
    -------
    roi_image : np.ndarray
        지정된 다각형 영역만 남기고 나머지 부분은 0(검정색)으로 마스크 처리된 이미지
    """

    h, w = image.shape[:2]

    # (1) 정규화 좌표를 실제 픽셀 좌표로 변환
    if normalized:
        vertices = np.array([
            [(int(x * w), int(y * h)) for (x, y) in vertices]
        ], dtype=np.int32)
    else:
        vertices = np.int32(vertices)

    # (2) 채널 수에 따라 마스크 색 결정
    white = (255, 255, 255) if len(image.shape) > 2 else 255

    # (3) 마스크 생성
    mask = cv2.fillPoly(np.zeros_like(image), vertices, white)

    # (4) AND 연산으로 ROI 부분만 남김
    roi_image = cv2.bitwise_and(image, mask)

    return roi_image


def to_bev(
    image,
    top=0.7, #: float = 0.7,
    bottom=0.0, #: float = 0.0,
    margin=0.2, #: float = 0.2,
    dst_size=None,
    normalized=True #: bool = True
): # -> tuple[np.ndarray, np.ndarray]:
    """
    Perform Bird’s Eye View (BEV) perspective transform based on proportional or pixel-based region settings.

    Parameters
    ----------
    image : np.ndarray
        입력 이미지 (BGR 형식)
    top : float or int, default=0.7
        BEV 변환 상단 경계 (비율 또는 픽셀값).
        normalized=True → 비율 (0~1)
        normalized=False → 픽셀 단위
    bottom : float or int, default=0.0
        BEV 변환 하단 경계 (비율 또는 픽셀값).
    margin : float or int, default=0.2
        윗변 좌우 여유폭 (비율 또는 픽셀 단위).
    dst_size : tuple(int, int) or None, default=None
        출력 BEV 이미지 크기 (width, height).
        None이면 원본 비율 기반으로 자동 계산됨.
    normalized : bool, default=True
        True → 비율로(top, bottom, margin) 계산
        False → 픽셀 단위로 직접 사용

    Returns
    -------
    bev_img : np.ndarray
        BEV 변환 결과 이미지
    Minv : np.ndarray
        BEV → 원본 시점으로 되돌리는 역변환 행렬
    """

    h, w = image.shape[:2]

    # ① 입력 단위 처리
    if normalized:
        y_top = int(h * (1 - top))
        y_bottom = int(h * (1 - bottom))
        x_margin = int(w * margin)
    else:
        y_top = int(h - top)
        y_bottom = int(h - bottom)
        x_margin = int(margin)

    # ② BEV 원본 영역 (사다리꼴)
    src = np.float32([
        [x_margin, y_top],          # 왼쪽 위
        [0, y_bottom],              # 왼쪽 아래
        [w - x_margin, y_top],      # 오른쪽 위
        [w, y_bottom]               # 오른쪽 아래
    ])

    # ③ BEV 출력 크기 설정
    if dst_size is None:
        dst_w = int(w * (1 - 2 * (margin if normalized else x_margin / w)))
        dst_h = int(h * (top - bottom) if normalized else y_bottom - y_top)
    else:
        dst_w, dst_h = dst_size

    dst = np.float32([
        [0, 0],
        [0, dst_h],
        [dst_w, 0],
        [dst_w, dst_h]
    ])

    # ④ BEV 변환 수행
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    bev_img = cv2.warpPerspective(image, M, (dst_w, dst_h), flags=cv2.INTER_LINEAR)

    return bev_img, Minv


# def color_filter(image: np.ndarray, hls_range: list, inverse: bool = False) -> np.ndarray:
def color_filter(image, hls_range, inverse=False):
    """
    Apply HLS-based color filtering on the input image with optional inversion.

    Parameters
    ----------
    image : np.ndarray
        BGR 형식의 입력 이미지
    hls_range : list
        필터링할 HLS 범위 리스트.
        예: [[(h1s, l1s, s1s), (h1e, l1e, s1e)], [(h2s, l2s, s2s), (h2e, l2e, s2e)]]
        각 튜플은 (H, L, S) 최소/최대값을 의미.
    inverse : bool, default=False
        False → 지정된 범위 내 픽셀만 남김 (정상 필터링)
        True  → 지정된 범위 내 픽셀을 제외하고 나머지를 남김 (역필터링)

    Returns
    -------
    masked : np.ndarray
        필터링된 결과 이미지
    """
    # BGR → HLS 변환
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)

    # 전체 마스크 초기화
    total_mask = np.zeros(hls.shape[:2], dtype=np.uint8)

    # 주어진 모든 범위에 대해 마스크 생성 (OR 연산)
    for lower, upper in hls_range:
        lower_np = np.array(lower, dtype=np.uint8)
        upper_np = np.array(upper, dtype=np.uint8)
        mask = cv2.inRange(hls, lower_np, upper_np)
        total_mask = cv2.bitwise_or(total_mask, mask)

    # inverse=True이면 마스크 반전
    if inverse:
        total_mask = cv2.bitwise_not(total_mask)

    # 원본 이미지에 마스크 적용
    masked = cv2.bitwise_and(image, image, mask=total_mask)

    return masked


def get_hough_image(canny_image, slope_threshold=10, min_votes=100):
    """
    Perform Hough Transform on a Canny edge image and remove near-horizontal lines.

    Parameters
    ----------
    canny_image : np.ndarray
        Binary image (result of cv2.Canny).
    slope_threshold : float, default=5
        Degree threshold: lines with |slope| < slope_threshold are considered horizontal and will be filtered out.
    min_votes : int, default=100
        Minimum number of votes required to consider a line (Hough accumulator threshold).

    Returns
    -------
    hough_img : np.ndarray
        Binary image (0 or 255) containing only filtered Hough lines.
    """

    # --- (1) Hough Transform 수행 ---
    lines = cv2.HoughLines(canny_image, 1, np.pi / 180, min_votes)
    h, w = canny_image.shape[:2]

    # --- (2) 결과 이미지 초기화 ---
    hough_img = np.zeros((h, w), dtype=np.uint8)

    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            a, b = np.cos(theta), np.sin(theta)
            x0, y0 = a * rho, b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))

            # --- (3) 기울기 계산 (0~90°) ---
            slope = 90 - abs(math.degrees(math.atan2(b, a)))

            # --- (4) 수평선 필터링 ---
            if abs(slope) < slope_threshold or abs(slope - 180) < slope_threshold:
                continue  # 수평에 가까운 선은 무시

            # --- (5) 유효한 선만 그리기 ---
            cv2.line(hough_img, (x1, y1), (x2, y2), 255, 2)

    return hough_img


def save_image(img, save_dir, filename_prefix="image", file_extension=".jpg"):
    """
    이미지를 파일로 저장합니다.
    
    Parameters
    ----------
    img : np.ndarray
        저장할 이미지 (BGR 형식)
    save_dir : str
        저장할 디렉토리 경로
    filename_prefix : str, default="image"
        파일명 접두사
    file_extension : str, default=".jpg"
        파일 확장자
    
    Returns
    -------
    str or None
        저장된 파일의 전체 경로. 실패 시 None
    """
    import os
    import cv2
    
    if img is None:
        return None
    
    # 디렉토리가 없으면 생성
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 파일명 생성 (타임스탬프 포함)
    import time
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = "{}_{}{}".format(filename_prefix, timestamp, file_extension)
    filepath = os.path.join(save_dir, filename)
    
    # 이미지 저장
    success = cv2.imwrite(filepath, img)
    
    if success:
        return filepath
    else:
        return None


def save_image_with_counter(img, save_dir, marker_id, counter_dict, prefix="triggered_object", file_extension=".jpg"):
    """
    마커 ID와 카운터를 사용하여 이미지를 저장합니다.
    
    Parameters
    ----------
    img : np.ndarray
        저장할 이미지
    save_dir : str
        저장할 디렉토리 경로
    marker_id : int or str
        마커 ID
    counter_dict : dict
        마커 ID별 카운터를 저장하는 딕셔너리 (참조로 전달)
    prefix : str, default="triggered_object"
        파일명 접두사
    file_extension : str, default=".jpg"
        파일 확장자
    
    Returns
    -------
    str or None
        저장된 파일의 전체 경로. 실패 시 None
    """
    import os
    import cv2
    
    if img is None:
        return None
    
    # 디렉토리가 없으면 생성
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 카운터 증가
    if marker_id not in counter_dict:
        counter_dict[marker_id] = 0
    counter_dict[marker_id] += 1
    
    # 파일명 생성
    filename = "{}{}_{}{}".format(prefix, marker_id, counter_dict[marker_id], file_extension)
    filepath = os.path.join(save_dir, filename)
    
    # 이미지 저장
    success = cv2.imwrite(filepath, img)
    
    if success:
        return filepath
    else:
        return None


def should_capture_frame(frame_index, last_capture_index, interval=30):
    """
    이전 캡처 이후 일정 프레임 간격이 지났는지 확인합니다.
    
    Parameters
    ----------
    frame_index : int
        현재 프레임 인덱스
    last_capture_index : int
        마지막 캡처 프레임 인덱스
    interval : int, default=30
        캡처 간격 (프레임 수)
    
    Returns
    -------
    bool
        캡처해야 하면 True, 아니면 False
    """
    return (frame_index - last_capture_index) >= interval


def should_capture_time(current_time, last_capture_time, cooldown=5.0):
    """
    이전 캡처 이후 일정 시간(초)이 지났는지 확인합니다.
    
    Parameters
    ----------
    current_time : float
        현재 시간 (time.time() 값)
    last_capture_time : float
        마지막 캡처 시간 (time.time() 값)
    cooldown : float, default=5.0
        쿨다운 시간 (초)
    
    Returns
    -------
    bool
        캡처해야 하면 True, 아니면 False
    """
    return (current_time - last_capture_time) >= cooldown
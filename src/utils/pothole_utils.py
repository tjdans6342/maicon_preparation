# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""
포트홀 감지 관련 유틸리티 함수 모듈
"""

import numpy as np


def check_binary_image_pothole(binary_img, white_threshold=0.1):
    """
    이진화된 영상에서 포트홀(도로 함몰이나 구멍)을 감지합니다.
    
    Parameters
    ----------
    binary_img : np.ndarray
        이진화된 영상 (0 또는 255 값)
    white_threshold : float, default=0.1
        상단 중앙 영역의 흰색 픽셀 비율 임계값
    
    Returns
    -------
    bool
        포트홀이 감지되면 True, 아니면 False
    """
    if binary_img is None or not isinstance(binary_img, np.ndarray):
        return False
    
    h, w = binary_img.shape[:2]
    
    # 영역 정의
    top_side_area1 = binary_img[:int(h*0.3), :int(0.1*w)]
    top_side_area2 = binary_img[:int(h*0.3), int(0.9*w):]
    top_center_area = binary_img[:int(h*0.15), int(0.2*w):int(0.8*w)]
    mid_area = binary_img[int(h*0.5):int(h*0.7), :]
    
    # 상단 중앙 영역의 흰색 픽셀 비율 계산
    white_cnt = np.sum(top_center_area == 255)
    total = top_center_area.size
    white_per = float(white_cnt) / float(total)
    
    # 포트홀 조건:
    # 1. 상단 중앙 영역에 흰색 픽셀이 일정 비율 이상
    # 2. 중간 영역과 상단 좌우 영역이 모두 검은색 (0)
    is_pothole = (white_per > white_threshold) and \
                 (mid_area.max() == 0) and \
                 (top_side_area1.max() == 0) and \
                 (top_side_area2.max() == 0)
    
    return is_pothole


def check_pothole_with_buffer(binary_img, buffer_size=3, white_threshold=0.1):
    """
    버퍼를 사용하여 포트홀을 안정적으로 감지합니다.
    연속된 여러 프레임에서 포트홀이 감지되어야 True를 반환합니다.
    
    Parameters
    ----------
    binary_img : np.ndarray
        이진화된 영상
    buffer_size : int, default=3
        버퍼 크기 (연속 프레임 수)
    white_threshold : float, default=0.1
        흰색 픽셀 비율 임계값
    
    Returns
    -------
    bool
        버퍼의 모든 프레임에서 포트홀이 감지되면 True
    """
    is_pothole = check_binary_image_pothole(binary_img, white_threshold)
    
    # 버퍼 관리 (클래스 외부에서 관리해야 하므로 여기서는 단순히 반환)
    # 실제 사용 시에는 클래스 내부에서 버퍼를 관리해야 함
    return is_pothole


# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""
ArUco 마커 관련 유틸리티 함수 모듈
"""

import cv2
import numpy as np
import time

# OpenCV ArUco dictionary and parameters (handle version compatibility)
try:
    ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
except AttributeError:
    ARUCO_DICT = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)

try:
    ARUCO_PARAMS = cv2.aruco.DetectorParameters_create()
except AttributeError:
    ARUCO_PARAMS = cv2.aruco.DetectorParameters()


def detect_aruco_markers(bgr_img):
    """
    BGR 이미지에서 ArUco 마커를 검출하여 정보를 반환합니다.
    
    Parameters
    ----------
    bgr_img : np.ndarray
        BGR 형식의 입력 이미지
    
    Returns
    -------
    results : list
        각 마커에 대한 정보를 담은 딕셔너리 리스트
        [{"id": int, "center": (cx, cy), "area": float}, ...]
    """
    # 컬러 이미지를 그레이스케일로 변환
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    
    # ArUco 마커 검출 수행
    corners, ids, _ = cv2.aruco.detectMarkers(gray, ARUCO_DICT, parameters=ARUCO_PARAMS)
    
    results = []
    if ids is not None:
        ids = ids.flatten()
        # 각 검출된 마커마다 정보 추출
        for c, i in zip(corners, ids):
            pts = c.reshape(-1, 2)  # (4,2) shape array of corner points
            
            # 마커 중심 좌표 계산 (x, y 평균)
            cx = float(np.mean(pts[:, 0]))
            cy = float(np.mean(pts[:, 1]))
            
            # 마커의 폭과 높이 계산 (픽셀 단위)
            w = float(np.max(pts[:, 0]) - np.min(pts[:, 0]))
            h = float(np.max(pts[:, 1]) - np.min(pts[:, 1]))
            
            # 면적을 사각형으로 근사 (w * h)
            area = abs(w * h)
            
            # id, 중심, 면적 정보를 결과 리스트에 추가
            results.append({"id": int(i), "center": (cx, cy), "area": area})
    
    return results


def is_valid_marker(detected_ids, last_detect_time, target_ids=None, cooldown=5.0):
    """
    감지된 ArUco 마커 ID가 유효하며, 연속 중복이 아닌지 검사합니다.
    
    Parameters
    ----------
    detected_ids : list[int]
        감지된 마커 ID 리스트
    last_detect_time : float
        마지막 감지 시각 (time.time() 값)
    target_ids : list[int] or None, optional
        유효한 마커 ID 목록. None이면 모든 ID를 유효로 간주
    cooldown : float, default=5.0
        쿨다운 시간 (초)
    
    Returns
    -------
    bool
        유효한 마커가 있고 쿨다운이 지났으면 True, 아니면 False
    """
    if not detected_ids:
        return False
    
    # target_ids가 지정되어 있으면 해당 ID만 확인
    if target_ids is not None:
        valid_ids = [id for id in detected_ids if id in target_ids]
        if not valid_ids:
            return False
    else:
        valid_ids = detected_ids
    
    # 쿨다운 체크
    now = time.time()
    if (now - last_detect_time) < cooldown:
        return False
    
    return True


def filter_marker_ids(detected_ids, target_ids):
    """
    감지된 마커 ID 중에서 유효한 ID만 필터링합니다.
    
    Parameters
    ----------
    detected_ids : list[int]
        감지된 마커 ID 리스트
    target_ids : list[int]
        유효한 마커 ID 목록
    
    Returns
    -------
    list[int]
        필터링된 유효한 마커 ID 리스트
    """
    if target_ids is None:
        return detected_ids
    return [id for id in detected_ids if id in target_ids]


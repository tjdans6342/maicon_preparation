# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""
로봇 동작 관련 유틸리티 함수 모듈
"""

import math


def compute_turn_duration(angle_deg, angular_speed=1.0):
    """
    주어진 각도와 각속도로부터 필요한 회전 시간을 계산합니다.
    
    Parameters
    ----------
    angle_deg : float
        회전할 각도 (도 단위)
    angular_speed : float, default=1.0
        각속도 (라디안/초)
    
    Returns
    -------
    float
        회전에 필요한 시간 (초)
    """
    if angular_speed <= 0:
        return 0.0
    
    # 회전 지속 시간 = 회전 각도(라디안) / 각속도(라디안/초)
    angle_rad = abs(angle_deg) * math.pi / 180.0
    duration = angle_rad / abs(angular_speed)
    
    return duration


def compute_drive_duration(distance, speed):
    """
    주어진 거리와 속도로부터 필요한 주행 시간을 계산합니다.
    
    Parameters
    ----------
    distance : float
        이동할 거리 (미터)
    speed : float
        주행 속도 (미터/초)
    
    Returns
    -------
    float
        주행에 필요한 시간 (초)
    """
    if speed <= 0:
        return 0.0
    
    duration = abs(distance / speed)
    return duration


def compute_circle_duration(diameter, speed):
    """
    반원형 주행 경로의 주행 시간을 계산합니다.
    
    Parameters
    ----------
    diameter : float
        회피 경로의 지름 (미터)
    speed : float
        주행 속도 (미터/초)
    
    Returns
    -------
    float
        반원 주행에 필요한 시간 (초)
    """
    if speed <= 0:
        return 0.0
    
    # 반원의 호 길이 = π * (diameter / 2)
    arc_length = math.pi * diameter / 2.0
    duration = arc_length / speed
    
    return duration


def compute_circle_angular_speed(diameter, curvature, speed, arrow="left"):
    """
    원형 주행을 위한 각속도를 계산합니다.
    
    Parameters
    ----------
    diameter : float
        회피 경로의 지름 (m)
    curvature : float
        곡률 (1.0 → 정확한 반원, 0.5 → 완만한 곡선)
    speed : float
        주행 속도 (m/s)
    arrow : str, default="left"
        회전 방향 ("left" or "right")
    
    Returns
    -------
    tuple
        (angular_speed, radius) - 각속도 (rad/s)와 반지름 (m)
    """
    import numpy as np
    curvature += np.finfo(float).eps
    
    # 회전 방향 부호
    sign = 1.0 if arrow == "left" else -1.0
    
    # 실제 반지름 계산
    R = (1.0 / curvature) * (diameter / 2.0)
    
    # 각속도(ω = v / R)
    angular_speed = sign * (speed / R)
    
    return angular_speed, R


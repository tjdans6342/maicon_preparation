# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""
ArUco 트리거 관련 유틸리티 함수
"""

import time


def check_cooldown(marker_id, last_trigger_times, cooldown_per_id, cooldown_default):
    """
    마커의 쿨다운이 지났는지 확인합니다.
    
    Parameters
    ----------
    marker_id : int
        마커 ID
    last_trigger_times : dict
        마커 ID별 마지막 트리거 시각 {id: time}
    cooldown_per_id : dict
        마커 ID별 쿨다운 시간 {id: seconds}
    cooldown_default : float
        기본 쿨다운 시간 (초)
    
    Returns
    -------
    bool
        쿨다운이 지났으면 True, 아니면 False
    """
    now = time.time()
    last = last_trigger_times.get(marker_id, 0.0)
    cooldown = cooldown_per_id.get(marker_id, cooldown_default)
    return (now - last) >= cooldown


def check_consecutive_frames(marker_id, required_consecutive, consec_dict):
    """
    연속 프레임 조건을 만족하는지 확인합니다.
    
    Parameters
    ----------
    marker_id : int
        마커 ID
    required_consecutive : int
        필요한 연속 프레임 수
    consec_dict : dict
        마커 ID별 연속 프레임 카운트 {id: count}
    
    Returns
    -------
    bool
        조건을 만족하면 True
    """
    current_count = consec_dict.get(marker_id, 0)
    return current_count >= required_consecutive


def get_marker_action(rules, marker_id, nth):
    """
    마커 ID와 등장 횟수에 해당하는 액션을 가져옵니다.
    
    Parameters
    ----------
    rules : dict
        규칙 딕셔너리
    marker_id : int or str
        마커 ID
    nth : int
        등장 횟수
    
    Returns
    -------
    list or tuple or None
        액션 리스트/튜플. 없으면 None
    """
    if marker_id not in rules:
        return None
    if nth not in rules[marker_id]:
        return None
    return rules[marker_id][nth]


def normalize_action(action):
    """
    액션을 리스트 형태로 정규화합니다.
    
    Parameters
    ----------
    action : tuple or list
        액션
    
    Returns
    -------
    list
        정규화된 액션 리스트
    """
    if isinstance(action, tuple):
        return [action]
    return list(action)


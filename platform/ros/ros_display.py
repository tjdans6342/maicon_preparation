# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""
ROS 환경용 디스플레이 구현체
ROS 환경에서는 OLED가 없으므로 rospy.loginfo로 대체
"""

import rospy
import sys
import os

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
from interface.display_interface import DisplayInterface


class ROSDisplay(DisplayInterface):
    """
    ROS 환경에서 디스플레이 출력을 처리하는 구현체
    OLED가 없으므로 ROS 로그로 대체
    """
    
    def __init__(self):
        rospy.loginfo("[ROSDisplay] Initialized (using ROS log)")
    
    def log(self, message):
        """
        메시지를 ROS 로그로 출력합니다.
        
        Parameters
        ----------
        message : str
            출력할 메시지
        """
        rospy.loginfo("[Display] {}".format(message))
    
    def log_clear(self):
        """
        디스플레이를 초기화합니다.
        ROS 환경에서는 의미 없지만 호환성을 위해 구현
        """
        # ROS 환경에서는 화면 지우기 기능이 없음
        pass


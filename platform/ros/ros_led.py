# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""
ROS 환경용 LED 제어 구현체
ROS 환경에서 LED가 없을 수 있으므로 로그로 대체
"""

import rospy
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))
from interface.led_interface import LEDInterface


class ROSLedController(LEDInterface):
    """
    ROS 환경에서 LED를 제어하는 구현체
    LED가 없을 경우 로그로만 출력
    """
    
    def __init__(self):
        rospy.loginfo("[ROSLedController] Initialized (LED may not be available)")
    
    def set_color(self, position, index, color):
        """
        LED 색상을 설정합니다.
        ROS 환경에서는 로그로만 출력
        
        Parameters
        ----------
        position : str
            LED 위치
        index : int
            LED 인덱스
        color : tuple
            RGB 색상 값 (R, G, B)
        """
        rospy.loginfo("[LED] Set {}[{}] to RGB{}".format(position, index, color))
        # 실제 LED 제어는 ROS 토픽/서비스로 구현 가능
    
    def set_all(self, color):
        """
        모든 LED를 동일한 색상으로 설정합니다.
        
        Parameters
        ----------
        color : tuple
            RGB 색상 값
        """
        rospy.loginfo("[LED] Set all to RGB{}".format(color))
    
    def off(self):
        """
        모든 LED를 끕니다.
        """
        rospy.loginfo("[LED] All LEDs off")


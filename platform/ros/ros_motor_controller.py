# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""
ROS 환경용 모터 제어 구현체
Adapter 패턴: MotorInterface -> ROS Twist 메시지
"""

import rospy
from geometry_msgs.msg import Twist
import sys
import os

# 프로젝트 루트 경로 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))
from interface.motor_interface import MotorInterface


class ROSMotorController(MotorInterface):
    """
    ROS 환경에서 모터를 제어하는 구현체
    /cmd_vel 토픽에 Twist 메시지를 발행하여 로봇을 제어
    """
    
    def __init__(self, topic_name="/cmd_vel"):
        """
        Parameters
        ----------
        topic_name : str, default="/cmd_vel"
            ROS 토픽 이름
        """
        self.pub = rospy.Publisher(topic_name, Twist, queue_size=1)
        self._last_cmd = Twist()
        rospy.loginfo("[ROSMotorController] Initialized → publishing to {}".format(topic_name))
    
    def set_speed(self, left, right):
        """
        왼쪽/오른쪽 모터 속도를 설정합니다.
        Differential drive 변환: linear.x = 평균 속도, angular.z = 차이
        
        Parameters
        ----------
        left : float
            왼쪽 모터 속도 (-1.0 ~ 1.0)
        right : float
            오른쪽 모터 속도 (-1.0 ~ 1.0)
        """
        # Differential drive 변환
        # 단순 예시: 평균 속도를 직진속도로, 차이를 회전으로
        linear = (left + right) / 2.0
        angular = (right - left) / 0.5  # 베이스폭 0.5m 가정 (실제 값으로 조정 필요)
        
        self.set_linear_angular(linear, angular)
    
    def set_linear_angular(self, linear, angular):
        """
        선속도와 각속도로 모터를 제어합니다.
        
        Parameters
        ----------
        linear : float
            선속도 (m/s)
        angular : float
            각속도 (rad/s, +좌회전 / -우회전)
        """
        msg = Twist()
        msg.linear.x = float(linear)
        msg.angular.z = float(angular)
        self.pub.publish(msg)
        self._last_cmd = msg
    
    def stop(self):
        """
        모터를 정지시킵니다.
        """
        msg = Twist()
        msg.linear.x = 0.0
        msg.angular.z = 0.0
        self.pub.publish(msg)
        self._last_cmd = msg
    
    def get_last_command(self):
        """
        마지막 명령을 조회합니다 (디버깅용).
        
        Returns
        -------
        Twist
            마지막으로 발행한 Twist 메시지
        """
        return self._last_cmd


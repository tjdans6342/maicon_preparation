#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
from geometry_msgs.msg import Twist
import sys
import os

# 프로젝트 루트 경로 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))
from interface.motor_interface import MotorInterface


class Controller:
    """
    ✅ 로봇 제어 모듈 (리팩토링됨)
    - MotorInterface를 사용하여 플랫폼 독립적으로 동작
    - 기존 호환성 유지: topic_name으로 초기화 시 자동으로 ROSMotorController 생성
    - 또는 motor_interface를 직접 주입 가능
    """

    def __init__(self, topic_name="/cmd_vel", motor_interface=None):
        """
        Parameters
        ----------
        topic_name : str, default="/cmd_vel"
            ROS 토픽 이름 (motor_interface가 None일 때만 사용)
        motor_interface : MotorInterface, optional
            모터 제어 인터페이스. None이면 자동으로 ROSMotorController 생성
        """
        if motor_interface is None:
            # 기존 호환성 유지: topic_name으로 초기화
            from platform.ros.ros_motor_controller import ROSMotorController
            self.motor = ROSMotorController(topic_name=topic_name)
        else:
            # 인터페이스 주입 방식
            self.motor = motor_interface
        
        self._last_linear = 0.0
        self._last_angular = 0.0
        rospy.loginfo("🕹️ Controller initialized")

    # -------------------------------------------------------
    #  퍼블리시 함수 (로봇 이동 명령)
    # -------------------------------------------------------
    def publish(self, linear=0.0, angular=0.0):
        """
        linear:  m/s 단위 선속도
        angular: rad/s 단위 각속도 (+좌회전 / -우회전)
        """
        self.motor.set_linear_angular(linear, angular)
        self._last_linear = linear
        self._last_angular = angular

    # -------------------------------------------------------
    #  로봇 정지
    # -------------------------------------------------------
    def stop(self):
        """
        로봇을 정지시킵니다.
        """
        self.motor.stop()
        self._last_linear = 0.0
        self._last_angular = 0.0

    # -------------------------------------------------------
    #  마지막 명령 조회 (디버깅용)
    # -------------------------------------------------------
    def get_last_command(self):
        """
        마지막 명령을 조회합니다.
        
        Returns
        -------
        dict
            {"linear": float, "angular": float}
        """
        return {"linear": self._last_linear, "angular": self._last_angular}

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
from geometry_msgs.msg import Twist


class Controller:
    """
    ✅ /cmd_vel 퍼블리셔 모듈
    - 모든 주행 명령의 단일 인터페이스
    - publish(linear, angular)로 간단히 사용
    """

    def __init__(self, topic_name="/cmd_vel"):
        self.pub = rospy.Publisher(topic_name, Twist, queue_size=1)
        self._last_cmd = Twist()
        rospy.loginfo("🕹️ Controller initialized → publishing to {}".format(topic_name))

    # -------------------------------------------------------
    #  퍼블리시 함수 (로봇 이동 명령)
    # -------------------------------------------------------
    def publish(self, linear=0.0, angular=0.0):
        """
        linear:  m/s 단위 선속도
        angular: rad/s 단위 각속도 (+좌회전 / -우회전)
        """
        msg = Twist()
        msg.linear.x = float(linear)
        msg.angular.z = float(angular)
        self.pub.publish(msg)

        self._last_cmd = msg

    # -------------------------------------------------------
    #  로봇 정지
    # -------------------------------------------------------
    def stop(self):
        self.publish(0.0, 0.0)
        # rospy.loginfo("🛑 Controller: STOP command sent")

    # -------------------------------------------------------
    #  마지막 명령 조회 (디버깅용)
    # -------------------------------------------------------
    def get_last_command(self):
        return self._last_cmd

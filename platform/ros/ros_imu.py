# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""
ROS 환경용 IMU 센서 구현체
"""

import rospy
from sensor_msgs.msg import Imu
import sys
import os

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
from interface.imu_interface import IMUInterface


class ROSImu(IMUInterface):
    """
    ROS 환경에서 IMU 데이터를 받는 구현체
    sensor_msgs/Imu 토픽을 구독
    """
    
    def __init__(self, imu_topic="/imu/data"):
        """
        Parameters
        ----------
        imu_topic : str, default="/imu/data"
            IMU 토픽 이름
        """
        self.accel = [0.0, 0.0, 0.0]
        self.gyro = [0.0, 0.0, 0.0]
        self.yaw = 0.0
        
        self.sub = rospy.Subscriber(imu_topic, Imu, self._imu_callback)
        rospy.loginfo("[ROSImu] Subscribed to {}".format(imu_topic))
    
    def _imu_callback(self, msg):
        """IMU 메시지 콜백"""
        # 가속도 (m/s^2)
        self.accel = [msg.linear_acceleration.x, 
                      msg.linear_acceleration.y, 
                      msg.linear_acceleration.z]
        
        # 자이로 (rad/s)
        self.gyro = [msg.angular_velocity.x,
                     msg.angular_velocity.y,
                     msg.angular_velocity.z]
        
        # Yaw는 quaternion에서 계산 필요 (간단히 angular_velocity.z 사용)
        # 실제로는 quaternion을 euler angle로 변환해야 함
        self.yaw = msg.angular_velocity.z  # 임시
    
    def get_acceleration(self):
        """
        가속도 값을 가져옵니다.
        
        Returns
        -------
        tuple
            (ax, ay, az) 가속도 값 (m/s^2)
        """
        return tuple(self.accel)
    
    def get_gyro(self):
        """
        자이로 값을 가져옵니다.
        
        Returns
        -------
        tuple
            (gx, gy, gz) 자이로 값 (rad/s)
        """
        return tuple(self.gyro)
    
    def get_yaw(self):
        """
        Yaw 각도를 가져옵니다.
        
        Returns
        -------
        float
            Yaw 각도 (라디안)
        """
        return self.yaw


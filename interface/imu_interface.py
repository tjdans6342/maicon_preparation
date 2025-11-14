# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""
IMU 센서 인터페이스 추상 클래스
"""


class IMUInterface(object):
    """
    IMU 센서 데이터를 위한 추상 인터페이스
    """
    
    def get_acceleration(self):
        """
        가속도 값을 가져옵니다.
        
        Returns
        -------
        tuple or list
            (ax, ay, az) 가속도 값 (m/s^2)
        """
        raise NotImplementedError("Subclass must implement get_acceleration")
    
    def get_gyro(self):
        """
        자이로 값을 가져옵니다.
        
        Returns
        -------
        tuple or list or None
            (gx, gy, gz) 자이로 값 (rad/s). 지원되지 않으면 None
        """
        raise NotImplementedError("Subclass must implement get_gyro")
    
    def get_yaw(self):
        """
        Yaw 각도를 가져옵니다.
        
        Returns
        -------
        float or None
            Yaw 각도 (라디안). 지원되지 않으면 None
        """
        raise NotImplementedError("Subclass must implement get_yaw")


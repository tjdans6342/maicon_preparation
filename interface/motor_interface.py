# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""
모터 제어 인터페이스 추상 클래스
Adapter 패턴 적용: ROS/Tiki 환경을 통일된 인터페이스로 추상화
"""


class MotorInterface(object):
    """
    모터 제어를 위한 추상 인터페이스
    ROS와 Tiki 환경 모두에서 동일한 인터페이스로 모터를 제어할 수 있도록 함
    """
    
    def set_speed(self, left, right):
        """
        왼쪽/오른쪽 모터 속도 설정
        
        Parameters
        ----------
        left : float
            왼쪽 모터 속도 (범위: -1.0 ~ 1.0 일반화)
        right : float
            오른쪽 모터 속도 (범위: -1.0 ~ 1.0 일반화)
        """
        raise NotImplementedError("Subclass must implement set_speed")
    
    def stop(self):
        """
        모터 정지
        """
        raise NotImplementedError("Subclass must implement stop")
    
    def set_linear_angular(self, linear, angular):
        """
        선속도와 각속도로 모터 제어 (differential drive 변환)
        
        Parameters
        ----------
        linear : float
            선속도 (m/s)
        angular : float
            각속도 (rad/s, +좌회전 / -우회전)
        """
        raise NotImplementedError("Subclass must implement set_linear_angular")


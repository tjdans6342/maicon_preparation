# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""
디스플레이(OLED/로그) 인터페이스 추상 클래스
"""


class DisplayInterface(object):
    """
    디스플레이 출력을 위한 추상 인터페이스
    ROS 환경에서는 로그 출력, Tiki 환경에서는 OLED 출력
    """
    
    def log(self, message):
        """
        메시지를 디스플레이에 출력합니다.
        
        Parameters
        ----------
        message : str
            출력할 메시지
        """
        raise NotImplementedError("Subclass must implement log")
    
    def log_clear(self):
        """
        디스플레이를 초기화합니다.
        (ROS 환경에서는 의미 없을 수 있음)
        """
        raise NotImplementedError("Subclass must implement log_clear")


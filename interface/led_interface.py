# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""
LED 제어 인터페이스 추상 클래스
"""


class LEDInterface(object):
    """
    LED 제어를 위한 추상 인터페이스
    """
    
    def set_color(self, position, index, color):
        """
        LED 색상을 설정합니다.
        
        Parameters
        ----------
        position : str
            LED 위치 (예: "top", "front", "back")
        index : int
            LED 인덱스
        color : tuple
            RGB 색상 값 (R, G, B) 각각 0-255 범위
        """
        raise NotImplementedError("Subclass must implement set_color")
    
    def set_all(self, color):
        """
        모든 LED를 동일한 색상으로 설정합니다.
        
        Parameters
        ----------
        color : tuple
            RGB 색상 값 (R, G, B)
        """
        raise NotImplementedError("Subclass must implement set_all")
    
    def off(self):
        """
        모든 LED를 끕니다.
        """
        raise NotImplementedError("Subclass must implement off")


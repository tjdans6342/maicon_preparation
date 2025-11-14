# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""
카메라 인터페이스 추상 클래스
"""

import numpy as np


class CameraInterface(object):
    """
    카메라 이미지 캡처를 위한 추상 인터페이스
    """
    
    def capture(self):
        """
        한 프레임의 이미지를 캡처합니다.
        
        Returns
        -------
        np.ndarray or None
            BGR 형식의 이미지. 실패 시 None
        """
        raise NotImplementedError("Subclass must implement capture")
    
    def is_available(self):
        """
        카메라가 사용 가능한지 확인합니다.
        
        Returns
        -------
        bool
            카메라가 사용 가능하면 True
        """
        raise NotImplementedError("Subclass must implement is_available")


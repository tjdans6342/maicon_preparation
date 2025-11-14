# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""
ROS 환경용 카메라 구현체
"""

import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CompressedImage
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))
from interface.camera_interface import CameraInterface


class ROSCamera(CameraInterface):
    """
    ROS 환경에서 카메라 이미지를 받는 구현체
    sensor_msgs/Image 또는 CompressedImage 토픽을 구독
    """
    
    def __init__(self, image_topic="/usb_cam/image_raw/compressed", use_compressed=True):
        """
        Parameters
        ----------
        image_topic : str, default="/usb_cam/image_raw/compressed"
            이미지 토픽 이름
        use_compressed : bool, default=True
            CompressedImage 사용 여부
        """
        self.bridge = CvBridge()
        self.image = None
        self.use_compressed = use_compressed
        
        if use_compressed:
            self.sub = rospy.Subscriber(image_topic, CompressedImage, self._compressed_callback)
        else:
            self.sub = rospy.Subscriber(image_topic, Image, self._image_callback)
        
        rospy.loginfo("[ROSCamera] Subscribed to {}".format(image_topic))
    
    def _compressed_callback(self, msg):
        """CompressedImage 콜백"""
        try:
            np_arr = np.fromstring(msg.data, np.uint8)
            self.image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        except Exception as e:
            rospy.logwarn("[ROSCamera] Failed to decode compressed image: {}".format(e))
    
    def _image_callback(self, msg):
        """Image 콜백"""
        try:
            self.image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            rospy.logwarn("[ROSCamera] Failed to convert image: {}".format(e))
    
    def capture(self):
        """
        최신 프레임을 반환합니다.
        
        Returns
        -------
        np.ndarray or None
            BGR 형식의 이미지. 없으면 None
        """
        return self.image
    
    def is_available(self):
        """
        카메라가 사용 가능한지 확인합니다.
        
        Returns
        -------
        bool
            이미지가 있으면 True
        """
        return self.image is not None


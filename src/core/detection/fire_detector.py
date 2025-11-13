#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fire Building Detector - ROS wrapper for fire detection pipeline
Optimized for Jetson Nano
"""

import os
import random
# import torch
from pathlib import Path
from typing import List

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

## TODO: Adjust import path if fire_detection_pipeline is in different location
from fire.fire_detection_pipeline import detect_fire_buildings


class FireBuildingDetector:
    """
    Wrapper class for fire detection in buildings
    - Loads YOLO model once at initialization
    - Provides simple interface to detect fires in images
    - Optimized for Jetson Nano performance
    """
    
    def __init__(self, weights_path, device=None):
        """
        Initialize fire detection model
        
        Args:
            weights_path: Path to YOLO weights file (best.pt)
            device: 'cuda' or 'cpu', auto-detect if None
        """
        self.weights_path = weights_path
        
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        print("[FireBuildingDetector] Using device: {}".format(self.device))
        
    def detect(self, image_path, conf_threshold=0.25, img_size=416):
        """
        Detect fire in buildings from image
        
        Args:
            image_path: Path to input image
            conf_threshold: Confidence threshold
            img_size: Image size for inference
        
        Returns:
            List[int]: Building numbers on fire (1-9)
        """
        fire_buildings = detect_fire_buildings(
            image_path=image_path,
            weights_path=self.weights_path,
            device=self.device,
            conf_threshold=conf_threshold,
            img_size=img_size
        )
        
        return fire_buildings
    
    def detect_random_from_folder(self, folder_path, conf_threshold=0.25, img_size=416):
        """
        Select random image from folder and detect fires
        
        Args:
            folder_path: Path to folder containing building images
            conf_threshold: Confidence threshold
            img_size: Image size for inference
        
        Returns:
            tuple: (selected_image_path, fire_buildings_list)
        """
        folder = Path(folder_path)
        image_extensions = ('.jpg', '.jpeg', '.png')
        images = [f for f in folder.iterdir() if f.suffix.lower() in image_extensions]
        
        if not images:
            print("[FireBuildingDetector] No images found in {}".format(folder_path))
            return None, []
        
        selected_image = random.choice(images)
        print("[FireBuildingDetector] Selected image: {}".format(selected_image.name))
        
        fire_buildings = self.detect(
            image_path=str(selected_image),
            conf_threshold=conf_threshold,
            img_size=img_size
        )
        
        return str(selected_image), fire_buildings
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LaneConfig
- LaneDetector에서 사용하는 모든 설정값을 한 곳에서 관리.
- 실험 환경(조명, 카메라, 시야각)에 따라 손쉽게 수정 가능.
"""

import os
import yaml

class LaneConfig:
    """✅ Lane Detection Configuration"""

    def __init__(self, yaml_path=None):
        """
        Parameters
        ----------
        yaml_path : str, optional
            YAML 설정 파일 경로. 없으면 기본값을 사용.
        """

        # --- Default Configuration ---
        self.display_mode = False
        self.image_names = ["Original", "BEV", "Filtered", "Canny", "Hough"]

        # --- ROI / BEV 설정 ---
        self.bev_normalized = True
        self.roi_top = 0.75
        self.roi_bottom = 0.0
        self.roi_margin = 0.1

        # --- 색상 필터 (HLS 범위) ---
        self.hls = [[(0, 160, 0), (180, 255, 255)]] # default: white line

        # --- 이미지 전처리 ---
        self.binary_threshold = (20, 255)

        # --- Sliding Window 설정 ---
        self.nwindows = 15
        self.width = 100
        self.minpix = 15

        # --- YAML 파일이 주어졌다면 override ---
        if yaml_path and os.path.exists(yaml_path):
            self.load_from_yaml(yaml_path)


    # -------------------------------------------------------
    #  동적 설정 업데이트
    # -------------------------------------------------------
    def update(self, **kwargs):
        """
        동적으로 설정값 수정 (예: cfg.update(display_mode=False, hls=새값))
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print("[LaneConfig] ⚠️ Unknown key ignored: {}".format(key))

    # -------------------------------------------------------
    #  YAML 파일 로드
    # -------------------------------------------------------
    def load_from_yaml(self, yaml_path):
        with open(yaml_path, "r") as f:
            cfg = yaml.safe_load(f)

        for key, value in cfg.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print("[LaneConfig] ⚠️ Unknown key ignored: {}".format(key))

    # -------------------------------------------------------
    #  YAML로 저장 (선택)
    # -------------------------------------------------------
    def save_to_yaml(self, yaml_path):
        cfg_dict = {
            k: v for k, v in self.__dict__.items()
            if not k.startswith("_")
        }
        with open(yaml_path, "w") as f:
            yaml.safe_dump(cfg_dict, f, allow_unicode=True)
        print("[LaneConfig] Saved to {}".format(yaml_path))

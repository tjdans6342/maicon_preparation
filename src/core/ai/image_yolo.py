#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
from collections import defaultdict

import cv2
from ultralytics import YOLO

# Optional: check for GPU availability
try:
    import torch
except ImportError:
    print("Warning: torch module not found. YOLO will use CPU.")
    torch = None

# Select inference device
if torch and torch.cuda.is_available():
    INFERENCE_DEVICE = 0
    print(f"[YOLO] Using GPU (device {INFERENCE_DEVICE})")
else:
    INFERENCE_DEVICE = 'cpu'
    print("[YOLO] Using CPU")

# Load YOLO model
MODEL = YOLO('config/best_rokaf.pt')
CONFIDENCE_THRESHOLD = 0.3

# Directory to watch
IMAGE_DIR = os.path.expanduser("~/catkin_ws/src/ROKAF_Autonomous_Car_2025/yolo_images")
SLEEP_SEC = 1.0

def main():
    class_counts = defaultdict(int)
    processed_files = set()

    if not os.path.isdir(IMAGE_DIR):
        raise RuntimeError(f"[ERROR] Image directory does not exist: {IMAGE_DIR}")

    print(f"[YOLO] Watching directory: {IMAGE_DIR}")
    print("[YOLO] Press Ctrl+C to stop\n")

    while True:
        filenames = sorted(f for f in os.listdir(IMAGE_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png")))
        new_files = [f for f in filenames if f not in processed_files]

        for fname in new_files:
            full_path = os.path.join(IMAGE_DIR, fname)
            img = cv2.imread(full_path)

            if img is None:
                print(f"[WARN] Failed to read image: {full_path}")
                processed_files.add(fname)
                continue

            results = MODEL(img, verbose=False, device=INFERENCE_DEVICE)
            per_image_counts = defaultdict(int)

            for r in results:
                if not r.boxes:
                    continue
                for box in r.boxes:
                    conf = float(box.conf[0])
                    if conf < CONFIDENCE_THRESHOLD:
                        continue
                    class_id = int(box.cls[0])
                    label = MODEL.names[class_id]
                    class_counts[label] += 1
                    per_image_counts[label] += 1

            processed_files.add(fname)

            print(f"=== Image processed: {fname} ===")
            if per_image_counts:
                print("  [Per-image counts]")
                for label, count in sorted(per_image_counts.items()):
                    print(f"   - {label}: {count}")
            else:
                print(f"  No objects detected above conf {CONFIDENCE_THRESHOLD:.2f}")

            print("\n  [Cumulative counts so far]")
            for label, count in sorted(class_counts.items()):
                print(f"   - {label}: {count}")
            print("================================\n")

        time.sleep(SLEEP_SEC)

if __name__ == "__main__":
    main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Lane Analysis Video Recorder
- Records all lane detection pipeline stages in a grid layout
- Side-by-side comparison for performance analysis
"""

import cv2
import numpy as np
import rospy
import threading
import Queue as queue
import os
from datetime import datetime


# ============================================================
# CONFIGURATION - Customize parameters here
# ============================================================

# Recording settings
ENABLED = True  # Enable/disable recording
OUTPUT_DIR = "robot_videos/lane_analysis"  # Output directory
FPS = 15  # Video frame rate
CODEC = "XVID"  # Video codec (XVID, mp4v, MJPG)

# Grid layout
GRID_ROWS = 3  # Number of rows in grid
GRID_COLS = 3  # Number of columns in grid
CELL_WIDTH = 320  # Width of each cell (pixels)
CELL_HEIGHT = 240  # Height of each cell (pixels)

# Visual settings
ADD_LABELS = True  # Add text labels to each cell
ADD_TIMESTAMP = True  # Add timestamp to video
LABEL_FONT_SCALE = 0.7  # Font size for labels
LABEL_COLOR = (255, 255, 255)  # White text
LABEL_BG_COLOR = (0, 0, 0)  # Black background

# Queue settings
MAX_QUEUE_SIZE = 30  # Maximum frames in buffer

# Pipeline image order (must match LaneDetector output)
PIPELINE_NAMES = [
    "Original", "BEV", "Filtered",
    "gray", "Blurred", "binary",
    "Canny", "Hough", "Lane Detection"
]

# ============================================================


class LaneAnalysisRecorder:
    """
    Records lane detection pipeline stages as grid video
    
    Usage:
        recorder = LaneAnalysisRecorder()
        recorder.start_recording()
        recorder.add_pipeline_frame(image_dict)
        recorder.stop_recording()
    """
    
    def __init__(self):
        """Initialize lane analysis video recorder"""
        # Threading components
        self.frame_queue = queue.Queue(maxsize=MAX_QUEUE_SIZE)
        self.recording_thread = None
        self.stop_event = threading.Event()
        
        # Recording state
        self.is_recording_flag = False
        self.video_writer = None
        self.current_file = None
        self.frame_count = 0
        
        # Create output directory (including parent directories)
        if ENABLED and not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR, exist_ok=True)  # exist_ok=True prevents error if already exists
            rospy.loginfo("[LaneAnalysisRecorder] Created directory: {}".format(OUTPUT_DIR))
        
        rospy.loginfo("[LaneAnalysisRecorder] Initialized")
    
    def start_recording(self):
        """Start recording lane analysis video"""
        if self.is_recording_flag:
            rospy.logwarn("[LaneAnalysisRecorder] Already recording")
            return False
        
        if not ENABLED:
            rospy.loginfo("[LaneAnalysisRecorder] Recording disabled in config")
            return False
        
        # Reset stop event
        self.stop_event.clear()
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = "lane_analysis_{}.avi".format(timestamp)
        self.current_file = os.path.join(OUTPUT_DIR, filename)
        
        # Start recording thread
        self.recording_thread = threading.Thread(target=self._recording_loop)
        self.recording_thread.daemon = True
        self.recording_thread.start()
        
        self.is_recording_flag = True
        rospy.loginfo("[LaneAnalysisRecorder] Recording started: {}".format(self.current_file))
        return True
    
    def stop_recording(self):
        """Stop recording video"""
        if not self.is_recording_flag:
            return False
        
        rospy.loginfo("[LaneAnalysisRecorder] Stopping...")
        
        # Signal thread to stop
        self.stop_event.set()
        
        # Wait for thread to finish
        if self.recording_thread is not None:
            self.recording_thread.join(timeout=5.0)
        
        # Close video file
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
        
        self.is_recording_flag = False
        
        # Log final info
        if self.current_file and os.path.exists(self.current_file):
            size_mb = os.path.getsize(self.current_file) / (1024 * 1024)
            rospy.loginfo("[LaneAnalysisRecorder] Saved: {} ({:.1f} MB, {} frames)".format(
                self.current_file, size_mb, self.frame_count
            ))
        
        return True
    
    def add_pipeline_frame(self, image_dict):
        """
        Add pipeline images to recording
        
        Parameters
        ----------
        image_dict : dict
            Dictionary with keys matching PIPELINE_NAMES
            e.g., {"Original": img1, "BEV": img2, ...}
        
        Returns
        -------
        bool : True if added, False if dropped
        """
        if not self.is_recording_flag or image_dict is None:
            return False
        
        try:
            # Add to queue without blocking
            self.frame_queue.put_nowait(image_dict.copy())
            return True
        except queue.Full:
            # Queue full, drop frame
            return False
    
    def is_recording(self):
        """Check if currently recording"""
        return self.is_recording_flag
    
    def _create_grid_frame(self, image_dict):
        """
        Create grid layout from pipeline images
        
        Parameters
        ----------
        image_dict : dict
            Dictionary of pipeline images
        
        Returns
        -------
        np.ndarray : Grid image (BGR)
        """
        # Calculate grid size
        grid_width = GRID_COLS * CELL_WIDTH
        grid_height = GRID_ROWS * CELL_HEIGHT
        grid = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
        
        # Place each image in grid
        for idx, name in enumerate(PIPELINE_NAMES):
            if name not in image_dict:
                continue
            
            img = image_dict[name]
            if img is None:
                continue
            
            # Calculate grid position
            row = idx // GRID_COLS
            col = idx % GRID_COLS
            y_start = row * CELL_HEIGHT
            x_start = col * CELL_WIDTH
            
            # Convert grayscale to BGR if needed
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            
            # Resize to cell size
            img_resized = cv2.resize(img, (CELL_WIDTH, CELL_HEIGHT))
            
            # Add label if enabled
            if ADD_LABELS:
                # Add background rectangle for text
                label_height = 30
                cv2.rectangle(img_resized, (0, 0), (CELL_WIDTH, label_height), 
                            LABEL_BG_COLOR, -1)
                
                # Add text label
                cv2.putText(img_resized, name, (10, 20),
                          cv2.FONT_HERSHEY_SIMPLEX, LABEL_FONT_SCALE,
                          LABEL_COLOR, 2, cv2.LINE_AA)
            
            # Place in grid
            grid[y_start:y_start+CELL_HEIGHT, x_start:x_start+CELL_WIDTH] = img_resized
        
        # Add timestamp if enabled
        if ADD_TIMESTAMP:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(grid, timestamp, (10, grid_height - 10),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                      (0, 255, 0), 2, cv2.LINE_AA)
        
        return grid
    
    def _recording_loop(self):
        """
        Recording loop running in separate thread
        Gets frames from queue and writes to video file
        """
        rospy.loginfo("[LaneAnalysisRecorder] Recording thread started")
        
        while not self.stop_event.is_set():
            try:
                # Get frame from queue (timeout to check stop_event)
                image_dict = self.frame_queue.get(timeout=0.1)
                
                # Create grid frame
                grid_frame = self._create_grid_frame(image_dict)
                
                # Initialize video writer on first frame
                if self.video_writer is None:
                    h, w = grid_frame.shape[:2]
                    
                    fourcc = cv2.VideoWriter_fourcc(*CODEC)
                    self.video_writer = cv2.VideoWriter(
                        self.current_file,
                        fourcc,
                        FPS,
                        (w, h)
                    )
                    
                    if not self.video_writer.isOpened():
                        rospy.logerr("[LaneAnalysisRecorder] Failed to create video file")
                        break
                    
                    rospy.loginfo("[LaneAnalysisRecorder] Writing {}x{} @ {} fps".format(
                        w, h, FPS
                    ))
                
                # Write frame
                self.video_writer.write(grid_frame)
                self.frame_count += 1
                
                # Mark task done
                self.frame_queue.task_done()
                
            except queue.Empty:
                # No frame available, continue
                continue
            except Exception as e:
                rospy.logerr("[LaneAnalysisRecorder] Error: {}".format(e))
                break
        
        # Flush remaining frames
        while not self.frame_queue.empty():
            try:
                image_dict = self.frame_queue.get_nowait()
                grid_frame = self._create_grid_frame(image_dict)
                if self.video_writer is not None:
                    self.video_writer.write(grid_frame)
                    self.frame_count += 1
                self.frame_queue.task_done()
            except queue.Empty:
                break
        
        rospy.loginfo("[LaneAnalysisRecorder] Recording thread finished")
    
    def __del__(self):
        """Cleanup on destruction"""
        if self.is_recording_flag:
            self.stop_recording()


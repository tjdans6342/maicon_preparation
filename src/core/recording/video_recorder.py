#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simple Video Recorder with Threading
- Records video from camera in background thread
- Non-blocking operation
"""

import cv2
import rospy
import threading
import queue
import os
from datetime import datetime


class VideoRecorder:
    """
    Simple video recorder with threading
    
    Usage:
        recorder = VideoRecorder(config)
        recorder.start_recording()
        recorder.add_frame(frame)  # Call this from main loop
        recorder.stop_recording()
    """

    def __init__(self, config):
        """
        Initialize video recorder
        
        Parameters
        ----------
        config : VideoConfig
            Configuration object
        """
        self.cfg = config
        
        # Threading components
        self.frame_queue = queue.Queue(maxsize=self.cfg.max_queue_size)
        self.recording_thread = None
        self.stop_event = threading.Event()
        
        # Recording state
        self.is_recording_flag = False
        self.video_writer = None
        self.current_file = None
        self.frame_count = 0
        
        rospy.loginfo("VideoRecorder initialized")

    def start_recording(self):
        """Start recording video"""
        if self.is_recording_flag:
            rospy.logwarn("[VideoRecorder] Already recording")
            return False
        
        if not self.cfg.enabled:
            rospy.loginfo("[VideoRecorder] Recording disabled in config")
            return False
        
        # Reset stop event
        self.stop_event.clear()
        
        # Generate filename
        self.current_file = self.cfg.generate_filename()
        
        # Start recording thread
        self.recording_thread = threading.Thread(
            target=self._recording_loop,
            daemon=True
        )
        self.recording_thread.start()
        
        self.is_recording_flag = True
        rospy.loginfo("Recording started: {}".format(self.current_file))
        return True

    def stop_recording(self):
        """Stop recording video"""
        if not self.is_recording_flag:
            return False
        
        rospy.loginfo("[VideoRecorder] Stopping...")
        
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
            rospy.loginfo("[VideoRecorder] Saved: {} ({:.1f} MB, {} frames)".format(
                self.current_file, size_mb, self.frame_count
            ))
        
        return True

    def add_frame(self, frame):
        """
        Add frame to recording (called from main thread)
        
        Parameters
        ----------
        frame : np.ndarray
            BGR image
            
        Returns
        -------
        bool : True if added, False if dropped
        """
        if not self.is_recording_flag or frame is None:
            return False
        
        try:
            # Add to queue without blocking
            self.frame_queue.put_nowait(frame.copy())
            return True
        except queue.Full:
            # Queue full, drop frame
            return False

    def is_recording(self):
        """Check if currently recording"""
        return self.is_recording_flag

    def _add_timestamp(self, frame):
        """Add timestamp overlay to frame"""
        if not self.cfg.add_timestamp:
            return frame
        
        frame_copy = frame.copy()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        cv2.putText(
            frame_copy,
            timestamp,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )
        
        return frame_copy

    def _recording_loop(self):
        """
        Recording loop running in separate thread
        Gets frames from queue and writes to video file
        """
        rospy.loginfo("[VideoRecorder] Recording thread started")
        
        while not self.stop_event.is_set():
            try:
                # Get frame from queue (timeout to check stop_event)
                frame = self.frame_queue.get(timeout=0.1)
                
                # Initialize video writer on first frame
                if self.video_writer is None:
                    h, w = frame.shape[:2]
                    
                    # Apply resolution if specified
                    if self.cfg.resolution is not None:
                        w, h = self.cfg.resolution
                        frame = cv2.resize(frame, (w, h))
                    
                    # Create video writer
                    fourcc = cv2.VideoWriter_fourcc(*self.cfg.codec)
                    self.video_writer = cv2.VideoWriter(
                        self.current_file,
                        fourcc,
                        self.cfg.fps,
                        (w, h)
                    )
                    
                    if not self.video_writer.isOpened():
                        rospy.logerr("[VideoRecorder] Failed to create video file")
                        break
                    
                    rospy.loginfo("[VideoRecorder] Writing {}x{} @ {} fps".format(
                        w, h, self.cfg.fps
                    ))
                
                # Resize if needed
                if self.cfg.resolution is not None:
                    frame = cv2.resize(frame, self.cfg.resolution)
                
                # Add timestamp
                frame = self._add_timestamp(frame)
                
                # Write frame
                self.video_writer.write(frame)
                self.frame_count += 1
                
                # Mark task done
                self.frame_queue.task_done()
                
            except queue.Empty:
                # No frame available, continue
                continue
            except Exception as e:
                rospy.logerr("[VideoRecorder] Error: {}".format(e))
                break
        
        # Flush remaining frames
        while not self.frame_queue.empty():
            try:
                frame = self.frame_queue.get_nowait()
                if self.video_writer is not None:
                    if self.cfg.resolution is not None:
                        frame = cv2.resize(frame, self.cfg.resolution)
                    frame = self._add_timestamp(frame)
                    self.video_writer.write(frame)
                    self.frame_count += 1
                self.frame_queue.task_done()
            except queue.Empty:
                break
        
        rospy.loginfo("[VideoRecorder] Recording thread finished")

    def __del__(self):
        """Cleanup on destruction"""
        if self.is_recording_flag:
            self.stop_recording()
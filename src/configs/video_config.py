
"""
Simple Video Recording Configuration
"""

import os
from datetime import datetime


class VideoConfig:
    """Simple video recording configuration"""

    def __init__(self):
        # Enable/disable recording
        self.enabled = True  # Set to True to enable video recording
        
        # Output directory for videos
        self.output_dir = "robot_videos"  # Changed to relative path
        
        # Video settings
        self.fps = 15  # Frames per second
        self.codec = "XVID"  # Video codec (XVID, mp4v, MJPG)
        self.resolution = (316, 316)  # None = keep original size
        
        # Overlay settings
        self.add_timestamp = True  # Add timestamp text to video
        
        # Queue settings
        self.max_queue_size = 50  # Maximum frames in buffer
        
        # Create output directory if needed
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print("[VideoConfig] Created directory: {}".format(self.output_dir))
    
    def generate_filename(self):
        """Generate filename with timestamp"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = "robot_video_{}.avi".format(timestamp)
        return os.path.join(self.output_dir, filename)
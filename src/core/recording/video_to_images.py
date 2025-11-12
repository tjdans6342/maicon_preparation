#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Video to Images Converter
- Extract frames from recorded videos for AI training
- 1 frame per second by default
"""

import cv2
import os
import argparse
import re
from datetime import datetime


# ============================================================
# CONFIGURATION - Customize parameters here
# ============================================================

# Default paths (can be changed)
DEFAULT_VIDEO_DIR = "robot_videos"  # Directory containing videos
DEFAULT_OUTPUT_DIR = None  # None = save with video, or specify different path

# Frame extraction parameters
DEFAULT_FPS = 1  # Frames per second (1 = extract 1 image per second)
JPEG_QUALITY = 95  # JPEG quality (0-100, 95 = high quality)

# Supported video formats
SUPPORTED_VIDEO_EXTENSIONS = ['.avi', '.mp4', '.mov', '.mkv', '.MP4', '.AVI']

# Output filename format
# Can change pattern: "frame_{timestamp}_{number}.jpg"
OUTPUT_FILENAME_PATTERN = "frame_{timestamp}_{number:03d}.jpg"

# Create subfolder for each video?
CREATE_SUBFOLDER_PER_VIDEO = True  # True = separate folder, False = save directly

# ============================================================


def extract_timestamp_from_filename(video_filename):
    """
    Extract timestamp from video filename
    e.g., 'robot_video_20250112_143022.avi' -> '20250112_143022'
    
    Returns:
        str: timestamp string or current time if not found
    """
    match = re.search(r'(\d{8}_\d{6})', video_filename)
    if match:
        return match.group(1)
    else:
        # Fallback to current time
        return datetime.now().strftime("%Y%m%d_%H%M%S")


def process_video(video_path, output_dir=None, fps=1):
    """
    Convert video to images (1 frame per second)
    
    Parameters
    ----------
    video_path : str
        Path to video file
    output_dir : str, optional
        Output directory (default: same as video)
    fps : int
        Number of frames to extract per second (default: 1)
    
    Returns
    -------
    int : Number of frames extracted
    """
    if not os.path.exists(video_path):
        print("[ERROR] Video not found: {}".format(video_path))
        return 0
    
    # Get video info
    video_name = os.path.basename(video_path)
    video_name_no_ext = os.path.splitext(video_name)[0]
    
    # Set output directory
    if output_dir is None:
        output_dir = os.path.dirname(video_path)
    
    # Create output subfolder for this video (if configured)
    if CREATE_SUBFOLDER_PER_VIDEO:
        output_folder = os.path.join(output_dir, "{}_frames".format(video_name_no_ext))
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            print("[INFO] Created output folder: {}".format(output_folder))
    else:
        output_folder = output_dir
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("[ERROR] Cannot open video: {}".format(video_path))
        return 0
    
    # Get video properties
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if video_fps == 0:
        print("[ERROR] Invalid video FPS")
        cap.release()
        return 0
    
    # Calculate frame interval (extract 1 frame per second)
    frame_interval = int(video_fps / fps)
    if frame_interval < 1:
        frame_interval = 1
    
    print("[INFO] Processing: {}".format(video_name))
    print("       Video FPS: {:.1f}, Total frames: {}".format(video_fps, total_frames))
    print("       Extracting 1 frame every {} frames ({} fps)".format(frame_interval, fps))
    
    # Extract timestamp from filename
    timestamp = extract_timestamp_from_filename(video_name)
    
    # Extract frames
    frame_count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Save frame at interval
        if frame_count % frame_interval == 0:
            # Generate filename using pattern from config
            frame_filename = OUTPUT_FILENAME_PATTERN.format(
                timestamp=timestamp, 
                number=saved_count + 1
            )
            frame_path = os.path.join(output_folder, frame_filename)
            
            # Save with specified JPEG quality
            cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            saved_count += 1
            
            # Progress indicator
            if saved_count % 10 == 0:
                print("       ... extracted {} frames".format(saved_count))
        
        frame_count += 1
    
    cap.release()
    
    print("[SUCCESS] Extracted {} images to: {}".format(saved_count, output_folder))
    return saved_count


def process_directory(input_dir, output_dir=None, fps=1):
    """
    Process all videos in a directory
    
    Parameters
    ----------
    input_dir : str
        Directory containing video files
    output_dir : str, optional
        Output directory
    fps : int
        Frames per second to extract
    
    Returns
    -------
    int : Total number of frames extracted
    """
    if not os.path.isdir(input_dir):
        print("[ERROR] Directory not found: {}".format(input_dir))
        return 0
    
    # Find all video files (using extensions from config)
    video_files = []
    
    for filename in os.listdir(input_dir):
        ext = os.path.splitext(filename)[1]
        if ext in SUPPORTED_VIDEO_EXTENSIONS:
            video_files.append(os.path.join(input_dir, filename))
    
    if not video_files:
        print("[INFO] No video files found in: {}".format(input_dir))
        return 0
    
    print("="*60)
    print("[INFO] Found {} video(s) to process".format(len(video_files)))
    print("="*60)
    
    # Process each video
    total_frames = 0
    for i, video_path in enumerate(video_files, 1):
        print("\n[{}/{}]".format(i, len(video_files)))
        frames = process_video(video_path, output_dir, fps)
        total_frames += frames
    
    print("\n" + "="*60)
    print("[COMPLETE] Total frames extracted: {}".format(total_frames))
    print("="*60)
    
    return total_frames


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Convert robot videos to training images (1 frame per second)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single video
  python video_to_images.py --input robot_videos/robot_video_20250112_143022.avi
  
  # Process all videos in directory
  python video_to_images.py --input robot_videos/
  
  # Extract 2 frames per second
  python video_to_images.py --input robot_videos/ --fps 2
  
  # Specify output directory
  python video_to_images.py --input robot_videos/ --output training_data/
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Path to video file or directory containing videos'
    )
    
    parser.add_argument(
        '--output', '-o',
        default=None,
        help='Output directory (default: same as input)'
    )
    
    parser.add_argument(
        '--fps',
        type=int,
        default=DEFAULT_FPS,
        help='Number of frames to extract per second (default: {})'.format(DEFAULT_FPS)
    )
    
    args = parser.parse_args()
    
    # Check if input is file or directory
    if os.path.isfile(args.input):
        # Process single video
        process_video(args.input, args.output, args.fps)
    elif os.path.isdir(args.input):
        # Process directory
        process_directory(args.input, args.output, args.fps)
    else:
        print("[ERROR] Input path does not exist: {}".format(args.input))
        return 1
    
    return 0


if __name__ == "__main__":
    main()


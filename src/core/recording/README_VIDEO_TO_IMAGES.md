
```bash
cd C:\Maicon\code_main\maicon_preparation

python src/core/recording/video_to_images.py --input robot_videos/robot_video_20250112_143022.avi
```


```bash
python src/core/recording/video_to_images.py --input robot_videos/
```



```bash
python src/core/recording/video_to_images.py --input robot_videos/ --fps 2
```


```bash
python src/core/recording/video_to_images.py --input robot_videos/ --output training_data/
```

## Output


```
robot_videos/
├── robot_video_20250112_143022.avi
└── robot_video_20250112_143022_frames/
    ├── frame_20250112_143022_001.jpg
    ├── frame_20250112_143022_002.jpg
    ├── frame_20250112_143022_003.jpg
    └── ...
```

```
[INFO] Found 3 video(s) to process
==================================================
[1/3]
[INFO] Processing: robot_video_20250112_143022.avi
       Video FPS: 15.0, Total frames: 450
       Extracting 1 frame every 15 frames (1 fps)
       ... extracted 10 frames
       ... extracted 20 frames
       ... extracted 30 frames
[SUCCESS] Extracted 30 images to: robot_videos/robot_video_20250112_143022_frames
==================================================
[COMPLETE] Total frames extracted: 90
==================================================
```



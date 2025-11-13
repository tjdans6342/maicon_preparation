
### Configuration
File: `src/configs/video_config.py`

```python
self.enabled = True  # Enable/disable
self.output_dir = "robot_videos"  # Output directory
self.fps = 15  # Frame rate
self.codec = "XVID"  # Video codec
self.resolution = (316, 316)  # Video resolution
self.add_timestamp = True  # Add timestamp overlay
```

### Output
```
robot_videos/
├── robot_video_20250112_143022.avi
├── robot_video_20250112_150315.avi
└── ...
```

### Features
- ✅ Original camera feed
- ✅ Timestamp overlay
- ✅ Configurable resolution (default: 316x316)
- ✅ Non-blocking threading

### Use Cases
- General overview recording
- Convert to images for training (`video_to_images.py`)
- Share/replay runs
- Backup original footage

---

## 2. Lane Analysis Video Recording

### Description
Ghi video grid 3x3 hiển thị tất cả các bước xử lý lane detection pipeline.

### Configuration
File: `src/core/recording/lane_analysis_recorder.py`

```python
ENABLED = True  # Enable/disable
OUTPUT_DIR = "robot_videos/lane_analysis"  # Output directory
FPS = 15  # Frame rate
GRID_ROWS = 3  # Grid rows
GRID_COLS = 3  # Grid columns
CELL_WIDTH = 320  # Cell width
CELL_HEIGHT = 240  # Cell height
ADD_LABELS = True  # Add labels to each cell
```

### Grid Layout
```
┌─────────────┬─────────────┬─────────────┐
│  Original   │     BEV     │  Filtered   │
├─────────────┼─────────────┼─────────────┤
│    Gray     │   Blurred   │   Binary    │
├─────────────┼─────────────┼─────────────┤
│    Canny    │    Hough    │Lane Detection│
└─────────────┴─────────────┴─────────────┘
```

### Output
```
robot_videos/
└── lane_analysis/
    ├── lane_analysis_20250112_143022.avi
    ├── lane_analysis_20250112_150315.avi
    └── ...
```

### Features
- ✅ 9 pipeline stages in one video
- ✅ Side-by-side comparison
- ✅ Labels for each stage
- ✅ Timestamp overlay
- ✅ High resolution (960x720 default)

### Use Cases
- Debug lane detection issues
- Optimize detection parameters
- Analyze control performance
- Visualize processing pipeline
- Training/presentation material

---

## Usage

### Start Robot (Both recordings auto-start)
```bash
rosrun your_package robot.py
```

Both recorders will start automatically and save to their respective directories.

### Stop Recording
Recording automatically stops when robot shuts down (Ctrl+C).

---

## Disable/Enable Recordings

### Disable Original Video
In `src/configs/video_config.py`:
```python
self.enabled = False
```

### Disable Lane Analysis Video
In `src/core/recording/lane_analysis_recorder.py`:
```python
ENABLED = False
```

### Disable Both
Set both to `False` in their respective config files.

---

## Output Directory Structure

```
robot_videos/
├── robot_video_20250112_143022.avi        # Original video
├── robot_video_20250112_150315.avi
├── robot_video_20250112_163045.avi
│
└── lane_analysis/                          # Lane analysis videos
    ├── lane_analysis_20250112_143022.avi
    ├── lane_analysis_20250112_150315.avi
    └── lane_analysis_20250112_163045.avi
```

---

## Performance Impact

### Original Video Recording
- **CPU Impact:** ~1-2% (minimal)
- **Memory:** ~10-20 MB queue buffer
- **Storage:** ~50-100 MB per minute (depends on resolution)

### Lane Analysis Recording
- **CPU Impact:** ~3-5% (grid composition)
- **Memory:** ~20-30 MB queue buffer
- **Storage:** ~150-300 MB per minute (higher resolution)

### Combined
- **Total CPU:** ~4-7% overhead
- **Total Memory:** ~30-50 MB
- **Non-blocking:** Does not affect control loop performance

---

## Comparison

| Feature | Original Video | Lane Analysis Video |
|---------|---------------|---------------------|
| **Content** | Raw camera feed | 9 pipeline stages |
| **Resolution** | 316x316 (default) | 960x720 (default) |
| **File Size** | Smaller | Larger |
| **Use Case** | General recording | Debug & analysis |
| **CPU Impact** | Lower | Slightly higher |
| **Configuration** | VideoConfig | lane_analysis_recorder.py |

---

## Tips

### For Long Recordings (Storage Saving)
```python
# Original video - lower resolution
self.resolution = (240, 240)

# Lane analysis - smaller cells
CELL_WIDTH = 240
CELL_HEIGHT = 180
FPS = 10
```

### For High Quality Analysis
```python
# Original video - higher resolution
self.resolution = (480, 480)

# Lane analysis - larger cells
CELL_WIDTH = 400
CELL_HEIGHT = 300
FPS = 20
```

### For Normal Use (Balanced)
Use default settings - already optimized for balance.

---

## Additional Tools

### Convert Video to Images
Use `video_to_images.py` to extract frames from original videos:

```bash
python src/core/recording/video_to_images.py --input robot_videos/
```

See: [`README_VIDEO_TO_IMAGES.md`](README_VIDEO_TO_IMAGES.md)

---

## Troubleshooting

### Videos not recording
1. Check `enabled = True` in configs
2. Verify output directories have write permissions
3. Check ROS logs for error messages

### High CPU usage
1. Lower FPS in both configs
2. Reduce resolution/cell size
3. Increase queue sizes to reduce drops

### Large file sizes
1. Use MJPG codec (better compression)
2. Lower resolution
3. Reduce FPS

### Queue full / dropped frames
1. Increase `MAX_QUEUE_SIZE`
2. Lower FPS to match processing speed
3. Check disk write speed

---

## File References

- **Original Video Recorder:** `src/core/recording/video_recorder.py`
- **Original Video Config:** `src/configs/video_config.py`
- **Lane Analysis Recorder:** `src/core/recording/lane_analysis_recorder.py`
- **Lane Detector:** `src/core/detection/lane_detector.py`
- **Main Robot:** `src/main/robot.py`
- **Video to Images Converter:** `src/core/recording/video_to_images.py`

---

## Quick Start Checklist

- [x] Original video recording enabled in `video_config.py`
- [x] Lane analysis recording enabled in `lane_analysis_recorder.py`
- [x] Output directories configured
- [x] Run robot: `rosrun your_package robot.py`
- [x] Videos saved automatically on shutdown

**Happy recording!** 📹🚗


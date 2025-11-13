
```
┌─────────────┬─────────────┬─────────────┐
│  Original   │     BEV     │  Filtered   │
├─────────────┼─────────────┼─────────────┤
│    Gray     │   Blurred   │   Binary    │
├─────────────┼─────────────┼─────────────┤
│    Canny    │    Hough    │Lane Detection│
└─────────────┴─────────────┴─────────────┘
```

**Pipeline stages:**
1. **Original** - Ảnh gốc từ camera
2. **BEV** - Bird's Eye View (nhìn từ trên xuống)
3. **Filtered** - Sau khi lọc màu (HLS color filtering)
4. **Gray** - Grayscale conversion
5. **Blurred** - Gaussian blur để giảm noise
6. **Binary** - Binary thresholding
7. **Canny** - Canny edge detection
8. **Hough** - Hough line detection
9. **Lane Detection** - Kết quả cuối với sliding windows & fitted curve

## Configuration

Mở file `src/core/recording/lane_analysis_recorder.py` và chỉnh sửa phần **CONFIGURATION** (dòng 17-55):

### 1. Enable/Disable Recording

```python
ENABLED = True  # True = bật ghi video, False = tắt
```

### 2. Output Directory

```python
OUTPUT_DIR = "robot_videos/lane_analysis"  # Thư mục lưu video
```

### 3. Video Settings

```python
FPS = 15  # Frame rate của video (15-20 recommended)
CODEC = "XVID"  # Video codec (XVID, mp4v, MJPG)
```

### 4. Grid Layout

```python
GRID_ROWS = 3  # Số hàng
GRID_COLS = 3  # Số cột
CELL_WIDTH = 320  # Chiều rộng mỗi ô (pixels)
CELL_HEIGHT = 240  # Chiều cao mỗi ô (pixels)
```

**Total video resolution:** 960x720 (3x320 x 3x240)

### 5. Visual Options

```python
ADD_LABELS = True  # Thêm text label cho mỗi ảnh
ADD_TIMESTAMP = True  # Thêm timestamp vào video
LABEL_FONT_SCALE = 0.7  # Kích thước chữ
LABEL_COLOR = (255, 255, 255)  # Màu chữ (white)
LABEL_BG_COLOR = (0, 0, 0)  # Màu nền label (black)
```

## Usage

### Automatic Recording (Default)

Recording tự động bật khi robot start:

```bash
rosrun your_package robot.py
```

Video sẽ được lưu tại: `robot_videos/lane_analysis/lane_analysis_YYYYMMDD_HHMMSS.avi`

### Manual Control (Advanced)

Nếu muốn tắt/bật thủ công, sửa trong `robot.py`:

```python
# Tắt auto-start
# self.lane_analysis_recorder.start_recording()  # Comment out

# Bật recording thủ công khi cần
self.lane_analysis_recorder.start_recording()

# Tắt recording
self.lane_analysis_recorder.stop_recording()
```

### Disable Completely

Set `ENABLED = False` trong config (dòng 21 của `lane_analysis_recorder.py`)

## Output

### Video File

- **Filename:** `lane_analysis_YYYYMMDD_HHMMSS.avi`
- **Location:** `robot_videos/lane_analysis/`
- **Format:** AVI (XVID codec)
- **Resolution:** 960x720 (default)
- **FPS:** 15 (default)

### Example

```
robot_videos/
├── lane_analysis/
│   ├── lane_analysis_20250112_143022.avi
│   ├── lane_analysis_20250112_150315.avi
│   └── lane_analysis_20250112_163045.avi
└── ...
```

## Analysis Use Cases

### 1. Debug Lane Detection
- Xem từng bước xử lý để tìm lỗi
- So sánh các tham số color filtering
- Kiểm tra threshold values

### 2. Optimize Parameters
- Điều chỉnh HLS range và xem kết quả real-time
- Test các giá trị binary threshold
- Tune Canny/Hough parameters

### 3. Performance Analysis
- Phân tích control response với visual feedback
- Xem xe xử lý cua như thế nào
- Identify failure cases (mất làn, sai lệch)

### 4. Training Data Collection
- Thu thập video cho machine learning
- Annotate các trường hợp thành công/thất bại
- Create dataset cho model improvement

## Technical Details

### Threading Model
- Non-blocking recording (giống `VideoRecorder`)
- Queue-based với max size 30 frames
- Automatic frame dropping nếu queue full
- Clean shutdown với thread join

### Performance
- Minimal overhead (~1-2ms per frame)
- Không ảnh hưởng đến control loop
- Automatic resize và color conversion
- Efficient grid composition

## Troubleshooting

### Recording không bắt đầu
1. Check `ENABLED = True` trong config
2. Kiểm tra OUTPUT_DIR có quyền write
3. Xem log: `[LaneAnalysisRecorder]` messages

### Video bị lag/dropped frames
1. Giảm FPS: `FPS = 10`
2. Giảm cell size: `CELL_WIDTH = 240`, `CELL_HEIGHT = 180`
3. Tăng queue size: `MAX_QUEUE_SIZE = 50`

### File size quá lớn
1. Giảm FPS: `FPS = 10`
2. Giảm cell size
3. Thay codec: `CODEC = "MJPG"` (compression tốt hơn)

### Video quality kém
1. Tăng cell size: `CELL_WIDTH = 400`, `CELL_HEIGHT = 300`
2. Thay codec: `CODEC = "mp4v"`
3. Dùng higher FPS: `FPS = 20`

## Notes

- Video tự động stop khi robot shutdown
- Cleanup được handle tự động
- Thread-safe implementation
- Memory efficient với queue management

## Quick Configuration Examples

### High Quality (for detailed analysis)
```python
FPS = 20
CELL_WIDTH = 400
CELL_HEIGHT = 300
```

### Low Storage (for long recordings)
```python
FPS = 10
CELL_WIDTH = 240
CELL_HEIGHT = 180
CODEC = "MJPG"
```

### Balanced (default)
```python
FPS = 15
CELL_WIDTH = 320
CELL_HEIGHT = 240
CODEC = "XVID"
```

---

**Enjoy analyzing your lane detection performance!** 🚗📹



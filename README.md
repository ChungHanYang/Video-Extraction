# Video Extraction Service

A Python service for processing school bus camera videos to track vehicles, detect movement direction, and extract frames with license plate visibility.

## Features

- **Vehicle Tracking**: Uses YOLO11 to track vehicles in video footage
- **Direction Detection**: Determines if vehicle is moving toward or away from camera
- **Smart Frame Selection**: Selects two frames with specified frame difference
- **License Plate Detection**: Finds the best frame for plate visibility from direction-specific video
- **Base64 Encoding**: Returns frames as base64-encoded strings for easy integration

## Architecture

The service consists of several modules:

- `video_processor.py`: Main video processing and vehicle tracking logic
- `plate_detector.py`: License plate detection and quality assessment
- `service.py`: High-level service API and CLI interface
- `example_usage.py`: Usage examples and integration patterns

## Requirements

- Python 3.8+
- OpenCV
- PyTorch
- Ultralytics YOLO11

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download YOLO11 model (will auto-download on first run):
```python
from ultralytics import YOLO
model = YOLO('yolo11n.pt')
```

3. (Optional) For better plate detection, use a specialized plate detection model:
   - Update `config.json` with path to your plate model
   - Or download a pre-trained plate model

## Usage

### Command Line

```bash
python service.py <main_video> <toward_video> <away_video> [start_second] [frame_difference]
```

**Arguments:**
- `main_video`: Path to main video for vehicle tracking
- `toward_video`: Path to video for plate detection when vehicle moves toward camera
- `away_video`: Path to video for plate detection when vehicle moves away
- `start_second`: (Optional) Starting time in seconds (default: 0.0)
- `frame_difference`: (Optional) Frame difference between selected frames (default: 30)

**Example:**
```bash
python service.py videos/main.mp4 videos/front.mp4 videos/rear.mp4 5.0 30
```

### Python API

```python
from service import VideoExtractionService

# Initialize service
service = VideoExtractionService(config_path='config.json')

# Process video
result = service.extract_frames(
    main_video_path='videos/main.mp4',
    toward_video_path='videos/front.mp4',
    away_video_path='videos/rear.mp4',
    start_second=5.0,
    frame_difference=30
)

if result:
    # Access base64-encoded frames
    frame1_b64 = result['frame1']
    frame2_b64 = result['frame2']
    plate_frame_b64 = result['plate_frame']

    # Get metadata
    direction = result['direction']  # "toward" or "away"
    frame_indices = [result['frame1_index'], result['frame2_index']]
    plate_index = result['plate_frame_index']

    # Save frames as images
    service.save_frames_as_images(result, 'output')

    # Save metadata
    service.save_result(result, 'output/metadata.json')
```

## Configuration

Edit `config.json` to customize behavior:

```json
{
  "vehicle_model": "yolo11n.pt",
  "plate_model": null,
  "confidence_threshold": 0.5,
  "min_track_length": 10,
  "default_frame_difference": 30,
  "skip_frame": 2
}
```

**Parameters:**
- `vehicle_model`: Path to YOLO model for vehicle detection
- `plate_model`: Path to YOLO model for plate detection (null for heuristic-based)
- `confidence_threshold`: Minimum confidence for detections (0.0-1.0)
- `min_track_length`: Minimum frames required for valid vehicle track
- `default_frame_difference`: Default frame difference when not specified
- `skip_frame`: Process every Nth frame for faster detection (1 = all frames, 2 = every other frame, etc.)

## How It Works

### 1. Vehicle Tracking
- Processes video starting from specified second
- Uses YOLO11 tracking to follow vehicles across frames
- Maintains bounding box and center position history

### 2. Direction Detection
- Analyzes vehicle bounding box size over time
- Increasing size = moving toward camera
- Decreasing size = moving away from camera

### 3. Frame Selection
- Selects first frame when vehicle bounding box center is closest to frame center
- Selects second frame exactly 1 second after the first frame
- This center-based approach ensures the vehicle is optimally positioned for visibility
- Defaults to frames 1 and 2 if track is too short for 1-second spacing

### 4. Plate Detection
- Based on detected direction, uses appropriate video (toward/away)
- Scans frames between selected pair using same frame indices
- Scores frames based on:
  - Plate detection confidence (if model available)
  - Rectangular contours with plate-like aspect ratio
  - Image sharpness and clarity
  - Optimal lighting conditions

### 5. Output
- Returns three base64-encoded frames
- Includes metadata (direction, frame indices, track ID)

## Output Format

The service returns a dictionary with:

```python
{
    'frame1': 'base64_encoded_jpeg...',
    'frame2': 'base64_encoded_jpeg...',
    'plate_frame': 'base64_encoded_jpeg...',
    'frame1_index': 150,
    'frame2_index': 180,
    'plate_frame_index': 165,
    'direction': 'toward',
    'track_id': 1,
    'plate_video_used': 'videos/front.mp4'
}
```

## Advanced Usage

### Custom Plate Detection Model

Train or download a YOLO model specifically for license plates:

```python
service = VideoExtractionService()
service.processor.plate_model = YOLO('custom_plate_model.pt')
```

### Batch Processing

```python
import glob

service = VideoExtractionService()
video_sets = [
    ('main1.mp4', 'front1.mp4', 'rear1.mp4'),
    ('main2.mp4', 'front2.mp4', 'rear2.mp4'),
]

for i, (main, front, rear) in enumerate(video_sets):
    result = service.extract_frames(main, front, rear)
    if result:
        service.save_frames_as_images(result, f'output/set_{i}')
```

## Performance Tips

1. **GPU Acceleration**: Ensure PyTorch is installed with CUDA support for faster processing
2. **Model Selection**: Use `yolo11n.pt` (nano) for speed, `yolo11x.pt` (extra-large) for accuracy
3. **Frame Skipping**: Set `skip_frame` to 2 or 3 to process every 2nd or 3rd frame for 2-3x faster processing
4. **Frame Difference**: Larger frame differences process faster but may miss quick movements
5. **Start Second**: Skip initial seconds to avoid processing unnecessary footage

## Troubleshooting

**No vehicles detected:**
- Lower `confidence_threshold` in config
- Ensure video quality is sufficient
- Check that vehicles occupy significant portion of frame

**Poor plate detection:**
- Use a specialized plate detection model
- Ensure plate is visible in the direction-specific video
- Increase video resolution if possible

**Slow processing:**
- Use smaller YOLO model (yolo11n.pt)
- Enable GPU acceleration
- Increase frame skip rate for initial tracking

## Examples

See `example_usage.py` for detailed examples including:
- Basic usage
- Frame display and visualization
- Batch processing multiple videos
- Custom configuration
- API integration patterns

## License

This project uses the Ultralytics YOLO11 model, which is licensed under AGPL-3.0.

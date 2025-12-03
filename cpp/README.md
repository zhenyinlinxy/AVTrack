# Robust Visual Object Tracker (C++)

C++ implementation of a robust visual object tracker with ReID (Re-Identification) support, designed for deployment on embedded platforms like HiSilicon.

## Features

- ONNX Runtime based inference
- ReID (Re-Identification) for target verification
- HiSilicon platform aligned memory operations (16-byte alignment)
- Multi-state tracking (TRACKING, UNCERTAIN, LOST)
- Video and image sequence processing
- Configurable confidence thresholds

## Requirements

### Dependencies

- **CMake** >= 3.10
- **OpenCV** >= 4.0
- **ONNX Runtime** >= 1.10

### Optional Dependencies

- **CUDA** (for GPU acceleration)
- **TensorRT** (for optimized inference on NVIDIA platforms)

## Building

### Standard Build

```bash
cd cpp
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### Build with CUDA Support

```bash
cmake -DUSE_CUDA=ON ..
make -j$(nproc)
```

### Build with TensorRT Support

```bash
cmake -DUSE_CUDA=ON -DUSE_TENSORRT=ON ..
make -j$(nproc)
```

### Specifying ONNX Runtime Path

If ONNX Runtime is installed in a non-standard location:

```bash
cmake -DONNXRUNTIME_ROOT=/path/to/onnxruntime ..
```

## Usage

### Command Line Options

```
Robust Visual Object Tracker with ReID Support

Usage: robust_tracker [options]

Required:
  --onnx <path>              Path to tracking ONNX model
  --video <path>             Path to video file
  OR
  --folder <path>            Path to image folder

Optional:
  --providers <list>         ONNX execution providers (default: CPUExecutionProvider)
  --save <path>              Output path for saving results
  --template-size <int>      Template size (default: 128)
  --search-size <int>        Search region size (default: 256)
  --template-factor <float>  Template area factor (default: 2.0)
  --search-factor <float>    Search area factor (default: 4.0)
  --conf-high <float>        High confidence threshold (default: 0.8)
  --conf-medium <float>      Medium confidence threshold (default: 0.3)
  --conf-low <float>         Low confidence threshold (default: 0.15)
  --max-lost-frames <int>    Maximum lost frames before giving up (default: 3000)
  --reid-onnx <path>         Path to ReID ONNX model
  --reid-bank-size <int>     ReID feature bank size (default: 10)
  --reid-update-interval <int>  ReID update interval in frames (default: 20)
  --reid-similarity-threshold <float>  ReID similarity threshold (default: 0.7)
  --no-display               Disable visualization window
  --help                     Show help message
```

### Examples

#### Basic Video Tracking

```bash
./robust_tracker --onnx model.onnx --video input.mp4
```

#### Save Output Video

```bash
./robust_tracker --onnx model.onnx --video input.mp4 --save output.mp4
```

#### Track with ReID Verification

```bash
./robust_tracker --onnx model.onnx --video input.mp4 --reid-onnx osnet.onnx
```

#### Process Image Folder

```bash
./robust_tracker --onnx model.onnx --folder images/ --save output/
```

#### Use GPU Acceleration

```bash
./robust_tracker --onnx model.onnx --video input.mp4 --providers CUDAExecutionProvider CPUExecutionProvider
```

#### Custom Thresholds

```bash
./robust_tracker --onnx model.onnx --video input.mp4 \
    --conf-high 0.85 --conf-medium 0.4 --conf-low 0.2
```

#### Full Example with All Options

```bash
./robust_tracker \
    --onnx tracking_model.onnx \
    --video input.mp4 \
    --save output.mp4 \
    --reid-onnx osnet.onnx \
    --template-size 128 \
    --search-size 256 \
    --template-factor 2.0 \
    --search-factor 4.0 \
    --conf-high 0.8 \
    --conf-medium 0.3 \
    --conf-low 0.15 \
    --reid-bank-size 10 \
    --reid-update-interval 20 \
    --reid-similarity-threshold 0.7 \
    --providers CUDAExecutionProvider CPUExecutionProvider
```

## Project Structure

```
cpp/
├── CMakeLists.txt          # Build configuration
├── README.md               # This file
├── include/
│   ├── tracking_state.h    # Tracking state enumeration
│   ├── reid_feature_bank.h # ReID feature management
│   ├── osnet_infer.h       # OSNet ReID inference
│   ├── utils.h             # Utility functions
│   └── robust_tracker.h    # Main tracker class
└── src/
    ├── main.cpp            # Main entry point
    ├── robust_tracker.cpp  # Tracker implementation
    ├── reid_feature_bank.cpp # ReID feature bank
    ├── osnet_infer.cpp     # OSNet inference
    └── utils.cpp           # Utilities
```

## API Usage

### Using as a Library

```cpp
#include "robust_tracker.h"

using namespace robust_tracker;

// Configure tracker
TrackerConfig config;
config.template_size = 128;
config.search_size = 256;
config.conf_high = 0.8f;
config.conf_medium = 0.3f;
config.conf_low = 0.15f;

// Create tracker
RobustTracker tracker("model.onnx", config);

// Optional: Add ReID model
tracker.setReIDModel("osnet.onnx");

// Initialize with first frame and bounding box
cv::Mat first_frame = cv::imread("frame0.jpg");
BBox init_box(100, 100, 50, 80);  // x, y, width, height
tracker.initialize(first_frame, init_box);

// Track in subsequent frames
cv::Mat frame = cv::imread("frame1.jpg");
TrackingResult result = tracker.track(frame);

// Access results
std::cout << "State: " << trackingStateToString(result.state) << std::endl;
std::cout << "Confidence: " << result.confidence << std::endl;
std::cout << "Box: " << result.box.x << ", " << result.box.y 
          << ", " << result.box.w << ", " << result.box.h << std::endl;
```

## Tracking States

- **INIT**: Tracker not yet initialized
- **TRACKING**: Normal tracking with high confidence
- **UNCERTAIN**: Tracking with medium confidence, target may be occluded
- **LOST**: Target is lost

## Notes

### HiSilicon Platform

The tracker is optimized for HiSilicon embedded platforms:
- Memory alignment to 16 bytes
- Efficient image sampling with padding
- ONNX Runtime with appropriate providers

### ReID Verification

ReID (Re-Identification) is used to verify target identity:
- Features are extracted using OSNet model
- Features are stored in a circular buffer (feature bank)
- Cosine similarity is used for matching
- Helps recover from occlusions and similar distractors

### Performance Tips

1. Use CUDA/TensorRT providers for GPU acceleration
2. Adjust `search_factor` for faster/wider search
3. Tune confidence thresholds based on your scenario
4. Increase `reid_update_interval` to reduce computation

## License

This project follows the license of the original AVTrack repository.

# Pothole Detection and Severity Scoring

This project utilizes spatial AI to detect potholes on street surfaces, accurately measure their depth, and assign a severity score. It uses an OAK 4 PRO camera mounted perpendicular (pointing straight down) to the road. This simplifies detection by using the flat street surface as a consistent depth baseline.

## Setup Instructions

1. **Create and activate a virtual environment**:
   ```powershell
   # Windows
   python -m venv .venv
   .venv\Scripts\Activate.ps1
    ```

    ```bash
    # Linux/macOS
    python3 -m venv .venv
    source .venv/bin/activate
    ```
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the application**:
   ```bash
   python main.py
   ```
   _(Alternatively, you can run this as an OAK App using the provided `oakapp.toml` and `oakctl`)_
   
  **Kill the app**
  lsof -ti :8082 -ti :8765 | xargs kill -9 2>/dev/null && pkill -f "python3 main.py" 2>/dev/null; echo "done"

## DepthAI Pipeline

The application builds a robust DepthAI pipeline focusing on high-accuracy depth sensing:

1. **IR Illumination**: The IR laser dot projector and flood lights are enabled. This provides texture to textureless surfaces (like asphalt), crucial for accurate Active Stereo Depth matching.
2. **Cameras**:
   - Left and Right cameras stream to the `StereoDepth` node.
   - The central RGB camera feeds the neural network and provides the visualization stream.
3. **Stereo Depth with Filtering**: High-detail preset is enabled, along with Spatial and Temporal filtering configurations to smooth the depth map and reduce noise.
4. **Spatial Detection Network**: A pre-trained object detection neural network recognizes the 2D bounding boxes of potholes and incorporates the depth map.
5. **Custom Host Node (`PotholeDetector`)**:
   - Calculates a flat-road baseline depth by sampling the median depth of the remaining surface (excluding detected pothole bounding boxes).
   - Iterates through detected potholes, extracting the max depth within each bounding box.
   - Compares the max pothole depth against the flat-road baseline to calculate true depth, mapping it to a 1-10 Severity Score.
6. **Video Encoder**: Streams the annotated RGB and Depth components.

## Custom / Retrained Models

This pipeline leverages custom model configurations for pothole object detection:

- **`pothole_2`**: A specially trained model adapted for identifying pothole edges from a straight-down perspective. This is used by default in `main.py`.
- **`yolov6`**: Additional YOLO object detection models (`yolov6_nano_r2_coco`) are provided for basic standard detection references and comparisons.

The application leverages the Luxonis Model Zoo capabilities, resolving the actual model weights automatically during pipeline initialization using the yaml definition files (`dai.NNModelDescription.fromYamlFile`).

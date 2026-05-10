# Pothole Detection and Severity Scoring

## Project Context

This project was developed during the weekend-long [GDG AI HACK 2026](https://gdgaihack.com/). Specifically, it was created for the [Luxonis "See Beyond" track](https://gdgaihack.com/guidebook/tracks/luxonis), which challenged participants to solve practical real-world vision problems utilizing on-device spatial AI and the Luxonis OAK 4 D camera.

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
5. **Depth Masking & Plane Fitting**:
   - For every detected pothole, the depth frame is isolated, and bounding boxes are evaluated to filter noisy pixels using a 3x3 median filter.
   - We fit a 3D plane (`z = a*x + b*y + c`) to the road pixels _outside_ the pothole bounding box using Least Squares. This provides an accurate predicted baseline road depth underneath the center of the pothole, accounting for road slopes or uneven camera mounting.
   - We extract the 95th percentile depth inside the pothole's bounding box (to ignore noise spikes) and compare it against the modeled plane baseline to measure true physical depth.
6. **Video Encoder**: Streams the annotated RGB and Depth components.

## Custom / Retrained Models

This pipeline leverages custom model configurations for pothole object detection:

- **`pothole_2`**: A specially trained model adapted for identifying pothole edges from a straight-down perspective. This is used by default in `main.py`. [hugging face model](https://huggingface.co/cazzz307/Pothole-Finetuned-YoloV8)
- **`yolov6`**: Additional YOLO object detection models (`yolov6_nano_r2_coco`) are provided for basic standard detection references and comparisons.

The application leverages the Luxonis Model Zoo capabilities, resolving the actual model weights automatically during pipeline initialization using the yaml definition files (`dai.NNModelDescription.fromYamlFile`).

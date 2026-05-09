# Pothole Detection and Severity Scoring

## Project Description

This project aims to detect potholes on street surfaces, accurately measure their depth, and assign a severity score. The system relies on an OAK 4 PRO camera mounted pointing straight down at the road. This straight-down orientation simplifies the detection task significantly, as the normal flat road surface establishes a consistent baseline distance. Any significant increase in distance compared to this baseline indicates a pothole or depression.

## Specifications

- **Hardware:** OAK 4 PRO
- **Camera Setup:** Downward-facing (perpendicular to the road).
- **Depth Perception:** Active Stereo Depth leveraging the OAK 4 PRO's Infrared (IR) capabilities.

Since asphalt and concrete can sometimes lack distinct visual features, the IR dot projector and flood LED are crucial. Projecting an IR pattern onto the road surface guarantees high texture for the stereo matching algorithm, resulting in highly accurate depth measurements.

## Step-by-Step Task Line

### Phase 1: Camera and Depth Setup

1. **Initialize the Pipeline:** Set up the basic DepthAI pipeline.
2. **Configure Camera Nodes:** Initialize the Left and Right Mono cameras for stereo vision, and the Color camera for visualization.
3. **Enable IR Capabilities:** Configure the Device to activate the IR dot projector and flood light to ensure accurate depth mapping on the street surface.
4. **Configure Stereo Depth Node:** Set up the `StereoDepth` node. Enable spatial and temporal filtering to smooth the depth map and reduce noise.

### Phase 2: Detection and Measurement

5. **Establish Road Baseline:** Continuously calculate the average depth of the flat road surface in the frame. Since the camera points straight down, the majority of the frame will represent this baseline distance.
6. **Detect Potholes:** Identify regions of interest (ROIs) where the depth values are significantly higher (further away) than the baseline road surface. This can be done using depth contouring, thresholding, or a Spatial Detection Network if a trained AI model is preferred.
7. **Measure Pothole Depth:** For each detected ROI, calculate the relative depth by subtracting the baseline road distance from the maximum depth value found inside the pothole.

### Phase 3: Analysis and Output

8. **Severity Scoring:** Implement an algorithmic scoring system (e.g., 1 to 10 or Low/Medium/High). The score should be a function of:
   - **Maximum Depth:** Deeper potholes are more severe.
   - **Surface Area:** Calculated using the spatial coordinates/boundaries of the detected pothole.
9. **Visualization:** Overlay the original RGB/Depth frame with bounding boxes, maximum depth texts, and the severity score for each detected pothole.
10. **Data Logging (Optional):** Save images of severe potholes along with their metrics for reporting purposes.

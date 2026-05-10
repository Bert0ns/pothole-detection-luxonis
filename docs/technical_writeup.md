# Tackling the Tarmac: Building an AI-Powered Pothole Detector with DepthAI

Have you ever driven down a seemingly perfect stretch of road, only to be jolted by a hidden crater? Potholes are the nemesis of drivers and city planners alike. For this hackathon, we set out to build a smarter way to detect, measure, and score the severity of potholes in real-time.

Here is a look under the hood at how we built our Pothole Detection system using the Luxonis OAK 4 PRO, DepthAI, and a bit of spatial geometry.

## The Architecture: Bringing the DepthAI Pipeline to Life

At the heart of our project is the Luxonis OAK 4 PRO camera. We quickly realized that a standard 2D camera wouldn't cut it—we needed to know exactly _how deep_ a pothole was to understand its severity.

Our pipeline (`main.py`) does the heavy lifting:

1. **Camera Sensor Setup**: We tap into the left and right mono cameras for stereo vision, while the center RGB camera handles visualization and AI inference.
2. **Active Stereo Depth**: Asphalt is incredibly difficult for cameras to read because it lacks distinct texture. To fix this, we fire up the IR Laser Dot Projector and IR Floodlight. This bathes the road in a high-contrast pattern, acting as a cheat code for the stereo matching algorithm.
3. **Filtering**: Raw depth data can be noisy. We enable internal Spatial and Temporal filtering within the `StereoDepth` node to smooth the depth map and eliminate the jittering effect from frame to frame.
4. **Spatial Detection**: Instead of doing things the hard way, we use DepthAI’s incredible `SpatialDetectionNetwork`. It combines the RGB output and Stereo Depth to give us bounding regions of interest (ROI) alongside X/Y/Z spatial coordinates.
5. **Severity Evaluation & Streaming**: Our measurement scripts crunch the numbers on the host side, and we stream the combined visualizations out smoothly via an H264 encoded remote connection.

## Decoding the Depth: How We Measure Potholes

One of our biggest breakthroughs was our camera mounting strategy. By mounting the camera **straight down**, the flat ground plane remains mostly parallel to the sensor. However, since roads slope, crown, and camera mounts shift, relying on a simple flat baseline distance measurement was flawed.

Here is how we precisely calculate depth:

1. **The Dynamic Baseline**: Instead of assuming a completely flat ground baseline for the entire frame, we dynamically isolate the valid depth pixels on the road surface—explicitly masking out any identified pothole bounding boxes. We then use Least Squares to fit a mathematical 3D plane (`z = ax + by + c`) to that road surface. This allows us to predict what the road's true depth _should_ be right at the exact center `(cx, cy)` of the crater!
2. **Finding the Bottom (Smartly)**: We look exclusively at the depth pixels _inside_ each pothole's bounding box. We run a 3x3 median filter over the region to discard isolated noise spikes. Then, instead of grabbing the absolute maximum depth, we evaluate the **95th percentile** depth. This ensures we are measuring the true bottom of the pothole while ignoring any sensor anomalies.
3. **Calculating the Physical Depth**: We subtract the computed road-plane baseline from the 95th percentile pothole distance. That difference is the true, physical depth of the pothole in millimeters, which can be visualized back on the UI!

## Models: From COCO to Custom

When we started, we threw a standard `yolov6_nano_r2_coco` model at the problem. Predictably, an AI trained to find dogs, cars, and stop signs didn't know what to make of a straight-down view of cracked asphalt.

We pivoted to a custom, retrained model called `pothole_2`. To make deployment seamless, we utilized the Luxonis Model Zoo mechanism (`dai.NNModelDescription.fromYamlFile`). By loading a simple `.yaml` file, DepthAI dynamically downloads the correct model compilation for the active RVC platform (RVC2 or RVC4). No clunky `.blob` file management required!

## The Lab: What We Tried (and What Failed)

- **Dashcam vs. Straight-Down**: Originally, we wanted a forward-facing mount like a standard dashcam. We quickly found ourselves drowning in projective geometry trying to map a flat plane at an angle. Switching to a straight-down perspective made the math beautiful and straightforward.
- **Passive vs. Active Stereo**: Our early tests on fresh blacktop yielded hilariously bad, noisy depth maps. Stereo matching algorithms need texture to find common points. Turning on the IR Dot Projector was a "eureka!" moment. It works flawlessly, proving why Pro-tier hardware is essential for road surfaces.
- **Where to Crunch the Numbers**: We originally used a plain median depth outside the potholes for a baseline. It was extremely unstable if the car drove over the crest of a hill or hit a bump. Upgrading our host calculation logic to a localized 3D Plane Fit made the math completely independent of mounting jitter or uneven roadways!

## Takeaways and Learnings

Building this project taught us some invaluable lessons about spatial AI in the real world:

- **IR is Non-Negotiable**: You simply cannot rely on passive stereo vision when looking at uniformly textured surfaces like concrete or asphalt. Active illumination is a must.
- **Dynamic Plane Fitting**: Simple flat math doesn't work in the wild. Accounting for road crowns and vehicle suspensions via dynamic localized plane fitting (`scipy` + `numpy.linalg.lstsq`) drastically elevated the robustness of our software.
- **Always Filter Anomalies**: Between standard `DepthAI` spatial filters, local median filters, and utilizing the 95th percentile instead of pure maxes, smoothing away noise is half the battle in stereo depth applications!

We came out of this hackathon with a robust, functional prototype that turns an unassuming camera into a powerful infrastructure-analysis tool. Watch out, potholes—we see you!

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
5. **Severity Evaluation & Streaming**: A custom `PotholeDetector` node crunches the numbers on the host side, and we stream it all out smoothly via an H264 encoded remote connection.

## Decoding the Depth: How We Measure Potholes

One of our biggest breakthroughs was our camera mounting strategy. By mounting the camera **straight down**, the flat ground plane remains parallel to the sensor. This simple geometric constraint was a game-changer.

Here is how we calculate depth without complex point cloud manipulation:

1. **The Dynamic Baseline**: Inside our `PotholeDetector` node, we take the depth frame and _mask out_ all the AI-detected potholes. We grab the median depth of whatever is left. This gives us an incredibly reliable baseline distance to the flat road surface, adjusting dynamically to every bump or suspension shift.
2. **Finding the Bottom**: For each detected pothole, we look exclusively at the depth pixels _inside_ its bounding box. We seek out the maximum depth reading.
3. **Scoring the Severity**: We subtract the baseline distance from the pothole's maximum distance. That difference is the true, physical depth of the pothole in millimeters. We scale this on a 1–10 scale—where 1 is a minor depression and 10 means you might lose a hubcap.

## Models: From COCO to Custom

When we started, we threw a standard `yolov6_nano_r2_coco` model at the problem. Predictably, an AI trained to find dogs, cars, and stop signs didn't know what to make of a straight-down view of cracked asphalt.

We pivoted to a custom, retrained model called `pothole_2`. To make deployment seamless, we utilized the Luxonis Model Zoo mechanism (`dai.NNModelDescription.fromYamlFile`). By loading a simple `.yaml` file, DepthAI dynamically downloads the correct model compilation for the active RVC platform (RVC2 or RVC4). No clunky `.blob` file management required!

## The Lab: What We Tried (and What Failed)

- **Dashcam vs. Straight-Down**: Originally, we wanted a forward-facing mount like a standard dashcam. We quickly found ourselves drowning in projective geometry trying to map a flat plane at an angle. Switching to a straight-down perspective made the math beautiful and straightforward.
- **Passive vs. Active Stereo**: Our early tests on fresh blacktop yielded hilariously bad, noisy depth maps. Stereo matching algorithms need texture to find common points. Turning on the IR Dot Projector was a "eureka!" moment. It works flawlessly, proving why Pro-tier hardware is essential for road surfaces.
- **Where to Crunch the Numbers**: We tried pulling all the depth calculation logic out to the host PC after running standard 2D inference on-device. It worked, but keeping the `SpatialDetectionNetwork` on the camera saved us massive amounts of processing overhead.

## Takeaways and Learnings

Building this project taught us some invaluable lessons about spatial AI in the real world:

- **IR is Non-Negotiable**: You simply cannot rely on passive stereo vision when looking at uniformly textured surfaces like concrete or asphalt. Active illumination is a must.
- **Dynamic Baselines Rule**: Trying to hardcode a distance to the road failed the second the vehicle hit a bump. Dynamically masking out the defects to find a per-frame ground truth proved incredibly resilient.
- **Filter Tuning is an Art**: Finding the sweet spot between temporal `alpha` values and `holeFillingRadius` is crucial. Too little filtering leaves your severity scores bouncing around; too much, and you introduce latency that misses fast-moving potholes.

We came out of this hackathon with a robust, functional prototype that turns an unassuming camera into a powerful infrastructure-analysis tool. Watch out, potholes—we see you!

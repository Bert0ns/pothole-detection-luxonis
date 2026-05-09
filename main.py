#!/usr/bin/env python3

import cv2
import depthai as dai
from utils.arguments import initialize_argparser
from depthai_nodes.node import ApplyDepthColormap
from utils.pothole_detector import PotholeDetector

_, args = initialize_argparser()

visualizer = dai.RemoteConnection(httpPort=8082)
device = dai.Device(dai.DeviceInfo(args.device)) if args.device else dai.Device()
platform = device.getPlatform().name
print(f"Platform: {platform}")

# Enable IR projector and floodlight to get accurate depth on textureless asphalt
try:
    device.setIrLaserDotProjectorIntensity(1.0)
    device.setIrFloodLightIntensity(1.0)
except Exception as e:
    print(
        f"Warning: Could not set IR capabilities (might not be supported on this device). Error: {e}"
    )

with dai.Pipeline(device) as pipeline:
    det_model_description = dai.NNModelDescription.fromYamlFile(
        f"pothole_2.{platform}.yaml"
    )
    det_model_nn_archive = dai.NNArchive(dai.getModelFromZoo(det_model_description))
    nn_size = det_model_nn_archive.getInputSize()

    # camera input
    cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)

    left_cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
    right_cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)

    stereo = pipeline.create(dai.node.StereoDepth).build(
        left=left_cam.requestOutput(nn_size),
        right=right_cam.requestOutput(nn_size),
        presetMode=dai.node.StereoDepth.PresetMode.HIGH_DETAIL,
    )
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
    if platform == "RVC2":
        stereo.setOutputSize(*nn_size)
    stereo.setLeftRightCheck(True)
    stereo.setRectification(True)

    # Enable spatial and temporal filtering to smooth the depth map and reduce noise
    config = stereo.initialConfig
    config.postProcessing.spatialFilter.enable = True
    config.postProcessing.spatialFilter.holeFillingRadius = 2
    config.postProcessing.spatialFilter.numIterations = 1

    config.postProcessing.temporalFilter.enable = True
    config.postProcessing.temporalFilter.alpha = 0.4
    config.postProcessing.temporalFilter.delta = 20

    nn = pipeline.create(dai.node.SpatialDetectionNetwork).build(
        input=cam,
        stereo=stereo,
        nnArchive=det_model_nn_archive,
    )
    if platform == "RVC2":
        nn.setNNArchive(det_model_nn_archive, numShaves=7)
    nn.setBoundingBoxScaleFactor(0.7)

    # Secondary visualization (RGB + AI Boxes)
    from utils.annotation_node import AnnotationNode

    classes = det_model_nn_archive.getConfig().model.heads[0].metadata.classes
    annotation_node = pipeline.create(AnnotationNode).build(
        input_detections=nn.out, depth=stereo.depth, labels=classes
    )

    # Encode the RGB Camera explicitly for streaming
    cam_nv12 = cam.requestOutput(
        size=nn_size,
        type=dai.ImgFrame.Type.NV12,
    )
    video_encoder = pipeline.create(dai.node.VideoEncoder)
    video_encoder.setMaxOutputFrameSize(nn_size[0] * nn_size[1] * 3)
    video_encoder.setDefaultProfilePreset(
        30, dai.VideoEncoderProperties.Profile.H264_MAIN
    )
    cam_nv12.link(video_encoder.input)

    depth_color_transform = pipeline.create(ApplyDepthColormap).build(stereo.disparity)
    depth_color_transform.setColormap(cv2.COLORMAP_JET)

    # Custom HostNode that measures median depth for baseline
    # and detects potholes based on depth thresholding
    pothole_detector = pipeline.create(PotholeDetector).build(
        disparity_frames=depth_color_transform.out,
        depth_frames=stereo.depth,
        detections=nn.out,
    )

    visualizer.addTopic("Pothole Details", pothole_detector.passthrough)
    visualizer.addTopic("Spatial Calculations", pothole_detector.annotation_output)

    # Adding secondary visualization for RGB + pure Object Detection
    visualizer.addTopic("RGB Camera", video_encoder.out)
    visualizer.addTopic("Detections overlay", annotation_node.out_annotations)

    print("Pipeline created.")
    pipeline.start()
    visualizer.registerPipeline(pipeline)

    while pipeline.isRunning():
        key = visualizer.waitKey(1)
        if key == ord("q"):
            print("Got q key from the remote connection!")
            break

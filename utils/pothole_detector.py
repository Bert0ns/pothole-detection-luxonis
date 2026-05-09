import cv2
import numpy as np
import depthai as dai
from depthai_nodes.utils import AnnotationHelper


class PotholeDetector(dai.node.HostNode):
    def __init__(self):
        super().__init__()

        self.passthrough = self.createOutput(
            possibleDatatypes=[
                dai.Node.DatatypeHierarchy(dai.DatatypeEnum.ImgFrame, True)
            ]
        )
        self.annotation_output = self.createOutput(
            possibleDatatypes=[
                dai.Node.DatatypeHierarchy(dai.DatatypeEnum.ImgAnnotations, True)
            ]
        )

        self._min_depth = 200  # 20cm
        self._max_depth = 5000  # 5m

        self._depth_threshold_mm = 30  # A change of 3 cm to count as a pothole
        self._min_contour_area = 500  # Ignore tiny noise

    def build(
        self,
        disparity_frames: dai.Node.Output,
        depth_frames: dai.Node.Output,
        detections: dai.Node.Output,
    ) -> "PotholeDetector":
        self.link_args(disparity_frames, depth_frames, detections)
        return self

    def process(
        self,
        disparity: dai.ImgFrame,
        depth_frame: dai.ImgFrame,
        in_det: dai.SpatialImgDetections,
    ) -> None:
        depth = depth_frame.getFrame()
        dets = in_det.detections

        height, width = depth.shape
        background_mask = np.ones((height, width), dtype=bool)

        for det in dets:
            x1 = int(det.xmin * width)
            y1 = int(det.ymin * height)
            x2 = int(det.xmax * width)
            y2 = int(det.ymax * height)
            background_mask[
                max(0, y1) : min(height, y2), max(0, x1) : min(width, x2)
            ] = False

        # 1. Establish baseline depth (flat road) explicitly ignoring the detected objects
        valid_mask = (depth > self._min_depth) & (depth < self._max_depth)
        bg_valid = background_mask & valid_mask

        baseline_depth = 0
        if bg_valid.any():
            baseline_depth = np.median(depth[bg_valid])

        annotations_builder = AnnotationHelper()
        if baseline_depth == 0:
            # invalid or no depth points
            annotations_builder.draw_text(
                "Baseline: --", (0.02, 0.05), (0, 0, 0, 1), (1, 1, 1, 0.7), size=4
            )
            self.annotation_output.send(
                annotations_builder.build(
                    disparity.getTimestamp(), disparity.getSequenceNum()
                )
            )
            self.passthrough.send(disparity)
            return

        # 2. Extract bounding boxes from YOLO and calculate severity
        for det in dets:
            x1 = int(det.xmin * width)
            y1 = int(det.ymin * height)
            x2 = int(det.xmax * width)
            y2 = int(det.ymax * height)

            roi_depth = depth[max(0, y1) : min(height, y2), max(0, x1) : min(width, x2)]
            roi_valid = valid_mask[
                max(0, y1) : min(height, y2), max(0, x1) : min(width, x2)
            ]

            if roi_valid.any():
                max_depth = np.max(roi_depth[roi_valid])
                diff_mm = max_depth - baseline_depth

                if diff_mm >= self._depth_threshold_mm:
                    # 3. Severity scoring: based on depth difference
                    severity = min(10, max(1, int((diff_mm - 30) / 10) + 1))
                    color = (1, 0.2, 0.2, 1)  # Red-ish for pothole

                    annotations_builder.draw_rectangle(
                        top_left=(det.xmin, det.ymin),
                        bottom_right=(det.xmax, det.ymax),
                        outline_color=color,
                        thickness=2,
                    )

                    text = f"Sev: {severity}/10 Diff: {diff_mm:.0f}mm"
                    annotations_builder.draw_text(
                        text=text,
                        position=(det.xmin, max(0.01, det.ymin - 0.02)),
                        color=color,
                        background_color=(1, 1, 1, 0.7),
                        size=4,
                    )

        annotations_builder.draw_text(
            text=f"Baseline Depth: {baseline_depth:.0f} mm",
            position=(0.02, 0.05),
            color=(0, 0, 0, 1),
            background_color=(1, 1, 1, 0.7),
            size=4,
        )

        annotations = annotations_builder.build(
            disparity.getTimestamp(), disparity.getSequenceNum()
        )
        self.annotation_output.send(annotations)
        self.passthrough.send(disparity)

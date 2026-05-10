from typing import List

import cv2
import depthai as dai
import numpy as np
from depthai_nodes import PRIMARY_COLOR, SECONDARY_COLOR, TRANSPARENT_PRIMARY_COLOR
from depthai_nodes.utils import AnnotationHelper

from utils.depth_filter import DepthFilter, DepthFilterConfig
from utils.plane_fit import fit_road_plane, road_depth_at

_MIN_VALID_PIXELS = 30  # skip depth measurement if fewer valid pixels in bbox


class AnnotationNode(dai.node.HostNode):
    def __init__(self) -> None:
        super().__init__()
        self.input_detections = self.createInput()
        self.out_annotations = self.createOutput(
            possibleDatatypes=[
                dai.Node.DatatypeHierarchy(dai.DatatypeEnum.ImgAnnotations, True)
            ]
        )
        self.out_depth = self.createOutput(
            possibleDatatypes=[
                dai.Node.DatatypeHierarchy(dai.DatatypeEnum.ImgFrame, True)
            ]
        )
        self.labels: List[str] = []
        self._depth_filter = DepthFilter(DepthFilterConfig())

    def build(
        self,
        input_detections: dai.Node.Output,
        depth: dai.Node.Output,
        labels: List[str],
    ) -> "AnnotationNode":
        self.labels = labels
        self.link_args(input_detections, depth)
        return self

    def process(
        self, detections_message: dai.Buffer, depth_message: dai.ImgFrame
    ) -> None:
        assert isinstance(detections_message, dai.SpatialImgDetections)

        detections_list: List[dai.SpatialImgDetection] = detections_message.detections
        depth_raw = depth_message.getCvFrame().astype(np.float32)
        h, w = depth_raw.shape

        annotation_helper = AnnotationHelper()

        for detection in detections_list:
            xmin_n, ymin_n, xmax_n, ymax_n = (
                detection.xmin,
                detection.ymin,
                detection.xmax,
                detection.ymax,
            )

            annotation_helper.draw_rectangle(
                top_left=(xmin_n, ymin_n),
                bottom_right=(xmax_n, ymax_n),
                outline_color=PRIMARY_COLOR,
                fill_color=TRANSPARENT_PRIMARY_COLOR,
                thickness=2.0,
            )

            pothole_depth_str = self._measure_pothole_depth(
                depth_raw, w, h, xmin_n, ymin_n, xmax_n, ymax_n
            )

            annotation_helper.draw_text(
                text=(
                    f"{self.labels[detection.label]} {int(detection.confidence * 100)}%\n"
                    f"x: {detection.spatialCoordinates.x:.0f}mm  "
                    f"y: {detection.spatialCoordinates.y:.0f}mm  "
                    f"z: {detection.spatialCoordinates.z:.0f}mm\n"
                    f"depth: {pothole_depth_str}"
                ),
                position=(xmin_n + 0.01, ymin_n + 0.2),
                size=12,
                color=SECONDARY_COLOR,
            )

        annotations = annotation_helper.build(
            timestamp=detections_message.getTimestamp(),
            sequence_num=detections_message.getSequenceNum(),
        )

        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_raw, alpha=0.3), cv2.COLORMAP_JET
        )
        depth_frame = dai.ImgFrame()
        depth_frame.setCvFrame(depth_colormap, dai.ImgFrame.Type.BGR888i)
        depth_frame.setTimestamp(depth_message.getTimestamp())
        depth_frame.setSequenceNum(depth_message.getSequenceNum())

        self.out_annotations.send(annotations)
        self.out_depth.send(depth_frame)

    def _measure_pothole_depth(
        self,
        depth_raw: np.ndarray,
        w: int,
        h: int,
        xmin_n: float,
        ymin_n: float,
        xmax_n: float,
        ymax_n: float,
    ) -> str:
        """Return pothole depth string, or '---' if measurement is unreliable."""
        bbox_px = (
            int(xmin_n * w),
            int(ymin_n * h),
            int(xmax_n * w),
            int(ymax_n * h),
        )
        cx = (bbox_px[0] + bbox_px[2]) / 2
        cy = (bbox_px[1] + bbox_px[3]) / 2

        roi, mask = self._depth_filter.apply(depth_raw, bbox_px)
        valid_depths = roi[mask]

        if len(valid_depths) < _MIN_VALID_PIXELS:
            return "---"

        coeffs = fit_road_plane(depth_raw, bbox_px)
        if coeffs is None:
            return "---"

        pothole_max = float(np.percentile(valid_depths, 95))
        road_at_center = road_depth_at(coeffs, cx, cy)
        depth_mm = pothole_max - road_at_center

        if depth_mm <= 0:
            return "---"

        return f"{depth_mm:.0f}mm"

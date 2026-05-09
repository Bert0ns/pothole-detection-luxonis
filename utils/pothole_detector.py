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
        self, disparity_frames: dai.Node.Output, depth_frames: dai.Node.Output
    ) -> "PotholeDetector":
        self.link_args(disparity_frames, depth_frames)
        return self

    def process(self, disparity: dai.ImgFrame, depth_frame: dai.ImgFrame) -> None:
        depth = depth_frame.getFrame()

        # 1. Establish baseline depth (flat road)
        # For a camera pointing straight down at a flat surface, the median
        # depth is likely the road surface.
        valid_mask = (depth > self._min_depth) & (depth < self._max_depth)

        baseline_depth = 0
        if valid_mask.any():
            baseline_depth = np.median(depth[valid_mask])

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

        # 2. Detect potholes (areas deeper than baseline + threshold)
        pothole_mask = (
            depth > (baseline_depth + self._depth_threshold_mm)
        ) & valid_mask

        # Convert to 8-bit for contour finding
        pothole_mask_8u = (pothole_mask * 255).astype(np.uint8)

        # Optionally perform morphological ops to clean up noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        pothole_mask_8u = cv2.morphologyEx(pothole_mask_8u, cv2.MORPH_OPEN, kernel)
        pothole_mask_8u = cv2.morphologyEx(pothole_mask_8u, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            pothole_mask_8u, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        height, width = depth.shape

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > self._min_contour_area:
                x, y, w, h = cv2.boundingRect(cnt)

                # Compute max depth inside this contour
                roi_depth = depth[y : y + h, x : x + w]
                roi_mask = pothole_mask[y : y + h, x : x + w]

                if roi_mask.any():
                    max_depth = np.max(roi_depth[roi_mask])
                    diff_mm = max_depth - baseline_depth

                    # 3. Severity scoring: based on depth difference and area
                    # simple heuristic: 30mm = Low (1), > 120mm = High (10)
                    severity = min(10, max(1, int((diff_mm - 30) / 10) + 1))

                    color = (1, 0.2, 0.2, 1)  # Red-ish for pothole

                    # draw bounding box
                    rel_xmin = x / width
                    rel_ymin = y / height
                    rel_xmax = (x + w) / width
                    rel_ymax = (y + h) / height

                    annotations_builder.draw_rectangle(
                        top_left=(rel_xmin, rel_ymin),
                        bottom_right=(rel_xmax, rel_ymax),
                        outline_color=color,
                        thickness=2,
                    )

                    # draw text
                    text = f"Sev: {severity}/10 Diff: {diff_mm:.0f}mm"
                    annotations_builder.draw_text(
                        text=text,
                        position=(
                            rel_xmin,
                            max(0.01, rel_ymin - 0.02),
                        ),  # Ensure it stays in view
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

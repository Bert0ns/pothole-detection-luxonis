from dataclasses import dataclass

import numpy as np
from scipy.ndimage import median_filter


@dataclass
class DepthFilterConfig:
    threshold_enabled: bool = True
    median_enabled: bool = True
    depth_min: int = 200   # mm — below this is invalid depthai output
    depth_max: int = 2000  # mm — above this is noise at ~1m camera height
    kernel_size: int = 3   # median filter kernel (odd number)


class DepthFilter:
    """Extracts and filters depth pixels within a bounding box.

    Applies, in order:
      1. Median filter on the ROI (removes isolated noisy pixels).
      2. Threshold filter (removes invalid and out-of-range values).

    Each step is independently toggleable via DepthFilterConfig.
    """

    def __init__(self, config: DepthFilterConfig | None = None) -> None:
        self.config = config or DepthFilterConfig()

    def apply(
        self,
        depth_frame: np.ndarray,
        bbox: tuple[int, int, int, int],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Filter depth pixels within bbox.

        Args:
            depth_frame: Full depth frame (H×W), values in mm.
            bbox: (xmin, ymin, xmax, ymax) in pixel coordinates.

        Returns:
            roi:  2D float32 array of depth values in the bbox region,
                  median-filtered if enabled.
            mask: 2D boolean array — True where pixels are valid.
                  Use roi[mask] for flat valid values (percentile etc.).
                  Use np.where(mask) for pixel positions (plane fitting etc.).
        """
        xmin, ymin, xmax, ymax = bbox
        roi = depth_frame[ymin:ymax, xmin:xmax].astype(np.float32)

        if self.config.median_enabled:
            roi = median_filter(roi, size=self.config.kernel_size)

        if self.config.threshold_enabled:
            mask = (roi >= self.config.depth_min) & (roi <= self.config.depth_max)
        else:
            mask = roi > 0  # always exclude invalid depthai zeros

        return roi, mask

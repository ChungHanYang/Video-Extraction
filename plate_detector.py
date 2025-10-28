"""
License Plate Detection Module
Provides specialized methods for detecting and analyzing license plates
"""

import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PlateDetector:
    """Specialized detector for license plates"""

    def __init__(self, plate_model_path: Optional[str] = None):
        """
        Initialize PlateDetector

        Args:
            plate_model_path: Path to YOLO model trained on license plates
                            If None, will use heuristic-based detection
        """
        self.plate_model = None
        if plate_model_path:
            try:
                self.plate_model = YOLO(plate_model_path)
                logger.info(f"Loaded plate detection model: {plate_model_path}")
            except Exception as e:
                logger.warning(f"Could not load plate model: {e}. Using heuristics.")

    def detect_plates(self, frame: np.ndarray, conf_threshold: float = 0.5) -> List[dict]:
        """
        Detect license plates in a frame

        Args:
            frame: Input frame
            conf_threshold: Confidence threshold for detections

        Returns:
            List of detected plates with bounding boxes and confidence
        """
        if self.plate_model:
            return self._detect_with_model(frame, conf_threshold)
        else:
            return self._detect_with_heuristics(frame)

    def _detect_with_model(self, frame: np.ndarray, conf_threshold: float) -> List[dict]:
        """Detect plates using YOLO model"""
        results = self.plate_model(frame, verbose=False)
        plates = []

        if results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()

            for box, conf in zip(boxes, confidences):
                if conf >= conf_threshold:
                    x1, y1, x2, y2 = map(int, box)
                    plates.append({
                        'bbox': (x1, y1, x2, y2),
                        'confidence': float(conf),
                        'area': (x2 - x1) * (y2 - y1)
                    })

        return plates

    def _detect_with_heuristics(self, frame: np.ndarray) -> List[dict]:
        """Detect plates using heuristic methods"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply bilateral filter to reduce noise while keeping edges sharp
        filtered = cv2.bilateralFilter(gray, 11, 17, 17)

        # Detect edges
        edges = cv2.Canny(filtered, 30, 200)

        # Find contours
        contours, _ = cv2.findContours(
            edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

        # Sort contours by area
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]

        plates = []

        for contour in contours:
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

            # License plates are typically rectangular (4 corners)
            if len(approx) >= 4:
                x, y, w, h = cv2.boundingRect(contour)

                # Check aspect ratio (typical plate ratios)
                aspect_ratio = w / float(h) if h > 0 else 0

                # Area threshold
                area = w * h

                # License plate characteristics:
                # - Aspect ratio between 2:1 and 5:1
                # - Minimum area
                # - Not too large (relative to frame)
                if (2.0 <= aspect_ratio <= 5.0 and
                    area > 500 and
                    area < frame.shape[0] * frame.shape[1] * 0.3):

                    # Calculate confidence based on aspect ratio and rectangularity
                    ideal_ratio = 3.5  # Common plate ratio
                    ratio_score = 1.0 - abs(aspect_ratio - ideal_ratio) / ideal_ratio
                    rectangularity = cv2.contourArea(contour) / area if area > 0 else 0

                    confidence = (ratio_score * 0.5 + rectangularity * 0.5)

                    plates.append({
                        'bbox': (x, y, x + w, y + h),
                        'confidence': float(confidence),
                        'area': area
                    })

        # Remove overlapping detections (keep higher confidence ones)
        plates = self._non_max_suppression(plates)

        return plates

    def _non_max_suppression(self, plates: List[dict], iou_threshold: float = 0.5) -> List[dict]:
        """Remove overlapping plate detections"""
        if len(plates) == 0:
            return []

        # Sort by confidence
        plates = sorted(plates, key=lambda x: x['confidence'], reverse=True)

        keep = []

        while plates:
            current = plates.pop(0)
            keep.append(current)

            # Remove overlapping plates
            plates = [
                p for p in plates
                if self._calculate_iou(current['bbox'], p['bbox']) < iou_threshold
            ]

        return keep

    def _calculate_iou(self, box1: Tuple[int, int, int, int],
                       box2: Tuple[int, int, int, int]) -> float:
        """Calculate Intersection over Union between two boxes"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        # Intersection area
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)

        if x2_i < x1_i or y2_i < y1_i:
            return 0.0

        intersection = (x2_i - x1_i) * (y2_i - y1_i)

        # Union area
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def calculate_plate_quality(self, frame: np.ndarray, plate_bbox: Tuple[int, int, int, int]) -> float:
        """
        Calculate quality score for a detected plate region

        Args:
            frame: Full frame image
            plate_bbox: Bounding box of the plate (x1, y1, x2, y2)

        Returns:
            Quality score (higher is better)
        """
        x1, y1, x2, y2 = plate_bbox

        # Extract plate region
        plate_region = frame[y1:y2, x1:x2]

        if plate_region.size == 0:
            return 0.0

        # Convert to grayscale
        gray_plate = cv2.cvtColor(plate_region, cv2.COLOR_BGR2GRAY)

        # 1. Sharpness (Laplacian variance)
        sharpness = cv2.Laplacian(gray_plate, cv2.CV_64F).var()

        # 2. Contrast (standard deviation)
        contrast = gray_plate.std()

        # 3. Brightness (should be well-lit, not too dark or bright)
        mean_brightness = gray_plate.mean()
        brightness_score = 1.0 - abs(mean_brightness - 128) / 128

        # 4. Size (larger plates are generally better)
        area = (x2 - x1) * (y2 - y1)
        size_score = min(area / 10000.0, 1.0)  # Normalize

        # Combine scores
        quality = (
            sharpness * 0.4 +
            contrast * 0.3 +
            brightness_score * 100 * 0.2 +
            size_score * 100 * 0.1
        )

        return quality

"""
Main Video Processor Module
Handles video processing, vehicle tracking, and frame extraction using YOLO11
Plate detection from separate video based on vehicle direction
"""

import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Tuple, Optional, Dict
import base64
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class TrackInfo:
    """Store tracking information for a vehicle"""
    track_id: int
    bbox_history: List[Tuple[int, int, int, int]]  # (x1, y1, x2, y2)
    frame_indices: List[int]
    center_history: List[Tuple[float, float]]


@dataclass
class ProcessingResult:
    """Result of video processing"""
    frame1: str  # base64 encoded
    frame2: str  # base64 encoded
    plate_frame: str  # base64 encoded from separate video
    frame1_index: int
    frame2_index: int
    plate_frame_index: int
    direction: str  # "toward" or "away"
    track_id: int
    plate_video_used: str  # which video was used for plate


class VideoProcessor:
    """Process video to extract vehicle frames and detect license plates"""

    def __init__(self,
                 vehicle_model_path: str = "yolo11n.pt",
                 plate_model_path: Optional[str] = None,
                 confidence_threshold: float = 0.5,
                 skip_frame: int = 1):
        """
        Initialize VideoProcessor with YOLO models

        Args:
            vehicle_model_path: Path to YOLO11 model for vehicle detection
            plate_model_path: Path to YOLO model for plate detection (optional)
            confidence_threshold: Minimum confidence for detections
            skip_frame: Process every Nth frame (1 = all frames, 2 = every other frame, etc.)
        """
        self.vehicle_model = YOLO(vehicle_model_path)
        self.plate_model = YOLO(plate_model_path) if plate_model_path else None
        self.confidence_threshold = confidence_threshold
        self.skip_frame = max(1, skip_frame)  # Ensure at least 1

        # Vehicle class IDs in COCO dataset
        self.vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck

    def process_video(self,
                     main_video_path: str,
                     toward_video_path: str,
                     away_video_path: str,
                     start_second: float,
                     frame_difference: int,
                     min_track_length: int = 10) -> Optional[ProcessingResult]:
        """
        Process video to extract vehicle frames and plate from direction-specific video

        Args:
            main_video_path: Path to main video for vehicle tracking
            toward_video_path: Path to video for plate when vehicle moves toward camera
            away_video_path: Path to video for plate when vehicle moves away
            start_second: Starting time in seconds
            frame_difference: Frame difference between selected frames
            min_track_length: Minimum number of frames to consider a valid track

        Returns:
            ProcessingResult or None if no valid vehicle found
        """
        cap = cv2.VideoCapture(main_video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {main_video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        start_frame = int(start_second * fps)

        # Set to start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        # Track vehicles
        tracks = self._track_vehicles(cap, start_frame, total_frames)

        if not tracks:
            cap.release()
            return None

        # Select best track
        best_track = self._select_best_track(tracks, min_track_length)

        if not best_track:
            cap.release()
            return None

        # Determine direction
        direction = self._determine_direction(best_track)

        # Select two frames (first = closest to center, second = 1 second after)
        frame1_idx, frame2_idx = self._select_frame_pair(
            best_track, fps, frame_width, frame_height
        )

        if frame1_idx is None or frame2_idx is None:
            cap.release()
            return None

        # Select appropriate video for plate detection based on direction
        plate_video_path = toward_video_path if direction == "toward" else away_video_path

        # Find best plate frame from the appropriate video using same frame indices
        plate_frame_idx = self._find_best_plate_frame(
            plate_video_path, frame1_idx, frame2_idx
        )

        # Extract and encode frames
        frame1_b64 = self._extract_and_encode_frame(main_video_path, frame1_idx)
        frame2_b64 = self._extract_and_encode_frame(main_video_path, frame2_idx)
        plate_frame_b64 = self._extract_and_encode_frame(plate_video_path, plate_frame_idx)

        cap.release()

        return ProcessingResult(
            frame1=frame1_b64,
            frame2=frame2_b64,
            plate_frame=plate_frame_b64,
            frame1_index=frame1_idx,
            frame2_index=frame2_idx,
            plate_frame_index=plate_frame_idx,
            direction=direction,
            track_id=best_track.track_id,
            plate_video_used=plate_video_path
        )

    def _track_vehicles(self,
                       cap: cv2.VideoCapture,
                       start_frame: int,
                       total_frames: int) -> Dict[int, TrackInfo]:
        """
        Track vehicles in video using YOLO11

        Args:
            cap: OpenCV VideoCapture object
            start_frame: Starting frame index
            total_frames: Total frames in video

        Returns:
            Dictionary of track_id to TrackInfo
        """
        tracks = defaultdict(lambda: TrackInfo(
            track_id=0,
            bbox_history=[],
            frame_indices=[],
            center_history=[]
        ))

        frame_idx = start_frame
        frame_count = 0

        while cap.isOpened() and frame_idx < total_frames:
            ret, frame = cap.read()
            if not ret:
                break

            # Skip frames based on skip_frame parameter
            if frame_count % self.skip_frame == 0:
                # Run YOLO tracking
                results = self.vehicle_model.track(
                    frame,
                    persist=True,
                    classes=self.vehicle_classes,
                    conf=self.confidence_threshold,
                    verbose=False
                )

                if results[0].boxes is not None and results[0].boxes.id is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                    confidences = results[0].boxes.conf.cpu().numpy()

                    for box, track_id, conf in zip(boxes, track_ids, confidences):
                        if conf >= self.confidence_threshold:
                            x1, y1, x2, y2 = map(int, box)
                            center_x = (x1 + x2) / 2
                            center_y = (y1 + y2) / 2

                            if tracks[track_id].track_id == 0:
                                tracks[track_id].track_id = track_id

                            tracks[track_id].bbox_history.append((x1, y1, x2, y2))
                            tracks[track_id].frame_indices.append(frame_idx)
                            tracks[track_id].center_history.append((center_x, center_y))

            frame_idx += 1
            frame_count += 1

        return dict(tracks)

    def _select_best_track(self,
                          tracks: Dict[int, TrackInfo],
                          min_track_length: int) -> Optional[TrackInfo]:
        """
        Select the most prominent vehicle track

        Args:
            tracks: Dictionary of tracks
            min_track_length: Minimum track length

        Returns:
            Best TrackInfo or None
        """
        valid_tracks = [
            track for track in tracks.values()
            if len(track.frame_indices) >= min_track_length
        ]

        if not valid_tracks:
            return None

        # Select track with largest average bounding box area
        best_track = max(
            valid_tracks,
            key=lambda t: np.mean([
                (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                for bbox in t.bbox_history
            ])
        )

        return best_track

    def _determine_direction(self, track: TrackInfo) -> str:
        """
        Determine if vehicle is moving toward or away from camera

        Args:
            track: TrackInfo object

        Returns:
            "toward" or "away"
        """
        # Calculate average box area over time
        areas = [
            (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            for bbox in track.bbox_history
        ]

        # Use linear regression on areas
        if len(areas) < 2:
            return "away"  # default

        x = np.arange(len(areas))
        z = np.polyfit(x, areas, 1)
        slope = z[0]

        # Positive slope means getting larger (toward)
        # Negative slope means getting smaller (away)
        return "toward" if slope > 0 else "away"

    def _select_frame_pair(self,
                          track: TrackInfo,
                          fps: float,
                          frame_width: int,
                          frame_height: int) -> Tuple[Optional[int], Optional[int]]:
        """
        Select first frame where bounding box is closest to center,
        and second frame 1 second after it

        Args:
            track: TrackInfo object
            fps: Video frames per second
            frame_width: Width of video frame
            frame_height: Height of video frame

        Returns:
            Tuple of (frame1_index, frame2_index) or (None, None)
        """
        frame_indices = track.frame_indices

        # Calculate frame difference for 1 second
        frame_difference = int(fps * 1.0)

        if len(frame_indices) < frame_difference + 1:
            return None, None

        # Calculate center of frame
        frame_center_x = frame_width / 2
        frame_center_y = frame_height / 2

        # Find frame where bbox center is closest to frame center
        min_distance = float('inf')
        best_idx = None

        for i in range(len(frame_indices) - frame_difference):
            bbox = track.bbox_history[i]

            # Calculate bbox center
            bbox_center_x = (bbox[0] + bbox[2]) / 2
            bbox_center_y = (bbox[1] + bbox[3]) / 2

            # Calculate Euclidean distance to frame center
            distance = np.sqrt(
                (bbox_center_x - frame_center_x) ** 2 +
                (bbox_center_y - frame_center_y) ** 2
            )

            if distance < min_distance:
                min_distance = distance
                best_idx = i

        if best_idx is None:
            return None, None

        # Select first frame and frame 1 second after it
        frame1_idx = frame_indices[best_idx]
        frame2_idx = frame_indices[best_idx + frame_difference]

        return frame1_idx, frame2_idx

    def _find_best_plate_frame(self,
                               plate_video_path: str,
                               frame1_idx: int,
                               frame2_idx: int) -> int:
        """
        Find the frame with best license plate visibility from plate video

        Args:
            plate_video_path: Path to plate detection video
            frame1_idx: First frame index
            frame2_idx: Second frame index

        Returns:
            Frame index with best plate visibility
        """
        cap = cv2.VideoCapture(plate_video_path)

        if not cap.isOpened():
            raise ValueError(f"Cannot open plate video: {plate_video_path}")

        # Scan frames between frame1_idx and frame2_idx
        start_frame = min(frame1_idx, frame2_idx)
        end_frame = max(frame1_idx, frame2_idx)

        best_frame_idx = start_frame
        best_score = -1

        for frame_idx in range(start_frame, end_frame + 1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()

            if not ret:
                continue

            # Calculate plate score for full frame
            score = self._calculate_plate_score(frame)

            if score > best_score:
                best_score = score
                best_frame_idx = frame_idx

        cap.release()

        # If no good plate found, return middle frame
        if best_score == -1:
            return (start_frame + end_frame) // 2

        return best_frame_idx

    def _calculate_plate_score(self, frame: np.ndarray) -> float:
        """
        Calculate score for plate visibility in a frame

        Args:
            frame: Full frame image

        Returns:
            Score indicating plate visibility
        """
        score = 0.0

        # If we have a plate detection model, use it
        if self.plate_model:
            results = self.plate_model(frame, verbose=False)
            if results[0].boxes is not None and len(results[0].boxes) > 0:
                # Score based on plate detection confidence and size
                confidences = results[0].boxes.conf.cpu().numpy()
                boxes = results[0].boxes.xyxy.cpu().numpy()

                for conf, box in zip(confidences, boxes):
                    plate_area = (box[2] - box[0]) * (box[3] - box[1])
                    score += conf * plate_area
        else:
            # Heuristic: look for rectangular regions that might be a plate
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect edges
            edges = cv2.Canny(gray, 50, 150)

            # Find contours
            contours, _ = cv2.findContours(
                edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            # Look for rectangular contours with plate-like aspect ratio
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 100:  # Too small
                    continue

                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / float(h) if h > 0 else 0

                # License plates typically have aspect ratio 2:1 to 4:1
                if 2.0 <= aspect_ratio <= 4.5:
                    score += area * aspect_ratio * 0.01

            # Add sharpness score (plates should be in focus)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            score += laplacian_var * 0.1

        return score

    def _extract_and_encode_frame(self,
                                  video_path: str,
                                  frame_idx: int) -> str:
        """
        Extract frame from video and encode to base64

        Args:
            video_path: Path to video
            frame_idx: Frame index to extract

        Returns:
            Base64 encoded image string
        """
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            raise ValueError(f"Cannot extract frame {frame_idx} from {video_path}")

        # Encode frame to JPEG
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

        # Convert to base64
        frame_b64 = base64.b64encode(buffer).decode('utf-8')

        return frame_b64

"""
Video Extraction Service
Main service API for processing videos and extracting vehicle frames
"""

import os
import json
from typing import Optional, Dict, Any
from video_processor import VideoProcessor, ProcessingResult
from plate_detector import PlateDetector
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VideoExtractionService:
    """Service for processing videos and extracting vehicle information"""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the service

        Args:
            config_path: Path to configuration file (JSON)
        """
        self.config = self._load_config(config_path)

        # Initialize components
        self.processor = VideoProcessor(
            vehicle_model_path=self.config.get('vehicle_model', 'yolo11n.pt'),
            plate_model_path=self.config.get('plate_model', None),
            confidence_threshold=self.config.get('confidence_threshold', 0.5),
            skip_frame=self.config.get('skip_frame', 1)
        )

        self.plate_detector = PlateDetector(
            plate_model_path=self.config.get('plate_model', None)
        )

        logger.info("VideoExtractionService initialized")

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file or use defaults"""
        default_config = {
            'vehicle_model': 'yolo11n.pt',
            'plate_model': None,
            'confidence_threshold': 0.5,
            'min_track_length': 10,
            'default_frame_difference': 30,
            'skip_frame': 1
        }

        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                default_config.update(user_config)
                logger.info(f"Loaded configuration from {config_path}")
            except Exception as e:
                logger.warning(f"Could not load config file: {e}. Using defaults.")

        return default_config

    def extract_frames(self,
                      main_video_path: str,
                      toward_video_path: str,
                      away_video_path: str,
                      start_second: float = 0.0,
                      frame_difference: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """
        Extract vehicle frames from video

        Args:
            main_video_path: Path to main video for tracking
            toward_video_path: Path to video for plate when moving toward
            away_video_path: Path to video for plate when moving away
            start_second: Starting time in seconds
            frame_difference: Frame difference between selected frames

        Returns:
            Dictionary containing results or None if processing failed
        """
        if frame_difference is None:
            frame_difference = self.config.get('default_frame_difference', 30)

        logger.info(f"Processing video: {main_video_path}")
        logger.info(f"Start time: {start_second}s, Frame difference: {frame_difference}")

        try:
            result = self.processor.process_video(
                main_video_path=main_video_path,
                toward_video_path=toward_video_path,
                away_video_path=away_video_path,
                start_second=start_second,
                frame_difference=frame_difference,
                min_track_length=self.config.get('min_track_length', 10)
            )

            if result is None:
                logger.warning("No valid vehicle tracks found")
                return None

            # Convert to dictionary
            result_dict = {
                'frame1': result.frame1,
                'frame2': result.frame2,
                'plate_frame': result.plate_frame,
                'frame1_index': result.frame1_index,
                'frame2_index': result.frame2_index,
                'plate_frame_index': result.plate_frame_index,
                'direction': result.direction,
                'track_id': result.track_id,
                'plate_video_used': result.plate_video_used
            }

            logger.info(f"Successfully processed. Direction: {result.direction}, "
                       f"Frames: {result.frame1_index}, {result.frame2_index}, "
                       f"Plate: {result.plate_frame_index}")

            return result_dict

        except Exception as e:
            logger.error(f"Error processing video: {e}", exc_info=True)
            return None

    def save_result(self, result: Dict[str, Any], output_path: str) -> bool:
        """
        Save processing result to file

        Args:
            result: Processing result dictionary
            output_path: Path to save result (JSON)

        Returns:
            True if successful
        """
        try:
            # Don't save base64 images to JSON (too large)
            # Save metadata only
            metadata = {
                k: v for k, v in result.items()
                if k not in ['frame1', 'frame2', 'plate_frame']
            }

            with open(output_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            logger.info(f"Result metadata saved to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Error saving result: {e}")
            return False

    def save_frames_as_images(self,
                             result: Dict[str, Any],
                             output_dir: str,
                             prefix: str = "frame") -> bool:
        """
        Save extracted frames as image files

        Args:
            result: Processing result dictionary
            output_dir: Directory to save images
            prefix: Prefix for filenames

        Returns:
            True if successful
        """
        import base64

        try:
            os.makedirs(output_dir, exist_ok=True)

            # Save frame 1
            frame1_data = base64.b64decode(result['frame1'])
            with open(os.path.join(output_dir, f"{prefix}_1.jpg"), 'wb') as f:
                f.write(frame1_data)

            # Save frame 2
            frame2_data = base64.b64decode(result['frame2'])
            with open(os.path.join(output_dir, f"{prefix}_2.jpg"), 'wb') as f:
                f.write(frame2_data)

            # Save plate frame
            plate_data = base64.b64decode(result['plate_frame'])
            with open(os.path.join(output_dir, f"{prefix}_plate.jpg"), 'wb') as f:
                f.write(plate_data)

            logger.info(f"Frames saved to {output_dir}")
            return True

        except Exception as e:
            logger.error(f"Error saving frames: {e}")
            return False


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) < 4:
        print("Usage: python service.py <main_video> <toward_video> <away_video> [start_second] [frame_diff]")
        sys.exit(1)

    main_video = sys.argv[1]
    toward_video = sys.argv[2]
    away_video = sys.argv[3]
    start_sec = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
    frame_diff = int(sys.argv[5]) if len(sys.argv) > 5 else 30

    # Initialize service
    service = VideoExtractionService()

    # Process video
    result = service.extract_frames(
        main_video_path=main_video,
        toward_video_path=toward_video,
        away_video_path=away_video,
        start_second=start_sec,
        frame_difference=frame_diff
    )

    if result:
        # Save frames
        service.save_frames_as_images(result, "output")
        service.save_result(result, "output/metadata.json")
        print("Processing complete! Check 'output' directory for results.")
    else:
        print("Processing failed. Check logs for details.")

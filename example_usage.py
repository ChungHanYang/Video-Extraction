"""
Example usage of the Video Extraction Service
"""

from service import VideoExtractionService
import base64
import cv2
import numpy as np


def decode_and_display_frame(frame_b64: str, window_name: str = "Frame"):
    """Helper function to decode and display a base64 frame"""
    # Decode base64 to bytes
    frame_bytes = base64.b64decode(frame_b64)

    # Convert to numpy array
    nparr = np.frombuffer(frame_bytes, np.uint8)

    # Decode image
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Display
    cv2.imshow(window_name, frame)


def example_basic_usage():
    """Basic usage example"""
    print("=== Basic Usage Example ===\n")

    # Initialize service with default config
    service = VideoExtractionService(config_path='config.json')

    # Process video
    result = service.extract_frames(
        main_video_path='videos/main.mp4',
        toward_video_path='videos/front_camera.mp4',
        away_video_path='videos/rear_camera.mp4',
        start_second=5.0,
        frame_difference=30
    )

    if result:
        print(f"✓ Processing successful!")
        print(f"  Direction: {result['direction']}")
        print(f"  Frame 1 Index: {result['frame1_index']}")
        print(f"  Frame 2 Index: {result['frame2_index']}")
        print(f"  Plate Frame Index: {result['plate_frame_index']}")
        print(f"  Plate Video Used: {result['plate_video_used']}")
        print(f"  Track ID: {result['track_id']}")

        # Save frames as images
        service.save_frames_as_images(result, 'output/example1')
        service.save_result(result, 'output/example1/metadata.json')

        print(f"\n✓ Frames saved to 'output/example1'")
    else:
        print("✗ Processing failed")


def example_display_frames():
    """Example showing how to display the extracted frames"""
    print("\n=== Display Frames Example ===\n")

    service = VideoExtractionService()

    result = service.extract_frames(
        main_video_path='videos/main.mp4',
        toward_video_path='videos/front_camera.mp4',
        away_video_path='videos/rear_camera.mp4',
        start_second=5.0,
        frame_difference=30
    )

    if result:
        print("Displaying frames (press any key to continue)...")

        # Display frame 1
        decode_and_display_frame(result['frame1'], "Frame 1")
        cv2.waitKey(0)

        # Display frame 2
        decode_and_display_frame(result['frame2'], "Frame 2")
        cv2.waitKey(0)

        # Display plate frame
        decode_and_display_frame(result['plate_frame'], "Plate Frame")
        cv2.waitKey(0)

        cv2.destroyAllWindows()


def example_batch_processing():
    """Example of processing multiple videos"""
    print("\n=== Batch Processing Example ===\n")

    service = VideoExtractionService()

    # List of video sets to process
    video_sets = [
        {
            'name': 'video_set_1',
            'main': 'videos/set1_main.mp4',
            'toward': 'videos/set1_front.mp4',
            'away': 'videos/set1_rear.mp4',
            'start_second': 5.0
        },
        {
            'name': 'video_set_2',
            'main': 'videos/set2_main.mp4',
            'toward': 'videos/set2_front.mp4',
            'away': 'videos/set2_rear.mp4',
            'start_second': 10.0
        },
    ]

    results = []

    for i, video_set in enumerate(video_sets):
        print(f"Processing {video_set['name']}...")

        result = service.extract_frames(
            main_video_path=video_set['main'],
            toward_video_path=video_set['toward'],
            away_video_path=video_set['away'],
            start_second=video_set['start_second'],
            frame_difference=30
        )

        if result:
            # Save to separate directory
            output_dir = f"output/{video_set['name']}"
            service.save_frames_as_images(result, output_dir)
            service.save_result(result, f"{output_dir}/metadata.json")

            results.append({
                'name': video_set['name'],
                'direction': result['direction'],
                'success': True
            })
            print(f"  ✓ Success - Direction: {result['direction']}")
        else:
            results.append({
                'name': video_set['name'],
                'success': False
            })
            print(f"  ✗ Failed")

    print(f"\n=== Summary ===")
    print(f"Total: {len(results)}")
    print(f"Success: {sum(1 for r in results if r['success'])}")
    print(f"Failed: {sum(1 for r in results if not r['success'])}")


def example_custom_config():
    """Example with custom configuration"""
    print("\n=== Custom Configuration Example ===\n")

    # Create custom configuration
    custom_config = {
        'vehicle_model': 'yolo11s.pt',  # Use small model
        'confidence_threshold': 0.6,     # Higher threshold
        'min_track_length': 15,          # Require longer tracks
        'default_frame_difference': 45   # Larger frame difference
    }

    # Save custom config
    import json
    with open('custom_config.json', 'w') as f:
        json.dump(custom_config, f, indent=2)

    # Initialize with custom config
    service = VideoExtractionService(config_path='custom_config.json')

    result = service.extract_frames(
        main_video_path='videos/main.mp4',
        toward_video_path='videos/front_camera.mp4',
        away_video_path='videos/rear_camera.mp4',
        start_second=5.0
    )

    if result:
        print("✓ Processing with custom config successful!")
        service.save_frames_as_images(result, 'output/custom_config')


def example_api_integration():
    """Example showing how to integrate with an API"""
    print("\n=== API Integration Example ===\n")

    service = VideoExtractionService()

    # Simulate receiving request from API
    request_data = {
        'main_video': 'videos/main.mp4',
        'toward_video': 'videos/front_camera.mp4',
        'away_video': 'videos/rear_camera.mp4',
        'start_second': 5.0,
        'frame_difference': 30
    }

    # Process request
    result = service.extract_frames(
        main_video_path=request_data['main_video'],
        toward_video_path=request_data['toward_video'],
        away_video_path=request_data['away_video'],
        start_second=request_data['start_second'],
        frame_difference=request_data['frame_difference']
    )

    if result:
        # Prepare API response
        api_response = {
            'status': 'success',
            'data': {
                'frames': {
                    'frame1': result['frame1'],  # base64
                    'frame2': result['frame2'],  # base64
                    'plate': result['plate_frame']  # base64
                },
                'metadata': {
                    'direction': result['direction'],
                    'frame_indices': [
                        result['frame1_index'],
                        result['frame2_index']
                    ],
                    'plate_frame_index': result['plate_frame_index'],
                    'track_id': result['track_id']
                }
            }
        }

        print("API Response prepared:")
        print(f"  Status: {api_response['status']}")
        print(f"  Direction: {api_response['data']['metadata']['direction']}")
        print(f"  Frame indices: {api_response['data']['metadata']['frame_indices']}")

        # In real API, would return JSON response
        # return jsonify(api_response)
    else:
        api_response = {
            'status': 'error',
            'message': 'No vehicle detected in video'
        }
        print(f"API Error: {api_response['message']}")


if __name__ == "__main__":
    print("Video Extraction Service - Examples\n")
    print("=" * 50)

    # Run examples (comment out examples you don't want to run)

    # Example 1: Basic usage
    # example_basic_usage()

    # Example 2: Display frames
    # example_display_frames()

    # Example 3: Batch processing
    # example_batch_processing()

    # Example 4: Custom configuration
    # example_custom_config()

    # Example 5: API integration
    example_api_integration()

    print("\n" + "=" * 50)
    print("Examples complete!")

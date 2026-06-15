#!/usr/bin/env python
"""
MediaPipe Pose Testing Script
Tests pose estimation on images and videos with visual output.
"""

import sys
import cv2
import numpy as np
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from backend.services.pose import MediaPipePoseService

# MediaPipe landmark connections for visualization
POSE_CONNECTIONS = [
    # Face
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    # Torso
    (9, 10),
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24),
    # Legs
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32)
]


def draw_landmarks(image, keypoints, confidence_threshold=0.5):
    """
    Draw pose landmarks on image.
    
    Args:
        image: Input image (BGR)
        keypoints: List of [x, y, z, visibility] for 33 landmarks
        confidence_threshold: Minimum visibility to draw landmark
        
    Returns:
        Image with landmarks drawn
    """
    h, w, _ = image.shape
    output = image.copy()
    
    # Draw landmarks
    for i, kp in enumerate(keypoints):
        x, y = kp[0], kp[1]
        visibility = kp[3] if len(kp) > 3 else 1.0
        
        # Convert normalized coordinates to pixels
        px, py = int(x * w), int(y * h)
        
        # Color based on visibility
        if visibility >= confidence_threshold:
            color = (0, 255, 0)  # Green for high confidence
            radius = 5
        else:
            color = (0, 165, 255)  # Orange for low confidence
            radius = 3
        
        cv2.circle(output, (px, py), radius, color, -1)
        
        # Draw landmark index for key points
        if i in [0, 11, 12, 23, 24, 15, 16, 27, 28]:  # Key landmarks
            cv2.putText(output, str(i), (px + 5, py - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    
    # Draw connections
    for connection in POSE_CONNECTIONS:
        start_idx, end_idx = connection
        
        if start_idx >= len(keypoints) or end_idx >= len(keypoints):
            continue
        
        start_kp = keypoints[start_idx]
        end_kp = keypoints[end_idx]
        
        start_vis = start_kp[3] if len(start_kp) > 3 else 1.0
        end_vis = end_kp[3] if len(end_kp) > 3 else 1.0
        
        if start_vis >= confidence_threshold and end_vis >= confidence_threshold:
            start_pt = (int(start_kp[0] * w), int(start_kp[1] * h))
            end_pt = (int(end_kp[0] * w), int(end_kp[1] * h))
            cv2.line(output, start_pt, end_pt, (255, 200, 0), 2)
    
    return output


def test_on_image(image_path: str, service: MediaPipePoseService):
    """Test pose estimation on a single image."""
    print(f"\n{'='*60}")
    print(f"Testing on image: {image_path}")
    print('='*60)
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Failed to load image: {image_path}")
        return
    
    print(f"✅ Image loaded: {image.shape}")
    
    # Extract pose
    pose_record = service.extract(image, track_id=1, frame_id=0)
    
    if pose_record is None:
        print("❌ No pose detected in image")
        return
    
    print(f"✅ Pose detected!")
    print(f"   Landmarks: {len(pose_record.keypoints)}")
    
    # Calculate statistics
    keypoints = pose_record.keypoints
    visibilities = [kp[3] for kp in keypoints if len(kp) > 3]
    
    if visibilities:
        print(f"   Visibility - Min: {min(visibilities):.3f}, "
              f"Max: {max(visibilities):.3f}, "
              f"Mean: {np.mean(visibilities):.3f}")
        print(f"   High confidence (>0.7): {sum(1 for v in visibilities if v > 0.7)}/{len(visibilities)}")
    
    # Draw landmarks
    output = draw_landmarks(image, keypoints)
    
    # Add info text
    h, w = output.shape[:2]
    cv2.rectangle(output, (10, 10), (400, 120), (0, 0, 0), -1)
    cv2.putText(output, "MediaPipe Pose Test", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(output, f"Landmarks: {len(keypoints)}", (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    if visibilities:
        cv2.putText(output, f"Avg Confidence: {np.mean(visibilities):.2f}", (20, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(output, "Green=High conf, Orange=Low conf", (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    # Save output
    output_path = Path(image_path).stem + "_pose.jpg"
    cv2.imwrite(output_path, output)
    print(f"✅ Output saved: {output_path}")
    
    # Display
    cv2.imshow("MediaPipe Pose Test", output)
    print("\nPress any key to continue...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def test_on_video(video_path: str, service: MediaPipePoseService, max_frames=100):
    """Test pose estimation on video."""
    print(f"\n{'='*60}")
    print(f"Testing on video: {video_path}")
    print('='*60)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Failed to open video: {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"✅ Video opened: {width}x{height} @ {fps:.1f} FPS, {total_frames} frames")
    
    # Create output video
    output_path = Path(video_path).stem + "_pose.mp4"
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'),
                             fps, (width, height))
    
    frame_count = 0
    success_count = 0
    
    print(f"\nProcessing up to {max_frames} frames...")
    
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Extract pose
        pose_record = service.extract(frame, track_id=1, frame_id=frame_count)
        
        if pose_record:
            success_count += 1
            keypoints = pose_record.keypoints
            output_frame = draw_landmarks(frame, keypoints)
            
            # Add frame info
            visibilities = [kp[3] for kp in keypoints if len(kp) > 3]
            if visibilities:
                avg_conf = np.mean(visibilities)
                color = (0, 255, 0) if avg_conf > 0.7 else (0, 165, 255)
            else:
                avg_conf = 0.0
                color = (0, 0, 255)
            
            cv2.putText(output_frame, f"Frame: {frame_count}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(output_frame, f"Confidence: {avg_conf:.2f}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        else:
            output_frame = frame.copy()
            cv2.putText(output_frame, f"Frame: {frame_count} - No pose detected",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        writer.write(output_frame)
        frame_count += 1
        
        if frame_count % 10 == 0:
            print(f"  Processed {frame_count} frames, {success_count} poses detected...")
    
    cap.release()
    writer.release()
    
    print(f"\n✅ Processed {frame_count} frames")
    print(f"   Poses detected: {success_count}/{frame_count} ({100*success_count/frame_count:.1f}%)")
    print(f"✅ Output saved: {output_path}")
    
    # Show statistics
    stats = service.get_stats()
    print(f"\nService Statistics:")
    print(f"   Total extractions: {stats['total_extractions']}")
    print(f"   Total failures: {stats['total_failures']}")
    print(f"   Success rate: {stats['success_rate']:.2%}")


def test_on_webcam(service: MediaPipePoseService):
    """Test pose estimation on webcam feed."""
    print(f"\n{'='*60}")
    print("Testing on webcam (Press 'q' to quit)")
    print('='*60)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Failed to open webcam")
        return
    
    print("✅ Webcam opened")
    
    frame_count = 0
    success_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Extract pose
        pose_record = service.extract(frame, track_id=1, frame_id=frame_count)
        
        if pose_record:
            success_count += 1
            keypoints = pose_record.keypoints
            output = draw_landmarks(frame, keypoints)
            
            # Add info
            visibilities = [kp[3] for kp in keypoints if len(kp) > 3]
            if visibilities:
                avg_conf = np.mean(visibilities)
                cv2.putText(output, f"Confidence: {avg_conf:.2f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            output = frame
            cv2.putText(output, "No pose detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Show stats
        if frame_count > 0:
            success_rate = success_count / frame_count
            cv2.putText(output, f"Success: {success_rate:.1%}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow("MediaPipe Pose - Webcam", output)
        
        frame_count += 1
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\n✅ Processed {frame_count} frames")
    print(f"   Poses detected: {success_count}/{frame_count} ({100*success_count/frame_count:.1f}%)")


def main():
    print("="*60)
    print("MEDIAPIPE POSE ESTIMATION TEST")
    print("="*60)
    
    # Initialize service
    print("\nInitializing MediaPipe Pose service...")
    service = MediaPipePoseService()
    
    if not service.available:
        print("\n❌ MediaPipe Pose is NOT available!")
        print("   Please install: pip install mediapipe==0.10.14")
        return
    
    print("✅ MediaPipe Pose service initialized successfully!")
    
    # Test options
    print("\nTest Options:")
    print("  1. Test on image")
    print("  2. Test on video")
    print("  3. Test on webcam")
    print("  4. Run all tests")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == "1":
        image_path = input("Enter image path: ").strip()
        test_on_image(image_path, service)
        
    elif choice == "2":
        video_path = input("Enter video path: ").strip()
        max_frames = input("Max frames to process (default 100): ").strip()
        max_frames = int(max_frames) if max_frames else 100
        test_on_video(video_path, service, max_frames)
        
    elif choice == "3":
        test_on_webcam(service)
        
    elif choice == "4":
        # Try to find sample files
        sample_image = "backend/uploads/*.jpg"
        sample_video = "backend/uploads/*.mp4"
        
        import glob
        images = glob.glob(sample_image)
        videos = glob.glob(sample_video)
        
        if images:
            test_on_image(images[0], service)
        else:
            print("\nNo sample images found in backend/uploads/")
        
        if videos:
            test_on_video(videos[0], service, max_frames=50)
        else:
            print("\nNo sample videos found in backend/uploads/")
    
    else:
        print("Invalid choice")
    
    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()

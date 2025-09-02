import argparse
from models import load_yolo_model
from video_processor import track_and_detect_in_video

def main():
    parser = argparse.ArgumentParser(description="YOLO-based car and license plate detection in videos")
    parser.add_argument('--video_path', type=str, required=True, help="Path to input video")
    parser.add_argument('--output_path', type=str, required=True, help="Path to output video")
    parser.add_argument('--car_model_path', type=str, default="yolo11x.pt", help="Path to car YOLO model")
    parser.add_argument('--plate_model_path', type=str, required=True, help="Path to plate YOLO model")
    parser.add_argument('--car_conf_threshold', type=float, default=0.55, help="Confidence threshold for car detection")
    parser.add_argument('--plate_conf_threshold', type=float, default=0.78, help="Confidence threshold for plate detection")
    args = parser.parse_args()

    # Load models
    car_model = load_yolo_model(args.car_model_path)
    plate_model = load_yolo_model(args.plate_model_path)

    # Process video
    track_and_detect_in_video(
        video_path=args.video_path,
        output_path=args.output_path,
        car_model=car_model,
        plate_model=plate_model,
        car_conf_threshold=args.car_conf_threshold,
        plate_conf_threshold=args.plate_conf_threshold
    )

if __name__ == "__main__":
    main()

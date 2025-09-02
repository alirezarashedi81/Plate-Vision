import cv2
from utils import fancy_bbox

def track_and_detect_in_video(video_path, output_path, car_model, plate_model,
                             car_conf_threshold=0.65, plate_conf_threshold=0.70, display_func=None):
    """
    Process a video to detect and track cars and license plates using YOLO models.
    
    Args:
        video_path (str): Path to input video.
        output_path (str): Path to save output video.
        car_model (YOLO): YOLO model for car detection.
        plate_model (YOLO): YOLO model for plate detection.
        car_conf_threshold (float): Confidence threshold for car detection.
        plate_conf_threshold (float): Confidence threshold for plate detection.
        display_func (callable, optional): Function to display frames (e.g., cv2_imshow for Colab).
    """
    # Get class labels
    car_class_list = car_model.names
    plate_class_list = plate_model.names
    plate_class_name = plate_class_list[0]  # single-class plate model

    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    # Video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO tracking for cars
        car_results = car_model.track(frame, persist=True, conf=car_conf_threshold)

        if car_results[0].boxes.data is not None:
            car_boxes = car_results[0].boxes.xyxy.cpu()
            car_track_ids = car_results[0].boxes.id.int().cpu().tolist() if car_results[0].boxes.id is not None else []
            car_class_indices = car_results[0].boxes.cls.int().cpu().tolist()
            car_confidences = car_results[0].boxes.conf.cpu()

            for i, (box, class_idx, conf) in enumerate(zip(car_boxes, car_class_indices, car_confidences)):
                class_name = car_class_list[class_idx]
                if class_name != "car":
                    continue

                x1, y1, x2, y2 = map(int, box)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                # Fancy bounding box for cars
                track_id = car_track_ids[i] if car_track_ids else "N/A"
                label = f"Car {track_id} ({conf:.2f})"
                fancy_bbox(frame, x1, y1, x2, y2, label, color=(255, 0, 0))

                # Center point
                cv2.circle(frame, (cx, cy), 4, (60, 191, 255), -1)

        # Run YOLO detection for plates
        plate_results = plate_model(frame, conf=plate_conf_threshold)

        if plate_results[0].boxes.data is not None:
            plate_boxes = plate_results[0].boxes.xyxy.cpu()
            plate_class_indices = plate_results[0].boxes.cls.int().cpu().tolist()
            plate_confidences = plate_results[0].boxes.conf.cpu()

            for box, class_idx, conf in zip(plate_boxes, plate_class_indices, plate_confidences):
                class_name = plate_class_list[class_idx]
                if class_name != plate_class_name:
                    continue

                x1, y1, x2, y2 = map(int, box)

                # Fancy bounding box for plates
                label = f"Plate ({conf:.2f})"
                fancy_bbox(frame, x1, y1, x2, y2, label, color=(90, 255, 60))

        # Write frame to output
        out.write(frame)

        # Optional display
        if display_func:
            display_func(frame)

        # Exit with ESC if running locally
        if cv2.waitKey(1) & 0xFF == 27:
            break

    # Cleanup
    cap.release()
    out.release()
    if not display_func:
        cv2.destroyAllWindows()

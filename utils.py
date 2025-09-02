def fancy_bbox(frame, x1, y1, x2, y2, label, color):
    """
    Draw a fancy bounding box with label on the frame.
    
    Args:
        frame (np.ndarray): Input frame to draw on.
        x1, y1, x2, y2 (int): Bounding box coordinates.
        label (str): Label to display.
        color (tuple): RGB color for the bounding box.
    """
    # Draw rectangle
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    
    # Draw label background
    (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (x1, y1 - label_height - 10), (x1 + label_width, y1), color, -1)
    
    # Draw label text
    cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

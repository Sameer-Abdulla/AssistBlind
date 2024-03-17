import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv


def perform_closest_detection(model: YOLO, frame: np.ndarray, box_annotator: sv.BoxAnnotator) -> tuple:
    result = model(frame, agnostic_nms=True)[0]
    detections = sv.Detections.from_yolov8(result)
    
    # Find the largest bounding box
    largest_box = None
    largest_area = 0
    largest_label = None
    for box, confidence, class_id, _ in detections:
        x1, y1, x2, y2 = box
        area = (x2 - x1) * (y2 - y1)
        if area > largest_area:
            largest_area = area
            largest_box = box
            largest_label = model.model.names[class_id]
            
    # Annotate only the largest bounding box
    if largest_box is not None:
        labels = [f"{largest_label} (Closest Object)"]
        annotated_frame = box_annotator.annotate(scene=frame, detections=[(largest_box, 1.0, 0, None)], labels=labels)
    else:
        annotated_frame = frame
    
    return annotated_frame, detections

def closest_object(cap):
    
    model = YOLO("yolov8l.pt")
    box_annotator = sv.BoxAnnotator(thickness=2, text_thickness=2, text_scale=1)

    while True:
        ret, frame = cap.read()
        
        # Check if any text is present in the frame
        annotated_frame, detections = perform_closest_detection(model, frame, box_annotator)
        #cv2.imshow("yolov8", annotated_frame)
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame = buffer.tobytes()
        return frame

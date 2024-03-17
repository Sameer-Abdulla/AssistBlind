import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv



def perform_object_detection(model: YOLO, frame: np.ndarray, box_annotator: sv.BoxAnnotator) -> tuple:
    result = model(frame, agnostic_nms=True)[0]
    detections = sv.Detections.from_yolov8(result)
    labels = [
        f"{model.model.names[class_id]} {confidence:0.2f}"
        for _, confidence, class_id, _
        in detections
    ]
    annotated_frame = box_annotator.annotate(scene=frame, detections=detections, labels=labels)
    return annotated_frame, detections



def test_object(cap):
    
    model = YOLO("yolov8l.pt")
    box_annotator = sv.BoxAnnotator(thickness=2, text_thickness=2, text_scale=1)

    while True:
        ret, frame = cap.read()
        
        # Check if any text is present in the frame
        annotated_frame, detections = perform_object_detection(model, frame, box_annotator)
        #cv2.imshow("yolov8", annotated_frame)
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame = buffer.tobytes()
        return frame

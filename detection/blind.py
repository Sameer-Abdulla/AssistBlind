import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
import pyttsx3
from google.cloud import vision
from google.oauth2 import service_account


def detect_text_with_vision_api(image):
    credentials = service_account.Credentials.from_service_account_file('detection/work.json')
    client = vision.ImageAnnotatorClient(credentials=credentials)
    success, encoded_image = cv2.imencode('.jpg', image)
    if not success:
        raise ValueError("Failed to encode the image")
    content = encoded_image.tobytes()
    image = vision.Image(content=content)
    response = client.text_detection(image=image)
    texts = response.text_annotations
    if texts:
        detected_text = texts[0].description
        bounding_box = texts[0].bounding_poly.vertices
        return detected_text, bounding_box
    else:
        return None, None

engine = pyttsx3.init()
last_detected_object = None

def text_to_speech(text):
    
    engine.say(text)
    engine.runAndWait()
    
def perform_closest_detection(model: YOLO, frame: np.ndarray, box_annotator: sv.BoxAnnotator) -> tuple:
    global last_detected_object
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
        last_detected_object = largest_label  # Update last detected object
        annotated_frame = box_annotator.annotate(scene=frame, detections=[(largest_box, 1.0, 0, None)], labels=labels)
    else:
        annotated_frame = frame
    
    return annotated_frame, detections

def blindFunction(cap):
    global engine
    count =20
    model = YOLO("yolov8l.pt")
    box_annotator = sv.BoxAnnotator(thickness=2, text_thickness=2, text_scale=1)
    while True:
        ret, frame = cap.read()
        detected_text, bounding_box = detect_text_with_vision_api(frame)
        if detected_text:
            print("Text detected:", detected_text)    
            text_to_speech(detected_text)
            count = 0
            # Skip object detection if text is detected
            continue
        # Check if any text is present in the frame
        count +=1
        annotated_frame, detections = perform_closest_detection(model, frame, box_annotator)
        #cv2.imshow("yolov8", annotated_frame)
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame = buffer.tobytes()
        return frame

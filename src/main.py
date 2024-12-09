import cv2
from camera import capture_camera_feed
from face_detection import detect_faces
from preprocess import preprocess_face
from emotion_recognition import EmotionRecognizer

def process_frame(frame):
    recognizer = EmotionRecognizer("models/emotion_model.h5")
    results = detect_faces(frame)
    if results and results.detections:
        for detection in results.detections:
            face = preprocess_face(frame, detection)
            emotion = recognizer.predict_emotion(face)
            annotate_emotion(frame, detection, emotion)
    return frame

def annotate_emotion(frame, detection, emotion):
    bboxC = detection.location_data.relative_bounding_box
    ih, iw, _ = frame.shape
    x, y = int(bboxC.xmin * iw), int(bboxC.ymin * ih)
    cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

if __name__ == "__main__":
    capture_camera_feed(process_frame)

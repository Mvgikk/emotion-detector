import cv2

def preprocess_face(frame, detection):
    bboxC = detection.location_data.relative_bounding_box
    ih, iw, _ = frame.shape
    x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
    cropped_face = frame[y:y+h, x:x+w]
    resized_face = cv2.resize(cropped_face, (48, 48))
    gray_face = cv2.cvtColor(resized_face, cv2.COLOR_BGR2GRAY)
    normalized_face = gray_face / 255.0
    return normalized_face.reshape(1, 48, 48, 1)

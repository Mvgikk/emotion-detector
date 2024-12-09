import cv2

def capture_camera_feed(process_frame):
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        processed_frame = process_frame(frame)
        cv2.imshow("Emotion Detection", processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Wciśnij 'q' aby zakończyć
            break
    cap.release()
    cv2.destroyAllWindows()

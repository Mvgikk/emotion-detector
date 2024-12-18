from deepface import DeepFace
import cv2

def analyze_image(image_path):

    # Odczyt obrazu do wyświetlenia
    frame = cv2.imread(image_path)
    if frame is None:
        print("Nie można otworzyć pliku ze zdjęciem.")
        return
    
    results = DeepFace.analyze(img_path=image_path,
                                actions=['emotion'],
                                enforce_detection=False,
                                detector_backend='mtcnn')
    
    print("Wykryto twarze:", len(results))

    for face_info in results:
        dominant_emotion = face_info['dominant_emotion']
        region = face_info['region']
        x, y, w, h = region['x'], region['y'], region['w'], region['h']

        # Rysujemy prostokąt wokół twarzy
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

        # Wyświetlamy emocję nad kwadratem
        cv2.putText(frame, dominant_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.9, (0,255,0), 2)

    cv2.imshow("Emotion Recognition - Multiple Faces", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def analyze_webcam():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Brak obrazu z kamerki.")
            break

        try:
            #Funkcja analyze przyjmuje obraz i wybrane akcje do wykonania. Tutaj actions=['emotion'] informuje, że chcemy wykryć emocje.
            #enforce_detection=False pozwala kontynuować analizę nawet, gdy twarz nie jest wykrywana na obrazie.
            
            results = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            dominant_emotion = results[0]['dominant_emotion']

            region = results[0]['region']
            x, y, w, h = region['x'], region['y'], region['w'], region['h']

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(frame, dominant_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
        except:
            # Nie wykryto twarzy lub wystąpił błąd
            pass

        cv2.imshow("Emotion Recognition - Webcam", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def analyze_video(video_path): 
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Nie można otworzyć pliku wideo.")
        return

    scaling_factor = 0.5  # Zmniejszamy rozdzielczość

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        try:
            # Zmniejszamy rozdzielczość klatki przed analizą
            frame_small = cv2.resize(frame, (0, 0), fx=scaling_factor, fy=scaling_factor)

            # Analiza na pomniejszonej klatce
            results = DeepFace.analyze(frame_small, actions=['emotion'], enforce_detection=False,)
            
            # Iterujemy po wszystkich twarzach wykrytych w klatce
            for face_info in results:
                dominant_emotion = face_info['dominant_emotion']
                region = face_info['region']
                x, y, w, h = region['x'], region['y'], region['w'], region['h']

                # Skalowanie współrzędnych prostokąta z powrotem do oryginalnego rozmiaru
                x = int(x / scaling_factor)
                y = int(y / scaling_factor)
                w = int(w / scaling_factor)
                h = int(h / scaling_factor)

                cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
                cv2.putText(frame, dominant_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

        except Exception as e:
            # Jeśli nie wykryto twarzy lub wystąpił inny błąd, ignorujemy tę klatkę
            pass

        cv2.imshow("Emotion Recognition - Video", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    cap = cv2.VideoCapture(video_path)


if __name__ == "__main__":
    print("Wybierz źródło obrazu:")
    print("1 - Kamerka")
    print("2 - Zdjęcie")
    print("3 - Wideo")
    choice = input("Podaj numer opcji: ")

    if choice == '1':
        analyze_webcam()
    elif choice == '2':
        image_path = "resources/test.png"
        #image_path = input("Podaj ścieżkę do zdjęcia: ")
        analyze_image(image_path)
    elif choice == '3':
        video_path = "resources/white_chicks.mp4"
        # video_path = input("Podaj ścieżkę do pliku wideo: ")
        analyze_video(video_path)
    else:
        print("Nieznana opcja. Zakończenie programu.")

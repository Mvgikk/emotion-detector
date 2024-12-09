from tensorflow.keras.models import load_model

class EmotionRecognizer:
    def __init__(self, model_path):
        self.model = load_model(model_path)
        self.emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

    def predict_emotion(self, face):
        prediction = self.model.predict(face)
        return self.emotions[prediction.argmax()]

# File: utils.py
import cv2
from tensorflow.keras.models import model_from_json

# Load the emotion recognition model
def load_emotion_model():
    model = model_from_json(open("Facial Expression Recognition.json", "r").read())
    model.load_weights("fer.h5")
    return model

emotion_model = load_emotion_model()
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def process_video(input_path, output_path=None, extract_emotions=False):
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Format for saving processed video

    out = None
    if output_path:
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        out = cv2.VideoWriter(output_path, fourcc, 20.0, (frame_width, frame_height))

    detected_emotions = []
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)

        for (x, y, w, h) in faces:
            roi_gray = gray_frame[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48, 48))
            roi_gray = roi_gray / 255.0
            roi_gray = roi_gray.reshape(1, 48, 48, 1)

            prediction = emotion_model.predict(roi_gray)
            max_index = prediction[0].argmax()
            detected_emotion = emotions[max_index]
            detected_emotions.append(detected_emotion)

            # Annotate frame
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, detected_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        if out:
            out.write(frame)

    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()

    if extract_emotions:
        return detected_emotions

import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Path to model file (check local dir first, then kaggle_model dir)
MODEL_PATH = "emotion_model.h5"
if not os.path.exists(MODEL_PATH) and os.path.exists("kaggle_model/emotion_model.h5"):
    MODEL_PATH = "kaggle_model/emotion_model.h5"

EMOTIONS = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Sad",
    "Surprise",
    "Neutral"
]

def load_emotion_keras_model(model_path=MODEL_PATH):
    """Load the trained Keras H5 model."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file '{model_path}' not found!")
    print(f"[Emotion Model] Loading {model_path}...")
    model = load_model(model_path, compile=False)
    print("[Emotion Model] Successfully loaded.")
    return model

def predict_emotion_from_face(face_gray, model):
    """
    Given a grayscale face crop, preprocess to 48x48 normalized array
    and return predicted emotion label, confidence score, and raw probability distribution.
    """
    face = cv2.resize(face_gray, (48, 48))
    face = face.astype("float32") / 255.0
    face = np.expand_dims(face, axis=-1)
    face = np.expand_dims(face, axis=0)

    prediction = model.predict(face, verbose=0)
    emotion_id = int(np.argmax(prediction[0]))
    emotion = EMOTIONS[emotion_id]
    confidence = float(prediction[0][emotion_id])
    
    return emotion, confidence, prediction[0]

def main():
    # Load model
    model = load_emotion_keras_model(MODEL_PATH)

    # Face detector
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Could not open webcam (VideoCapture 0 failed)")
        return

    print("[Emotion Detection] Running... Press 'q' to quit.")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Failed to grab frame from webcam.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(48, 48)
        )

        for (x, y, w, h) in faces:
            # Crop face
            face = gray[y:y+h, x:x+w]

            # Predict emotion
            emotion, confidence, probs = predict_emotion_from_face(face, model)

            # Draw face bounding box
            cv2.rectangle(
                frame,
                (x, y),
                (x+w, y+h),
                (0, 255, 0),
                2
            )

            # Display emotion and confidence
            text = f"{emotion}: {confidence * 100:.1f}%"

            cv2.putText(
                frame,
                text,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

        cv2.imshow("Emotion Detection", frame)

        # Press Q to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

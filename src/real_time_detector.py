import cv2
import numpy as np
import tensorflow as tf
from face_detection import load_face_detector, detect_faces

def run_real_time_detection(model_path):
    model = tf.keras.models.load_model(model_path)
    face_cascade = load_face_detector()

    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = detect_faces(frame, face_cascade)
        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            face_resized = cv2.resize(face, (224,224)) / 255.0
            face_input = np.expand_dims(face_resized, axis=0)
            pred = model.predict(face_input)[0][0]
            label = "Mask" if pred > 0.5 else "No Mask"
            color = (0,255,0) if label=="Mask" else (0,0,255)
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)

        cv2.imshow('Face Mask Detector', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_real_time_detection("models/face_mask_cnn.h5")

import cv2
import os
import urllib.request
import json
import numpy as np


class FaceRecognizerDNN:
    def __init__(self, trainer_path='trainer/trainer.yml', label_map_path='trainer/label_map.json'):
        # Load the label map
        with open(label_map_path, 'r') as f:
            self.label_map = json.load(f)

        # Reverse map: index → name
        self.rev_label_map = {v: k for k, v in self.label_map.items()}

        # Load LBPH face recognizer
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        if os.path.exists(trainer_path):
            self.recognizer.read(trainer_path)
        else:
            raise FileNotFoundError(f"Trainer file not found at {trainer_path}")

        # Load DNN model
        self.proto_path, self.model_path = self._download_dnn_model()
        self.net = cv2.dnn.readNetFromCaffe(self.proto_path, self.model_path)

        self.confidence_threshold = 0.5

    def _download_dnn_model(self):
        proto_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
        model_url = "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000_fp16.caffemodel"

        proto_path = "models/deploy.prototxt"
        model_path = "models/res10_300x300_ssd_iter_140000_fp16.caffemodel"

        os.makedirs("models", exist_ok=True)

        if not os.path.exists(proto_path):
            print("[INFO] Downloading deploy.prototxt...")
            urllib.request.urlretrieve(proto_url, proto_path)

        if not os.path.exists(model_path):
            print("[INFO] Downloading res10_300x300 model...")
            urllib.request.urlretrieve(model_url, model_path)

        return proto_path, model_path


    def recognize_faces(self):
        cap = cv2.VideoCapture(0)
        print("[INFO] Starting face recognition...")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Failed to grab frame")
                break

            h, w = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                         (104.0, 177.0, 123.0))
            self.net.setInput(blob)
            detections = self.net.forward()

            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence < self.confidence_threshold:
                    continue

                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x1, y1, x2, y2) = box.astype("int")

                face_gray = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
                try:
                    face_resized = cv2.resize(face_gray, (200, 200))
                    id_, conf = self.recognizer.predict(face_resized)
                    name = self.rev_label_map.get(id_, "Unknown")
                    label = f"{name} ({conf:.2f})"
                except:
                    label = "Face Error"

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imshow("Face Recognition", frame)

            if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    recognizer = FaceRecognizerDNN(
        trainer_path="trainer/trainer.yml",
        label_map_path="trainer/label_map.json"
    )
    recognizer.recognize_faces()

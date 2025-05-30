import cv2
import os
import numpy as np
import json

class FaceTrainer:
    def __init__(self, dataset_path='dataset', model_save_path='trainer.yml', label_map_path='label_map.json'):
        self.dataset_path = dataset_path
        self.model_save_path = model_save_path
        self.label_map_path = label_map_path
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.label_map = {}
        self.reverse_label_map = {}

    def _get_images_and_labels(self):
        face_samples = []
        ids = []
        current_label = 0

        for root, dirs, files in os.walk(self.dataset_path):
            for file in files:
                if file.endswith(".jpg") or file.endswith(".png"):
                    path = os.path.join(root, file)
                    # Filename format assumed: User.<name>.<sample_no>.jpg
                    parts = file.split(".")
                    if len(parts) < 3:
                        print(f"Skipping invalid filename: {file}")
                        continue
                    user_name = parts[1]

                    # Map string user_name to a numeric label
                    if user_name not in self.label_map:
                        self.label_map[user_name] = current_label
                        current_label += 1
                    user_id = self.label_map[user_name]

                    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        print(f"Failed to load image: {path}")
                        continue

                    face_samples.append(img)
                    ids.append(user_id)

        return face_samples, ids

    def train(self):
        print("[INFO] Preparing data for training...")
        faces, ids = self._get_images_and_labels()
        print(f"[INFO] Number of faces: {len(faces)}")
        print(f"[INFO] Number of unique labels: {len(set(ids))}")

        if len(faces) == 0:
            print("[ERROR] No valid face images found. Aborting training.")
            return

        self.recognizer.train(faces, np.array(ids))
        self.recognizer.save(self.model_save_path)
        print(f"[INFO] Model saved at {self.model_save_path}")

        # Save label map to JSON
        with open(self.label_map_path, 'w') as f:
            json.dump(self.label_map, f)
        print(f"[INFO] Label map saved at {self.label_map_path}")

if __name__ == "__main__":
    trainer = FaceTrainer()
    trainer.train()

import cv2
import os
import urllib.request

class FaceDatasetDNN:
    def __init__(self, face_id: str, use_droidcam=False, droidcam_url=None, save_path: str='dataset', width: int=640, height: int=480):
        self.face_id = face_id
        self.save_path = save_path
        self.use_droidcam = use_droidcam
        self.droidcam_url = droidcam_url
        os.makedirs(self.save_path, exist_ok=True)

        if use_droidcam:
            self.cam = cv2.VideoCapture(f"http://{self.droidcam_url}")
            print("[INFO] Using DroidCam stream.")
        else:
            self.cam = cv2.VideoCapture(0)
            self.cam.set(3, width)
            self.cam.set(4, height)
            print("[INFO] Using built-in webcam.")

        self._download_dnn_model()
        self.face_net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel")
        self.count = 0

    def _download_dnn_model(self):
        if not os.path.isfile("deploy.prototxt"):
            print("[INFO] Downloading deploy.prototxt...")
            urllib.request.urlretrieve(
                "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
                "deploy.prototxt"
            )
        if not os.path.isfile("res10_300x300_ssd_iter_140000.caffemodel"):
            print("[INFO] Downloading res10_300x300_ssd_iter_140000.caffemodel...")
            urllib.request.urlretrieve(
                "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
                "res10_300x300_ssd_iter_140000.caffemodel"
            )

    def capture_faces(self, samples: int=30):
        print(f"\n[INFO] Initializing DNN face capture for User {self.face_id}. Please look at the camera...")

        while self.count < samples:
            ret, frame = self.cam.read()
            if not ret:
                print("[ERROR] Failed to grab frame.")
                break

            h, w = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                         (104.0, 177.0, 123.0))
            self.face_net.setInput(blob)
            detections = self.face_net.forward()

            face_found = False

            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]

                if confidence > 0.6:
                    face_found = True
                    box = detections[0, 0, i, 3:7] * [w, h, w, h]
                    (x1, y1, x2, y2) = box.astype("int")

                    face = frame[y1:y2, x1:x2]
                    self.count += 1
                    filename = f"{self.save_path}/User.{self.face_id}.{self.count}.jpg"
                    cv2.imwrite(filename, face)
                    print(f"[INFO] Saved image {self.count}: {filename}")
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            if not face_found:
                print("[DEBUG] No faces detected in this frame.")

            cv2.imshow("Face Capture", frame)

            if cv2.waitKey(100) & 0xFF == 27:  # ESC key
                print("[INFO] Capture interrupted.")
                break

        print(f"[INFO] {self.count} face samples captured successfully.")
        self.cam.release()
        cv2.destroyAllWindows()


# --- Usage Example ---
if __name__ == "__main__":
    face_id = input("Enter user ID and press <return>: ")
    use_droidcam = input("Use DroidCam? (y/n): ").strip().lower() == 'y'
    droidcam_url = None

    if use_droidcam:
        droidcam_url = input("Enter DroidCam URL (e.g. 192.168.1.7:4747/video): ").strip()

    dataset = FaceDatasetDNN(face_id=face_id, use_droidcam=use_droidcam, droidcam_url=droidcam_url)
    dataset.capture_faces()



# Face-Detection-System

## FaceDatasetDNN

A Python class for building custom face image datasets using OpenCV’s DNN face detector. Supports both built-in webcams and DroidCam streams. Automatically downloads the required DNN model files if not present.

## Features

- **Face detection** using OpenCV’s pre-trained DNN model (`res10_300x300_ssd_iter_140000.caffemodel`)
- **Supports built-in webcam or DroidCam IP camera**
- **Saves detected face images** to a user-specified dataset directory
- **Automatic model download** if files are missing
- **Configurable image size and sample count**


## How It Works

1. **Initialization**:
    - Specify a user ID, camera source (webcam or DroidCam), and dataset folder.
    - The class checks for the required DNN model files and downloads them if needed.
    - Opens the video stream for capturing frames.
2. **Face Capture**:
    - For each frame, detects faces using the DNN model.
    - Crops and saves each detected face as an image in the dataset directory.
    - Draws a bounding box around detected faces for visualization.
    - Stops after collecting the specified number of samples or when ESC is pressed.

## Installation

```bash
pip install opencv-python
```


## Usage

```python
from facedatasetdnn import FaceDatasetDNN  # or paste the class in your script

if __name__ == "__main__":
    face_id = input("Enter user ID and press <return>: ")
    use_droidcam = input("Use DroidCam? (y/n): ").strip().lower() == 'y'
    droidcam_url = None

    if use_droidcam:
        droidcam_url = input("Enter DroidCam URL (e.g. 192.168.1.7:4747/video): ").strip()

    dataset = FaceDatasetDNN(face_id=face_id, use_droidcam=use_droidcam, droidcam_url=droidcam_url)
    dataset.capture_faces()
```


### Arguments

- `face_id` (str): Unique identifier for the person.
- `use_droidcam` (bool): Use DroidCam stream if True; otherwise, use the default webcam.
- `droidcam_url` (str): IP address and port for DroidCam (e.g., `192.168.1.7:4747/video`).
- `save_path` (str): Directory to save face images (default: `dataset`).
- `width`, `height` (int): Video frame dimensions.


## Example

**Webcam:**

```bash
python your_script.py
# Enter user ID and press <return>: 1
# Use DroidCam? (y/n): n
```

**DroidCam:**

```bash
python your_script.py
# Enter user ID and press <return>: 2
# Use DroidCam? (y/n): y
# Enter DroidCam URL (e.g. 192.168.1.7:4747/video): 192.168.1.7:4747/video
```


## Output

- Cropped face images saved as `User.<face_id>.<count>.jpg` in the specified dataset folder.


## References

- [OpenCV DNN Face Detector](https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector)
- [PyImageSearch: How to build a custom face recognition dataset][^2]

---

**Tip:**
This approach is widely used for collecting diverse face datasets for machine learning, as described in [PyImageSearch tutorials][^2]. The DNN-based detector is robust and suitable for real-world conditions.

[^2]: https://pyimagesearch.com/2018/06/11/how-to-build-a-custom-face-recognition-dataset/

<div style="text-align: center">⁂</div>

[^1]: https://www.datacamp.com/tutorial/face-detection-python-opencv

[^2]: https://pyimagesearch.com/2018/06/11/how-to-build-a-custom-face-recognition-dataset/

[^3]: https://realpython.com/face-recognition-with-python/

[^4]: https://github.com/ritika26/How-to-build-custom-face-recognition-dataset/blob/master/build_face_dataset.py

[^5]: https://www.codingal.com/coding-for-kids/blog/build-face-recognition-app-with-python/

[^6]: https://github.com/topics/face-recognition-python

[^7]: https://ai.google.dev/edge/mediapipe/solutions/vision/face_detector/python

[^8]: https://www.kaggle.com/datasets/fareselmenshawii/face-detection-dataset



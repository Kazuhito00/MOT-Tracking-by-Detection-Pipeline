[[Japanese](https://github.com/Kazuhito00/MOT-Tracking-by-Detection-Pipeline)/English] 

# MOT-Tracking-by-Detection-Pipeline
Tracking-by-Detection method MOT(Multi Object Tracking) is a framework that separates the processing of Detection and Tracking.<br>

---

<img src="https://user-images.githubusercontent.com/37477845/154173587-51b7773e-4e7b-46c0-b8a7-da2bf774edb4.png" loading="lazy" width="100%">

---

https://user-images.githubusercontent.com/37477845/154089051-708b70c7-661a-4754-a5d8-556f9291e4c9.mp4

---

# Requirement
```
opencv-python 4.5.5.62 or later
onnxruntime 1.10.0     or later
mediapipe 0.8.9.1      or later ※When using MediaPipe
filterpy 1.4.5         or later ※When using motpy
lap 0.4.0              or later ※When using ByteTrack
Cython 0.29.27         or later ※When using ByteTrack
cython_bbox 0.1.3      or later ※When using ByteTrack
rich 11.2.0            or later ※When using Norfair
gdown 4.3.0            or later ※When using YoutuReID
tensorflow 2.8.0       or later ※When using Light Person Detector with tflite
```

*If the installation of cython_bbox fails on Windows, please try the installation from GitHub (as of 2022/02/16).<br>

```
pip install -e git+https://github.com/samson-wang/cython_bbox.git#egg=cython-bbox
```

# Usage
Here's how to run the demo.
```bash
python main.py
```
* --device<br>
Specifying the camera device number<br>
Default：0
* --movie<br>
Specify video file *When specified, priority is given to the camera device<br>
Default：unspecified
* --detector<br>
Specifying the Object Detection method to use<br>
Specify one of yolox, efficientdet, ssd, centernet, nanodet, mediapipe_face, mediapipe_hand, light_person_detector<br>
Default：yolox
* --tracker<br>
Specifying the Tracking algorithm to use<br>
Specify one of motpy, bytetrack, mc_bytetrack, norfair, mc_norfair, person_reid, youtureid, sface<br>
Default：motpy
* --target_id<br>
Specify the class ID to be tracked<br>If you specify more than one, specify them separated by commas. *If None, all are targeted.<br>
example：--target_id=1<br>example：--target_id=1,3<br>
Default：None
* --use_gpu<br>
Whether to use GPU<br>
Default：unspecified

# Direcotry
```
│  main.py
│  test.mp4
├─Detector
│  │  detector.py
│  └─xxxxxxxx
│      │  xxxxxxxx.py
│      │  config.json
│      │  LICENSE
│      └─model
│          xxxxxxxx.onnx
└─Tracker
    │  tracker.py
    └─yyyyyyyy
        │  yyyyyyyy.py
        │  config.json
        │  LICENSE
        └─tracker
```
In the directory where each model and tracking algorithm are stored, <br>
Includes license terms and config.

# Detector

| Model name | Source repository | License | Remarks |
| :--- | :--- | :--- | :--- |
| YOLOX | [Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) | Apache-2.0 | Use the model that has been converted to <br> ONNX in [YOLOX-ONNX-TFLite-Sample](https://github.com/Kazuhito00/YOLOX-ONNX-TFLite-Sample) |
| EfficientDet | [tensorflow/models](https://github.com/tensorflow/models/tree/master/research/object_detection) | Apache-2.0 | Use the model that has been converted to <br> ONNX in [Object-Detection-API-TensorFlow2ONNX](https://github.com/Kazuhito00/Object-Detection-API-TensorFlow2ONNX) |
| SSD MobileNet v2 FPNLite | [tensorflow/models](https://github.com/tensorflow/models/tree/master/research/object_detection) | Apache-2.0 |  Use the model that has been converted to <br> ONNX in [Object-Detection-API-TensorFlow2ONNX](https://github.com/Kazuhito00/Object-Detection-API-TensorFlow2ONNX) |
| CenterNet | [tensorflow/models](https://github.com/tensorflow/models/tree/master/research/object_detection) | Apache-2.0 |  Use the model that has been converted to <br> ONNX in [Object-Detection-API-TensorFlow2ONNX](https://github.com/Kazuhito00/Object-Detection-API-TensorFlow2ONNX) |
| NanoDet | [RangiLyu/nanodet](https://github.com/RangiLyu/nanodet) | Apache-2.0 | Use the model that has been converted to <br> ONNX in [NanoDet-ONNX-Sample](https://github.com/Kazuhito00/NanoDet-ONNX-Sample) |
| MediaPipe Face Detection | [google/mediapipe](https://github.com/google/mediapipe) | Apache-2.0 | Unused eye, nose, mouth and ear key points|
| MediaPipe Hands | [google/mediapipe](https://github.com/google/mediapipe) | Apache-2.0 | Calculate and use the circumscribed rectangle from the landmark |
| Light Person Detector | [Person-Detection-using-RaspberryPi-CPU](https://github.com/Kazuhito00/Person-Detection-using-RaspberryPi-CPU) | Apache-2.0 | - |

# Tracker

| Algorithm name | Source repository | License | Remarks |
| :--- | :--- | :--- | :--- |
| motpy<br> (0.0.10) | [wmuron/motpy](https://github.com/wmuron/motpy) | MIT | - |
| ByteTrack<br> (2022/01/26) | [ifzhang/ByteTrack](https://github.com/ifzhang/ByteTrack) | MIT | ByteTrack is a tracker for single class <br> If you want to use the multi-class extended version, please specify "mc_bytetrack". |
| Norfair<br> (0.4.0) | [tryolabs/norfair](https://github.com/tryolabs/norfair) | MIT | Norfair is a single-class tracker <br> If you want to use the multi-class extended version, specify "mc_norfiar" |
| person-reidentification-retail  | [openvinotoolkit/open_model_zoo](https://github.com/openvinotoolkit/open_model_zoo/blob/2020.2/models/intel/person-reidentification-retail-0300/description/person-reidentification-retail-0300.md) | Apache-2.0 | ONNX model is obtained from [PINTO0309/PINTO_model_zoo](https://github.com/PINTO0309/PINTO_model_zoo/tree/main/083_Person_Reidentification) <br> Because it is a human model, please specify the class with the target_id option when using it. |
| YoutuReID | [opencv/opencv_zoo](https://github.com/opencv/opencv_zoo/tree/master/models/person_reid_youtureid) | Apache-2.0 | Since it is a human model, please specify the class with the target_id option when using it. |
| SFace  | [opencv/opencv_zoo](https://github.com/opencv/opencv_zoo/tree/master/models/face_recognition_sface) | Apache-2.0 | Please specify face detection for Detector <br> Also, SFace should perform processing to correct the angle of the face vertically before inference, but this source does not support it. |

# Author
Kazuhito Takahashi(https://twitter.com/KzhtTkhs)
 
# License 
MOT-Tracking-by-Detection-Pipeline is under [MIT License](LICENSE).<br><br>
The source code of MOT-Tracking-by-Detection-Pipeline itself is [MIT License](LICENSE), but <br>
The source code for each algorithm is subject to its own license. <br>
For details, please check the LICENSE file included in each directory.

# License(Movie)
The sample video uses the "[Pedestrian Crossing in Milan, Italy](https://www2.nhk.or.jp/archives/creative/material/view.cgi?m=D0002011299_00000)" from the [NHK Creative Library](https://www.nhk.or.jp/archives/creative/).

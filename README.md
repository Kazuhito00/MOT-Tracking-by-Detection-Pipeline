# MOT-Tracking-by-Detection-Pipeline
Tracking-by-Detection形式のMOT(Multi Object Tracking)について、<br>DetectionとTrackingの処理を分離して寄せ集めたフレームワークです。<br>

---

<img src="https://user-images.githubusercontent.com/37477845/154173587-51b7773e-4e7b-46c0-b8a7-da2bf774edb4.png" loading="lazy" width="100%">

---

https://user-images.githubusercontent.com/37477845/154089051-708b70c7-661a-4754-a5d8-556f9291e4c9.mp4

---

# Requirement
```
opencv-python 4.5.5.62 or later
onnxruntime 1.10.0     or later
mediapipe 0.8.9.1      or later ※MediaPipeを実行する場合
filterpy 1.4.5         or later ※motpyを実行する場合
lap 0.4.0              or later ※ByteTrackを実行する場合
Cython 0.29.27         or later ※ByteTrackを実行する場合
cython_bbox 0.1.3      or later ※ByteTrackを実行する場合
rich 11.2.0            or later ※Norfairを実行する場合
```

※Windowsでcython_bbox のインストールが失敗する場合は、GitHubからのインストールをお試しください(2022/02/16時点)<br>

```
pip install -e git+https://github.com/samson-wang/cython_bbox.git#egg=cython-bbox
```

# Usage
デモの実行方法は以下です。
```bash
python main.py
```
* --device<br>
カメラデバイス番号の指定<br>
デフォルト：0
* --movie<br>
動画ファイルの指定 ※指定時はカメラデバイスより優先<br>
デフォルト：指定なし
* --detector<br>
Object Detectionのモデル選択<br>
yolox, efficientdet, ssd, centernet, nanodet, mediapipe_face, mediapipe_hand の何れかを指定<br>
デフォルト：yolox
* --tracker<br>
トラッキングアルゴリズムの選択<br>
motpy, bytetrack, norfair, person_reid の何れかを指定<br>
デフォルト：bytetrack
* --target_id<br>
トラッキング対象のクラスIDを指定<br>複数指定する場合はカンマ区切りで指定　※Noneの場合は全てを対象とする<br>
例：--target_id=1<br>例：--target_id=1,3<br>
デフォルト：None
* --use_gpu<br>
GPU推論するか否か<br>
デフォルト：指定なし

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
各モデル、トラッキングアルゴリズムを格納しているディレクトリには、<br>
ライセンス条項とコンフィグを同梱しています。

# Detector

| モデル名 | 取得元リポジトリ | ライセンス | 備考 |
| :--- | :--- | :--- | :--- |
| YOLOX | [Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) | Apache-2.0 | [YOLOX-ONNX-TFLite-Sample](https://github.com/Kazuhito00/YOLOX-ONNX-TFLite-Sample)にて<br>ONNX化したモデルを使用 |
| EfficientDet | [tensorflow/models](https://github.com/tensorflow/models/tree/master/research/object_detection) | Apache-2.0 | [Object-Detection-API-TensorFlow2ONNX](https://github.com/Kazuhito00/Object-Detection-API-TensorFlow2ONNX)にて<br>ONNX化したモデルを使用 |
| SSD MobileNet v2 FPNLite | [tensorflow/models](https://github.com/tensorflow/models/tree/master/research/object_detection) | Apache-2.0 |  [Object-Detection-API-TensorFlow2ONNX](https://github.com/Kazuhito00/Object-Detection-API-TensorFlow2ONNX)にて<br>ONNX化したモデルを使用 |
| CenterNet | [tensorflow/models](https://github.com/tensorflow/models/tree/master/research/object_detection) | Apache-2.0 |  [Object-Detection-API-TensorFlow2ONNX](https://github.com/Kazuhito00/Object-Detection-API-TensorFlow2ONNX)にて<br>ONNX化したモデルを使用 |
| NanoDet | [RangiLyu/nanodet](https://github.com/RangiLyu/nanodet) | Apache-2.0 | [NanoDet-ONNX-Sample](https://github.com/Kazuhito00/NanoDet-ONNX-Sample)にて<br>ONNX化したモデルを使用 |
| MediaPipe Face Detection | [google/mediapipe](https://github.com/google/mediapipe) | Apache-2.0 | 目、鼻、口、耳のキーポイントは未使用|
| MediaPipe Hands | [google/mediapipe](https://github.com/google/mediapipe) | Apache-2.0 | ランドマークから外接矩形を算出し使用 |

# Tracker

| アルゴリズム名 | 取得元リポジトリ | ライセンス | 備考 |
| :--- | :--- | :--- | :--- |
| motpy | [wmuron/motpy](https://github.com/wmuron/motpy) | MIT | マルチクラス対応 |
| ByteTrack | [ifzhang/ByteTrack](https://github.com/ifzhang/ByteTrack) | MIT | - |
| Norfair  | [tryolabs/norfair](https://github.com/tryolabs/norfair) | MIT | - |
| person-reidentification-retail  | [openvinotoolkit/open_model_zoo](https://github.com/openvinotoolkit/open_model_zoo/blob/2020.2/models/intel/person-reidentification-retail-0300/description/person-reidentification-retail-0300.md) | Apache-2.0 | ONNXモデルは[PINTO0309/PINTO_model_zoo](https://github.com/PINTO0309/PINTO_model_zoo/tree/main/083_Person_Reidentification)から取得 |

# Author
高橋かずひと(https://twitter.com/KzhtTkhs)
 
# License 
MOT-Tracking-by-Detection-Pipeline is under [MIT License](LICENSE).<br><br>
MOT-Tracking-by-Detection-Pipelineのソースコード自体は[MIT License](LICENSE)ですが、<br>
各アルゴリズムのソースコードは、それぞれのライセンスに従います。<br>
詳細は各ディレクトリ同梱のLICENSEファイルをご確認ください。

# License(Movie)
サンプル動画は[NHKクリエイティブ・ライブラリー](https://www.nhk.or.jp/archives/creative/)の[イタリア ミラノの横断歩道](https://www2.nhk.or.jp/archives/creative/material/view.cgi?m=D0002011299_00000)を使用しています。

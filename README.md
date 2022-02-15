# MOT-Tracking-by-Detection-Pipeline
Tracking-by-Detection形式のMOT(Multi Object Tracking)について、<br>DetectionとTrackingの処理を分離したフレームワークです。<br>

---

<img src="https://user-images.githubusercontent.com/37477845/154084140-aa156ba7-7461-4701-8346-a1411eb08e63.png" loading="lazy" width="100%">

---

https://user-images.githubusercontent.com/37477845/154089051-708b70c7-661a-4754-a5d8-556f9291e4c9.mp4

---

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
[yolox, efficientdet, ssd, centernet, nanodet, mediapipe_face, mediapipe_hand」の何れかを指定<br>
デフォルト：yolox
* --tracker<br>
トラッキングアルゴリズムの選択<br>
[motpy, bytetrack, norfair」の何れかを指定<br>
デフォルト：bytetrack

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
| SSD | [tensorflow/models](https://github.com/tensorflow/models/tree/master/research/object_detection) | Apache-2.0 |  [Object-Detection-API-TensorFlow2ONNX](https://github.com/Kazuhito00/Object-Detection-API-TensorFlow2ONNX)にて<br>ONNX化したモデルを使用 |
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

# Author
高橋かずひと(https://twitter.com/KzhtTkhs)
 
# License 
MOT-Tracking-by-Detection-Pipeline is under [MIT License](LICENSE).<br><br>
※MOT-Tracking-by-Detection-Pipelineのソースコード自体は[MIT License](LICENSE)での提供ですが、<br>
各アルゴリズムのソースコードは、それぞれのライセンスに従います。

# License(Movie)
サンプル動画は[NHKクリエイティブ・ライブラリー](https://www.nhk.or.jp/archives/creative/)の[イタリア ミラノの横断歩道](https://www2.nhk.or.jp/archives/creative/material/view.cgi?m=D0002011299_00000)を使用しています。

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

# Reference
* XXXX

# Author
高橋かずひと(https://twitter.com/KzhtTkhs)
 
# License 
MOT-Tracking-by-Detection-Pipeline is under [MIT License](LICENSE).<br><br>
※MOT-Tracking-by-Detection-Pipelineのソースコード自体は[MIT License](LICENSE)での提供ですが、<br>
各アルゴリズムのソースコードは、それぞれのライセンスに従います。

# License(Movie)
サンプル動画は[NHKクリエイティブ・ライブラリー](https://www.nhk.or.jp/archives/creative/)の[イタリア ミラノの横断歩道](https://www2.nhk.or.jp/archives/creative/material/view.cgi?m=D0002011299_00000)を使用しています。



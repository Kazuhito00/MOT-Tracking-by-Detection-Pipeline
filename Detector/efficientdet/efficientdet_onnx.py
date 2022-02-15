#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy

import cv2
import numpy as np
import onnxruntime


class EfficientDetONNX(object):
    def __init__(
        self,
        model_path='efficientdet_d0.onnx',
        input_shape=(512, 512),
        class_score_th=0.3,
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
    ):
        # 入力サイズ
        self.input_shape = input_shape

        # 閾値
        self.class_score_th = class_score_th

        # モデル読み込み
        self.onnx_session = onnxruntime.InferenceSession(
            model_path,
            providers=providers,
        )

        self.input_name = self.onnx_session.get_inputs()[0].name
        self.output_name = self.onnx_session.get_outputs()[0].name

    def __call__(self, image):
        image_height, image_width = image.shape[0], image.shape[1]

        # 前処理
        input_image = cv2.resize(
            image,
            dsize=(self.input_shape[1], self.input_shape[0]),
        )
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        input_image = np.expand_dims(input_image, axis=0).astype('uint8')

        # 推論実施
        result = self.onnx_session.run(
            None,
            {self.input_name: input_image},
        )

        # 後処理
        # num_detections = int(result[5][0])
        class_ids = result[2][0]
        bboxes = result[1][0]
        scores = result[4][0]

        # スコア閾値による対象抽出
        index_array = np.where(scores > self.class_score_th)
        class_ids = class_ids[index_array]
        bboxes = bboxes[index_array]
        scores = scores[index_array]

        # バウンディングボックスを相対座標から絶対座標に変換
        for index in range(len(bboxes)):
            y1 = int(bboxes[index][0] * image_height)
            x1 = int(bboxes[index][1] * image_width)
            y2 = int(bboxes[index][2] * image_height)
            x2 = int(bboxes[index][3] * image_width)

            bboxes[index][0] = x1
            bboxes[index][1] = y1
            bboxes[index][2] = x2
            bboxes[index][3] = y2

        return bboxes, scores, class_ids

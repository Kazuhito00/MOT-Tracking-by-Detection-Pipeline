#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy

import cv2
import numpy as np
import onnxruntime


class NanoDetONNX(object):
    # NanoDet後処理用定義
    STRIDES = (8, 16, 32)
    REG_MAX = 7
    PROJECT = np.arange(REG_MAX + 1)

    # 標準化用定義
    MEAN = np.array([103.53, 116.28, 123.675], dtype=np.float32)
    MEAN = MEAN.reshape(1, 1, 3)
    STD = np.array([57.375, 57.12, 58.395], dtype=np.float32)
    STD = STD.reshape(1, 1, 3)

    def __init__(
        self,
        model_path='nanodet_m.onnx',
        input_shape=(320, 320),
        class_score_th=0.35,
        nms_th=0.6,
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
    ):
        # 入力サイズ
        self.input_shape = (input_shape[1], input_shape[0])

        # 閾値
        self.class_score_th = class_score_th
        self.nms_th = nms_th

        # モデル読み込み
        self.onnx_session = onnxruntime.InferenceSession(
            model_path,
            providers=providers,
        )

        self.input_name = self.onnx_session.get_inputs()[0].name
        output_detail = self.onnx_session.get_outputs()
        self.output_names = []
        self.output_names.append(output_detail[0].name)  # cls_pred_stride_8
        self.output_names.append(output_detail[3].name)  # dis_pred_stride_8
        self.output_names.append(output_detail[1].name)  # cls_pred_stride_16
        self.output_names.append(output_detail[4].name)  # dis_pred_stride_16
        self.output_names.append(output_detail[2].name)  # cls_pred_stride_32
        self.output_names.append(output_detail[5].name)  # dis_pred_stride_32

        # ストライド毎のグリッド点を算出
        self.grid_points = []
        for index in range(len(self.STRIDES)):
            grid_point = self._make_grid_point(
                (int(self.input_shape[0] / self.STRIDES[index]),
                 int(self.input_shape[1] / self.STRIDES[index])),
                self.STRIDES[index],
            )
            self.grid_points.append(grid_point)

    def __call__(self, image):
        temp_image = copy.deepcopy(image)
        image_height, image_width = image.shape[0], image.shape[1]

        # 前処理：標準化、リシェイプ
        resize_image, new_height, new_width, top, left = self._resize_image(
            temp_image)
        x = self._pre_process(resize_image)

        # 推論実行
        results = self.onnx_session.run(self.output_names,
                                        {self.input_name: x})

        # 後処理：NMS、グリッド->座標変換
        bboxes, scores, class_ids = self._post_process(results)

        # 後処理：リサイズ前の座標に変換
        ratio_height = image_height / new_height
        ratio_width = image_width / new_width
        for i in range(bboxes.shape[0]):
            bboxes[i, 0] = max(int((bboxes[i, 0] - left) * ratio_width), 0)
            bboxes[i, 1] = max(int((bboxes[i, 1] - top) * ratio_height), 0)
            bboxes[i, 2] = min(
                int((bboxes[i, 2] - left) * ratio_width),
                image_width,
            )
            bboxes[i, 3] = min(
                int((bboxes[i, 3] - top) * ratio_height),
                image_height,
            )

        class_ids = class_ids + 1  # 1始まりのクラスIDに変更

        return bboxes, scores, class_ids

    def _make_grid_point(self, grid_size, stride):
        grid_height, grid_width = grid_size

        shift_x = np.arange(0, grid_width) * stride
        shift_y = np.arange(0, grid_height) * stride

        xv, yv = np.meshgrid(shift_x, shift_y)
        xv = xv.flatten()
        yv = yv.flatten()

        cx = xv + 0.5 * (stride - 1)
        cy = yv + 0.5 * (stride - 1)

        return np.stack((cx, cy), axis=-1)

    def _resize_image(self, image, keep_ratio=True):
        top, left = 0, 0
        new_height, new_width = self.input_shape[0], self.input_shape[1]

        if keep_ratio and image.shape[0] != image.shape[1]:
            hw_scale = image.shape[0] / image.shape[1]
            if hw_scale > 1:
                new_height = self.input_shape[0]
                new_width = int(self.input_shape[1] / hw_scale)

                resize_image = cv2.resize(
                    image,
                    (new_width, new_height),
                    interpolation=cv2.INTER_AREA,
                )

                left = int((self.input_shape[1] - new_width) * 0.5)

                resize_image = cv2.copyMakeBorder(
                    resize_image,
                    0,
                    0,
                    left,
                    self.input_shape[1] - new_width - left,
                    cv2.BORDER_CONSTANT,
                    value=0,
                )
            else:
                new_height = int(self.input_shape[0] * hw_scale)
                new_width = self.input_shape[1]

                resize_image = cv2.resize(
                    image,
                    (new_width, new_height),
                    interpolation=cv2.INTER_AREA,
                )

                top = int((self.input_shape[0] - new_height) * 0.5)

                resize_image = cv2.copyMakeBorder(
                    resize_image,
                    top,
                    self.input_shape[0] - new_height - top,
                    0,
                    0,
                    cv2.BORDER_CONSTANT,
                    value=0,
                )
        else:
            resize_image = cv2.resize(
                image,
                self.input_shape,
                interpolation=cv2.INTER_AREA,
            )

        return resize_image, new_height, new_width, top, left

    def _pre_process(self, image):
        # 標準化
        image = image.astype(np.float32)
        image = (image - self.MEAN) / self.STD

        # リシェイプ
        image = image.transpose(2, 0, 1).astype('float32')
        image = image.reshape(-1, 3, self.input_shape[0], self.input_shape[1])

        return image

    def _softmax(self, x, axis=1):
        x_exp = np.exp(x)
        x_sum = np.sum(x_exp, axis=axis, keepdims=True)
        s = x_exp / x_sum
        return s

    def _post_process(self, predict_results):
        class_scores = predict_results[::2]
        bbox_predicts = predict_results[1::2]

        bboxes, scores, class_ids = self._get_bboxes_single(
            class_scores,
            bbox_predicts,
            1,
            rescale=False,
        )

        return bboxes.astype(np.int32), scores, class_ids

    def _get_bboxes_single(
        self,
        class_scores,
        bbox_predicts,
        scale_factor,
        rescale=False,
        topk=1000,
    ):
        bboxes = []
        scores = []

        # ストライド毎にバウンディングボックスの座標を変換
        for stride, class_score, bbox_predict, grid_point in zip(
                self.STRIDES, class_scores, bbox_predicts, self.grid_points):
            # 次元調整
            if class_score.ndim == 3:
                class_score = class_score.squeeze(axis=0)
            if bbox_predict.ndim == 3:
                bbox_predict = bbox_predict.squeeze(axis=0)

            # バウンディングボックスを相対座標と相対距離に変換
            bbox_predict = bbox_predict.reshape(-1, self.REG_MAX + 1)
            bbox_predict = self._softmax(bbox_predict, axis=1)
            bbox_predict = np.dot(bbox_predict, self.PROJECT).reshape(-1, 4)
            bbox_predict *= stride

            # スコア降順で対象を絞る
            if 0 < topk < class_score.shape[0]:
                max_scores = class_score.max(axis=1)
                topk_indexes = max_scores.argsort()[::-1][0:topk]

                grid_point = grid_point[topk_indexes, :]
                bbox_predict = bbox_predict[topk_indexes, :]
                class_score = class_score[topk_indexes, :]

            # バウンディングボックスを絶対座標に変換
            x1 = grid_point[:, 0] - bbox_predict[:, 0]
            y1 = grid_point[:, 1] - bbox_predict[:, 1]
            x2 = grid_point[:, 0] + bbox_predict[:, 2]
            y2 = grid_point[:, 1] + bbox_predict[:, 3]
            x1 = np.clip(x1, 0, self.input_shape[1])
            y1 = np.clip(y1, 0, self.input_shape[0])
            x2 = np.clip(x2, 0, self.input_shape[1])
            y2 = np.clip(y2, 0, self.input_shape[0])
            bbox = np.stack([x1, y1, x2, y2], axis=-1)

            bboxes.append(bbox)
            scores.append(class_score)

        # スケール調整
        bboxes = np.concatenate(bboxes, axis=0)
        if rescale:
            bboxes /= scale_factor
        scores = np.concatenate(scores, axis=0)

        # Non-Maximum Suppression
        bboxes_wh = bboxes.copy()
        bboxes_wh[:, 2:4] = bboxes_wh[:, 2:4] - bboxes_wh[:, 0:2]
        class_ids = np.argmax(scores, axis=1)
        scores = np.max(scores, axis=1)

        indexes = cv2.dnn.NMSBoxes(
            bboxes_wh.tolist(),
            scores.tolist(),
            self.class_score_th,
            self.nms_th,
        )

        # NMS処理後の件数確認
        if len(indexes) > 0:
            if indexes.ndim == 2:
                bboxes = bboxes[indexes[:, 0]]
                scores = scores[indexes[:, 0]]
                class_ids = class_ids[indexes[:, 0]]
            elif indexes.ndim == 1:
                bboxes = bboxes[indexes]
                scores = scores[indexes]
                class_ids = class_ids[indexes]
        else:
            bboxes = np.array([])
            scores = np.array([])
            class_ids = np.array([])

        return bboxes, scores, class_ids

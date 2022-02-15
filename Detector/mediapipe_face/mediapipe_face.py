#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2
import mediapipe as mp


class MediaPipeFaceDetection(object):
    def __init__(
        self,
        model_selection=0,
        min_detection_confidence=0.5,
    ):
        # 入力サイズ
        mp_face_detection = mp.solutions.face_detection
        self.face_detection = mp_face_detection.FaceDetection(
            model_selection=model_selection,
            min_detection_confidence=min_detection_confidence,
        )

    def __call__(self, image):
        image_height, image_width = image.shape[0], image.shape[1]

        # 前処理
        input_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 推論実施
        results = self.face_detection.process(input_image)

        # 後処理
        bboxes = []
        class_ids = []
        scores = []
        if results.detections is not None:
            for detection in results.detections:
                score = detection.score[0]
                bbox = detection.location_data.relative_bounding_box
                x1 = int(bbox.xmin * image_width)
                y1 = int(bbox.ymin * image_height)
                x2 = x1 + int(bbox.width * image_width)
                y2 = y1 + int(bbox.height * image_height)

                scores.append(score)
                bboxes.append([x1, y1, x2, y2])
                class_ids.append(0)

        return bboxes, scores, class_ids

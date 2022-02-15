#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2
import numpy as np
import mediapipe as mp


class MediaPipeHands(object):
    def __init__(
        self,
        static_image_mode=False,
        max_num_hands=5,
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ):
        # 入力サイズ
        mp_hands = mp.solutions.hands
        self.hands = mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def __call__(self, image):
        image_height, image_width = image.shape[0], image.shape[1]

        # 前処理
        input_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 推論実施
        results = self.hands.process(input_image)

        # 後処理
        bboxes = []
        class_ids = []
        scores = []
        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                  results.multi_handedness):
                bbox = self._calc_bounding_box(image, hand_landmarks)

                bboxes.append(bbox)
                scores.append(handedness.classification[0].score)
                if handedness.classification[0].label == "Left":
                    class_ids.append(0)
                elif handedness.classification[0].label == "Right":
                    class_ids.append(1)

        return bboxes, scores, class_ids

    def _calc_bounding_box(self, image, landmarks):
        image_width, image_height = image.shape[1], image.shape[0]

        landmark_array = np.empty((0, 2), int)

        for _, landmark in enumerate(landmarks.landmark):
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)

            landmark_point = [np.array((landmark_x, landmark_y))]

            landmark_array = np.append(landmark_array, landmark_point, axis=0)

        x, y, w, h = cv2.boundingRect(landmark_array)

        return [x, y, x + w, y + h]

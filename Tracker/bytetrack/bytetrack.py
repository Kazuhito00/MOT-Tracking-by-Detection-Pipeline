#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy

import numpy as np

from Tracker.bytetrack.tracker.byte_tracker import BYTETracker


class dict_dot_notation(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


class ByteTrack(object):
    def __init__(
        self,
        fps,
        track_thresh=0.5,
        track_buffer=30,
        match_thresh=0.8,
        min_box_area=10,
        mot20=False,
    ):
        self.min_box_area = min_box_area

        # ByteTrackerインスタンス生成
        self.tracker = BYTETracker(
            args=dict_dot_notation({
                'track_thresh': track_thresh,
                'track_buffer': track_buffer,
                'match_thresh': match_thresh,
                'mot20': mot20,
            }),
            frame_rate=fps,
        )

    def __call__(
        self,
        image,
        bboxes,
        scores,
        class_ids,
    ):
        detections = [[*b, s, l] for b, s, l in zip(bboxes, scores, class_ids)]
        detections = np.array(detections)

        # トラッカー更新
        bboxes, scores, ids = self._tracker_update(
            image,
            detections,
        )

        return ids, bboxes, scores, np.zeros(len(ids))

    def _tracker_update(self, image, detections):
        image_info = {'id': 0}
        image_info['image'] = copy.deepcopy(image)
        image_info['width'] = image.shape[1]
        image_info['height'] = image.shape[0]

        # トラッカー更新
        online_targets = []
        if detections is not None and len(detections) != 0:
            online_targets = self.tracker.update(
                detections[:, :-1],
                [image_info['height'], image_info['width']],
                [image_info['height'], image_info['width']],
            )

        online_tlwhs = []
        online_ids = []
        online_scores = []
        for online_target in online_targets:
            tlwh = online_target.tlwh
            track_id = online_target.track_id
            vertical = tlwh[2] / tlwh[3] > 1.6
            if tlwh[2] * tlwh[3] > self.min_box_area and not vertical:
                online_tlwhs.append(
                    np.array([
                        tlwh[0], tlwh[1], tlwh[0] + tlwh[2], tlwh[1] + tlwh[3]
                    ]))
                online_ids.append(track_id)
                online_scores.append(online_target.score)

        return online_tlwhs, online_scores, online_ids

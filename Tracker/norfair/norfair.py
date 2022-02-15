# -*- coding: utf-8 -*-
import numpy as np

from Tracker.norfair.tracker import Detection
from Tracker.norfair.tracker import Tracker as NorfairTracker


def euclidean_distance(detection, tracked_object):
    return np.linalg.norm(detection.points - tracked_object.estimate)


class Norfair(object):
    def __init__(
        self,
        fps=30,
        max_distance_between_points=30,
    ):
        self.tracker = NorfairTracker(
            distance_function=euclidean_distance,
            distance_threshold=max_distance_between_points,
        )

    def __call__(self, _, bboxes, scores, class_ids):
        detections = []
        for bbox, score in zip(bboxes, scores):
            bbox = np.array([[bbox[0], bbox[1]], [bbox[2], bbox[3]]])
            scores = np.array([score, score])
            detections.append(Detection(points=bbox, scores=scores))

        results = self.tracker.update(detections=detections)

        tracker_ids = []
        tracker_bboxes = []
        tracker_class_ids = []
        tracker_scores = []

        for result in results:
            x1 = result.estimate[0][0]
            y1 = result.estimate[0][1]
            x2 = result.estimate[1][0]
            y2 = result.estimate[1][1]

            tracker_ids.append(result.id)
            tracker_bboxes.append([x1, y1, x2, y2])
            tracker_class_ids.append(0)
            tracker_scores.append(0)

            # print(result.id, result.estimate, result.live_points)

        return tracker_ids, tracker_bboxes, tracker_scores, tracker_class_ids

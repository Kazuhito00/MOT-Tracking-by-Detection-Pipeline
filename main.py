import copy
import time
import argparse

import cv2

from Detector.detector import ObjectDetector
from Tracker.tracker import MultiObjectTracker


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--movie", type=str, default=None)

    parser.add_argument(
        '--detector',
        choices=[
            'yolox',
            'efficientdet',
            'ssd',
            'centernet',
            'nanodet',
            'mediapipe_face',
            'mediapipe_hand',
            'light_person_detector',
        ],
        default='yolox',
    )
    parser.add_argument(
        '--tracker',
        choices=[
            'motpy',
            'bytetrack',
            'mc_bytetrack',
            'norfair',
            'mc_norfair',
            'person_reid',
            'youtureid',
            'sface',
        ],
        default='motpy',
    )

    parser.add_argument("--target_id", type=str, default=None)

    parser.add_argument('--use_gpu', action='store_true')

    args = parser.parse_args()

    return args


def main():
    args = get_args()

    cap_device = args.device
    if args.movie is not None:
        cap_device = args.movie

    detector_name = args.detector
    tracker_name = args.tracker

    target_id = args.target_id
    if target_id is not None:
        target_id = [int(i) for i in target_id.split(',')]

    use_gpu = args.use_gpu

    # VideoCapture初期化
    cap = cv2.VideoCapture(cap_device)
    cap_fps = cap.get(cv2.CAP_PROP_FPS)

    # Object Detection
    detector = ObjectDetector(
        detector_name,
        target_id,
        use_gpu=use_gpu,
    )
    detector.print_info()

    # Multi Object Tracking
    tracker = MultiObjectTracker(
        tracker_name,
        cap_fps,
        use_gpu=use_gpu,
    )
    tracker.print_info()

    # トラッキングID保持用変数
    track_id_dict = {}

    while True:
        start_time = time.time()

        # フレーム読み込み
        ret, frame = cap.read()
        if not ret:
            break
        debug_image = copy.deepcopy(frame)

        # Object Detection
        d_bboxes, d_scores, d_class_ids = detector(frame)

        # Multi Object Tracking
        track_ids, t_bboxes, t_scores, t_class_ids = tracker(
            frame,
            d_bboxes,
            d_scores,
            d_class_ids,
        )

        # トラッキングIDと連番の紐付け
        for track_id in track_ids:
            if track_id not in track_id_dict:
                new_id = len(track_id_dict)
                track_id_dict[track_id] = new_id

        elapsed_time = time.time() - start_time

        # 描画
        debug_image = draw_debug_info(
            debug_image,
            elapsed_time,
            track_ids,
            t_bboxes,
            t_scores,
            t_class_ids,
            track_id_dict,
        )

        key = cv2.waitKey(1)
        if key == 27:  # ESC
            break
        cv2.imshow('MOT Tracking by Detection Pipeline Sample', debug_image)

    cap.release()
    cv2.destroyAllWindows()


def get_id_color(index):
    temp_index = abs(int(index + 1)) * 3
    color = ((37 * temp_index) % 255, (17 * temp_index) % 255,
             (29 * temp_index) % 255)
    return color


def draw_debug_info(
    debug_image,
    elapsed_time,
    track_ids,
    bboxes,
    scores,
    class_ids,
    track_id_dict,
):
    for id, bbox, score, class_id in zip(track_ids, bboxes, scores, class_ids):
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

        color = get_id_color(track_id_dict[id])

        # バウンディングボックス
        debug_image = cv2.rectangle(
            debug_image,
            (x1, y1),
            (x2, y2),
            color,
            thickness=2,
        )

        # トラックID、スコア
        score = '%.2f' % score
        text = 'TID:%s(%s)' % (str(int(track_id_dict[id])), str(score))
        debug_image = cv2.putText(
            debug_image,
            text,
            (x1, y1 - 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            thickness=2,
        )

        # クラスID
        text = 'CID:%s' % (str(int(class_id)))
        debug_image = cv2.putText(
            debug_image,
            text,
            (x1, y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            thickness=2,
        )

    # 経過時間(キャプチャ、物体検出、トラッキング)
    cv2.putText(
        debug_image,
        "Elapsed Time : " + '{:.1f}'.format(elapsed_time * 1000) + "ms",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        1,
        cv2.LINE_AA,
    )

    return debug_image


if __name__ == '__main__':
    main()

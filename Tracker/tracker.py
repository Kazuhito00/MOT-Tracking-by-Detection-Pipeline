import json


class MultiObjectTracker(object):
    def __init__(
        self,
        tracker_name='motpy',
        fps=30,
    ):
        self.fps = round(fps, 2)
        self.tracker_name = tracker_name
        self.tracker = None
        self.config = None

        if self.tracker_name == 'motpy':
            from Tracker.motpy.motpy import Motpy

            with open('Tracker/motpy/config.json') as fp:
                self.config = json.load(fp)

            if self.config is not None:
                self.tracker = Motpy(
                    fps=self.fps,
                    min_steps_alive=self.config['min_steps_alive'],
                    max_staleness=self.config['max_staleness'],
                    order_pos=self.config['order_pos'],
                    dim_pos=self.config['dim_pos'],
                    order_size=self.config['order_size'],
                    dim_size=self.config['dim_size'],
                    q_var_pos=self.config['q_var_pos'],
                    r_var_pos=self.config['r_var_pos'],
                    min_iou=self.config['min_iou'],
                    multi_match_min_iou=self.config['multi_match_min_iou'],
                )

        elif self.tracker_name == 'bytetrack':
            from Tracker.bytetrack.bytetrack import ByteTrack

            with open('Tracker/bytetrack/config.json') as fp:
                self.config = json.load(fp)

            if self.config is not None:
                self.tracker = ByteTrack(
                    fps=self.fps,
                    track_thresh=self.config['track_thresh'],
                    track_buffer=self.config['track_buffer'],
                    match_thresh=self.config['match_thresh'],
                    min_box_area=self.config['min_box_area'],
                    mot20=self.config['mot20'],
                )

        elif self.tracker_name == 'norfair':
            from Tracker.norfair.norfair import Norfair

            with open('Tracker/norfair/config.json') as fp:
                self.config = json.load(fp)

            if self.config is not None:
                self.tracker = Norfair(
                    fps=self.fps,
                    max_distance_between_points=self.
                    config['max_distance_between_points'],
                )

        else:
            raise ValueError('Invalid Tracker Name')

    def __call__(self, image, bboxes, scores, class_ids):
        if self.tracker is not None:
            results = self.tracker(
                image,
                bboxes,
                scores,
                class_ids,
            )
        else:
            raise ValueError('Tracker is None')

        # 0:Tracker ID, 1:Bounding Box, 2:Score, 3:Class ID
        return results[0], results[1], results[2], results[3]

    def print_info(self):
        from pprint import pprint

        print('Tracker:', self.tracker_name)
        print('FPS:', self.fps)
        pprint(self.config, indent=4)
        print()

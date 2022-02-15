import copy
import json


class ObjectDetector(object):
    def __init__(
        self,
        model_name='yolox',
        providers=['CPUExecutionProvider'],
    ):
        self.model_name = model_name
        self.model = None
        self.config = None

        if self.model_name == 'yolox':
            from Detector.yolox.yolox_onnx import YoloxONNX

            with open('Detector/yolox/config.json') as fp:
                self.config = json.load(fp)

            if self.config is not None:
                self.model = YoloxONNX(
                    model_path=self.config['model_path'],
                    input_shape=[
                        int(i) for i in self.config['input_shape'].split(',')
                    ],
                    class_score_th=self.config['class_score_th'],
                    nms_th=self.config['nms_th'],
                    nms_score_th=self.config['nms_score_th'],
                    with_p6=self.config['with_p6'],
                    providers=providers,
                )

        elif self.model_name == 'efficientdet':
            from Detector.efficientdet.efficientdet_onnx import EfficientDetONNX

            with open('Detector/efficientdet/config.json') as fp:
                self.config = json.load(fp)

            if self.config is not None:
                self.model = EfficientDetONNX(
                    model_path=self.config['model_path'],
                    input_shape=[
                        int(i) for i in self.config['input_shape'].split(',')
                    ],
                    class_score_th=self.config['class_score_th'],
                    providers=providers,
                )

        elif self.model_name == 'ssd':
            from Detector.ssd.ssd_onnx import SsdONNX

            with open('Detector/ssd/config.json') as fp:
                self.config = json.load(fp)

            if self.config is not None:
                self.model = SsdONNX(
                    model_path=self.config['model_path'],
                    input_shape=[
                        int(i) for i in self.config['input_shape'].split(',')
                    ],
                    class_score_th=self.config['class_score_th'],
                    providers=providers,
                )

        elif self.model_name == 'centernet':
            from Detector.centernet.centernet_onnx import CenterNetONNX

            with open('Detector/centernet/config.json') as fp:
                self.config = json.load(fp)

            if self.config is not None:
                self.model = CenterNetONNX(
                    model_path=self.config['model_path'],
                    input_shape=[
                        int(i) for i in self.config['input_shape'].split(',')
                    ],
                    class_score_th=self.config['class_score_th'],
                    providers=providers,
                )

        elif self.model_name == 'nanodet':
            from Detector.nanodet.nanodet_onnx import NanoDetONNX

            with open('Detector/nanodet/config.json') as fp:
                self.config = json.load(fp)

            if self.config is not None:
                self.model = NanoDetONNX(
                    model_path=self.config['model_path'],
                    input_shape=[
                        int(i) for i in self.config['input_shape'].split(',')
                    ],
                    class_score_th=self.config['class_score_th'],
                    nms_th=self.config['nms_th'],
                    providers=providers,
                )

        elif self.model_name == 'mediapipe_face':
            from Detector.mediapipe_face.mediapipe_face import MediaPipeFaceDetection

            with open('Detector/mediapipe_face/config.json') as fp:
                self.config = json.load(fp)

            if self.config is not None:
                self.model = MediaPipeFaceDetection(
                    model_selection=self.config['model_selection'],
                    min_detection_confidence=self.
                    config['min_detection_confidence'],
                )

        elif self.model_name == 'mediapipe_hand':
            from Detector.mediapipe_hand.mediapipe_hand import MediaPipeHands

            with open('Detector/mediapipe_hand/config.json') as fp:
                self.config = json.load(fp)

            if self.config is not None:
                self.model = MediaPipeHands(
                    static_image_mode=self.config['static_image_mode'],
                    max_num_hands=self.config['max_num_hands'],
                    model_complexity=self.config['model_complexity'],
                    min_detection_confidence=self.
                    config['min_detection_confidence'],
                    min_tracking_confidence=self.
                    config['min_tracking_confidence'],
                )

        else:
            raise ValueError('Invalid Model Name')

    def __call__(self, image):
        input_image = copy.deepcopy(image)
        bboxes, scores, class_ids = None, None, None

        if self.model is not None:
            bboxes, scores, class_ids = self.model(input_image)
        else:
            raise ValueError('Model is None')

        return bboxes, scores, class_ids

    def print_info(self):
        from pprint import pprint

        print('Detector:', self.model_name)
        pprint(self.config, indent=4)
        print()

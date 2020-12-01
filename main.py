import os
import cv2
import numpy as np
import tensorflow as tf
# from pjtlibs.yolov3.yolov3 import Create_Yolov3
from pjtlibs.yolov3.yolov4 import Create_Yolo
from pjtlibs.yolov3.utils import load_yolo_weights, image_preprocess, postprocess_boxes, nms, draw_bbox, read_class_names

from pjtlibs.deep_sort import nn_matching
from pjtlibs.deep_sort.detection import Detection
from pjtlibs.deep_sort.tracker import Tracker
from pjtlibs.deep_sort import generate_detections as gdet

YOLO_COCO_CLASSES = "./pjtlibs/coco.names"  # coco 클래스 경로
input_size = 512  # 인풋 사이즈
Darknet_weights = "./pjtlibs/yolov4.weights"  # your darknet weight path

yolo = Create_Yolo(input_size=input_size)  # 텐서플로우 네트워크 모델
load_yolo_weights(yolo, Darknet_weights)  # 다크넷 웨이트를 텐서플로우 모델로 로드


def Object_tracking(YoloV3, video_path, output_path, input_size, show=False, CLASSES=YOLO_COCO_CLASSES,
                    score_threshold=0.3, iou_threshold=0.1, rectangle_colors='', Track_only=[]):
    # Definition of the parameters
    max_cosine_distance = 0.4
    nn_budget = None
    # framecount = 0

    # initialize deep sort object
    model_filename = "./pjtlibs/mars-small128.pb"  # deep sort 웨이트
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    if video_path:
        vid = cv2.VideoCapture(video_path)  # detect on video

    NUM_CLASS = read_class_names(CLASSES)  # name strip 하는 커스텀함수 from utils
    key_list = list(NUM_CLASS.keys())
    val_list = list(NUM_CLASS.values())

    while True:
        _, img = vid.read()

        try:
            original_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        except:
            break

        image_data = image_preprocess(np.copy(original_image), [input_size, input_size])  # 인풋 프레임 전처리
        image_data = tf.expand_dims(image_data, 0)

        pred_bbox = YoloV3.predict(image_data)

        pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
        pred_bbox = tf.concat(pred_bbox, axis=0)

        bboxes = postprocess_boxes(pred_bbox, original_image, input_size, score_threshold)
        bboxes = nms(bboxes, iou_threshold, method='nms')  # 신뢰도 낮은 박스 제거하는 커스텀 함수 from utils

        # extract bboxes to boxes (x, y, width, height), scores and names
        boxes, scores, names = [], [], []
        for bbox in bboxes:
            if len(Track_only) != 0 and NUM_CLASS[int(bbox[5])] in Track_only or len(Track_only) == 0:
                boxes.append([bbox[0].astype(int), bbox[1].astype(int), bbox[2].astype(int) - bbox[0].astype(int), bbox[3].astype(int) - bbox[1].astype(int)])
                scores.append(bbox[4])
                names.append(NUM_CLASS[int(bbox[5])])

        # Obtain all the detections for the given frame.
        boxes = np.array(boxes)
        names = np.array(names)
        scores = np.array(scores)
        features = np.array(encoder(original_image, boxes))
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(boxes, scores, names, features)]

        # Pass detections to the deepsort object and obtain the track information.
        tracker.predict()
        tracker.update(detections)

        # Obtain info from the tracks
        tracked_bboxes = []
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()  # Get the corrected/predicted bounding box
            class_name = track.get_class()  # Get the class name of particular object
            tracking_id = track.track_id  # Get the ID for the particular track
            index = key_list[val_list.index(class_name)]  # Get predicted object index by object name
            tracked_bboxes.append(bbox.tolist() + [tracking_id, index])  # Structure data, that we could use it with our draw_bbox function

        if len(tracked_bboxes) != 0:
            image = draw_bbox(original_image, tracked_bboxes, CLASSES=CLASSES, tracking=True)
            if not os.path.isdir('./captured'):
                os.mkdir('./captured')
            cv2.imwrite("./captured/frame.jpg", image)
            return

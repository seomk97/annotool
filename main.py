import os
import colorsys
import random
import threading

import cv2
import numpy as np
import torch
from ultralytics import YOLO


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.environ.get("ANNOTOOL_YOLO_MODEL", "yolo26s.pt")
TRACKER_CONFIG = os.environ.get(
    "ANNOTOOL_TRACKER_CONFIG",
    os.path.join(BASE_DIR, "configs", "tracktrack_reid.yaml"),
)
YOLO_COCO_CLASSES = os.path.join(BASE_DIR, "pjtlibs", "coco.names")
input_size = int(os.environ.get("ANNOTOOL_IMGSZ", "960"))
DEVICE = os.environ.get("ANNOTOOL_DEVICE", "0" if torch.cuda.is_available() else "cpu")
USE_HALF = torch.cuda.is_available() and DEVICE.lower() != "cpu"

# Keep low-confidence person detections available to TrackTrack's second-stage
# association. A larger inference size helps small/distant person detections.
score_threshold = 0.05
iou_threshold = 0.70


def read_class_names(class_file_name=YOLO_COCO_CLASSES):
    names = {}
    with open(class_file_name, "r", encoding="utf-8") as data:
        for class_id, name in enumerate(data):
            names[class_id] = name.strip()
    return names


NUM_CLASS = read_class_names()


class YOLOTrackerAdapter:
    """Adapter that preserves annotool's historical tracking interface.

    The UI expects each tracked box as:
        [x1, y1, x2, y2, track_id, class_id]

    The model, predictor, CUDA context, ReID hook and tracker objects are kept
    alive for the full application session. Session changes reset only tracker
    state, avoiding repeated predictor/model initialization.
    """

    def __init__(
        self,
        model_path=MODEL_PATH,
        conf=score_threshold,
        iou=iou_threshold,
        tracker_config=TRACKER_CONFIG,
        imgsz=input_size,
        device=DEVICE,
        half=USE_HALF,
    ):
        self.model_path = model_path
        self.conf = conf
        self.iou = iou
        self.tracker_config = tracker_config
        self.imgsz = imgsz
        self.device = device
        self.half = half
        self.model = None
        self._prepared = False
        self._lock = threading.RLock()

    @property
    def is_prepared(self):
        return self._prepared

    def _ensure_model(self):
        if self.model is None:
            self.model = YOLO(self.model_path)

    def _reset_tracker_state(self):
        if self.model is None or self.model.predictor is None:
            return

        predictor = self.model.predictor
        for active_tracker in getattr(predictor, "trackers", []) or []:
            reset = getattr(active_tracker, "reset", None)
            if callable(reset):
                reset()

        if hasattr(predictor, "vid_path"):
            predictor.vid_path = [None] * len(predictor.vid_path)

    def prepare(self, progress=None):
        """Load and warm the detector/tracker once per application session."""
        with self._lock:
            if self._prepared:
                if progress:
                    progress("Model ready")
                return

            if progress:
                progress("Loading YOLO26s model...")
            self._ensure_model()

            if progress:
                progress("Initializing GPU and TrackTrack ReID...")
            dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
            self.model.track(
                source=dummy,
                persist=True,
                tracker=self.tracker_config,
                conf=self.conf,
                iou=self.iou,
                classes=[0],
                imgsz=self.imgsz,
                device=self.device,
                half=self.half,
                verbose=False,
            )

            self._reset_tracker_state()
            self._prepared = True

            if progress:
                progress("Model ready")

    def reset(self):
        """Reset IDs and temporal state without rebuilding the predictor."""
        with self._lock:
            self._reset_tracker_state()

    def track_frame(self, frame, conf=None, iou=None, classes=(0,)):
        with self._lock:
            if not self._prepared:
                self.prepare()

            result = self.model.track(
                source=frame,
                persist=True,
                tracker=self.tracker_config,
                conf=self.conf if conf is None else conf,
                iou=self.iou if iou is None else iou,
                classes=list(classes) if classes is not None else None,
                imgsz=self.imgsz,
                device=self.device,
                half=self.half,
                verbose=False,
            )[0]

        boxes = result.boxes
        if boxes is None or len(boxes) == 0 or boxes.id is None:
            return []

        xyxy = boxes.xyxy.cpu().numpy()
        track_ids = boxes.id.int().cpu().tolist()
        class_ids = boxes.cls.int().cpu().tolist()

        return [
            [
                float(box[0]),
                float(box[1]),
                float(box[2]),
                float(box[3]),
                int(track_id),
                int(class_id),
            ]
            for box, track_id, class_id in zip(xyxy, track_ids, class_ids)
        ]


tracker = YOLOTrackerAdapter()
# Historical qt.py imports this name; keep it as an alias so the UI code does
# not need to know which detection/tracking backend is active.
yolo = tracker


def draw_bbox(
    image,
    bboxes,
    CLASSES=YOLO_COCO_CLASSES,
    show_label=True,
    show_confidence=True,
    Text_colors=(255, 255, 0),
    rectangle_colors='',
    tracking=False,
):
    """Legacy annotool renderer kept for UI compatibility."""
    num_class = read_class_names(CLASSES)
    num_classes = len(num_class)
    image_h, image_w, _ = image.shape

    hsv_tuples = [(1.0 * x / num_classes, 1.0, 1.0) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(
        map(
            lambda x: (
                int(x[0] * 255),
                int(x[1] * 255),
                int(x[2] * 255),
            ),
            colors,
        )
    )

    random.seed(0)
    random.shuffle(colors)

    for bbox in bboxes:
        coor = np.array(bbox[:4], dtype=np.int32)
        score = bbox[4]
        class_ind = int(bbox[5])
        bbox_color = rectangle_colors if rectangle_colors != '' else colors[class_ind]
        bbox_thick = int(0.6 * (image_h + image_w) / 1000)
        if bbox_thick < 1:
            bbox_thick = 1
        font_scale = 0.75 * bbox_thick

        (x1, y1), (x2, y2) = (coor[0], coor[1]), (coor[2], coor[3] + 3)
        cv2.rectangle(image, (x1, y1), (x2, y2), bbox_color, bbox_thick * 2)

        if show_label:
            score_str = " {:.2f}".format(score) if show_confidence else ""
            if tracking:
                score_str = " " + str(int(score))

            label = "{}".format(num_class[class_ind]) + score_str
            (text_width, text_height), baseline = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_COMPLEX_SMALL,
                font_scale,
                thickness=bbox_thick,
            )
            cv2.rectangle(
                image,
                (x1, y1),
                (x1 + text_width, y1 - text_height - baseline),
                bbox_color,
                thickness=cv2.FILLED,
            )
            cv2.putText(
                image,
                label,
                (x1, y1 - 4),
                cv2.FONT_HERSHEY_COMPLEX_SMALL,
                font_scale,
                Text_colors,
                bbox_thick,
                lineType=cv2.LINE_AA,
            )

    return image

def Object_tracking(
    model,
    video_path,
    output_path="",
    input_size=input_size,
    show=False,
    CLASSES=YOLO_COCO_CLASSES,
    score_threshold=score_threshold,
    iou_threshold=0.3,
    rectangle_colors="",
    Track_only=None,
):
    """Load and annotate the first frame used by the existing object-select UI."""
    del output_path, input_size, show, CLASSES

    vid = cv2.VideoCapture(video_path)
    ret, frame = vid.read()
    vid.release()
    if not ret:
        return []

    track_only = Track_only or ["person"]
    classes = [0] if "person" in track_only else None

    model.reset()
    tracked_bboxes = model.track_frame(
        frame,
        conf=score_threshold,
        iou=iou_threshold,
        classes=classes,
    )

    image = draw_bbox(
        frame.copy(),
        tracked_bboxes,
        tracking=True,
        rectangle_colors=rectangle_colors,
    )

    os.makedirs("./captured", exist_ok=True)
    cv2.imwrite("./captured/frame.jpg", image)

    # Preview must not leak its tracker state into the real worker-thread run.
    model.reset()
    return tracked_bboxes

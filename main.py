import os

import cv2
import numpy as np
from ultralytics import YOLO


MODEL_PATH = os.environ.get("ANNOTOOL_YOLO_MODEL", "yolo26n.pt")
YOLO_COCO_CLASSES = "./pjtlibs/coco.names"  # retained for UI compatibility
input_size = 640
score_threshold = 0.3
iou_threshold = 0.1


def read_class_names(class_file_name=YOLO_COCO_CLASSES):
    names = {}
    with open(class_file_name, "r", encoding="utf-8") as data:
        for class_id, name in enumerate(data):
            names[class_id] = name.strip()
    return names


NUM_CLASS = read_class_names()


class YOLOByteTracker:
    """Small adapter that preserves the old annotool tracking interface.

    The UI expects each tracked box as:
        [x1, y1, x2, y2, track_id, class_id]

    Ultralytics owns detector/tracker internals; this adapter deliberately keeps
    them out of qt.py so the original button/threading workflow can remain
    unchanged.
    """

    def __init__(
        self,
        model_path=MODEL_PATH,
        conf=score_threshold,
        iou=iou_threshold,
        tracker_config="bytetrack.yaml",
    ):
        self.model_path = model_path
        self.conf = conf
        self.iou = iou
        self.tracker_config = tracker_config
        self.model = None

    def _ensure_model(self):
        if self.model is None:
            self.model = YOLO(self.model_path)

    def reset(self):
        """Reset only stream/tracker state while keeping UI state untouched."""
        if self.model is not None:
            # Rebuilding the predictor recreates Ultralytics tracker state on
            # the next frame without reloading model weights.
            self.model.predictor = None

    def track_frame(self, frame, conf=None, iou=None, classes=(0,)):
        self._ensure_model()

        result = self.model.track(
            source=frame,
            persist=True,
            tracker=self.tracker_config,
            conf=self.conf if conf is None else conf,
            iou=self.iou if iou is None else iou,
            classes=list(classes) if classes is not None else None,
            verbose=False,
        )[0]

        boxes = result.boxes
        if boxes is None or len(boxes) == 0 or boxes.id is None:
            return []

        xyxy = boxes.xyxy.cpu().numpy()
        track_ids = boxes.id.int().cpu().tolist()
        class_ids = boxes.cls.int().cpu().tolist()

        tracked_bboxes = []
        for box, track_id, class_id in zip(xyxy, track_ids, class_ids):
            tracked_bboxes.append(
                [
                    float(box[0]),
                    float(box[1]),
                    float(box[2]),
                    float(box[3]),
                    int(track_id),
                    int(class_id),
                ]
            )
        return tracked_bboxes


def draw_bbox(
    image,
    bboxes,
    CLASSES=YOLO_COCO_CLASSES,
    show_label=True,
    show_confidence=True,
    Text_colors=(255, 255, 0),
    rectangle_colors="",
    tracking=False,
):
    """Draw boxes using the legacy annotool bbox layout."""
    class_names = NUM_CLASS if CLASSES == YOLO_COCO_CLASSES else read_class_names(CLASSES)
    image_h, image_w = image.shape[:2]
    bbox_thick = max(1, int(0.6 * (image_h + image_w) / 1000))
    font_scale = 0.75 * bbox_thick

    for bbox in bboxes:
        if len(bbox) < 6:
            continue

        x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
        value = bbox[4]
        class_id = int(bbox[5])

        bbox_color = rectangle_colors if rectangle_colors != "" else (0, 255, 0)
        cv2.rectangle(image, (x1, y1), (x2, y2), bbox_color, bbox_thick * 2)

        if not show_label:
            continue

        class_name = class_names.get(class_id, str(class_id))
        if tracking:
            label = f"{class_name} {int(value)}"
        else:
            suffix = f" {float(value):.2f}" if show_confidence else ""
            label = f"{class_name}{suffix}"

        (text_width, text_height), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_COMPLEX_SMALL,
            font_scale,
            thickness=bbox_thick,
        )
        top = max(0, y1 - text_height - baseline)
        cv2.rectangle(
            image,
            (x1, top),
            (x1 + text_width, y1),
            bbox_color,
            thickness=cv2.FILLED,
        )
        cv2.putText(
            image,
            label,
            (x1, max(text_height, y1 - 4)),
            cv2.FONT_HERSHEY_COMPLEX_SMALL,
            font_scale,
            Text_colors,
            bbox_thick,
            lineType=cv2.LINE_AA,
        )

    return image


tracker = YOLOByteTracker()
# Historical qt.py imports this name. Keep it as an alias so button/thread code
# does not need to know about the backend migration.
yolo = tracker


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

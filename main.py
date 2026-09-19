import os
import colorsys
import random
import threading

import cv2
import numpy as np
import torch
from boxmot import OccluBoost
from ultralytics import YOLO


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.environ.get("ANNOTOOL_YOLO_MODEL", "yolo26s.pt")
REID_MODEL = os.environ.get("ANNOTOOL_REID_MODEL", "osnet_x1_0_msmt17.pt")
YOLO_COCO_CLASSES = os.path.join(BASE_DIR, "pjtlibs", "coco.names")
input_size = int(os.environ.get("ANNOTOOL_IMGSZ", "960"))
DEVICE = os.environ.get("ANNOTOOL_DEVICE", "0" if torch.cuda.is_available() else "cpu")
USE_HALF = torch.cuda.is_available() and DEVICE.lower() != "cpu"
BOXMOT_DEVICE = (
    f"cuda:{DEVICE}" if DEVICE.isdigit() else DEVICE
)
# Keep low-confidence person detections available to the tracker's recovery
# stages. A larger inference size helps small/distant person detections.
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
    """YOLO26 detector + BoxMOT OccluBoost + OSNet person ReID adapter.

    The Qt UI keeps its historical box format:
        [x1, y1, x2, y2, track_id, class_id]

    YOLO performs detection only. OccluBoost owns temporal association,
    occlusion recovery and duplicate suppression, while OSNet x1.0 MSMT17
    supplies person appearance embeddings.
    """

    def __init__(
        self,
        model_path=MODEL_PATH,
        conf=score_threshold,
        iou=iou_threshold,
        imgsz=input_size,
        device=DEVICE,
        half=USE_HALF,
        reid_model=REID_MODEL,
    ):
        self.model_path = model_path
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.device = device
        self.half = half
        self.reid_model = reid_model

        self.model = None
        self.boxmot = None
        self._prepared = False
        self._lock = threading.RLock()

    @property
    def is_prepared(self):
        return self._prepared

    def _ensure_detector(self):
        if self.model is None:
            self.model = YOLO(self.model_path)

    def _build_tracker(self):
        # BoxMOT 25.0.0 public API: configure live ReID directly on the
        # tracker. The OSNet weights are downloaded lazily on first use.
        self.boxmot = OccluBoost(
            use_embeddings=True,
            reid_weights=self.reid_model,
            device=BOXMOT_DEVICE,
            half=self.half,
            per_class=False,
            class_ids=(0,),
            class_names={0: "person"),

            # BoxMOT v25 tuned AABB defaults.
            max_age=146,
            min_hits=1,
            det_thresh=0.5678626013369781,
            iou_threshold=0.2957128153631725,
            use_cmc=True,
            cmc_method="sof",
            min_box_area=73,
            aspect_ratio_thresh=1.4888137942764672,
            lambda_iou=1.0784558316374715,
            lambda_mhd=0.304435887183232,
            lambda_shape=1.6709449476805447,
            use_dlo_boost=True,
            use_duo_boost=False,
            use_rich_s=False,
            use_sb=True,
            use_vt=True,
            dlo_boost_coef=1.2061962091907352,
            recovery_appearance_thresh=0.6732855110134396,
            recovery_iou_thresh=0.24380051350243462,
            recovery_max_age=113,
            feat_alpha=0.8324072665785186,
            track_low_thresh=0.04473431588067598,
            use_second_pass=True,
            second_iou_thresh=0.8131671757478834,
            second_appearance_thresh=0.364089272226479,
            second_pass_max_age=8,
            second_pass_min_hits=7,
            new_track_thresh=0.7128242784621849,
            confirm_hits=2,
            instant_confirm_thresh=0.6783889178413256,
            tentative_max_age=3,
            duplicate_iou_thresh=0.9571823233925608,
            lambda_emb_multiplier=2.9476295884842885,
            gta_enabled=False,
        )

    def prepare(self, progress=None):
        """Load detector and initialize BoxMOT/OSNet once per app session."""
        with self._lock:
            if self._prepared:
                if progress:
                    progress("Model ready")
                return

            if progress:
                progress("Loading YOLO26s detector...")
            self._ensure_detector()

            # Warm the YOLO predictor/CUDA path.
            dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
            self.model.predict(
                source=dummy,
                conf=self.conf,
                iou=self.iou,
                classes=[0],
                imgsz=self.imgsz,
                device=self.device,
                half=self.half,
                verbose=False,
            )

            if progress:
                progress("Loading OccluBoost + OSNet x1.0 ReID...")
            self._build_tracker()

            # BoxMOT constructs the ReID backend lazily. A single synthetic
            # person detection initializes/downloads OSNet now, so normal
            # playback does not pay the first-use cost.
            dummy_det = np.array(
                [[
                    self.imgsz * 0.25,
                    self.imgsz * 0.10,
                    self.imgsz * 0.75,
                    self.imgsz * 0.90,
                    0.99,
                    0.0,
                ]],
                dtype=np.float32,
            )
            self.boxmot.update(dummy_det, frame=dummy)
            self.boxmot.reset()

            self._prepared = True
            if progress:
                progress("Model ready")

    def reset(self):
        """Reset temporal IDs while retaining loaded YOLO and OSNet weights."""
        with self._lock:
            if self.boxmot is not None:
                self.boxmot.reset()

    def _detect(self, frame, conf=None, iou=None, classes=(0,)):
        result = self.model.predict(
            source=frame,
            conf=self.conf if conf is None else conf,
            iou=self.iou if iou is None else iou,
            classes=list(classes) if classes is not None else None,
            imgsz=self.imgsz,
            device=self.device,
            half=self.half,
            verbose=False,
        )[0]

        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return np.empty((0, 6), dtype=np.float32)

        xyxy = boxes.xyxy.detach().cpu().numpy().astype(np.float32, copy=False)
        confidence = boxes.conf.detach().cpu().numpy().astype(np.float32, copy=False)
        class_ids = boxes.cls.detach().cpu().numpy().astype(np.float32, copy=False)

        return np.ascontiguousarray(
            np.column_stack((xyxy, confidence, class_ids)),
            dtype=np.float32,
        )

    def track_frame(
        self,
        frame,
        conf=None,
        iou=None,
        classes=(0,),
        preferred_track_id=None,
    ):
        del preferred_track_id  # BoxMOT association is independent of UI selection.

        with self._lock:
            if not self._prepared:
                self.prepare()

            detections = self._detect(frame, conf=conf, iou=iou, classes=classes)
            tracks = self.boxmot.update(detections, frame=frame)

        if tracks is None or len(tracks) == 0:
            return []

        # BoxMOT AABB output:
        # [x1, y1, x2, y2, track_id, confidence, class_id, detection_index]
        best_by_id = {}
        for row in np.asarray(tracks):
            track_id = int(row[4])
            confidence = float(row[5])
            candidate = (
                confidence,
                [
                    float(row[0]),
                    float(row[1]),
                    float(row[2]),
                    float(row[3]),
                    track_id,
                    int(row[6]),
                ],
            )
            if track_id not in best_by_id or confidence > best_by_id[track_id][0]:
                best_by_id[track_id] = candidate

        return [
            best_by_id[track_id][1]
            for track_id in sorted(best_by_id)
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

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
MODEL_PATH = os.environ.get("ANNOTOOL_YOLO_MODEL", "yolo26m.pt")
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
iou_threshold = 0.50
COAST_FRAMES = int(os.environ.get("ANNOTOOL_COAST_FRAMES", "2"))


def read_class_names(class_file_name=YOLO_COCO_CLASSES):
    names = {}
    with open(class_file_name, "r", encoding="utf-8") as data:
        for class_id, name in enumerate(data):
            names[class_id] = name.strip()
    return names


NUM_CLASS = read_class_names()


def _detection_iou(a, b):
    ax1, ay1, ax2, ay2 = a[:4]
    bx1, by1, bx2, by2 = b[:4]

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih

    area_a = max(1.0, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1.0, (bx2 - bx1) * (by2 - by1))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _detection_containment(a, b):
    ax1, ay1, ax2, ay2 = a[:4]
    bx1, by1, bx2, by2 = b[:4]

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih

    area_a = max(1.0, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1.0, (bx2 - bx1) * (by2 - by1))
    return inter / min(area_a, area_b)


def _suppress_duplicate_detections(detections):
    """Collapse only near-identical/nested person detections before tracking."""
    if len(detections) <= 1:
        return detections

    order = np.argsort(-detections[:, 4])
    kept = []

    for idx in order:
        candidate = detections[idx]
        duplicate = False
        for kept_det in kept:
            if (
                _detection_iou(candidate, kept_det) >= 0.65
                or _detection_containment(candidate, kept_det) >= 0.90
            ):
                duplicate = True
                break
        if not duplicate:
            kept.append(candidate)

    if not kept:
        return np.empty((0, 6), dtype=np.float32)

    return np.ascontiguousarray(np.stack(kept), dtype=np.float32)


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
            class_names={0: "person"},

            # Annotation work prioritizes person recall over MOT benchmark
            # precision. YOLO already filters to class=person at conf >= 0.05,
            # so do not discard those detections again inside the tracker.
            max_age=146,
            min_hits=0,
            det_thresh=0.15,
            iou_threshold=0.2957128153631725,
            use_cmc=True,
            cmc_method="sof",
            min_box_area=1,
            aspect_ratio_thresh=10.0,
            lambda_iou=1.0784558316374715,
            lambda_mhd=0.304435887183232,
            # Football players change pose/aspect ratio aggressively; do not
            # let bbox shape dominate identity association.
            lambda_shape=0.20,
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
            track_low_thresh=0.05,
            use_second_pass=True,
            second_iou_thresh=0.8131671757478834,
            second_appearance_thresh=0.364089272226479,
            second_pass_max_age=8,
            second_pass_min_hits=7,
            new_track_thresh=0.25,
            confirm_hits=2,
            instant_confirm_thresh=0.55,
            tentative_max_age=3,
            duplicate_iou_thresh=0.75,

            # Large legitimate pose changes (crouch, tackle, jump, fall) can
            # look like abnormal bbox shrink/motion to AMS. Disable it for
            # sports footage and rely more on motion + appearance association.
            ams_enabled=False,
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
                progress("Loading YOLO26m detector...")
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
                quantize=16 if self.half else None,
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

        detections = np.ascontiguousarray(
            np.column_stack((xyxy, confidence, class_ids)),
            dtype=np.float32,
        )
        return _suppress_duplicate_detections(detections)

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

        # BoxMOT AABB output:
        # [x1, y1, x2, y2, track_id, confidence, class_id, detection_index]
        #
        # OccluBoost keeps unmatched tracks alive internally, but only emits
        # tracks updated on the current frame. For annotation playback that
        # causes visible one-frame flicker whenever YOLO briefly misses a
        # person. Bridge only a very short miss with the tracker's predicted
        # state; the next real detection still corrects the trajectory.
        best_by_id = {}

        if tracks is not None and len(tracks) > 0:
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

        if COAST_FRAMES > 0:
            for active_track in getattr(self.boxmot, "trackers", []) or []:
                missed = int(getattr(active_track, "time_since_update", 0))
                if missed < 1 or missed > COAST_FRAMES:
                    continue
                if not getattr(active_track, "is_activated", True):
                    continue

                track_id = int(active_track.id)
                if track_id in best_by_id:
                    continue

                state = np.asarray(active_track.get_state()[0]).reshape(-1)
                if state.size < 4:
                    continue

                class_id = int(getattr(active_track, "cls", 0))
                confidence = float(getattr(active_track, "conf", 0.0))
                best_by_id[track_id] = (
                    confidence,
                    [
                        float(state[0]),
                        float(state[1]),
                        float(state[2]),
                        float(state[3]),
                        track_id,
                        class_id,
                    ],
                )

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

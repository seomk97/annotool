import os
import threading

import cv2
import numpy as np
import torch
from boxmot import OccluBoost
from ultralytics import YOLO


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.environ.get("ANNOTOOL_YOLO_MODEL", "yolo26m.pt")
REID_MODEL = os.environ.get("ANNOTOOL_REID_MODEL", "osnet_x1_0_msmt17.pt")
INPUT_SIZE = int(os.environ.get("ANNOTOOL_IMGSZ", "960"))
DEVICE = os.environ.get("ANNOTOOL_DEVICE", "0" if torch.cuda.is_available() else "cpu")
USE_HALF = torch.cuda.is_available() and DEVICE.lower() != "cpu"
BOXMOT_DEVICE = (
    f"cuda:{DEVICE}" if DEVICE.isdigit() else DEVICE
)
# Keep low-confidence person detections available to the tracker's recovery
# stages. A larger inference size helps small/distant person detections.
SCORE_THRESHOLD = 0.05
NMS_IOU_THRESHOLD = 0.50
COAST_FRAMES = int(os.environ.get("ANNOTOOL_COAST_FRAMES", "2"))


from camera import preprocess_frame


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


class PersonTracker:
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
        conf=SCORE_THRESHOLD,
        iou=NMS_IOU_THRESHOLD,
        imgsz=INPUT_SIZE,
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
            NMS_IOU_THRESHOLD=0.2957128153631725,
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

    def _detect(self, frame):
        result = self.model.predict(
            source=frame,
            conf=self.conf,
            iou=self.iou,
            classes=[0],
            imgsz=self.imgsz,
            device=self.device,
            quantize=16 if self.half else None,
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

    def track_frame(self, frame):
        with self._lock:
            if not self._prepared:
                self.prepare()

            detections = self._detect(frame)
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


tracker = PersonTracker()


def draw_tracks(image, tracks, text_color=(255, 255, 0), rectangle_color=None):
    """Draw person track IDs on a BGR frame."""
    image_h, image_w = image.shape[:2]
    bbox_thick = max(1, int(0.6 * (image_h + image_w) / 1000))
    font_scale = 0.75 * bbox_thick
    box_color = (50, 0, 255) if rectangle_color is None else rectangle_color

    for track in tracks:
        x1, y1, x2, y2 = np.asarray(track[:4], dtype=np.int32)
        track_id = int(track[4])
        cv2.rectangle(image, (x1, y1), (x2, y2 + 3), box_color, bbox_thick * 2)

        label = f"person {track_id}"
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
            box_color,
            thickness=cv2.FILLED,
        )
        cv2.putText(
            image,
            label,
            (x1, y1 - 4),
            cv2.FONT_HERSHEY_COMPLEX_SMALL,
            font_scale,
            text_color,
            bbox_thick,
            lineType=cv2.LINE_AA,
        )

    return image


def create_tracking_preview(video_path):
    """Track and write the first frame used by the object-selection UI."""
    video = cv2.VideoCapture(video_path)
    ok, frame = video.read()
    video.release()
    if not ok or frame is None:
        return []

    frame = preprocess_frame(frame)
    tracker.reset()
    tracks = tracker.track_frame(frame)
    image = draw_tracks(frame.copy(), tracks, rectangle_color=(255, 0, 0))

    os.makedirs("./captured", exist_ok=True)
    cv2.imwrite("./captured/frame.jpg", image)

    tracker.reset()
    return tracks

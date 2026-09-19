import os
import threading

import cv2
import numpy as np


FISHEYE_CALIB = os.environ.get("ANNOTOOL_FISHEYE_CALIB", "")
FISHEYE_BALANCE = float(os.environ.get("ANNOTOOL_FISHEYE_BALANCE", "0.2"))


class FisheyePreprocessor:
    """Optional OpenCV fisheye undistortion for calibrated cameras."""

    def __init__(self, calibration_path="", balance=0.2):
        self.calibration_path = calibration_path.strip()
        self.balance = float(balance)
        self._K = None
        self._D = None
        self._calibration_size = None
        self._maps = {}
        self._lock = threading.RLock()

    @property
    def enabled(self):
        return bool(self.calibration_path)

    def configure(self, calibration_path="", balance=None):
        """Switch fisheye correction at runtime and clear cached calibration maps."""
        with self._lock:
            self.calibration_path = str(calibration_path or "").strip()
            if balance is not None:
                self.balance = float(balance)
            self._K = None
            self._D = None
            self._calibration_size = None
            self._maps.clear()

    def _load(self):
        if self._K is not None:
            return

        path = os.path.abspath(os.path.expanduser(self.calibration_path))
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Fisheye calibration file not found: {path}")

        with np.load(path) as data:
            if "K" not in data or "D" not in data:
                raise ValueError("Fisheye calibration .npz must contain K and D arrays.")

            K = np.asarray(data["K"], dtype=np.float64)
            D = np.asarray(data["D"], dtype=np.float64).reshape(-1)

            size = None
            for key in ("DIM", "image_size", "size"):
                if key in data:
                    raw = np.asarray(data[key]).reshape(-1)
                    if raw.size >= 2:
                        size = (int(raw[0]), int(raw[1]))
                    break

        if K.shape != (3, 3):
            raise ValueError(f"Fisheye K must be 3x3, got {K.shape}.")
        if D.size != 4:
            raise ValueError(f"OpenCV fisheye D must contain 4 coefficients, got {D.size}.")

        self._K = K
        self._D = D.reshape(4, 1)
        self._calibration_size = size

    def _maps_for(self, width, height):
        key = (int(width), int(height))
        with self._lock:
            if key in self._maps:
                return self._maps[key]

            self._load()
            K = self._K.copy()

            if self._calibration_size:
                calib_w, calib_h = self._calibration_size
                if calib_w > 0 and calib_h > 0:
                    sx = width / calib_w
                    sy = height / calib_h
                    K[0, 0] *= sx
                    K[0, 2] *= sx
                    K[1, 1] *= sy
                    K[1, 2] *= sy

            size = (int(width), int(height))
            new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                K,
                self._D,
                size,
                np.eye(3),
                balance=self.balance,
                new_size=size,
            )
            map1, map2 = cv2.fisheye.initUndistortRectifyMap(
                K,
                self._D,
                np.eye(3),
                new_K,
                size,
                cv2.CV_16SC2,
            )
            self._maps[key] = (map1, map2)
            return map1, map2

    def apply(self, frame):
        if not self.enabled or frame is None:
            return frame

        height, width = frame.shape[:2]
        map1, map2 = self._maps_for(width, height)
        return cv2.remap(
            frame,
            map1,
            map2,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )


def configure_fisheye(calibration_path="", balance=None):
    """Enable fisheye correction with a calibration file, or disable it with an empty path."""
    fisheye.configure(calibration_path, balance=balance)


def preprocess_frame(frame):
    """Apply optional camera preprocessing while leaving normal video untouched."""
    return fisheye.apply(frame)


fisheye = FisheyePreprocessor(FISHEYE_CALIB, FISHEYE_BALANCE)


def configure_fisheye(calibration_path="", balance=None):
    """Enable fisheye correction, or disable it with an empty path."""
    fisheye.configure(calibration_path, balance=balance)


def preprocess_frame(frame):
    """Apply optional fisheye correction without affecting normal video."""
    return fisheye.apply(frame)

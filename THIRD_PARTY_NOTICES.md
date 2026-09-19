# Third-Party Notices

This project contains original annotation UI/workflow code and integrates open-source detection and tracking components.

## Current runtime: Ultralytics YOLO26 + BoT-SORT

- Upstream: https://github.com/ultralytics/ultralytics
- Used for: YOLO26 person detection and BoT-SORT multi-object tracking with ReID
- Integration: imported as the `ultralytics` Python dependency; not vendored into this repository
- License: GNU AGPL v3
- License copy: [THIRD_PARTY_LICENSES/Ultralytics-AGPL-3.0.txt](THIRD_PARTY_LICENSES/Ultralytics-AGPL-3.0.txt)

The current project is distributed under AGPL-3.0 to keep the runtime integration license-compatible.

## Legacy backend: PyLessons TensorFlow YOLOv3 / YOLOv4

- Upstream: https://github.com/pythonlessons/TensorFlow-2.x-YOLOv3
- Historical code retained under: `pjtlibs/yolov3/`
- License: MIT
- License copy: [THIRD_PARTY_LICENSES/PyLessons-MIT.txt](THIRD_PARTY_LICENSES/PyLessons-MIT.txt)

## Legacy backend: Deep SORT

- Upstream: https://github.com/nwojke/deep_sort
- Historical code retained under: `pjtlibs/deep_sort/`
- License: GNU GPL v3
- License copy: [THIRD_PARTY_LICENSES/Deep-SORT-GPL-3.0.txt](THIRD_PARTY_LICENSES/Deep-SORT-GPL-3.0.txt)
- Citation: Nicolai Wojke, Alex Bewley, Dietrich Paulus, *Simple Online and Realtime Tracking with a Deep Association Metric*, ICIP 2017.

The retained copy follows the Deep SORT integration used by the historical PyLessons-based implementation.

## Legacy Darknet / YOLOv4 weights

- Upstream: https://github.com/AlexeyAB/darknet
- Historical use: pretrained YOLOv4 Darknet weights loaded by the TensorFlow YOLO implementation
- License: YOLO License / Darknet public-domain notice
- License copy: [THIRD_PARTY_LICENSES/Darknet-YOLO-License.txt](THIRD_PARTY_LICENSES/Darknet-YOLO-License.txt)

## Legacy Deep SORT appearance model

- File: `pjtlibs/mars-small128.pb`
- Purpose: appearance descriptor used by the historical Deep SORT backend
- Related project/paper: Deep SORT / MARS person re-identification model

The legacy files are retained to preserve the original 2020 implementation history; the modern runtime path does not import TensorFlow, the PyLessons YOLO implementation, or the Deep SORT encoder.

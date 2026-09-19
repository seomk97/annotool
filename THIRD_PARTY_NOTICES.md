# Third-Party Notices

This project integrates open-source detection and tracking components. The annotation GUI and workflow are maintained in this repository; the components below retain their original upstream licenses.

## PyLessons TensorFlow YOLOv3 / YOLOv4

- Upstream: https://github.com/pythonlessons/TensorFlow-2.x-YOLOv3
- Used under: `pjtlibs/yolov3/` and related integration code
- License: MIT
- License copy: [THIRD_PARTY_LICENSES/PyLessons-MIT.txt](THIRD_PARTY_LICENSES/PyLessons-MIT.txt)

## Deep SORT

- Upstream: https://github.com/nwojke/deep_sort
- Used under: `pjtlibs/deep_sort/`
- License: GNU GPL v3
- License copy: [THIRD_PARTY_LICENSES/Deep-SORT-GPL-3.0.txt](THIRD_PARTY_LICENSES/Deep-SORT-GPL-3.0.txt)
- Citation: Nicolai Wojke, Alex Bewley, Dietrich Paulus, *Simple Online and Realtime Tracking with a Deep Association Metric*, ICIP 2017.

The copy included here follows the Deep SORT integration used by the PyLessons YOLO project and contains code matching the upstream Deep SORT implementation.

## Darknet / YOLOv4 weights

- Upstream: https://github.com/AlexeyAB/darknet
- Used for: pretrained YOLOv4 Darknet weights loaded by the TensorFlow YOLO implementation
- License: YOLO License / Darknet public-domain notice
- License copy: [THIRD_PARTY_LICENSES/Darknet-YOLO-License.txt](THIRD_PARTY_LICENSES/Darknet-YOLO-License.txt)

## Deep SORT appearance model

- File: `pjtlibs/mars-small128.pb`
- Purpose: appearance descriptor used by Deep SORT
- Related project/paper: Deep SORT / MARS person re-identification model

This notice is intended to document provenance and the licenses of third-party components redistributed or referenced by this repository.

# annotool

YOLOv4 + Deep SORT 기반의 영상 객체 / 행동 annotation 도구입니다.  
영상에서 검출·추적된 객체를 보면서 target을 지정하고, 필요한 frame과 action 구간을 기록해 이미지와 JSON으로 저장할 수 있도록 PyQt5 GUI를 구성했습니다.

> 2020년에 제작한 개인 프로젝트로, 당시의 TensorFlow / CUDA 환경을 기준으로 작성되어 있습니다.

## Pipeline

```text
Video
  → Object Detection (YOLOv4)
  → Object Tracking (Deep SORT)
  → PyQt Annotation UI
  → Captured Frames + JSON Annotation
```

## 구현 범위

이 repository의 detection / tracking backend는 기존 open-source 구현을 기반으로 하고, 그 위에 annotation workflow와 GUI를 구성했습니다.

직접 구현한 부분:

- PyQt5 기반 annotation GUI
- 영상 재생 / 일시정지 / 배속 / frame 이동
- tracking 결과에서 annotation 대상 object 선택
- tracking ID가 바뀌었을 때 target 변경
- action 시작 / 종료 frame 기록
- 저장할 frame 및 label 목록 관리
- 선택 항목 삭제 및 frame 이동
- captured image / JSON annotation 저장
- keyboard shortcut 중심의 annotation workflow

기존 구현을 활용한 부분:

| 구성 요소 | Upstream / 출처 |
|---|---|
| YOLOv3 / YOLOv4 TensorFlow implementation | [pythonlessons/TensorFlow-2.x-YOLOv3](https://github.com/pythonlessons/TensorFlow-2.x-YOLOv3) |
| Deep SORT tracking implementation | [nwojke/deep_sort](https://github.com/nwojke/deep_sort) 계열 구현 |
| YOLOv4 pretrained Darknet weights | [AlexeyAB/darknet](https://github.com/AlexeyAB/darknet) |
| Deep SORT appearance model | `mars-small128.pb`, Deep SORT / MARS appearance descriptor |

특히 `pjtlibs/yolov3/`의 YOLOv4 TensorFlow 코드와 Darknet weight loader는 PyLessons 구현을 기반으로 하며, `pjtlibs/deep_sort/` 역시 해당 프로젝트에서 사용한 Deep SORT integration을 기반으로 합니다.

이 repository는 GitHub의 fork 기능으로 생성한 repository는 아니지만, 위 backend 구현을 가져와 annotation tool에 맞게 통합한 프로젝트입니다.

## Screenshot

![annotool screenshot](https://user-images.githubusercontent.com/70502705/101143621-81594580-365a-11eb-9bcf-f81cfa04b7a7.png)

## 실행 환경

당시 확인한 환경:

- Ubuntu 18.04
- CUDA 10.1
- cuDNN 7.6.5
- TensorFlow / TensorFlow-GPU 2.3.1
- OpenCV 4.1.2
- PyQt5 5.15.1

의존성은 `requirements.txt`에 기록되어 있습니다.

## Setup

```bash
git clone https://github.com/seomk97/annotool.git
cd annotool

pip install -r requirements.txt
```

YOLOv4 Darknet weight를 준비합니다.

- Official upstream: [AlexeyAB/darknet YOLOv4 weights](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights)
- 저장 위치: `pjtlibs/yolov4.weights`

최종 구조:

```text
pjtlibs/
├─ yolov4.weights
├─ mars-small128.pb
├─ coco.names
├─ yolov3/
└─ deep_sort/
```

실행:

```bash
python qt.py
```

## 주요 UI

| 기능 | 설명 |
|---|---|
| File | annotation할 영상 선택 |
| Load | 첫 frame 로드 |
| Object | 추적 / 기록할 object 설정 |
| Target | tracking ID가 변경된 경우 target 변경 |
| Track start | detection + tracking 시작 |
| Play / Pause | 영상 재생 / 일시정지 |
| Arrow keys | 재생 속도 조절 |
| Make JSON | 현재 기록된 annotation을 JSON으로 저장 |
| Delete | 선택된 기록 삭제 |
| Action start | action 시작 / 종료 구간 기록 |
| Show target only | 선택한 target만 표시 |
| Open folder | 저장 폴더 열기 |
| Reset | 현재 작업 초기화 |

각 버튼의 shortcut은 GUI 버튼에 함께 표시됩니다.

## Output

annotation 결과는 `captured/` 아래에 object 단위 폴더로 저장됩니다.

- 선택된 frame 이미지
- frame 번호 / label 정보
- action 구간 정보
- JSON annotation

같은 object 번호를 다른 영상에서 사용할 경우 기존 폴더와 충돌하지 않도록 별도 폴더가 생성됩니다.

## Upstream / Acknowledgements

- PyLessons — TensorFlow 2.x YOLOv3 / YOLOv4 implementation  
  https://github.com/pythonlessons/TensorFlow-2.x-YOLOv3
- Alexey Bochkovskiy et al. — Darknet / YOLOv4  
  https://github.com/AlexeyAB/darknet
- Nicolai Wojke et al. — Deep SORT  
  https://github.com/nwojke/deep_sort
- Deep SORT paper: *Simple Online and Realtime Tracking with a Deep Association Metric*, ICIP 2017

## Notes

이 프로젝트는 detector나 tracker 자체를 새로 제안한 프로젝트가 아니라, 기존 YOLOv4 + Deep SORT pipeline을 실제 영상 annotation 작업에 사용할 수 있도록 GUI와 annotation workflow로 통합한 도구입니다.

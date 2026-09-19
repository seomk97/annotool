# annotool

영상에서 사람을 추적하면서 필요한 frame과 action 구간을 빠르게 기록하기 위해 만든 PyQt 기반 annotation tool입니다.

2020년에는 **TensorFlow YOLOv4 + Deep SORT**를 backend로 사용했고, 2026년에는 기존 버튼·단축키·threading 기반 annotation workflow를 유지한 채 backend를 **YOLO26 + BoxMOT OccluBoost + OSNet x1.0 MSMT17**로 현대화했습니다.

## Current pipeline

```text
Video
  → YOLO26 person detection
  → BoxMOT OccluBoost + OSNet x1.0 person ReID
  → PyQt annotation UI
  → Captured frames + JSON annotations
```

현재 runtime에서 UI는 detector / tracker 내부 구현을 직접 다루지 않습니다. `main.py`의 adapter가 tracking 결과를 기존 UI가 사용하던 다음 형식으로 변환합니다.

```text
[x1, y1, x2, y2, track_id, class_id]
```

덕분에 object 선택, target 변경, pause / seek, action 기록 같은 기존 UI state machine은 그대로 유지됩니다.

## 구현 범위

직접 구현한 부분:

- PyQt5 기반 annotation GUI
- 영상 재생 / 일시정지 / 배속 / frame 이동
- tracking ID를 이용한 annotation 대상 object 선택
- tracking ID가 바뀌었을 때 target 변경
- action 시작 / 종료 frame 기록
- frame / label 목록 관리와 선택 항목 삭제
- captured image / JSON annotation 저장
- keyboard shortcut 중심의 annotation workflow
- worker thread와 Qt signal을 이용한 영상 처리 / UI update 분리

현재 backend:

| 구성 요소 | 구현 |
|---|---|
| Object detection | Ultralytics YOLO26 |
| Object tracking | BoxMOT OccluBoost + OSNet x1.0 MSMT17 |
| GUI | PyQt5 |
| Video / image I/O | OpenCV |

## Backend modernization

기존 버전은 PyLessons의 TensorFlow YOLOv4 구현과 Deep SORT appearance encoder에 의존했습니다.

현대화하면서 다음 부분을 제거했습니다.

- TensorFlow 2.3 runtime dependency
- Darknet weight → TensorFlow model 변환
- 수동 YOLO post-processing / NMS 경로
- Deep SORT `mars-small128.pb` runtime dependency

대신 `main.py` adapter가 YOLO26 detection 결과를 BoxMOT OccluBoost에 넘기고, OSNet x1.0 person ReID 결과를 포함한 track을 기존 UI 형식으로 변환합니다.

pause / slider seek / list jump 시에는 UI 상태를 초기화하지 않고 tracker state만 reset하도록 유지했습니다.

> `pjtlibs/yolov3/`, `pjtlibs/deep_sort/`, `mars-small128.pb`는 2020년 구현 provenance를 보존하기 위해 repository에 남겨두었으며 현재 runtime에서는 import하지 않습니다.

## Screenshot

![annotool screenshot](https://user-images.githubusercontent.com/70502705/101143621-81594580-365a-11eb-9bcf-f81cfa04b7a7.png)

## Setup

권장 환경:

- Python 3.11+
- Windows / Linux
- CUDA GPU optional

```bash
git clone https://github.com/seomk97/annotool.git
cd annotool

python -m venv .venv

# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
python qt.py
```

Windows PowerShell:

```powershell
git clone https://github.com/seomk97/annotool.git
cd annotool

py -m venv .venv
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
python qt.py
```

`torch`와 `torchvision`은 runtime dependency로 명시되어 있으며, 깨끗한 환경에서는 `pip install -r requirements.txt`로 함께 설치됩니다. 이미 다른 Python 환경에 설치된 PyTorch가 깨져 있거나 CUDA build를 직접 선택해야 하는 경우에는 [PyTorch installation guide](https://docs.pytorch.org/get-started/locally/)에 따라 해당 환경의 PyTorch를 먼저 설치한 뒤 requirements를 설치하세요.

기본 모델은 `yolo26m.pt`이며 입력 크기는 `960`입니다. 첫 실행 시 Ultralytics가 weight를 준비합니다. 필요하면 `ANNOTOOL_IMGSZ` 환경변수로 입력 크기를 바꿀 수 있습니다.

Tracking에는 person ReID 전용 appearance encoder인 `osnet_x1_0_msmt17`을 사용합니다. BoxMOT가 첫 실행 시 weight를 자동으로 준비하며 이후 로컬 캐시를 재사용합니다.

다른 Ultralytics detection model을 사용하려면 환경변수로 지정할 수 있습니다.

```bash
ANNOTOOL_YOLO_MODEL=yolo26m.pt python qt.py
```

Windows PowerShell:

```powershell
$env:ANNOTOOL_YOLO_MODEL="yolo26m.pt"
python qt.py
```

### Fisheye camera prototype

기본 동작은 기존 일반 카메라 영상과 동일합니다. 보정된 fisheye 입력을 사용하려면 OpenCV fisheye calibration 결과를 `.npz`로 저장하고 환경변수로 지정합니다.

필수 배열은 `K` (3x3 camera matrix)와 `D` (4 fisheye distortion coefficients)이며, calibration 해상도를 알고 있으면 `DIM=[width, height]`도 함께 저장하는 것을 권장합니다.

```python
np.savez(
    "fisheye_calib.npz",
    K=K,
    D=D,
    DIM=np.array([width, height]),
)
```

Windows PowerShell:

```powershell
$env:ANNOTOOL_FISHEYE_CALIB="C:\path\to\fisheye_calib.npz"
$env:ANNOTOOL_FISHEYE_BALANCE="0.2"
python qt.py
```

`ANNOTOOL_FISHEYE_CALIB`을 지정하지 않으면 전처리는 완전히 비활성화됩니다. 활성화하면 해상도별 undistortion map을 한 번 생성해 캐시하고, 이후 화면 표시·detection·tracking·저장 crop에 동일한 `cv2.remap` 결과를 사용합니다. `ANNOTOOL_FISHEYE_BALANCE`는 기본 `0.2`이며 0에 가까울수록 검은 경계를 줄이는 쪽, 1에 가까울수록 원래 FOV를 더 보존하는 쪽입니다.

2020년 당시 TensorFlow / CUDA 환경은 `requirements-legacy.txt`에 보존했습니다.

## 주요 UI

| 기능 | 설명 |
|---|---|
| File | annotation할 영상 선택 |
| Load | 첫 frame에 detection / tracking box 번호 overlay 표시 |
| Object | 사용자 정의 object 이름과 초기 box 번호 지정 |
| Box No. | 화면의 현재 box 번호만 변경; object 이름/저장 identity는 유지 |
| Start / Pause / Resume | tracking worker 최초 시작과 재생/일시정지를 하나의 버튼에서 처리 |
| Arrow keys | 재생 속도 조절 (`0.1x` 단위, 최저 `0.1x`) |
| Make JSON | 현재 기록을 JSON으로 저장 |
| Delete | 선택된 기록 삭제 |
| Action Start / End | 액션명을 직접 입력해 시작/종료 프레임을 구간 라벨로 기록 |
| Action Snapshot | 현재 프레임을 단일 액션 라벨로 저장 |
| Show target only | 선택한 target만 표시 |
| Open folder | 저장 폴더 열기 |
| Reset | 현재 작업 초기화 |

각 버튼의 주요 shortcut은 GUI 버튼에 함께 표시됩니다.

현재 shortcut:

```text
F          File
L          Load
O          Object
Space      Start / Pause / Resume
Q          Reset
C          Box No.
← / →      Speed -0.1x / +0.1x
Home       Open folder
Tab        Target-only view
J          Make JSON
Delete     Delete selected annotation
B          Action Start / End
N          Action Snapshot
Ctrl+Q     Quit
```

기존 W/R/S 고정 action shortcut은 custom action 입력 방식으로 전환하면서 제거했습니다.

### Object identity and box 번호

`Object`는 사용자가 정하는 논리적인 이름입니다. 예를 들어 `person_A`, `customer_01`처럼 지정할 수 있고 결과 폴더와 JSON 파일도 이 이름을 기준으로 저장됩니다.

화면의 `person 10` 같은 숫자는 tracker가 현재 부여한 ID입니다. ID가 바뀌면 `Box No.`만 수정하며 object 이름과 annotation workspace는 유지됩니다.

영상 파일을 선택하면 detector를 기다리지 않고 raw 첫 frame을 먼저 표시합니다. 이후 `Load`를 누르면 첫 frame에 tracker ID가 overlay됩니다.

### Custom action annotation

`Action Start (B)` / `Action End`는 구간 액션을 기록합니다. `Action Snapshot (N)`은 현재 프레임 하나에 액션명을 바로 붙이는 단일-frame annotation입니다.

Snapshot은 구간이 아니라 액션명 자체를 label로 저장합니다. 예를 들어 `jump`를 입력하면 해당 프레임의 label은 `jump`가 됩니다.

`Action Start (B)`를 누르면 액션명을 직접 입력합니다. 입력을 확정한 현재 프레임이 `start_<action>`으로 저장되고 버튼은 해당 액션의 `Action End` 상태로 바뀝니다. 종료 시점에 다시 누르면 `end_<action>`이 저장됩니다.

예:

```text
Action Start → "jump"
  → start_jump

Action End
  → end_jump
```

액션명 입력/종료 시에는 정확한 프레임을 기록하기 위해 영상이 잠시 pause되고, 원래 재생 중이었다면 자동으로 다시 재생됩니다.

### Loading behavior

첫 영상을 선택하면 YOLO26m / OccluBoost / OSNet x1.0 backend를 백그라운드에서 미리 준비합니다. `Load`를 누르면 별도의 진행창에서 model loading, GPU / tracker initialization, first-frame tracking 단계를 표시합니다. 모델과 predictor는 애플리케이션 세션 동안 재사용하고, 영상 변경이나 seek 시에는 tracker state만 reset합니다.

### Playback and window scaling

- 기본 재생 속도는 `1.0x`이며 tracking 시작 전부터 좌/우 화살표로 `0.1x` 단위로 미리 조절할 수 있습니다.
- 최저 속도는 `0.1x`이며 `1.1x`, `1.7x` 같은 fractional speed도 지원합니다.
- 메인 창의 모서리/테두리를 드래그하면 영상 영역과 컨트롤 배치가 함께 확대·축소됩니다.
- 영상 자체는 화면 비율을 유지해 표시합니다.

## Output

annotation 결과는 `captured/` 아래에 object 단위 폴더로 저장됩니다.

- 선택된 frame 이미지
- frame 번호 / label
- action 시작 / 종료 정보
- JSON annotation

같은 object 번호를 다른 영상에서 사용할 경우 기존 폴더와 충돌하지 않도록 별도 폴더가 생성됩니다.

## Legacy implementation

2020년 원본 pipeline:

```text
Video
  → TensorFlow YOLOv4
  → Deep SORT
  → PyQt annotation UI
```

당시 환경:

- Ubuntu 18.04
- CUDA 10.1
- cuDNN 7.6.5
- TensorFlow / TensorFlow-GPU 2.3.1
- OpenCV 4.1.2
- PyQt5 5.15.1

관련 source와 dependency 기록은 현재 repository와 `requirements-legacy.txt`에 보존되어 있습니다.

## Upstream / licenses

현재 runtime:

- [Ultralytics](https://github.com/ultralytics/ultralytics) — YOLO26 detector, AGPL-3.0
- [BoxMOT](https://github.com/mikel-brostrom/boxmot) — OccluBoost tracker + OSNet ReID integration, AGPL-3.0

Legacy backend:

- [PyLessons TensorFlow-2.x-YOLOv3](https://github.com/pythonlessons/TensorFlow-2.x-YOLOv3) — YOLOv3 / YOLOv4 TensorFlow implementation, MIT
- [nwojke/deep_sort](https://github.com/nwojke/deep_sort) — Deep SORT, GPL-3.0
- [AlexeyAB/darknet](https://github.com/AlexeyAB/darknet) — YOLOv4 Darknet weights / implementation

프로젝트 라이선스는 AGPL-3.0이며, third-party provenance와 원문 라이선스 사본은 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)에 정리했습니다.

## Notes

이 프로젝트의 핵심은 detector나 tracker 자체를 새로 제안하는 것이 아니라, object tracking 결과를 사람이 빠르게 검토하고 frame / action annotation으로 기록할 수 있도록 만든 GUI와 interaction workflow입니다.

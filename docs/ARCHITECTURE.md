# Architecture

## Package layout

Annotool은 `src` layout을 사용하는 Python package입니다.

```text
run.py
  ↓
annotool.app
  ├─ annotool.tracking
  │    └─ annotool.camera
  ├─ annotool.camera
  └─ annotool.storage
```

실제 소스와 UI asset은 모두 `src/annotool/` 아래에 있으며, 루트의 `run.py`는 실행 진입점만 담당합니다.

## Entry point

### `run.py`

```python
from annotool.app import run_app
```

`app.py`는 tracking backend를 PyQt보다 먼저 import하므로 Windows에서 필요한 PyTorch DLL load order를 유지합니다.

## Application

### `src/annotool/app.py`

PyQt GUI와 annotation workflow를 담당합니다.

- video open / first-frame preview
- playback, pause, fractional speed
- slider preview / committed seek
- Object name과 current tracker ID 관리
- action interval / snapshot 입력
- worker thread → Qt signal 기반 UI update
- annotation session UI synchronization

Qt Designer asset은 package 내부의 `assets/annotool.ui`를 현재 module 경로 기준으로 불러옵니다.

## Tracking

### `src/annotool/tracking.py`

`PersonTracker`가 detection과 identity association을 하나의 interface로 제공합니다.

```text
BGR frame
  → YOLO26m person detections
  → duplicate detection suppression
  → OccluBoost association
  → OSNet x1.0 MSMT17 appearance embedding
  → [x1, y1, x2, y2, track_id, class_id]
```

기본 detector 설정:

- model: `yolo26m.pt`
- inference size: 960
- person confidence threshold: 0.05
- YOLO NMS IoU: 0.50

OccluBoost는 annotation/sports workflow에 맞게 조정합니다.

- low-confidence detection을 association 후보로 유지
- 별도의 new-track threshold 적용
- duplicate track suppression
- `lambda_shape=0.20`
- `ams_enabled=False`
- OSNet x1.0 appearance embedding

## Camera preprocessing

### `src/annotool/camera.py`

Normal mode에서는 no-op입니다. Fisheye mode에서는 OpenCV `fisheye` API로 undistortion map을 만들고 `cv2.remap()`을 적용합니다.

보정 map은 해상도별로 캐시되며 display, detection, tracking, saved crop이 같은 좌표계를 사용합니다.

## Annotation storage

### `src/annotool/storage.py`

저장 단위는 Object가 아니라 **video + labeling date**입니다.

```text
captured/
└─ <video_stem>_<YYYY-MM-DD>/
   ├─ images/
   └─ metadata.json
```

세션 directory는 첫 annotation 시점까지 생성하지 않습니다.

각 annotation은 다음 형태로 즉시 저장됩니다.

```json
{
  "video_title": "nfl.mp4",
  "object_name": "quarterback",
  "action": "start_throw",
  "frame_number": 153
}
```

동일 `object_name + frame_number`를 다시 기록하면 record를 교체합니다. 여러 Object는 같은 세션 metadata에 함께 저장되고, Delete는 image와 metadata를 함께 갱신합니다.

## Object identity

`Object`와 tracker ID는 서로 다른 개념입니다.

- **Object name:** persistent annotation identity
- **Box No.:** 현재 sequence의 transient tracker ID

긴 occlusion이나 seek 이후 track ID가 달라져도 Object name은 유지됩니다.

## Short detector misses

BoxMOT가 내부적으로 유지하는 unmatched track을 `ANNOTOOL_COAST_FRAMES` 범위에서 predicted bbox로 잠깐 표시해 짧은 detector flicker를 완화합니다. 기본값은 2 frame입니다.

## Seek semantics

Pause 상태의 slider interaction은 두 단계입니다.

1. **Preview:** slider 이동 중 frame을 빠르게 표시
2. **Commit:** slider release 시 tracker reset 후 해당 frame을 실제 detection/tracking

이 분리로 preview와 release request의 race를 피합니다.

## Threading

모델 준비와 video tracking은 worker thread에서 수행합니다. Worker는 Qt widget을 직접 수정하지 않고 signal로 다음 상태를 전달합니다.

- frame position
- button enabled state
- rendered QImage
- pause state
- video end

QPixmap 생성과 widget mutation은 GUI thread에서 수행합니다.

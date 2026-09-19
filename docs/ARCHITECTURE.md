# Architecture

## 1. Entry point

### `run.py`

실행 전용 entry point입니다.

```text
run.py
  → import app
  → app.run_app()
```

`app.py`가 tracking backend를 PyQt보다 먼저 import하므로 Windows에서 PyTorch DLL load order도 유지됩니다.

## 2. Application

### `app.py`

PyQt GUI와 annotation workflow를 담당합니다.

- video open / first-frame preview
- playback, pause, fractional speed
- slider preview / committed seek
- Object name과 current tracker ID 관리
- action interval / snapshot 입력
- worker thread → Qt signal 기반 UI update
- annotation session UI synchronization

Detection/tracking은 GUI thread에서 실행하지 않습니다.

## 3. Tracking

### `tracking.py`

사람 detection과 identity association을 `PersonTracker` interface로 제공합니다.

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

- 낮은-confidence detection을 association 후보로 유지
- 별도의 new-track threshold 적용
- duplicate track suppression 강화
- `lambda_shape=0.20`: 큰 pose/aspect-ratio 변화 허용
- `ams_enabled=False`: crouch/tackle/fall 같은 급격한 bbox 변화 허용
- OSNet x1.0 appearance embedding 유지

## 4. Camera preprocessing

### `camera.py`

Normal mode에서는 no-op입니다. Fisheye mode에서는 OpenCV `fisheye` API로 undistortion map을 생성하고 `cv2.remap()`을 적용합니다.

보정 map은 해상도별로 캐시하며 다음 단계가 모두 같은 보정 frame을 사용합니다.

- display
- detection
- tracking
- saved action crop

## 5. Annotation storage

### `storage.py`

저장 단위는 Object가 아니라 **video + labeling date**입니다.

```text
captured/
└─ <video_stem>_<YYYY-MM-DD>/
   ├─ images/
   └─ metadata.json
```

세션 directory는 첫 annotation 시점까지 생성하지 않습니다.

각 annotation은 다음 record로 즉시 persistence됩니다.

```json
{
  "video_title": "nfl.mp4",
  "object_name": "quarterback",
  "action": "start_throw",
  "frame_number": 153
}
```

따라서 별도의 수동 JSON export 단계가 없습니다.

동일 `object_name + frame_number`를 다시 기록하면 해당 record를 교체합니다. 여러 Object는 같은 세션 metadata에 함께 저장됩니다. Delete는 image와 metadata를 함께 갱신합니다.

## 6. Object identity

`Object`와 tracker ID는 별개입니다.

- **Object name:** 사용자가 정하는 persistent annotation identity
- **Box No.:** 현재 sequence에서 tracker가 부여한 transient ID

긴 occlusion이나 seek 이후 track ID가 달라져도 Object name은 유지됩니다. 사용자는 Box No.만 변경합니다.

## 7. Short detector misses

BoxMOT는 unmatched track을 내부적으로 유지하지만 현재 frame에 update되지 않은 track을 바로 출력하지 않습니다. Annotool은 최대 `ANNOTOOL_COAST_FRAMES` frame 동안 predicted bbox를 표시해 짧은 detection flicker를 완화합니다.

이 값은 장기 예측용이 아닙니다. 지나치게 크게 설정하면 빠른 움직임에서 ghost box가 늘 수 있습니다.

## 8. Seek semantics

Pause 상태의 slider interaction은 두 단계로 나눕니다.

1. **Preview:** slider 이동 중 frame을 빠르게 표시
2. **Commit:** slider release 시 tracker를 reset하고 해당 frame을 detection/tracking

preview request와 release request를 분리해 seek 직후 box가 사라지는 race를 피합니다.

## 9. Threading

모델 준비와 video tracking은 worker thread에서 수행합니다. Worker는 Qt widget을 직접 수정하지 않고 signal로 다음 상태만 전달합니다.

- frame position
- button enabled state
- rendered QImage
- pause state
- video end

QPixmap 생성과 실제 widget mutation은 GUI thread에서 수행합니다.

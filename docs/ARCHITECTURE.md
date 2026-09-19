# Architecture

## 1. Modules

### `app.py`

PyQt GUI와 annotation state를 담당합니다.

- video open / first-frame preview
- playback, pause, fractional speed
- slider preview / committed seek
- Object name과 current tracker ID 관리
- action interval / snapshot 저장
- JSON export
- worker thread → Qt signal 기반 UI update

Detection/tracking은 GUI thread에서 실행하지 않습니다.

### `tracking.py`

사람 detection과 identity association을 하나의 `PersonTracker` interface로 제공합니다.

```text
BGR frame
  → YOLO26m person detections
  → duplicate detection suppression
  → OccluBoost association
  → OSNet x1.0 MSMT17 appearance embedding
  → [x1, y1, x2, y2, track_id, class_id]
```

GUI는 backend 내부 representation을 알 필요 없이 위 6-field track format만 사용합니다.

### `camera.py`

기본적으로 no-op입니다. Fisheye calibration이 활성화된 경우에만 OpenCV `fisheye` API로 undistortion map을 생성하고 `cv2.remap()`을 적용합니다.

## 2. Tracking configuration

기본 detector:

- model: `yolo26m.pt`
- inference size: 960
- person confidence threshold: 0.05
- YOLO NMS IoU: 0.50

OccluBoost는 일반 MOT benchmark의 기본 설정을 그대로 사용하지 않고 annotation/sports workflow에 맞게 조정합니다.

- 낮은 confidence detection을 association 후보로 유지
- 새 track 생성 threshold는 별도로 제한
- duplicate track suppression 강화
- `lambda_shape=0.20`: 큰 pose/aspect-ratio 변화 허용
- `ams_enabled=False`: crouch/tackle/fall 같은 정상적인 급격한 bbox 변화 허용
- OSNet x1.0 appearance embedding 유지

## 3. Short detector misses

BoxMOT는 unmatched track을 내부적으로 유지하지만 현재 frame에 update되지 않은 track을 바로 출력하지 않습니다. Annotool은 최대 `ANNOTOOL_COAST_FRAMES` frame 동안 Kalman-predicted bbox를 표시해 1~2 frame detection flicker를 완화합니다.

이 값은 장기 예측을 위한 것이 아닙니다. 너무 크게 설정하면 빠른 움직임에서 ghost box가 늘 수 있습니다.

## 4. Seek semantics

Pause 상태의 slider interaction은 두 단계로 분리합니다.

1. **Preview:** 사용자가 slider를 움직이는 동안 raw frame을 빠르게 표시
2. **Commit:** slider를 놓는 순간 tracker를 reset하고 해당 frame을 실제 detection/tracking

이 구조는 preview 요청과 release 요청의 race를 피하면서 seek 직후 올바른 box를 보여주기 위한 것입니다.

## 5. Object identity

`Object`와 tracker ID는 다른 개념입니다.

- **Object name:** 사용자가 지정하는 persistent annotation identity
- **Box No.:** 현재 frame sequence에서 tracker가 부여한 transient track ID

긴 occlusion이나 seek 이후 ID가 달라져도 Object name은 유지됩니다. 사용자는 Box No.만 변경합니다.

## 6. Annotation storage

Object 선택 시에는 directory를 만들지 않습니다. 실제 annotation 이미지가 처음 기록될 때 output directory를 lazy-create합니다.

Action interval:

```text
start_<action>
end_<action>
```

Snapshot:

```text
<action>
```

동일 frame에는 하나의 annotation만 유지합니다.

## 7. Threading

모델 준비와 video tracking은 worker thread에서 수행합니다. Worker는 Qt widget을 직접 수정하지 않고 signal을 통해 다음 상태만 전달합니다.

- frame position
- button enabled state
- rendered QImage
- pause state
- video end

QPixmap 생성과 실제 widget mutation은 GUI thread에 남깁니다.

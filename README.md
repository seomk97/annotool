# Annotool

사람 중심 비디오 액션 라벨링을 위한 PyQt5 데스크톱 도구입니다. 영상에서 사람을 자동 검출·추적하고, 사용자가 하나의 논리적 Object를 선택해 액션 구간 또는 단일 프레임을 기록할 수 있습니다.

## 주요 기능

- **Person detection:** Ultralytics YOLO26m, 기본 입력 크기 960
- **Multi-object tracking:** BoxMOT OccluBoost
- **Person ReID:** OSNet x1.0 MSMT17
- **Object / track ID 분리:** Object 이름은 저장 identity로 유지하고 Box No.는 현재 tracker ID만 변경
- **Action interval:** `Action Start / End`로 `start_<action>`, `end_<action>` 기록
- **Action snapshot:** 현재 프레임을 단일 액션 라벨로 저장
- **Seek-aware workflow:** pause 상태에서 slider preview와 최종 seek tracking을 분리
- **Short miss bridging:** detector가 1~2 frame 놓쳐도 tracker prediction으로 짧게 연결
- **Fisheye prototype:** OpenCV calibration `.npz`를 이용한 optional undistortion
- **Lazy output creation:** 실제 액션을 하나 이상 저장할 때만 Object 출력 폴더 생성

## Architecture

```text
Video
  │
  ├─ Normal ─────────────────────┐
  │                              │
  └─ Fisheye calibration         │
       └─ cv2.remap() ───────────┤
                                 ▼
                         YOLO26m person detector
                                 │
                                 ▼
                         OccluBoost tracker
                                 │
                         OSNet x1.0 ReID
                                 │
                                 ▼
                         PyQt annotation UI
                                 │
                  ┌──────────────┴──────────────┐
                  ▼                             ▼
             action crops                  annotation JSON
```

세부 설계는 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)를 참고하세요.

## Repository

```text
.
├─ app.py                    # GUI, playback, seek, annotation workflow
├─ tracking.py               # YOLO26m + OccluBoost + OSNet adapter
├─ camera.py                 # normal/fisheye frame preprocessing
├─ assets/
│  └─ annotool.ui            # Qt Designer UI
├─ docs/
│  └─ ARCHITECTURE.md
├─ test_samples/             # sample videos
├─ requirements.txt
├─ THIRD_PARTY_NOTICES.md
└─ LICENSE
```

## Installation

Python 3.10+ 환경을 권장합니다.

### uv

```powershell
git clone https://github.com/seomk97/annotool.git
cd annotool

uv venv
uv pip install -r requirements.txt
uv run python app.py
```

### venv / pip

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

첫 실행 시 Ultralytics와 BoxMOT가 필요한 model weight를 준비할 수 있습니다.

## Workflow

1. **File (F)** 에서 영상을 선택합니다.
2. `Normal / Fisheye` camera mode를 선택합니다.
3. **Load (L)** 로 첫 프레임 detection / tracking ID를 확인합니다.
4. **Object (O)** 에서 저장할 Object 이름과 현재 **Box No.** 를 지정합니다.
5. **Space** 로 tracking을 시작하거나 pause/resume 합니다.
6. target ID가 바뀌면 **Box No. (C)** 로 현재 tracker ID만 수정합니다.
7. **Action Start / End (B)** 또는 **Action Snapshot (N)** 으로 annotation을 기록합니다.
8. 필요하면 **Make JSON (J)** 으로 현재 workspace를 JSON으로 저장합니다.

## Output

Object를 선택하는 것만으로는 폴더가 생성되지 않습니다. 첫 액션 이미지가 실제로 저장되는 순간 다음 구조가 생성됩니다.

```text
captured/
└─ <object_name>/
   ├─ 153.jpg
   ├─ 284.jpg
   └─ <object_name>.json
```

같은 이름의 폴더가 이미 있으면 `_1`, `_2`처럼 suffix를 붙입니다.

JSON은 frame number를 key, action label을 value로 저장합니다.

```json
{
  "153": "start_throw",
  "284": "end_throw",
  "417": "catch"
}
```

## Keyboard shortcuts

| Key | Action |
|---|---|
| F | File |
| L | Load |
| O | Object |
| Space | Start / Pause / Resume |
| Q | Reset |
| C | Box No. |
| ← / → | Speed - / + 0.1x |
| Home | Open output folder |
| Tab | Show target only |
| J | Make JSON |
| Delete | Delete selected annotation |
| B | Action Start / End |
| N | Action Snapshot |
| Ctrl+Q | Quit |

## Fisheye mode

Fisheye를 선택하면 OpenCV fisheye calibration `.npz` 파일을 요청합니다.

필수 값:

- `K`: 3×3 camera matrix
- `D`: 4개의 fisheye distortion coefficients
- `DIM`: calibration resolution `[width, height]` — 권장

예시:

```python
np.savez(
    "fisheye_calib.npz",
    K=K,
    D=D,
    DIM=np.array([width, height]),
)
```

보정 map은 해상도별로 한 번 생성한 뒤 캐시합니다. 화면 표시, detection, tracking, action crop 모두 같은 보정 프레임을 사용합니다.

## Runtime configuration

환경변수로 기본값을 조정할 수 있습니다.

| Variable | Default | Description |
|---|---:|---|
| `ANNOTOOL_YOLO_MODEL` | `yolo26m.pt` | Ultralytics detector |
| `ANNOTOOL_REID_MODEL` | `osnet_x1_0_msmt17.pt` | BoxMOT ReID model |
| `ANNOTOOL_IMGSZ` | `960` | detector inference size |
| `ANNOTOOL_DEVICE` | CUDA if available | inference device |
| `ANNOTOOL_COAST_FRAMES` | `2` | short detector-miss bridging |
| `ANNOTOOL_FISHEYE_CALIB` | empty | default fisheye calibration path |
| `ANNOTOOL_FISHEYE_BALANCE` | `0.2` | fisheye FOV/crop balance |
| `ANNOTOOL_DEBUG_FPS` | off | per-frame FPS console logging |

## Tracking notes

현재 backend는 사람 class만 추적합니다. 작은 선수나 겹침이 많은 스포츠 영상에서 recall을 확보하도록 detector threshold를 낮게 유지하고, 중복 detection suppress 및 short coast를 추가했습니다. 큰 자세 변화가 잦은 영상을 고려해 bbox shape 가중치는 낮추고 OccluBoost AMS는 비활성화했습니다.

온라인 tracker 특성상 긴 가림, scene cut, seek 이후에는 track ID가 바뀔 수 있습니다. 이 경우 Object 이름은 그대로 유지한 채 **Box No.** 만 새 ID로 바꾸면 됩니다.

## License

이 프로젝트는 [AGPL-3.0](LICENSE)으로 배포됩니다. 현재 runtime에서 사용하는 외부 프로젝트와 model lineage는 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)에 정리되어 있습니다.

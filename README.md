# Annotool

사람 중심 비디오 액션 라벨링을 위한 PyQt5 데스크톱 도구입니다. YOLO26m으로 사람을 검출하고, BoxMOT OccluBoost + OSNet ReID로 identity를 추적하며, 사용자가 선택한 Object의 액션을 프레임 단위로 기록합니다.

## Features

- YOLO26m person detection, 기본 inference size 960
- BoxMOT OccluBoost multi-object tracking
- OSNet x1.0 MSMT17 person ReID
- Object identity와 transient tracker ID(Box No.) 분리
- Action Start / End 및 Action Snapshot
- annotation 즉시 image + metadata 저장
- pause slider preview / committed seek 분리
- 짧은 detector miss를 tracker prediction으로 연결
- OpenCV fisheye calibration 기반 optional 보정
- 첫 annotation 전까지 output directory를 만들지 않는 lazy storage

## Repository layout

```text
.
├─ run.py
├─ pyproject.toml
├─ src/
│  └─ annotool/
│     ├─ __init__.py
│     ├─ app.py
│     ├─ tracking.py
│     ├─ camera.py
│     ├─ storage.py
│     └─ assets/
│        └─ annotool.ui
├─ docs/
│  └─ ARCHITECTURE.md
├─ test_samples/
├─ README.md
├─ THIRD_PARTY_NOTICES.md
└─ LICENSE
```

`run.py`는 실행 진입점만 담당하고, 실제 애플리케이션 코드는 `src/annotool/` 패키지 안에 있습니다.

## Installation

### uv

```powershell
git clone https://github.com/seomk97/annotool.git
cd annotool

uv sync
uv run python run.py
```

### pip

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
python run.py
```

의존성의 단일 기준은 `pyproject.toml`입니다.

## Workflow

1. **File (F)** 에서 영상을 선택합니다.
2. `Normal / Fisheye` camera mode를 선택합니다.
3. **Load (L)** 로 첫 프레임의 detection / tracking ID를 확인합니다.
4. **Object (O)** 에서 Object 이름과 현재 **Box No.** 를 지정합니다.
5. **Space** 로 tracking을 시작하거나 pause/resume 합니다.
6. target ID가 바뀌면 **Box No. (C)** 로 tracker ID만 수정합니다.
7. **Action Start / End (B)** 또는 **Action Snapshot (N)** 으로 annotation을 기록합니다.
8. annotation이 생길 때마다 crop image와 `metadata.json`이 즉시 갱신됩니다.

## Output

저장 단위는 Object별 폴더가 아니라 **영상 제목 + 라벨링 날짜 세션**입니다.

```text
captured/
└─ nfl_2026-09-19/
   ├─ images/
   │  ├─ quarterback_00000153.jpg
   │  ├─ quarterback_00000284.jpg
   │  └─ receiver_00000417.jpg
   └─ metadata.json
```

Object를 선택하거나 tracking만 하는 동안에는 세션 디렉토리를 만들지 않습니다. 첫 annotation이 저장될 때 생성됩니다.

`metadata.json`은 라벨링 1회당 record 1개를 저장합니다.

```json
[
  {
    "video_title": "nfl.mp4",
    "object_name": "quarterback",
    "action": "start_throw",
    "frame_number": 153
  },
  {
    "video_title": "nfl.mp4",
    "object_name": "receiver",
    "action": "catch",
    "frame_number": 417
  }
]
```

같은 영상 제목을 같은 날짜에 다시 열면 같은 세션을 이어서 사용합니다. 같은 Object의 같은 frame을 다시 기록하면 해당 record를 교체하며, Delete는 image와 metadata record를 함께 제거합니다.

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
| Home | Open current output/session folder |
| Tab | Show target only |
| Delete | Delete selected annotation |
| B | Action Start / End |
| N | Action Snapshot |
| Ctrl+Q | Quit |

## Fisheye mode

Fisheye를 선택하면 OpenCV fisheye calibration `.npz` 파일을 선택합니다.

필수 값:

- `K`: 3×3 camera matrix
- `D`: 4 fisheye distortion coefficients
- `DIM`: calibration resolution `[width, height]` — 권장

```python
np.savez(
    "fisheye_calib.npz",
    K=K,
    D=D,
    DIM=np.array([width, height]),
)
```

보정 map은 해상도별로 캐시하며 display, detection, tracking, saved crop이 모두 동일한 보정 프레임을 사용합니다.

## Runtime configuration

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

현재 backend는 person class만 추적합니다. 작은 선수와 겹침이 많은 스포츠 영상의 recall을 확보하도록 detector threshold를 낮게 유지하고, 중복 detection suppression과 short coast를 적용합니다. 큰 자세 변화가 잦은 영상을 고려해 bbox shape 가중치는 낮추고 OccluBoost AMS는 비활성화했습니다.

온라인 tracker 특성상 긴 가림, scene cut, seek 이후에는 track ID가 바뀔 수 있습니다. 이 경우 Object 이름은 유지한 채 **Box No.** 만 새 ID로 변경하면 됩니다.

세부 구현은 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)를 참고하세요.

## License

이 프로젝트는 [AGPL-3.0](LICENSE)으로 배포됩니다. 현재 runtime의 외부 프로젝트와 model lineage는 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)에 정리되어 있습니다.

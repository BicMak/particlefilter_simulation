# particle_viewer

손으로 그린 평면도 위에서 로봇을 방향키로 직접 움직이면서, **파티클 필터가 로봇 위치를 어떻게 추정하는지** 실시간으로 눈으로 확인하는 시뮬레이터입니다.

[PythonRobotics](https://github.com/AtsushiSakai/PythonRobotics)의 Particle Filter 예제(Atsushi Sakai)를 바탕으로, 맵 / 로봇 / 설정을 클래스로 분리하고 이미지 맵과 키보드 조작을 추가했습니다.

<!-- TODO: 실행 화면 스크린샷 또는 GIF -->

## 무엇을 볼 수 있나

- 로봇은 레퍼런스 포인트(벽 위의 점)까지의 **거리만** 관측합니다. 위치는 모릅니다.
- 화면에는 아래가 같이 그려집니다.

| 표시 | 의미 |
|---|---|
| 파란 삼각형 / 파란 점선 | 실제 로봇(ground truth)과 그 궤적 |
| 초록 점 + 초록 선 | 파티클의 위치와 진행 방향 |
| 빨간 선 | 파티클 필터의 추정 궤적 |
| 빨간 점선 타원 | 추정 위치의 불확실성 (x, y 공분산의 1σ 타원) |
| 연한 분홍 부채꼴 | 로봇의 시야각 (이 안의 레퍼런스만 관측됨) |
| 검은 점선 | 로봇 → 관측된 레퍼런스 포인트 |
| 검은 별 | 레퍼런스 포인트 (지도) |

## 파일 구성

```
particle_viewer/
├── particle_filter.py   # 메인. Robot(파티클 필터 포함), KeyboardController, main()
├── map.py               # 지도 생성. Map(사각형 방), GridMap(평면도 이미지), build_map()
├── config.py            # 모든 설정값 (dataclass)
├── input_image/         # 입력 평면도 이미지 (기본: Untitled.png)
└── output_image/        # map.py 결과 이미지 (기본: map.png)
```

## 설치

Python 3.9 이상을 권장합니다 (`dataclasses`, 타입 힌트 사용).

```bash
pip install numpy matplotlib opencv-python
```

키보드 입력을 받으려면 창이 뜨는 matplotlib 백엔드(TkAgg, QtAgg 등)가 필요합니다. 원격 서버나 헤드리스 환경에서는 동작하지 않습니다.

## 실행

```bash
python particle_filter.py
```

기본 설정은 `input_image/Untitled.png` 평면도를 맵으로 쓰고, 방향키로 로봇을 조작하는 모드입니다.

### 조작법

방향키를 **누르고 있는 동안** 입력이 들어갑니다.

| 키 | 동작 |
|---|---|
| ↑ / ↓ | 전진 / 후진 가속 (키를 떼면 서서히 정지) |
| ← / → | 좌회전(반시계) / 우회전 (키를 떼면 각속도 0으로 복귀) |
| space | 즉시 정지 |
| esc | 종료 |

가속도와 최대 속도는 `config.py`의 `KeyboardConfig`에서 바꿀 수 있습니다.

### 키보드 없이 실행

`KeyboardConfig.enabled = False`로 두면 일정한 속도와 각속도로 원을 그리며 움직이고, `sim_time`(기본 50초) 뒤에 끝납니다.

### 설정 바꿔서 실행

```python
from config import Config, RobotConfig, VizConfig
from particle_filter import main

main(Config(robot=RobotConfig(particle=500), viz=VizConfig(trail_time=None)))
```

## 설정

모든 값은 [config.py](config.py)에 dataclass로 모여 있습니다. 자주 만지는 값만 추렸습니다.

| 설정 | 기본값 | 설명 |
|---|---|---|
| `RobotConfig.particle` | 100 | 파티클 개수 |
| `RobotConfig.fov` | 60° | 시야각. 로봇 진행 방향 기준 좌우 `fov/2`만 관측 |
| `RobotConfig.max_range` | ∞ | 최대 관측 거리 [m] |
| `RobotConfig.Q` | σ = 0.5 m | **필터가 믿는** 거리 센서 노이즈. 가중치 계산에 사용 |
| `RobotConfig.R` | σ = 0.8 m/s, 45°/s | **필터가 믿는** 입력(속도, 각속도) 노이즈. 파티클을 퍼뜨리는 크기 |
| `SimNoiseConfig.Q` | σ = 0.02 m | 시뮬레이션에서 **실제로** 거리 센서에 섞이는 노이즈 |
| `SimNoiseConfig.R` | σ = 0.05 m/s, 2°/s | 시뮬레이션에서 **실제로** 입력(엔코더, IMU)에 섞이는 노이즈 |
| `ImageMapConfig.use_image_map` | True | True: 이미지 맵, False: 사각형 방(`RoomConfig`) |
| `ImageMapConfig.resolution` | 0.03 | 이미지 한 픽셀의 실제 길이 [m/px] |
| `ImageMapConfig.robot_start` | `[15, 7.5, 0, 0]` | 로봇 시작 `[x, y, yaw, v]` |
| `VizConfig.trail_time` | 10초 | 궤적을 남기는 시간 (`None`이면 안 지움) |
| `VizConfig.particle_draw_ratio` | 0.2 | 전체 파티클 중 화면에 그릴 비율 (그리기 전용) |

`RobotConfig`의 `Q`, `R`(필터의 믿음)과 `SimNoiseConfig`의 `Q`, `R`(실제 오차)은 **일부러 따로** 두었습니다. 필터는 실제 오차 크기를 모르기 때문에, 보통 필터 쪽을 더 크게 잡습니다.

## 동작 원리

한 time step(`dt = 0.1초`)마다 `main()`에서 아래 순서로 진행합니다.

1. **입력 결정**: 키보드(또는 고정값)로 지령 `u = [v, yaw_rate]`를 정합니다.
2. **실제 로봇 이동 + 관측** (`Robot.observation`): 노이즈 없는 `u`로 실제 로봇(`x_true`)이 움직이고, 시야각 안의 레퍼런스까지의 거리에 센서 노이즈를 섞어 관측 `z`를 만듭니다. 엔코더/IMU처럼 노이즈가 섞인 입력 `ud`도 함께 만들어지며, `ud`만으로 적분한 것이 Dead Reckoning(`x_dr`)입니다.
3. **예측** (`predict_particles`): 파티클마다 `ud`에 `R` 크기의 노이즈를 또 섞어서 운동 모델로 이동시킵니다. 이 단계에서 파티클이 퍼집니다.
4. **가중치 갱신** (`calc_delta` → `update_weights`): 각 파티클에서 레퍼런스까지의 예상 거리와 관측 거리의 차이 `delta`를 구하고, `Q`를 폭으로 하는 가우시안 우도로 가중치를 곱합니다. 우도가 0으로 언더플로우하지 않도록 로그 영역에서 계산합니다.
5. **추정 + 리샘플링** (`pf_localization`): 파티클의 가중 평균을 추정 위치로, 가중 공분산을 불확실성으로 계산합니다. 유효 파티클 수가 기준보다 작으면 리샘플링합니다.

상태 벡터는 `[x, y, yaw, v]`이고, 파티클은 `px` (4 × N 배열, 열 하나가 파티클 하나), 가중치는 `pw` (1 × N 배열)로 저장합니다.

## 지도 (map.py)

지도는 모두 `map.py`에서 만듭니다. `build_map(cfg)`가 설정(`ImageMapConfig.use_image_map`)에 따라 아래 둘 중 하나를 돌려줍니다.

- `Map`: 사각형 방. 벽 둘레에서 레퍼런스 포인트를 뽑습니다.
- `GridMap`: 손으로 그린 평면도 이미지를 변환한 지도. 아래 순서로 만들어집니다.

1. 이미지를 흑백으로 읽어서 어두운 픽셀을 벽, 밝은 픽셀을 빈 공간으로 봅니다. 투명 배경은 흰색으로 처리합니다.
2. 선이 살짝 끊긴 곳이 메워지도록 벽을 약간 두껍게 만듭니다.
3. 픽셀 좌표를 월드 좌표(m)로 바꿉니다. 이미지 아래쪽이 `y = 0`입니다.
4. 벽 픽셀에서 레퍼런스 포인트를 랜덤으로 뽑습니다.

`particle_filter.py`에 지도를 연결하면 실제 로봇은 벽을 통과하지 못하고, 벽에 가려진 레퍼런스는 관측되지 않습니다.

단독으로 실행해서 변환 결과만 확인할 수도 있습니다.

```bash
python map.py                               # input_image/Untitled.png -> output_image/map.png
python map.py my_plan.png --res 0.05 --n-ref 100 --show
```

| 옵션 | 설명 |
|---|---|
| `--out` | 결과 이미지 경로 |
| `--res` | 해상도 [m/px] |
| `--n-ref` | 레퍼런스 포인트 개수 |
| `--show` | 결과를 창으로 보기 |

그림은 **벽이 어둡고 배경이 밝은** 단순한 선화가 가장 잘 됩니다. 글자, 치수선, 가구처럼 벽이 아닌 어두운 요소가 있으면 벽으로 인식됩니다.

## 알려진 한계

- 문이나 개구부가 있는 평면도를 방 단위로 자동 분리하지는 않습니다. 벽 픽셀만 뽑아서 쓰며, 방 인식은 아직 없습니다.
- 파티클이 관측과 크게 어긋나면 필터가 실제 위치를 놓칠 수 있습니다. 로봇이 레퍼런스가 거의 안 보이는 곳으로 가면 추정이 실제와 크게 달라지고, 이때 불확실성 타원은 오히려 작게 나올 수 있습니다.
- 처음 시작 위치를 알고 있다고 가정합니다. 파티클이 전부 시작 위치에서 출발하며, 지도 전체에 흩뿌리는 전역 위치 추정(global localization)은 아닙니다.
- 지도(레퍼런스 좌표)는 정확하다고 가정합니다. 노이즈는 거리 측정과 입력에만 섞입니다.

## 참고

- [PythonRobotics](https://github.com/AtsushiSakai/PythonRobotics) — Particle Filter localization (Atsushi Sakai)
- Thrun, Burgard, Fox, *Probabilistic Robotics*, Chapter 4, 5, 8

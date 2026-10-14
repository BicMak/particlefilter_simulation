# particle_viewer

손으로 그린 평면도 위에서 로봇을 방향키로 직접 움직이면서, **파티클 필터가 로봇 위치를 어떻게 추정하는지** 실시간으로 눈으로 확인하는 시뮬레이터입니다.

[PythonRobotics](https://github.com/AtsushiSakai/PythonRobotics)의 Particle Filter 예제(Atsushi Sakai)를 바탕으로, 맵 / 로봇 / 설정을 클래스로 분리하고 이미지 맵과 키보드 조작을 추가했습니다.

<!-- TODO: 실행 화면 스크린샷 또는 GIF -->

## 무엇을 볼 수 있나

- 로봇은 시야각 안에 보이는 벽 위의 점(레퍼런스 포인트)까지의 **거리만** 관측합니다. 위치는 모릅니다.
  (`RobotConfig.n_rays`가 1 이상이면 시야각 안으로 광선을 쏴서 벽에 닿은 점을 관측하고, 0이면 미리 뽑아 둔 고정 레퍼런스 포인트를 관측합니다.)
- 화면에는 아래가 같이 그려집니다.

| 표시 | 의미 |
|---|---|
| 파란 삼각형 / 파란 점선 | 실제 로봇(ground truth)과 그 궤적 |
| 초록 점 + 초록 선 | 파티클의 위치와 진행 방향 |
| 빨간 선 | 파티클 필터의 추정 궤적 |
| 빨간 점선 타원 | 추정 위치의 불확실성 (x, y 공분산의 1σ 타원) |
| 연한 분홍 부채꼴 | 로봇의 시야각 (이 안의 점만 관측됨) |
| 검은 점선 | 지도에 쓰는 위치(`MappingConfig.use_estimated_pose`) → 로봇이 믿는 끝점 |
| 검은 별 | 로봇이 믿는 끝점. 그 위치에서 측정한 거리와 빔 각도로 계산한 점이라, 위치 오차와 거리 노이즈만큼 벽에서 벗어날 수 있음 (고정 레퍼런스 모드에서는 레퍼런스 포인트) |

## 파일 구성

```
particle_viewer/
├── particle_filter.py   # 메인. Robot(파티클 필터 포함), KeyboardController, main()
├── graphslam.py         # 그래프 SLAM. 랜드마크(번호를 아는 레퍼런스) 관측 + 오도메트리로 키프레임 위치를 한꺼번에 최적화
├── icpslam.py           # ICP SLAM. 랜드마크 없이 라이다 스캔끼리 맞춰(ICP) 이동량을 재고, 루프를 닫아 그래프로 최적화
├── map.py               # 지도 생성. Map(사각형 방), GridMap(평면도 이미지), build_map()
├── lidar_to_grid_map.py # 탐색 지도(점유 격자) 작성. MapBuilder
├── config.py            # 모든 설정값 (dataclass)
├── utils/               # 각도 계산, 그리기 보조 함수
├── input_image/         # 입력 평면도/미로 이미지 (기본: doolhof_03.jpg)
└── output_image/        # 결과 이미지 (map.py: map.png, 탐색 지도: explored_map.png)
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

기본 설정은 `input_image/doolhof_03.jpg` 미로 이미지를 맵으로 쓰고, 방향키로 로봇을 조작하는 모드입니다.

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
| `RobotConfig.sigma_range` | 0.5 m | **필터가 믿는** 거리 센서 노이즈. 가중치 계산에 사용 |
| `RobotConfig.sigma_v`, `sigma_yaw_rate` | 0.8 m/s, 45°/s | **필터가 믿는** 입력(속도, 각속도) 노이즈. 파티클을 퍼뜨리는 크기 |
| `SimNoiseConfig.sigma_range` | 0.02 m | 시뮬레이션에서 **실제로** 거리 센서에 섞이는 노이즈 |
| `SimNoiseConfig.sigma_bearing` | 0.5° | 시뮬레이션에서 **실제로** 랜드마크 각도 관측에 섞이는 노이즈 (그래프 SLAM 전용) |
| `SimNoiseConfig.sigma_v`, `sigma_yaw_rate` | 0.05 m/s, 2°/s | 시뮬레이션에서 **실제로** 입력(엔코더, IMU)에 섞이는 노이즈 |
| `ImageMapConfig.use_image_map` | True | True: 이미지 맵, False: 사각형 방(`RoomConfig`) |
| `ImageMapConfig.resolution` | 0.1 | 이미지 한 픽셀의 실제 길이 [m/px] (지도 크기 = 이미지 픽셀 수 × 이 값) |
| `ImageMapConfig.robot_start` | `[15, 7.5, 0, 0]` | 로봇 시작 `[x, y, yaw, v]` |
| `VizConfig.trail_time` | 10초 | 궤적을 남기는 시간 (`None`이면 안 지움) |
| `VizConfig.particle_draw_ratio` | 0.5 | 전체 파티클 중 화면에 그릴 비율 (그리기 전용) |

`RobotConfig`의 `sigma_*`(필터의 믿음)과 `SimNoiseConfig`의 `sigma_*`(실제 오차)는 **일부러 따로** 두었습니다. 필터는 실제 오차 크기를 모르기 때문에, 보통 필터 쪽을 더 크게 잡습니다.

## 동작 원리

한 time step(`dt = 0.1초`)마다 `main()`에서 아래 순서로 진행합니다.

1. **입력 결정**: 키보드(또는 고정값)로 지령 `u = [v, yaw_rate]`를 정합니다.
2. **실제 로봇 이동 + 관측** (`Robot.observation`): 노이즈 없는 `u`로 실제 로봇(`x_true`)이 움직이고, 시야각 안의 레퍼런스까지의 거리에 센서 노이즈를 섞어 관측 `z`(각 행: `[노이즈 낀 거리, 레퍼런스 x, 레퍼런스 y, 빔 각도]`)를 만듭니다. 엔코더/IMU처럼 노이즈가 섞인 입력 `ud`도 함께 만들어지며, `ud`만으로 적분한 것이 Dead Reckoning(`x_dr`)입니다.
3. **예측** (`predict_particles`): 파티클마다 `ud`에 `sigma_v`, `sigma_yaw_rate` 크기의 노이즈를 또 섞어서 운동 모델로 이동시킵니다. 이 단계에서 파티클이 퍼집니다.
4. **가중치 갱신** (`calc_delta` → `update_weights`): 각 파티클에서 레퍼런스까지의 예상 거리와 관측 거리의 차이 `delta`를 구하고, `sigma_range`를 폭으로 하는 가우시안 우도로 가중치를 곱합니다. 우도가 0으로 언더플로우하지 않도록 로그 영역에서 계산합니다.
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

## 지도 작성 복습용 자료 (lidar_to_grid_map.py)

코드에 나온 개념별로 어디를 보면 되는지만 적어 둔 목록입니다.

| 개념 | 코드 위치 | 볼 자료 |
|---|---|---|
| 점유 격자 지도, log-odds 갱신, inverse sensor model (`L_FREE`, `L_OCC`) | `MapBuilder.update` | *Probabilistic Robotics* Chapter 9 (Occupancy Grid Mapping) |
| 거리 센서가 어떻게 측정되는지 (beam model) | `Robot.observation`, `map.py`의 `raycast` | *Probabilistic Robotics* Chapter 6 (Robot Perception) |
| 직선이 지나가는 칸 구하기 | `bresenham` | Wikipedia "Bresenham's line algorithm"; Bresenham, "Algorithm for computer control of a digital plotter", IBM Systems Journal, 1965 |
| 점유 격자의 원전 | `MapBuilder` 전체 | Moravec & Elfes, "High resolution maps from wide angle sonar", ICRA 1985 / Elfes, "Using occupancy grids for mobile robot perception and navigation", IEEE Computer, 1989 |
| log-odds 범위 제한(clamping), 확률 임계값 | `L_MIN`, `L_MAX`, `OCC_THRESH`, `FREE_THRESH` | Hornung et al., "OctoMap: An efficient probabilistic 3D mapping framework based on octrees", Autonomous Robots, 2013 |
| 라이다 → 격자 지도 예제 (flood fill 방식) | `init_flood_fill`, `flood_fill` (파일에 남아 있지만 `MapBuilder`는 쓰지 않음) | PythonRobotics 문서의 Mapping → Lidar to grid map tutorial |
| 위치와 지도를 같이 추정하는 SLAM (이 코드에는 없음) | - | *Probabilistic Robotics* Chapter 10; Hess et al., "Real-time loop closure in 2D LIDAR SLAM", ICRA 2016 (Cartographer) |
| 위의 내용을 강의로 볼 때 | - | Cyrill Stachniss의 Robot Mapping 강의 (occupancy grid maps, mapping with known poses) |

## 참고

- [PythonRobotics](https://github.com/AtsushiSakai/PythonRobotics) — Particle Filter localization (Atsushi Sakai)
- Thrun, Burgard, Fox, *Probabilistic Robotics*, Chapter 4, 5, 8

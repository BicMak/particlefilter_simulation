# particle_viewer

손으로 그린 평면도 위에서 로봇을 움직이면서, **위치 추정 / SLAM 알고리즘이 어떻게 동작하는지** 실시간으로 눈으로 확인하는 시뮬레이터입니다.

[PythonRobotics](https://github.com/AtsushiSakai/PythonRobotics)(Atsushi Sakai)의 예제를 바탕으로, 맵 / 로봇 / 설정을 클래스로 분리하고 이미지 맵, 조작(키보드, 체크포인트), 점유 격자 지도 작성을 추가했습니다.

<!-- TODO: 실행 화면 스크린샷 또는 GIF -->

## 예제

세 예제가 같은 지도, 로봇 시뮬레이션, 설정 파일을 같이 씁니다. 자세한 내용은 각 문서를 보세요.

| 예제 | 실행 | 조작 | 한 줄 설명 | 문서 |
|---|---|---|---|---|
| 파티클 필터 | `python particle_filter.py` | 방향키 | 지도(레퍼런스 좌표)를 알고, 거리 관측만으로 현재 위치를 추정 | [docs/particle_filter.md](docs/particle_filter.md) |
| 그래프 SLAM | `python graphslam.py` | 마우스 클릭 (체크포인트) | 번호를 아는 랜드마크 관측 + 오도메트리로 키프레임 위치 전체를 최적화 | [docs/graphslam.md](docs/graphslam.md) |
| ICP SLAM | `python icpslam.py` | 마우스 클릭 (체크포인트) | 랜드마크 없이 라이다 스캔끼리 맞춰(ICP) 이동량을 재고, 루프를 닫아 그래프로 최적화 | [docs/icpslam.md](docs/icpslam.md) |

셋 다 **왼쪽이 점유 격자 지도, 오른쪽이 시뮬레이션**이고, 파란 삼각형이 실제 로봇, 파란 선이 실제 궤적입니다. 화면은 이렇게 다릅니다.

- 파티클 필터: matplotlib 창
- 그래프 SLAM / ICP SLAM: cv2 창. 가운데에 [지도 평가](#지도-평가) 칸이 하나 더 있고, 왼쪽 격자 지도에도 정답 벽 비교가 겹쳐집니다. matplotlib은 매 프레임 축과 글자까지 전부 다시 그려서 프레임당 100ms가 넘게 걸리는데, cv2 화면([utils/cv_viewer.py](utils/cv_viewer.py))은 지도 그림에서 보이는 부분만 잘라 붙이고 선과 점만 덧그려서 몇 ms면 끝납니다.

## 파일 구성

```
particle_viewer/
├── particle_filter.py   # 파티클 필터. Robot(시뮬레이션 로봇 + 파티클 필터), KeyboardController, main()
├── graphslam.py         # 그래프 SLAM. PoseGraph, MapEval(지도 평가 칸), 체크포인트 추종 / 화면 등 SLAM 공통 함수
├── icpslam.py           # ICP SLAM. icp(), PoseGraph(상대 위치 엣지만), 루프 후보 찾기
├── map.py               # 지도 생성. Map(사각형 방), GridMap(평면도 이미지, 벽 경계선 / 경계선까지 거리 / 라이다), build_map()
├── lidar_to_grid_map.py # 점유 격자 지도 작성(MapBuilder)과 실제 지도 비교(grid_match)
├── config.py            # 모든 설정값 (dataclass)
├── utils/               # 각도 계산, 그리기 보조 함수, cv_viewer.py (SLAM 화면 cv2: Panel, Window)
├── docs/                # 예제별 문서
├── input_image/         # 입력 평면도/미로 이미지 (기본: intel_like_1.png)
└── output_image/        # 결과 이미지 (map.py: map.png, 탐색 지도: explored_map.png)
```

`graphslam.py`와 `icpslam.py`는 `particle_filter.py`의 `Robot`(이동, Dead Reckoning, 벽 충돌)을 그대로 가져다 쓰고, `icpslam.py`는 `graphslam.py`의 공통 함수(상대 위치 계산, 체크포인트 추종, 지도 재작성)를 가져다 씁니다.

## 설치

Python 3.9 이상을 권장합니다 (`dataclasses`, 타입 힌트 사용).

```bash
pip install numpy matplotlib opencv-python scipy
```

`scipy`는 SLAM의 희소 행렬 풀이와 KD-tree, 지도 경계선까지 거리 계산(`map.py`)에 씁니다.

창을 띄우려면 파티클 필터는 창이 뜨는 matplotlib 백엔드(TkAgg, QtAgg 등)가, SLAM 두 예제는 GUI가 들어 있는 OpenCV가 필요합니다. `opencv-python-headless`에는 GUI가 없어서 SLAM 창이 뜨지 않으니 `opencv-python`을 설치하세요. 원격 서버나 헤드리스 환경에서는 `VizConfig.show_animation = False`로 두고 돌려야 합니다.

## 설정

모든 값은 [config.py](config.py)에 dataclass로 모여 있고, 전체는 `Config` 하나로 묶입니다. 기본값을 고치거나, 코드에서 직접 넘기면 됩니다.

```python
from config import Config, VizConfig
from particle_filter import main  # graphslam, icpslam 의 main 도 같은 방식

main(Config(viz=VizConfig(view_radius=None)))
```

| 묶음 | 쓰는 곳 |
|---|---|
| `RobotConfig`, `KeyboardConfig` | 파티클 필터 → [docs/particle_filter.md](docs/particle_filter.md#설정) |
| `GraphSlamConfig` | 그래프 SLAM → [docs/graphslam.md](docs/graphslam.md#설정) |
| `IcpSlamConfig` | ICP SLAM → [docs/icpslam.md](docs/icpslam.md#설정) |
| `SimNoiseConfig`, `RoomConfig`, `ImageMapConfig`, `MappingConfig`, `VizConfig` | 공통 (아래) |

공통 설정 중 자주 만지는 값만 추렸습니다.

| 설정 | 기본값 | 설명 |
|---|---|---|
| `Config.sim_time` | 50 s | 시뮬레이션 총 시간. 키보드 조작이나 화면이 있는 SLAM 실행에서는 무시 |
| `SimNoiseConfig.sigma_range` | 0.02 m | 시뮬레이션에서 **실제로** 거리 센서(랜드마크 관측, 라이다)에 섞이는 노이즈 |
| `SimNoiseConfig.sigma_bearing` | 0.5° | **실제로** 랜드마크 각도 관측에 섞이는 노이즈 (그래프 SLAM 전용) |
| `SimNoiseConfig.sigma_v`, `sigma_yaw_rate` | 0.05 m/s, 2°/s | **실제로** 입력(엔코더, IMU)에 섞이는 노이즈 |
| `ImageMapConfig.use_image_map` | True | True: 이미지 맵, False: 사각형 방(`RoomConfig`) |
| `ImageMapConfig.image` | `input_image/intel_like_1.png` | 지도로 쓸 평면도/미로 이미지 |
| `ImageMapConfig.resolution` | 0.1 | 이미지 한 픽셀의 실제 길이 [m/px] (지도 크기 = 이미지 픽셀 수 × 이 값) |
| `ImageMapConfig.n_ref` | 500 | 벽에서 뽑을 레퍼런스 포인트 개수 |
| `ImageMapConfig.robot_start` | `[15, 7.5, 0, 0]` | 로봇 시작 `[x, y, yaw, v]` |
| `MappingConfig.resolution` | 0.3 | 점유 격자 지도 한 칸 크기 [m] |
| `VizConfig.view_radius` | 8 m | 오른쪽 화면을 로봇 중심 ± 이 반경만 보여주고 따라다님 (`None`이면 지도 전체) |
| `VizConfig.figsize` | (7.5, 6.0) | 패널 하나 기준 창 크기 [inch] (파티클 필터) |
| `VizConfig.panel_px` | 520 | 패널 하나 높이 [px] (SLAM cv2 창). 창은 마우스로 늘리고 줄일 수 있음 |
| `VizConfig.sim_speed` | 3.0 | SLAM 화면에서 시뮬레이션 시간이 실제 시간의 몇 배로 흐르는지 (`None`이면 최대한 빨리) |
| `VizConfig.show_map_eval` | True | SLAM 화면에 지도 평가 칸을 보일지 |
| `VizConfig.eval_view_radius` | None | 지도 평가 칸을 로봇 주변 ± 이 반경[m]만 확대해서 보기 (`None`이면 지도 전체) |
| `MappingConfig.eval_color_max` | 0.2 m | 지도 평가 그림에서 끝점이 이 거리 이상 떨어지면 빨강 |
| `MappingConfig.eval_tol` | 0.05 m | 지도 평가(왼쪽 칸)에서 벽 칸이 맞았다고 보는 실제 벽 경계선까지 거리 |

`SimNoiseConfig`는 시뮬레이터가 **실제로** 섞는 오차이고, 각 알고리즘 설정의 `sigma_*`는 알고리즘이 **그렇다고 믿는** 오차입니다. 알고리즘은 실제 오차를 모른다는 가정이라 일부러 따로 두었습니다.

## 지도 (map.py)

지도는 모두 `map.py`에서 만듭니다. `build_map(cfg)`가 설정(`ImageMapConfig.use_image_map`)에 따라 아래 둘 중 하나를 돌려줍니다.

- `Map`: 사각형 방. 벽 둘레에서 레퍼런스 포인트를 뽑습니다.
- `GridMap`: 손으로 그린 평면도 이미지를 변환한 지도. 아래 순서로 만들어집니다.

1. 이미지를 흑백으로 읽어서 어두운 픽셀을 벽, 밝은 픽셀을 빈 공간으로 봅니다. 투명 배경은 흰색으로 처리합니다.
2. 선이 살짝 끊긴 곳이 메워지도록 벽을 약간 두껍게 만듭니다.
3. 픽셀 좌표를 월드 좌표(m)로 바꿉니다. 이미지 아래쪽이 `y = 0`입니다.
4. 벽 픽셀에서 레퍼런스 포인트를 랜덤으로 뽑습니다.

`GridMap`을 쓰면 실제 로봇은 벽을 통과하지 못하고(부딪히면 그 자리에 멈춤), 벽에 가려진 레퍼런스는 관측되지 않으며, 라이다 광선(`raycast`)을 쏠 수 있습니다. 광선은 벽과 방의 경계선과 처음 만나는 점을 정확히 계산해서 돌려주고, 지도 밖은 벽으로 봅니다 ([지도 평가](#지도-평가) 참고). ICP SLAM은 라이다가 필요해서 `GridMap`에서만 돕니다.

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

그림은 **벽이 어둡고 배경이 밝은** 단순한 선화가 가장 잘 됩니다. 글자, 치수선, 가구처럼 벽이 아닌 어두운 요소가 있으면 벽으로 인식됩니다. 문이나 개구부가 있는 평면도를 방 단위로 자동 분리하지는 않습니다 (벽 픽셀만 뽑아서 쓰며, 방 인식은 아직 없음).

## 점유 격자 지도 (lidar_to_grid_map.py)

화면 왼쪽의 지도는 `MapBuilder`가 그립니다. 관측한 점까지 광선을 그어서, 광선이 지나간 칸은 "비었다", 끝점 칸은 "막혔다" 쪽으로 log-odds를 누적합니다.

- 파티클 필터: 매 step 이번 관측으로 지도를 **누적 갱신**합니다. 위치는 `MappingConfig.use_estimated_pose`에 따라 실제 위치(깨끗한 지도) 또는 필터 추정 위치(위치 오차가 번진 지도)를 씁니다.
- 그래프 SLAM / ICP SLAM: 최적화할 때마다 **최적화된 키프레임 위치와 저장해 둔 스캔으로 처음부터 다시** 그립니다. log-odds 누적은 되돌릴 수 없어서, 위치가 바뀌면 새로 만드는 수밖에 없습니다.

| 설정 (`MappingConfig`) | 기본값 | 설명 |
|---|---|---|
| `enabled` | True | 지도 작성 여부 (파티클 필터) |
| `save_path` | `output_image/explored_map.png` | 파티클 필터 종료 시 / `s` 키로 저장할 경로 (`None`이면 저장 안 함) |
| `use_estimated_pose` | False | 파티클 필터에서 지도에 쓸 위치 |
| `use_flood_fill` | False | True면 광선 사이 빈틈도 부채꼴 안쪽을 flood fill로 빈 공간 처리 |
| `l_free`, `l_occ` | -0.2, 0.4 | 광선이 지나간 칸 / 끝점 칸이 받는 log-odds 변화량 |
| `l_min`, `l_max` | -6, 6 | log-odds 범위 제한 |
| `occ_thresh`, `free_thresh` | 0.7, 0.3 | 점유 확률이 이보다 크면 막힌 칸 / 작으면 빈 칸 (사이는 모름) |

### 지도 평가

그래프 SLAM / ICP SLAM 화면은 정답 지도(입력 이미지)의 **벽과 방의 경계선**을 기준으로 두 가지를 평가합니다. 최적화할 때마다 갱신되고, 같은 점수가 터미널에도 찍힙니다.

- **가운데 칸 (라이다 끝점)**: SLAM 위치가 맞나
- **왼쪽 칸 (점유 격자 지도)**: 최종 지도가 실제와 맞나

**경계선** (`GridMap.boundary_segments`): 입력 이미지의 픽셀 하나는 월드의 정사각형 하나입니다. 이웃한 두 픽셀이 하나는 벽, 하나는 빈 공간이면 그 사이 변(길이 = 해상도)이 경계선입니다. 지도 밖은 벽으로 봅니다.

**거리** (`GridMap.distance_to_boundary`): 점에서 가장 가까운 경계선 선분까지의 최단거리를 격자 없이 정확히 계산합니다. 방 쪽이면 +, 벽 안쪽이면 −입니다.

**라이다도 같은 경계선을 맞힙니다** (`GridMap.raycast`): 광선이 지나가는 픽셀을 하나씩 따라가서(DDA) 벽 픽셀로 넘어가는 경계의 정확한 점을 끝점으로 줍니다. 그래서 위치가 맞고 센서 노이즈가 0이면 끝점과 경계선 사이 거리는 정확히 0입니다.

#### 가운데 칸: 라이다 끝점

SLAM 위치에 그때의 스캔을 붙인 끝점이 경계선에서 몇 cm 떨어졌는지 봅니다. 그림은 정답 벽(회색) 위에 끝점을 거리에 따라 초록(0) ~ 빨강(`eval_color_max` 이상)으로 찍습니다. cm 단위로 보려면 `VizConfig.eval_view_radius`(예: 2.5)를 주면 로봇 주변만 확대해서 따라다닙니다.

| 줄 | 의미 |
|---|---|
| `SLAM: ... off on average, 95% are within ...` / `bias` | SLAM 위치에 올려놓은 라이다 점들이 실제 벽에서 떨어진 거리. 평균과 95% 값은 크기만 봅니다. bias는 부호를 살린 평균으로, +면 점들이 벽 앞(방 쪽)에 떠 있고 −면 벽 안쪽으로 파고든 쪽으로 치우쳤다는 뜻입니다 (`in front of the wall` / `inside the wall`) |
| `same scans on true pose` | 같은 스캔을 정답 위치로 찍은 값 (평균 / 95%). 센서 노이즈만 남은 바닥 |
| `on odometry only (no SLAM)` | 같은 스캔을 오도메트리(Dead Reckoning) 위치로 찍은 값 (평균 / 95%). SLAM 보정이 없을 때 |

#### 왼쪽 칸: 점유 격자 지도

점유 지도에서 정해진 칸(빈 칸 / 벽 칸) 중 **실제 지도와 맞게 찍힌 칸이 몇 %인지** 봅니다 (`lidar_to_grid_map.grid_match`). 아직 모르는 칸(흰색)은 세지 않습니다. 그림에는 실제 벽 경계선(검정)과 틀린 칸(파랑)이 겹쳐집니다.

- **빈 칸**(점유 확률 < `free_thresh`): 칸 중심이 실제 지도에서도 바닥이면 맞음
- **벽 칸**(점유 확률 > `occ_thresh`): 칸 중심이 실제 벽 경계선에서 `eval_tol`(기본 5cm, 칸 하나) 안이면 맞음. 라이다는 벽 표면을 맞혀서, 위치가 완벽해도 벽 칸이 경계선 바로 앞(바닥 쪽) 칸에 찍히기도 합니다. 그래서 "벽 픽셀 안인가"가 아니라 "경계선에 붙어 있나"로 봅니다

| 제목 줄 | 의미 |
|---|---|
| `SLAM map: ...% of cells match (free cells ...%, wall cells ...%)` | SLAM 지도에서 맞게 찍힌 칸의 비율. 전체, 빈 칸만, 벽 칸만 |
| `true pose map: ...%` | 같은 스캔을 정답 위치로 그린 지도(설정은 같음)의 같은 값. 위치가 완벽할 때의 값입니다 |

빈 칸이 벽 칸보다 훨씬 많아서 전체 비율은 빈 칸 쪽으로 끌려갑니다. 벽이 절반 틀려도 전체는 97%처럼 높게 나올 수 있으니, **벽 칸 비율을 꼭 같이 보세요.** SLAM 지도와 정답 위치 지도의 차이가 위치 오차 때문에 생긴 몫입니다.

다만 벽과 나란한 방향의 위치 오차(긴 복도에서 앞뒤로 미끄러지는 것)는 끝점과 벽이 여전히 겹쳐서 두 칸 모두에서 잘 드러나지 않습니다. 위치 정확도 자체는 `main()`이 돌려주는 키프레임 정답 위치(`kf_true`)와 최적화 결과(`graph.poses`)를 직접 비교해서 보세요.

### 복습용 자료

코드에 나온 개념별로 어디를 보면 되는지만 적어 둔 목록입니다.

| 개념 | 코드 위치 | 볼 자료 |
|---|---|---|
| 점유 격자 지도, log-odds 갱신, inverse sensor model (`L_FREE`, `L_OCC`) | `MapBuilder.update` | *Probabilistic Robotics* Chapter 9 (Occupancy Grid Mapping) |
| 거리 센서가 어떻게 측정되는지 (beam model) | `Robot.observation`, `map.py`의 `raycast` | *Probabilistic Robotics* Chapter 6 (Robot Perception) |
| 직선이 지나가는 칸 구하기 | `bresenham` | Wikipedia "Bresenham's line algorithm"; Bresenham, "Algorithm for computer control of a digital plotter", IBM Systems Journal, 1965 |
| 점유 격자의 원전 | `MapBuilder` 전체 | Moravec & Elfes, "High resolution maps from wide angle sonar", ICRA 1985 / Elfes, "Using occupancy grids for mobile robot perception and navigation", IEEE Computer, 1989 |
| log-odds 범위 제한(clamping), 확률 임계값 | `L_MIN`, `L_MAX`, `OCC_THRESH`, `FREE_THRESH` | Hornung et al., "OctoMap: An efficient probabilistic 3D mapping framework based on octrees", Autonomous Robots, 2013 |
| 라이다 → 격자 지도 예제 (flood fill 방식) | `init_flood_fill`, `flood_fill` | PythonRobotics 문서의 Mapping → Lidar to grid map tutorial |
| 위치와 지도를 같이 추정하는 SLAM | `graphslam.py`, `icpslam.py` | *Probabilistic Robotics* Chapter 10, 11; Hess et al., "Real-time loop closure in 2D LIDAR SLAM", ICRA 2016 (Cartographer) |
| 위의 내용을 강의로 볼 때 | - | Cyrill Stachniss의 Robot Mapping 강의 (occupancy grid maps, mapping with known poses, graph-based SLAM) |

## 참고

- [PythonRobotics](https://github.com/AtsushiSakai/PythonRobotics) (Atsushi Sakai) — Localization/particle_filter, SLAM/GraphBasedSLAM, SLAM/ICPMatching, Mapping/lidar_to_grid_map
- Thrun, Burgard, Fox, *Probabilistic Robotics*

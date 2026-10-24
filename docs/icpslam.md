# ICP SLAM (icpslam.py)

랜드마크 없이 **라이다 스캔끼리 겹쳐 보는 것(ICP)만으로** 이동량을 재고, 예전에 지나간 곳에 다시 오면 루프를 닫아 키프레임 위치 전체를 그래프로 최적화하는 예제입니다.

구조(키프레임 그래프 + Gauss-Newton + 최적화 후 지도 재작성)는 [그래프 SLAM](graphslam.md)과 같고, 엣지를 만드는 방법만 다릅니다.

공통 내용(설치, 지도, 점유 격자 지도, 공통 설정)은 [메인 README](../README.md)에 있습니다.

<!-- TODO: 실행 화면 스크린샷 또는 GIF -->

## 실행

```bash
python icpslam.py
```

조작은 [graphslam.py](graphslam.md#실행)와 같습니다.

| 조작 | 동작 |
|---|---|
| 왼쪽 클릭 | 체크포인트 추가. 로봇이 찍은 순서대로 따라감 |
| 오른쪽 클릭 | 체크포인트 전부 지움 |
| esc / 창 닫기 | 종료 |

화면 없이 돌릴 때는 `IcpSlamConfig.waypoints`에 체크포인트를 미리 넣어 두면 `Config.sim_time` 동안 돕니다.

```python
from config import Config, IcpSlamConfig, VizConfig
from icpslam import main

graph, kf_true, kf_dr, builder = main(Config(
    viz=VizConfig(show_animation=False),
    icpslam=IcpSlamConfig(waypoints=[[15, 10.5], [7, 10.5], [7, 7.5], [15, 7.5]])))  # 좌표는 지도마다 다름 (doolhof_03.jpg 기준)
```

라이다 스캔이 필요해서 **이미지 지도(`ImageMapConfig.use_image_map = True`)에서만** 돕니다. 사각형 방으로 실행하면 `RuntimeError`가 납니다.

## 무엇을 볼 수 있나

화면(cv2 창)은 [그래프 SLAM](graphslam.md#무엇을-볼-수-있나)과 같은 세 칸(격자 지도 / 지도 평가 / 시뮬레이션)입니다. 격자 지도 칸에는 루프로 닫힌 구간의 키프레임도 연두 점으로 찍힙니다. 시뮬레이션 칸에는 아래가 그려집니다.

| 표시 | 의미 |
|---|---|
| 파란 삼각형 / 파란 선 | 실제 로봇과 그 궤적 |
| 검은 실선 | Dead Reckoning 궤적 |
| 빨간 점 + 선 | ICP SLAM으로 최적화된 키프레임 위치 |
| 분홍 점 | 마지막 키프레임의 라이다 스캔 (그때의 실제 위치에 그림) |
| 연두 선 | 루프 엣지 (닫힌 두 키프레임을 이음) |
| 연두 점 | 루프로 닫힌 구간의 키프레임. 루프 엣지 i ↔ j가 생기면 i ~ j 사이가 전부 누적 오차를 나눠 갖기 때문에 구간 전체를 칠함 |
| 주황 X / 주황 선 | 남은 체크포인트와 경로 |

최적화할 때마다 터미널에 노드 수, 엣지 수(루프 엣지 수, ICP 실패 수 포함), 반복 횟수, 비용과 지도 평가 점수가 찍힙니다.

## 동작 원리

### 그래프 SLAM과 다른 점

- 그래프 SLAM은 "몇 번 랜드마크인지"를 알려주는 관측을 씁니다. 여기서는 그런 정보가 없습니다. 대신 두 키프레임의 스캔(벽 위의 점 구름)을 ICP로 겹쳐 보고, 겹치게 만드는 이동량을 "두 위치 사이의 상대 위치"로 씁니다.
- 그래프의 엣지가 전부 **상대 위치 한 종류**입니다 (오도메트리 엣지와 같은 모양). 어디서 왔느냐만 다릅니다.

| 엣지 | 만드는 방법 |
|---|---|
| 이웃 엣지 | 바로 이전 키프레임의 스캔과 ICP (초기값 = 오도메트리). ICP가 실패하면 오도메트리 값을 그대로 씀 |
| 루프 엣지 | 추정 위치가 예전 키프레임과 가까우면 그 스캔과 ICP (초기값 = 두 추정 위치의 차이). 더 엄격한 기준(`loop_max_err`)을 통과해야 추가 |

루프 엣지가 오래전 위치와 지금 위치를 직접 묶어서, 그동안 쌓인 오차를 그래프 전체에 나눠 없앱니다.

### ICP (Iterative Closest Point)

점 구름 A를 점 구름 B 위에 겹치는 이동량 `(dx, dy, dyaw)`를 찾습니다 (`icp()`).

1. 지금 이동량으로 A를 옮깁니다.
2. A의 각 점에 가장 가까운 B의 점을 짝으로 잡습니다 (KD-tree). `icp_max_corr_dist`보다 먼 짝은 겹치지 않는 부분이라 버립니다.
3. 짝지어진 점들을 가장 잘 겹치는 회전 + 이동을 닫힌 형태(SVD, Kabsch)로 풉니다 (`best_fit_transform()`).
4. 이동량을 갱신하고, 변화가 `icp_tol`보다 작아질 때까지 1로 돌아갑니다.

초기값이 정답에서 많이 벗어나면 엉뚱한 짝을 잡고 틀린 곳에 수렴합니다 (local minimum). 그래서 초기값으로 오도메트리를 쓰고, 수렴 후 대응점 평균 거리가 `icp_max_err`보다 크거나 대응점이 `icp_min_inliers`보다 적으면 실패로 봅니다.

### 한 step의 흐름

1. **이동**: 체크포인트로 가는 입력을 계산해 로봇을 움직입니다. 레퍼런스 관측은 쓰지 않습니다.
2. **키프레임 추가**: 마지막 키프레임에서 오도메트리로 `keyframe_dist`만큼 움직였거나 `keyframe_angle`만큼 돌았을 때만 추가합니다 (멈춰 있으면 안 생김). 스캔을 찍고 이웃 엣지를 만든 뒤, 루프 후보(`find_loop_candidate()`)가 있으면 루프 엣지를 시도합니다.
3. **최적화** (`optimize_every`마다): Gauss-Newton으로 전체 위치를 풀고(첫 노드 고정, 희소 행렬), 그 위치와 저장해 둔 스캔으로 격자 지도를 다시 만듭니다.

## 설정

[config.py](../config.py)의 `IcpSlamConfig`가 이 예제 전용입니다.

| 설정 | 기본값 | 설명 |
|---|---|---|
| `keyframe_dist` | 1.5 m | 이만큼 움직이면 키프레임 추가 |
| `keyframe_angle` | 30° | 이만큼 돌면 키프레임 추가 (제자리 회전 때도 스캔끼리 겹치는 부분이 남도록) |
| `optimize_every` | 3.0 s | 이 시간마다 그래프 전체를 다시 최적화 + 지도 재작성 |
| `max_iter` | 10 | Gauss-Newton 최대 반복 횟수 |
| `n_scan_rays` | 360 | 라이다 광선 수 (ICP와 지도 둘 다 이 스캔을 씀) |
| `icp_max_iter` | 10 | ICP 최대 반복 횟수 |
| `icp_max_corr_dist` | 0.5 m | 이보다 먼 짝은 대응점으로 안 씀 |
| `icp_tol` | 1e-4 | 한 반복의 이동량 변화가 이보다 작으면 수렴 |
| `icp_min_inliers` | 50 | 대응점이 이보다 적으면 실패 |
| `icp_max_err` | 0.1 m | 수렴 후 대응점 평균 거리가 이보다 크면 실패 |
| `loop_dist` | 0.5 m | 이 거리 안의 예전 키프레임을 루프 후보로 |
| `loop_min_gap` | 10 | 키프레임 번호 차이가 이보다 작으면 루프 후보에서 제외 |
| `loop_max_err` | 0.2 m | 루프 ICP 성공 기준 |
| `icp_sigma_xy`, `icp_sigma_yaw` | 0.01 m, 0.5° | **그래프가 믿는** ICP 이웃 엣지 노이즈 |
| `odo_sigma_xy`, `odo_sigma_yaw` | 0.005 m, 0.5° | **그래프가 믿는** 오도메트리 노이즈 (ICP 실패 시 대신 쓰는 엣지) |
| `loop_sigma_xy`, `loop_sigma_yaw` | 0.01 m, 0.5° | **그래프가 믿는** 루프 엣지 노이즈. 작을수록 루프를 닫을 때 고리 안 이웃 엣지들이 더 많이 휨 |
| `draw_every` | 1 | 화면을 이 step마다 한 번만 그림 (1 = 매 step) |
| `waypoints`, `goal_tolerance`, `v_max`, `yaw_rate_max`, `k_v`, `k_w`, `stuck_time` | | 체크포인트 추종. [그래프 SLAM](graphslam.md#설정)과 같음 |

## 알려진 한계

- 긴 복도처럼 한 방향으로 특징이 없는 곳에서는 ICP가 그 방향으로 미끄러집니다 (점-점 ICP의 알려진 약점).
- 루프를 잘못 닫으면(다른 곳을 같은 곳으로 착각) 지도가 크게 망가집니다. `loop_max_err`를 엄격하게 두는 이유입니다.
- 루프 후보는 추정 위치 기준 거리로만 찾습니다. 그동안 오차가 `loop_dist`보다 크게 쌓이면 실제로 같은 곳에 와도 루프를 못 찾습니다.
- 체크포인트 추종은 **실제 위치**를 보고 따라갑니다.

## 참고

- Besl & McKay, "A Method for Registration of 3-D Shapes", IEEE TPAMI, 1992 (ICP)
- Arun, Huang, Blostein, "Least-Squares Fitting of Two 3-D Point Sets", IEEE TPAMI, 1987 (SVD 정합)
- Lu & Milios, "Globally Consistent Range Scan Alignment for Environment Mapping", Autonomous Robots, 1997 (스캔 정합 + pose graph)
- [PythonRobotics](https://github.com/AtsushiSakai/PythonRobotics) — SLAM/ICPMatching
- Hess et al., "Real-time loop closure in 2D LIDAR SLAM", ICRA 2016 (Cartographer)

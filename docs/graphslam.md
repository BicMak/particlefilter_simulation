# 그래프 SLAM (graphslam.py)

번호를 아는 랜드마크(레퍼런스 포인트) 관측과 오도메트리로 **키프레임 위치 전체를 한꺼번에 최적화**하는 pose graph SLAM 예제입니다. 최적화된 위치로 점유 격자 지도를 처음부터 다시 그립니다.

[PythonRobotics](https://github.com/AtsushiSakai/PythonRobotics)의 `SLAM/GraphBasedSLAM/graph_based_slam.py`(Atsushi Sakai)를 이 프로젝트의 로봇 / 지도 / 설정에 맞게 옮겼습니다.

공통 내용(설치, 지도, 점유 격자 지도, 공통 설정)은 [메인 README](../README.md)에 있습니다.

<!-- TODO: 실행 화면 스크린샷 또는 GIF -->

## 실행

```bash
python graphslam.py
```

키보드 대신 **체크포인트를 찍어서** 로봇을 움직입니다.

| 조작 | 동작 |
|---|---|
| 왼쪽 클릭 | 체크포인트 추가 (세 패널 어디든). 로봇이 찍은 순서대로 따라감 |
| 오른쪽 클릭 | 체크포인트 전부 지움 |
| esc / 창 닫기 | 종료 |

- 체크포인트까지는 직선으로 가기 때문에, 벽이 있으면 모퉁이마다 찍어야 합니다.
- 가는 길에 벽에 막혀 `GraphSlamConfig.stuck_time`(기본 0.5초) 동안 못 움직이면 그 체크포인트는 도착한 것으로 보고 다음으로 넘어갑니다.
- 화면 없이 돌릴 때(`VizConfig.show_animation = False`)는 `GraphSlamConfig.waypoints`에 체크포인트를 미리 넣어 두면 `Config.sim_time` 동안 돕니다.

```python
from config import Config, GraphSlamConfig, VizConfig
from graphslam import main

graph, kf_true, kf_dr, builder = main(Config(
    viz=VizConfig(show_animation=False),
    graphslam=GraphSlamConfig(waypoints=[[15, 10.5], [7, 10.5], [7, 7.5], [15, 7.5]])))  # 좌표는 지도마다 다름 (doolhof_03.jpg 기준)
```

`main()`은 `(PoseGraph, 키프레임 정답 위치, 키프레임 Dead Reckoning 위치, MapBuilder)`를 돌려주므로, 정답 / Dead Reckoning / 최적화 결과를 직접 비교할 수 있습니다.

## 무엇을 볼 수 있나

화면(cv2 창)은 세 칸입니다.

1. **격자 지도**: 최적화된 위치로 다시 만든 점유 격자 지도 (`optimize_every`초마다 갱신). 빨간 점은 최적화된 키프레임, 파란 점은 실제 로봇 위치. 실제 벽 경계선(검정)과 실제 지도와 안 맞는 칸(파랑)이 겹쳐지고, 제목에 실제 지도와 맞는 칸의 비율(전체 / 빈 칸 / 벽 칸)이 나옴 ([메인 README의 지도 평가](../README.md#왼쪽-칸-점유-격자-지도) 참고)
2. **지도 평가**: 최적화된 위치로 찍은 라이다 끝점이 실제 벽 경계선에서 몇 cm 떨어졌나를 색과 점수로 보여 줌. 같은 스캔을 정답 위치 / Dead Reckoning으로 찍었을 때의 점수도 같이 나옴 ([메인 README의 지도 평가](../README.md#지도-평가) 참고). 이미지 지도에서만 나오고, `VizConfig.show_map_eval = False`면 숨김
3. **시뮬레이션**: 로봇을 따라다니는 화면. 아래가 그려짐

| 표시 | 의미 |
|---|---|
| 파란 삼각형 / 파란 선 | 실제 로봇과 그 궤적 |
| 검은 실선 | Dead Reckoning 궤적 (오도메트리만 적분) |
| 빨간 점 + 선 | 그래프 SLAM으로 최적화된 키프레임 위치 |
| 검은 별 | 랜드마크 (고정 레퍼런스 포인트) |
| 검은 가는 선 | 이번 step에 관측한 랜드마크 |
| 연한 분홍 부채꼴 | 시야각 |
| 주황 X / 주황 선 | 남은 체크포인트와 경로 |

최적화할 때마다 터미널에 노드 수, 엣지 수, 반복 횟수, 비용과 지도 평가 점수가 찍힙니다.

## 동작 원리

### 파티클 필터와 다른 점

- 랜드마크의 **정답 좌표를 쓰지 않습니다.** 로봇 기준의 거리, 각도, 그리고 "몇 번 랜드마크인지"만 씁니다.
- 매 step 현재 위치만 고치는 대신, 지금까지의 키프레임 위치 전체를 한꺼번에 최적화합니다.
- 최적화된 위치로 지도(`MapBuilder`)를 처음부터 다시 그립니다 (log-odds 누적은 되돌릴 수 없어서 매번 새로).

### 그래프

| 요소 | 내용 |
|---|---|
| 노드 | 키프레임(`keyframe_dt`초마다 하나)의 로봇 위치 `[x, y, yaw]` |
| 오도메트리 엣지 | 이웃한 두 키프레임 사이의 이동량 = Dead Reckoning으로 잰 값 |
| 랜드마크 엣지 | 같은 랜드마크를 본 두 키프레임. "위치1 + 관측1로 계산한 랜드마크 위치"와 "위치2 + 관측2로 계산한 위치"가 같아야 함 |

랜드마크 엣지의 가중치는 관측 하나가 랜드마크 위치에 주는 불확실성에서 나옵니다. 빔 방향은 거리 노이즈(`sigma_range`), 옆 방향은 거리 × 각도 노이즈(`d · sigma_bearing`)이고, 두 관측의 공분산을 더한 것의 역행렬을 정보 행렬로 씁니다.

### 한 step의 흐름

1. **이동 + 관측**: 체크포인트로 가는 입력을 계산해 로봇을 움직이고, 고정 레퍼런스를 관측합니다. 그래프 SLAM은 같은 대상을 다시 알아봐야 해서 `n_rays = 0`(고정 레퍼런스 모드)으로 강제합니다.
2. **키프레임 추가** (`keyframe_dt`마다): 직전 키프레임 대비 오도메트리 이동량, 랜드마크 관측 `[거리, 각도, 번호]`, 지도용 라이다 스캔을 저장합니다. 새 노드의 초기값은 직전 추정 위치 + 오도메트리입니다.
3. **최적화** (`optimize_every`마다): Gauss-Newton으로 전체 위치를 풀고, 그 위치와 저장해 둔 스캔으로 격자 지도를 다시 만듭니다.

### 최적화

- 첫 노드는 좌표계의 기준이라 고정합니다. 원본은 H에 단위행렬을 더해 고정하지만 그러면 첫 노드가 조금씩 밀려서, 여기서는 첫 노드의 행/열을 빼고 나머지만 풉니다.
- 같은 종류의 엣지는 numpy 배열로 한꺼번에 계산하고, H는 `scipy.sparse`로 만들어 `spsolve`로 풉니다. 꽉 찬 행렬은 노드 수의 제곱만큼 메모리를 쓰고 풀이도 거의 세제곱으로 느려지기 때문입니다.

### 원본 예제와 다른 점

- 원본은 랜드마크 엣지를 모든 키프레임 쌍으로 만듭니다 (키프레임 수의 제곱만큼 늘어남). 여기서는 랜드마크마다 **"바로 이전에 본 키프레임"과 "처음 본 키프레임"에만** 잇습니다. 처음 본 키프레임과의 엣지가 오래전 위치와 지금 위치를 직접 묶어서 루프를 닫는 역할을 합니다.
- 원본에 없는 **오도메트리 엣지**를 넣었습니다. 랜드마크가 잘 안 보이는 구간에서도 그래프가 끊기지 않습니다.

## 설정

[config.py](../config.py)의 `GraphSlamConfig`가 이 예제 전용입니다.

| 설정 | 기본값 | 설명 |
|---|---|---|
| `keyframe_dt` | 1.0 s | 이 시간마다 키프레임 하나 추가 |
| `optimize_every` | 10.0 s | 이 시간마다 그래프 전체를 다시 최적화 + 지도 재작성 |
| `max_iter` | 1 | Gauss-Newton 최대 반복 횟수 |
| `sigma_range` | 0.01 m | **그래프가 믿는** 랜드마크 거리 노이즈 |
| `sigma_bearing` | 0.5° | **그래프가 믿는** 랜드마크 각도 노이즈 |
| `odo_sigma_xy`, `odo_sigma_yaw` | 0.01 m, 0.5° | **그래프가 믿는** 키프레임 사이 오도메트리 노이즈 |
| `n_scan_rays` | 120 | 지도용 라이다 광선 수 |
| `draw_every` | 1 | 화면을 이 step마다 한 번만 그림 (1 = 매 step) |
| `waypoints` | `[]` | 미리 정해 둘 체크포인트 `[[x, y], ...]` |
| `goal_tolerance` | 0.4 m | 이 거리 안으로 들어오면 체크포인트 도착 |
| `v_max`, `yaw_rate_max` | 1.5 m/s, 1.5 rad/s | 체크포인트 추종의 최대 속도 / 각속도 |
| `k_v`, `k_w` | 1.0, 2.5 | 속도 = `k_v` × 남은 거리, 각속도 = `k_w` × 방향 오차 |
| `stuck_time` | 0.5 s | 이 시간 동안 벽에 막히면 그 체크포인트는 건너뜀 |

실제로 섞이는 노이즈는 공통 설정인 `SimNoiseConfig`(`sigma_range`, `sigma_bearing`, `sigma_v`, `sigma_yaw_rate`)에서 정합니다.

## 알려진 한계

- "어느 랜드마크인지"는 안다고 가정합니다 (data association을 풀지 않음). 시뮬레이터가 레퍼런스 좌표로 번호를 찾아서 넘겨 주며, 좌표 자체는 그래프에 넘기지 않습니다.
- 체크포인트 추종은 운전이 목적이라 **실제 위치**를 보고 따라갑니다. 추정 위치로 제어하는 구조는 아닙니다.
- 사각형 방(`ImageMapConfig.use_image_map = False`)에서도 돌지만, 이때는 라이다 스캔이 없어서 왼쪽 격자 지도는 그려지지 않습니다.

## 참고

- [PythonRobotics](https://github.com/AtsushiSakai/PythonRobotics) — SLAM/GraphBasedSLAM (Atsushi Sakai)
- Grisetti, Kümmerle, Stachniss, Burgard, "A Tutorial on Graph-Based SLAM", IEEE ITS Magazine, 2010
- Thrun, Burgard, Fox, *Probabilistic Robotics*, Chapter 11 (The GraphSLAM Algorithm)

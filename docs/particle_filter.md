# 파티클 필터 (particle_filter.py)

손으로 그린 평면도 위에서 로봇을 방향키로 직접 움직이면서, **파티클 필터가 로봇 위치를 어떻게 추정하는지** 실시간으로 보는 예제입니다.

[PythonRobotics](https://github.com/AtsushiSakai/PythonRobotics)의 Particle Filter localization 예제(Atsushi Sakai)를 바탕으로 했습니다.

공통 내용(설치, 지도, 점유 격자 지도, 공통 설정)은 [메인 README](../README.md)에 있습니다.

<!-- TODO: 실행 화면 스크린샷 또는 GIF -->

## 실행

```bash
python particle_filter.py
```

기본 설정은 `input_image/intel_like_1.png` 이미지를 지도로 쓰고, 방향키로 로봇을 조작하는 모드입니다.

### 조작법

방향키를 **누르고 있는 동안** 입력이 들어갑니다.

| 키 | 동작 |
|---|---|
| ↑ / ↓ | 전진 / 후진 가속 (키를 떼면 서서히 정지) |
| ← / → | 좌회전(반시계) / 우회전 (키를 떼면 각속도 0으로 복귀) |
| space | 즉시 정지 |
| s | 탐색 지도를 `MappingConfig.save_path`에 저장 |
| esc | 종료 (창을 닫아도 종료. 종료할 때 탐색 지도를 자동 저장) |

가속도와 최대 속도는 `config.py`의 `KeyboardConfig`에서 바꿀 수 있습니다.

### 키보드 없이 실행

`KeyboardConfig.enabled = False`로 두면 일정한 속도(1 m/s)와 각속도(0.1 rad/s)로 원을 그리며 움직이고, `Config.sim_time`(기본 50초) 뒤에 끝납니다.

### 설정 바꿔서 실행

```python
from config import Config, RobotConfig, VizConfig
from particle_filter import main

main(Config(robot=RobotConfig(particle=500), viz=VizConfig(trail_time=None)))
```

## 무엇을 볼 수 있나

로봇은 벽 위의 점(레퍼런스 포인트)까지의 **거리만** 관측합니다. 자기 위치는 모릅니다.

- `RobotConfig.n_rays`가 1 이상(기본 60)이면 매 step 시야각 안으로 광선을 쏴서 벽에 닿은 점을 관측합니다 (이미지 지도에서만).
- 0이면 지도에서 미리 뽑아 둔 고정 레퍼런스 포인트 중 시야각 안에 있고 벽에 가려지지 않은 점을 관측합니다.

화면은 왼쪽이 탐색 지도(점유 격자), 오른쪽이 시뮬레이션입니다. 오른쪽에는 아래가 같이 그려집니다.

| 표시 | 의미 |
|---|---|
| 파란 삼각형 / 파란 점선 | 실제 로봇(ground truth)과 그 궤적 |
| 초록 점 + 초록 선 | 파티클의 위치와 진행 방향 |
| 빨간 선 | 파티클 필터의 추정 궤적 |
| 빨간 점선 타원 | 추정 위치의 불확실성 (x, y 공분산의 1σ 타원) |
| 연한 분홍 부채꼴 | 로봇의 시야각 (이 안의 점만 관측됨) |
| 검은 점선 | 지도에 쓰는 위치(`MappingConfig.use_estimated_pose`) → 로봇이 믿는 끝점 |
| 검은 별 | 로봇이 믿는 끝점. 그 위치에서 측정한 거리와 빔 각도로 계산한 점이라, 위치 오차와 거리 노이즈만큼 벽에서 벗어날 수 있음 (고정 레퍼런스 모드에서는 레퍼런스 포인트) |

왼쪽 지도에서 파란 점은 실제 위치, 빨간 점은 필터 추정 위치입니다.

## 동작 원리

한 time step(`dt = 0.1초`)마다 `main()`에서 아래 순서로 진행합니다.

1. **입력 결정**: 키보드(또는 고정값)로 지령 `u = [v, yaw_rate]`를 정합니다.
2. **실제 로봇 이동 + 관측** (`Robot.observation`): 노이즈 없는 `u`로 실제 로봇(`x_true`)이 움직이고, 시야각 안의 레퍼런스까지의 거리에 센서 노이즈를 섞어 관측 `z`(각 행: `[노이즈 낀 거리, 레퍼런스 x, 레퍼런스 y, 빔 각도]`)를 만듭니다. 엔코더/IMU처럼 노이즈가 섞인 입력 `ud`도 함께 만들어지며, `ud`만으로 적분한 것이 Dead Reckoning(`x_dr`)입니다.
3. **예측** (`predict_particles`): 파티클마다 `ud`에 `sigma_v`, `sigma_yaw_rate` 크기의 노이즈를 또 섞어서 운동 모델로 이동시킵니다. 이 단계에서 파티클이 퍼집니다.
4. **가중치 갱신** (`calc_delta` → `update_weights`): 각 파티클에서 레퍼런스까지의 예상 거리와 관측 거리의 차이 `delta`를 구하고, `sigma_range`를 폭으로 하는 가우시안 우도로 가중치를 곱합니다. 우도가 0으로 언더플로우하지 않도록 로그 영역에서 계산합니다.
5. **추정 + 리샘플링** (`pf_localization`): 파티클의 가중 평균을 추정 위치로, 가중 공분산을 불확실성으로 계산합니다. 유효 파티클 수(N_eff)가 `particle × n_threshold_ratio`보다 작으면 low variance 방식으로 리샘플링합니다.
6. **지도 갱신**: 이번 관측으로 탐색 지도(`MapBuilder`)를 갱신합니다. 위치는 `MappingConfig.use_estimated_pose`에 따라 실제 위치 또는 추정 위치를 씁니다.

상태 벡터는 `[x, y, yaw, v]`이고, 파티클은 `px` (4 × N 배열, 열 하나가 파티클 하나), 가중치는 `pw` (1 × N 배열)로 저장합니다.

## 설정

[config.py](../config.py)의 `RobotConfig`, `KeyboardConfig`가 이 예제 전용입니다.

| 설정 | 기본값 | 설명 |
|---|---|---|
| `RobotConfig.particle` | 100 | 파티클 개수 |
| `RobotConfig.fov` | 360° | 시야각. 로봇 진행 방향 기준 좌우 `fov/2`만 관측 |
| `RobotConfig.max_range` | 20 m | 최대 관측 거리 |
| `RobotConfig.n_rays` | 60 | 0: 고정 레퍼런스 관측, 1 이상: 매 step 이만큼 광선을 쏴서 관측 |
| `RobotConfig.n_threshold_ratio` | 0.05 | N_eff가 `particle × 이 값`보다 작으면 리샘플링 |
| `RobotConfig.sigma_range` | 0.5 m | **필터가 믿는** 거리 센서 노이즈. 가중치 계산에 사용 |
| `RobotConfig.sigma_v`, `sigma_yaw_rate` | 0.8 m/s, 45°/s | **필터가 믿는** 입력(속도, 각속도) 노이즈. 파티클을 퍼뜨리는 크기 |
| `KeyboardConfig.enabled` | True | False면 원을 그리며 자동 이동 |
| `KeyboardConfig.v_max`, `yaw_max` | 3 m/s, 1.5 rad/s | 키보드 조작의 최대 속도 / 각속도 |
| `VizConfig.trail_time` | 10초 | 궤적을 남기는 시간 (`None`이면 안 지움) |
| `VizConfig.particle_draw_ratio` | 0.5 | 전체 파티클 중 화면에 그릴 비율 (그리기 전용) |

`RobotConfig`의 `sigma_*`(필터의 믿음)과 `SimNoiseConfig`의 `sigma_*`(실제 오차)는 **일부러 따로** 두었습니다. 필터는 실제 오차 크기를 모르기 때문에, 보통 필터 쪽을 더 크게 잡습니다.

## 알려진 한계

- 파티클이 관측과 크게 어긋나면 필터가 실제 위치를 놓칠 수 있습니다. 로봇이 레퍼런스가 거의 안 보이는 곳으로 가면 추정이 실제와 크게 달라지고, 이때 불확실성 타원은 오히려 작게 나올 수 있습니다.
- 처음 시작 위치를 알고 있다고 가정합니다. 파티클이 전부 시작 위치에서 출발하며, 지도 전체에 흩뿌리는 전역 위치 추정(global localization)은 아닙니다.
- 지도(레퍼런스 좌표)는 정확하다고 가정합니다. 노이즈는 거리 측정과 입력에만 섞입니다.

## 참고

- [PythonRobotics](https://github.com/AtsushiSakai/PythonRobotics) — Localization/particle_filter (Atsushi Sakai)
- Thrun, Burgard, Fox, *Probabilistic Robotics*, Chapter 4 (Nonparametric Filters), 5 (Robot Motion), 8 (Mobile Robot Localization)

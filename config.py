"""
particle_filter.py 에서 쓰는 설정(config) 모음.
값을 바꾸고 싶으면 여기 기본값을 고치거나, 코드에서 dataclass 를 만들 때 직접 넘기면 된다.
    cfg = Config(robot=RobotConfig(particle=500), viz=VizConfig(trail_time=None))
"""
import math
import os
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


@dataclass
class RobotConfig:
    """로봇 + 파티클 필터 파라미터"""
    particle: int = 100  # 파티클 개수
    dt: float = 0.1  # time tick [s]
    max_range: float = math.inf  # 최대 관측 거리 [m] (inf = 거리 제한 없음, 시야각으로만 결정)
    fov: float = field(default_factory=lambda: np.deg2rad(60.0))  # 시야각 [rad]: yaw 기준 좌우 fov/2 안쪽만 관측
    n_threshold_ratio: float = 0.05  # 유효 파티클 수(N_eff)가 particle * 이 비율보다 작으면 리샘플링 (0.5 = 절반)
    v: float = 1.0  # 초기 속도 [m/s]
    yaw_rate: float = 0.1  # 초기 각속도 [rad/s]
    # 필터가 "믿는" 노이즈 (설계 파라미터): 우도 계산(Q)과 파티클 퍼뜨리기(R)에만 쓰임
    Q: np.ndarray = field(default_factory=lambda: np.diag([0.5]) ** 2)  # 거리 관측 노이즈 분산
    R: np.ndarray = field(default_factory=lambda: np.diag([0.8, np.deg2rad(45.0)]) ** 2)  # 입력(v, yaw_rate) 노이즈 분산


@dataclass
class SimNoiseConfig:
    """시뮬레이션에서 "실제로" 들어가는 센서 노이즈 (실제 하드웨어라면 센서가 물리적으로 내는 오차). 필터는 이 값을 모른다"""
    Q: np.ndarray = field(default_factory=lambda: np.diag([0.02]) ** 2)  # 거리 센서 노이즈 분산
    R: np.ndarray = field(default_factory=lambda: np.diag([0.05, np.deg2rad(2.0)]) ** 2)  # 속도/각속도(엔코더, IMU) 노이즈 분산


@dataclass
class RoomConfig:
    """사각형 실내 공간(벽) (map.py 의 Map)"""
    x_min: float = -15.0
    x_max: float = 15.0
    y_min: float = -5.0
    y_max: float = 25.0
    n_ref: int = 200  # 벽 경계에서 샘플링할 레퍼런스 포인트 개수


@dataclass
class ImageMapConfig:
    """손그림 지도 (map.py 의 GridMap)"""
    use_image_map: bool = True  # True: 이미지에서 만든 지도 사용, False: 사각형 방(RoomConfig) 사용
    image: str = os.path.join(HERE, "input_image", "Untitled.png")  # 손으로 그린 평면도 이미지
    resolution: float = 0.03  # 이미지 한 픽셀의 실제 길이 [m/px]
    n_ref: int = 200  # 벽 픽셀에서 샘플링할 레퍼런스 포인트 개수
    robot_start: List[float] = field(default_factory=lambda: [15.0, 7.5, 0.0, 0.0])  # 로봇 시작 [x, y, yaw, v]


@dataclass
class KeyboardConfig:
    """키보드 조작"""
    enabled: bool = True  # True: 방향키로 조작 (esc로 종료), False: 자동으로 원 그리며 이동
    accel: float = 2.0  # 위/아래 키를 누르고 있는 동안의 가속도 [m/s^2]
    brake: float = 3.0  # 키를 떼면 속도가 0으로 줄어드는 감속도 [m/s^2]
    v_max: float = 3.0  # 최대 속도 [m/s]
    yaw_max: float = 1.5  # 좌/우 키를 누르고 있을 때의 목표 각속도 [rad/s]
    yaw_accel: float = 6.0  # 각속도가 목표값(키 누름=±yaw_max, 뗌=0)으로 변하는 속도 [rad/s^2]


@dataclass
class VizConfig:
    """화면 표시"""
    show_animation: bool = True
    trail_time: Optional[float] = 10.0  # 궤적을 남기는 시간 [s]. 이보다 오래된 건 지움 (None = 안 지움)
    robot_size: float = 0.7  # 로봇 삼각형 크기 [m]
    fov_draw_radius: float = 10.0  # 시야각 부채꼴을 그릴 반지름 [m] (그리기 전용, 관측 거리와 무관)
    particle_draw_ratio: float = 0.2  # 전체 파티클 중 화면에 그릴 비율 (0.05 = 5%, 1.0 = 전부). 매 프레임 랜덤 샘플링
    min_particles_draw: int = 10  # 비율로 계산한 개수가 이보다 적어도 최소 이만큼은 그림 (파티클이 이보다 적으면 전부)
    particle_dir_len: float = 0.5  # 파티클 방향선 길이 [m]


@dataclass
class Config:
    """전체 설정"""
    sim_time: float = 50.0  # 시뮬레이션 총 시간 [s] (키보드 모드에서는 무시)
    robot: RobotConfig = field(default_factory=RobotConfig)
    sim_noise: SimNoiseConfig = field(default_factory=SimNoiseConfig)
    room: RoomConfig = field(default_factory=RoomConfig)
    image_map: ImageMapConfig = field(default_factory=ImageMapConfig)
    keyboard: KeyboardConfig = field(default_factory=KeyboardConfig)
    viz: VizConfig = field(default_factory=VizConfig)

"""
particle_filter.py 에서 쓰는 설정(config) 모음.
값을 바꾸고 싶으면 여기 기본값을 고치거나, 코드에서 dataclass 를 만들 때 직접 넘기면 된다.
    cfg = Config(robot=RobotConfig(particle=500), viz=VizConfig(trail_time=None))
"""
import math
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


@dataclass
class RobotConfig:
    """로봇 + 파티클 필터 파라미터"""
    particle: int = 50  # 파티클 개수
    dt: float = 0.2  # time tick [s]
    max_range: float = 12 # math.inf  # 최대 관측 거리 [m] (inf = 거리 제한 없음, 시야각으로만 결정)
    fov: float = field(default_factory=lambda: np.deg2rad(360.0))  # 시야각 [rad]: yaw 기준 좌우 fov/2 안쪽만 관측
    n_rays: int = 60  # 0: 미리 뽑아둔 레퍼런스 포인트(rf_id)를 관측. 1 이상: 시야각 안의 랜덤한 방향으로 광선을 매 step 이만큼 쏴서 실제로 보이는 벽 위의 점을 관측 (이미지 지도에서만)
    n_threshold_ratio: float = 0.05  # 유효 파티클 수(N_eff)가 particle * 이 비율보다 작으면 리샘플링 (0.5 = 절반)
    v: float = 1.0  # 초기 속도 [m/s]
    yaw_rate: float = 0.1  # 초기 각속도 [rad/s]
    # 필터가 "믿는" 노이즈 표준편차 (설계 파라미터): 우도 계산(sigma_range)과 파티클 퍼뜨리기(sigma_v, sigma_yaw_rate)에만 쓰임
    sigma_range: float = 0.5  # 거리 관측 [m]
    sigma_v: float = 0.8  # 입력 속도 [m/s]
    sigma_yaw_rate: float = math.radians(45.0)  # 입력 각속도 [rad/s]


@dataclass
class SimNoiseConfig:
    """시뮬레이션에서 "실제로" 들어가는 센서 노이즈 (실제 하드웨어라면 센서가 물리적으로 내는 오차). 필터는 이 값을 모른다"""
    sigma_range: float = 0.03  # 거리 관측 노이즈 표준편차 [m] (랜드마크 관측 + 지도용 라이다 스캔)
    sigma_bearing: float = math.radians(0.5)  # 랜드마크 각도 관측 노이즈 표준편차 [rad] (그래프 SLAM 에서만 씀. 파티클 필터는 각도를 안 씀)
    sigma_v: float = 0.05  # 속도(엔코더) 노이즈 표준편차 [m/s]
    sigma_yaw_rate: float = math.radians(0.5)  # 각속도(IMU) 노이즈 표준편차 [rad/s]


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
    image: str = os.path.join(HERE, "input_image", "intel_like_1.png")  # 평면도/미로 이미지 (벽이 어둡고 배경이 밝은 그림)
    resolution: float = 0.05  # 이미지 한 픽셀의 실제 길이 [m/px]
    n_ref: int = 500  # 벽 픽셀에서 샘플링할 레퍼런스 포인트 개수
    robot_start: List[float] = field(default_factory=lambda: [10.0, 6.5, 0.0, 0.0])  # 로봇 시작 [x, y, yaw, v]


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
    figsize: Tuple[float, float] = (7.5, 6.0)  # 창 크기 [inch] (패널 하나 기준: 지도 패널이 같이 뜨면 가로가 2배). particle_filter.py (matplotlib) 전용
    panel_px: int = 520  # graphslam.py / icpslam.py (cv2 화면) 의 패널 하나 높이 [px]. 너비는 지도 가로세로 비율을 따름
    sim_speed: Optional[float] = 3.0  # cv2 화면에서 시뮬레이션 시간이 실제 시간의 몇 배로 흐르는지 (None = 최대한 빨리)
    show_map_eval: bool = True  # graphslam.py / icpslam.py 화면에 지도 평가 칸 추가: 라이다 끝점이 정답 벽 경계선에서 얼마나 떨어졌나 (이미지 지도에서만)
    eval_view_radius: Optional[float] = None  # 지도 평가 칸을 로봇 중심 +- 이 반경[m] 만 확대해서 따라다니며 보여줌 (cm 단위 오차를 눈으로 볼 때). None = 지도 전체
    view_radius: Optional[float] = 8  # 시뮬레이션(오른쪽) 화면을 로봇 중심 +- 이 반경 [m] 만큼만 보여주고 로봇을 따라다님 (작을수록 확대, 지도 가장자리 밖은 안 보임). None = 지도 전체. 왼쪽 지도는 항상 전체
    trail_time: Optional[float] = 10.0  # 궤적을 남기는 시간 [s]. 이보다 오래된 건 지움 (None = 안 지움)
    robot_size: float = 0.4  # 로봇 삼각형 크기 [m]
    fov_draw_radius: float = 10.0  # 시야각 부채꼴을 그릴 반지름 [m] (그리기 전용, 관측 거리와 무관)
    particle_draw_ratio: float = 0.2  # 전체 파티클 중 화면에 그릴 비율 (0.05 = 5%, 1.0 = 전부). 매 프레임 랜덤 샘플링
    min_particles_draw: int = 5  # 비율로 계산한 개수가 이보다 적어도 최소 이만큼은 그림 (파티클이 이보다 적으면 전부)
    particle_dir_len: float = 0.5  # 파티클 방향선 길이 [m]


@dataclass
class GraphSlamConfig:
    """그래프 SLAM (graphslam.py). 고정 레퍼런스(랜드마크)를 번호로 알아보고, 키프레임 위치들을 한꺼번에 최적화"""
    keyframe_dt: float = 1.0  # 이 시간[s]마다 키프레임(그래프 노드) 하나를 추가
    optimize_every: float = 10.0  # 이 시간[s]마다 그래프 전체를 다시 최적화
    max_iter: int = 1  # Gauss-Newton 최대 반복 횟수
    
    # 그래프가 "믿는" 노이즈 (엣지 가중치 = 이 분산의 역수)
    sigma_range: float = 0.01  # 랜드마크 거리 관측 [m]
    sigma_bearing: float = math.radians(0.5)  # 랜드마크 각도 관측 [rad]
    odo_sigma_xy: float = 0.01  # 키프레임 사이 오도메트리 이동량 (x, y) [m]
    odo_sigma_yaw: float = math.radians(0.5)  # 키프레임 사이 오도메트리 회전량 [rad]
    
    n_scan_rays: int = 500  # 지도용 라이다 광선 수 (최적화된 위치로 지도를 다시 그릴 때 사용)
    draw_every: int = 1  # 화면을 이 step 마다 한 번만 그림 (1 = 매 step)
    
    # 체크포인트 따라가기 (키보드 대신). 화면을 왼쪽 클릭하면 체크포인트 추가, 오른쪽 클릭하면 전부 지움
    waypoints: List[List[float]] = field(default_factory=list)  # 미리 정해 둘 체크포인트 [[x, y], ...] [m]
    goal_tolerance: float = 0.4  # 체크포인트에 이 거리[m] 안으로 들어오면 도착으로 보고 다음으로
    v_max: float = 1.5  # 최대 속도 [m/s]
    yaw_rate_max: float = 1.5  # 최대 각속도 [rad/s]
    k_v: float = 1.0  # 속도 = k_v * 남은 거리 (v_max 로 자름)
    k_w: float = 2.5  # 각속도 = k_w * 방향 오차 (yaw_rate_max 로 자름)
    stuck_time: float = 0.5  # 앞으로 가라고 했는데 이 시간[s] 동안 벽에 막혀 못 움직이면 그 체크포인트는 도착한 것으로 처리


@dataclass
class IcpSlamConfig:
    """ICP SLAM (icpslam.py). 랜드마크 없이 라이다 스캔만 쓴다. 이웃한 키프레임의 스캔을 ICP 로 맞춰 이동량을 재고,
    예전에 지나간 곳에 다시 오면(루프) 그 스캔과도 맞춰서, 키프레임 위치 전체를 그래프로 한꺼번에 최적화"""
    # 키프레임(스캔 + 그래프 노드)은 마지막 키프레임에서 오도메트리로 이만큼 움직이거나 돌았을 때 추가 (멈춰 있으면 안 생김)
    keyframe_dist: float = 1.5  # 이동 거리 [m]
    keyframe_angle: float = math.radians(30.0)  # 회전량 [rad] (제자리 회전 때도 스캔끼리 겹치는 부분이 남도록)
    optimize_every: float = 10.0  # 이 시간[s]마다 그래프 전체를 다시 최적화
    max_iter: int = 10  # Gauss-Newton 최대 반복 횟수
    n_scan_rays: int = 500  # 라이다 광선 수 (ICP 와 지도 둘 다 이 스캔을 씀)
    
    # ICP (scan matching)
    icp_max_iter: int = 10  # ICP 최대 반복 횟수
    icp_max_corr_dist: float =  0.5 # 가장 가까운 점이 이 거리[m]보다 멀면 대응점으로 안 씀 (겹치지 않는 부분을 버림)
    icp_tol: float = 1e-4  # 한 반복에서 이동량 변화가 이보다 작으면 수렴
    icp_min_inliers: int = 100  # 대응점이 이보다 적으면 실패로 보고 오도메트리를 씀
    icp_max_err: float = 0.1  # 수렴 후 대응점 평균 거리[m]가 이보다 크면 실패
    
    # 루프 닫기: 지금 추정 위치가 예전 키프레임과 가까우면 그 스캔과도 ICP 로 맞춰서 엣지를 추가
    loop_dist: float = 0.5  # 이 거리[m] 안의 예전 키프레임을 후보로
    loop_min_gap: int = 10  # 키프레임 번호 차이가 이보다 작으면 "예전"이 아니라고 봄 (바로 옆은 이미 이웃 엣지가 있음)
    loop_max_err: float = 0.2  # 루프 ICP 는 초기값이 나쁠 수 있어서 더 엄격하게 (잘못 닫으면 지도가 망가짐)
    
    # 그래프가 "믿는" 노이즈 (엣지 가중치 = 이 분산의 역수)
    icp_sigma_xy: float = 0.05  # ICP 로 잰 이동량 (x, y) [m]
    icp_sigma_yaw: float = math.radians(3)  # ICP 로 잰 회전량 [rad]
    odo_sigma_xy: float = 0.005  # ICP 실패 시 대신 쓰는 오도메트리 이동량 [m]
    odo_sigma_yaw: float = math.radians(0.5)  # 오도메트리 회전량 [rad]
    loop_sigma_xy: float = 0.01  # 루프 엣지 (예전 키프레임 스캔과 ICP 로 잰 상대 위치) 의 이동량 (x, y) [m]. 작을수록 루프를 닫을 때 고리 안 이웃 엣지들이 더 많이 휨
    loop_sigma_yaw: float = math.radians(1)  # 루프 엣지의 회전량 [rad]
    draw_every: int = 1  # 화면을 이 step 마다 한 번만 그림 (1 = 매 step)
    
    # 체크포인트 따라가기 (GraphSlamConfig 와 같음. 화면 왼쪽 클릭 = 추가, 오른쪽 클릭 = 전부 지움)
    waypoints: List[List[float]] = field(default_factory=list)
    goal_tolerance: float = 0.4
    v_max: float = 1.5
    yaw_rate_max: float = 1.5
    k_v: float = 1.0
    k_w: float = 2.5
    stuck_time: float = 0.5


@dataclass
class MappingConfig:
    """탐색하면서 지도 그리기 (lidar_to_grid_map.py 의 MapBuilder). 화면 왼쪽에 로봇이 지금까지 본 곳만 그려짐"""
    enabled: bool = True
    resolution: Optional[float] = 0.05  # 탐색 지도의 한 칸 크기 [m/px]. None 이면 원본 지도(image_map)의 해상도를 따름
    save_path: Optional[str] = "output_image/explored_map.png"  # 시뮬 종료/ s 키 입력 시 격자 지도를 이 이미지로 저장 (None 이면 저장 안 함)
    use_estimated_pose: bool = False  # False: 실제 위치(x_true)로 그림 = 깨끗한 지도, True: 필터 추정 위치(x_est)로 그림 = 위치 오차가 번진 지도
    use_flood_fill: bool = False  # True: 광선 사이 빈틈도 부채꼴 안쪽을 flood fill 로 빈 공간 처리 (관측점이 많을수록 잘 맞음). False: 광선이 지나간 칸만
    l_free: float = -0.2  # 광선이 지나간 칸이 받는 log-odds 변화량 (음수 = 빈 공간 쪽. 클수록 빨리 "빈 칸"이 됨)
    l_occ: float = 0.3  # 끝점 칸이 받는 log-odds 변화량 (양수 = 장애물 쪽. 클수록 빨리 "막힌 칸"이 됨)
    l_min: float = -6.0  # log-odds 하한 (너무 확신하면 나중에 고치기 힘들어서 제한)
    l_max: float = 6.0  # log-odds 상한
    occ_thresh: float = 0.7  # 점유 확률이 이 값을 넘으면 "막힌 칸"
    free_thresh: float = 0.3  # 점유 확률이 이 값 미만이면 "빈 칸" (두 값 사이는 "모름")
    eval_tol: float = 0.05  # 지도 평가(왼쪽 칸)에서 벽 칸이 맞았다고 보는 실제 벽 경계선까지 거리 [m]. 칸 하나 정도 (벽 표면을 맞힌 점은 경계선 바로 앞 칸에 찍히기도 해서 0 이면 안 됨)
    eval_color_max: float = 0.2  # 지도 평가 그림에서 라이다 끝점 색: 경계선까지 0 = 초록 ~ 이 거리[m] 이상 = 빨강


@dataclass
class Config:
    """전체 설정"""
    sim_time: float = 50.0  # 시뮬레이션 총 시간 [s] (키보드 모드에서는 무시)
    robot: RobotConfig = field(default_factory=RobotConfig)
    sim_noise: SimNoiseConfig = field(default_factory=SimNoiseConfig)
    room: RoomConfig = field(default_factory=RoomConfig)
    image_map: ImageMapConfig = field(default_factory=ImageMapConfig)
    keyboard: KeyboardConfig = field(default_factory=KeyboardConfig)
    mapping: MappingConfig = field(default_factory=MappingConfig)
    viz: VizConfig = field(default_factory=VizConfig)
    graphslam: GraphSlamConfig = field(default_factory=GraphSlamConfig)
    icpslam: IcpSlamConfig = field(default_factory=IcpSlamConfig)

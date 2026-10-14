"""

Particle Filter localization sample


[이 파일 설명]
python robotics 의 Particle Filter 예제 코드를 기반으로,
실내 공간(벽) 안에서 로봇이 움직이고, 벽에서 랜덤 샘플링한
레퍼런스 포인트까지의 "거리"만 관측해서 파티클 필터로 로봇 위치를 추정한다.

전체 흐름 (매 time step):
  1. 입력 u (속도, 각속도) 생성
  2. 실제 로봇(x_true) 이동 + 관측 z 생성 / 노이즈 낀 입력으로 Dead Reckoning(x_dr) 이동
  3. 파티클 필터: 예측(predict) -> 가중치 계산(update) -> 정규화 -> 필요시 리샘플링
  4. 가중 평균으로 추정 위치 x_est, 공분산 p_est 계산

"""
import math
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon, Wedge

from config import Config, KeyboardConfig, RobotConfig, SimNoiseConfig
from lidar_to_grid_map import MapBuilder, scan_to_world  # 탐색한 정보로 만드는 픽셀맵 (객체로 관리)
from map import build_map


def rot_mat_2d(angle):
    """2D 회전행렬 (PythonRobotics utils.angle.rot_mat_2d 와 동일, scipy 없이 numpy로 구현)"""
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s],
                     [s, c]])


class Robot:
    """
    시뮬레이션 상의 로봇. 상태 벡터는 [x, y, yaw, v]^T
      x_true : 실제 로봇 상태 (노이즈 없는 입력으로 이동) -> ground truth
      x_dr   : Dead Reckoning 상태 (노이즈 낀 입력으로 이동) -> 필터 없이 odometry만 쓴 결과
    """

    def __init__(self, cfg: RobotConfig, sim_noise: SimNoiseConfig, x0=None, world=None):
        """
        설정은 전부 RobotConfig 로 주입한다 (전역 상수 참조 없음).
          cfg       : RobotConfig (particle, sigma_*, dt, max_range, fov, n_threshold_ratio, v, yaw_rate). sigma_* = 필터가 믿는 노이즈
          sim_noise : SimNoiseConfig. 시뮬레이션에서 센서 측정값/입력에 실제로 섞이는 노이즈 (필터는 모름)
          x0    : 초기 상태 [x, y, yaw, v] (기본: 원점)
          world : GridMap(map.py). 주면 실제 로봇이 벽을 못 뚫고, 벽에 가린 레퍼런스는 관측 안 됨
        """
        x0 = np.zeros((4, 1)) if x0 is None else np.array(x0, dtype=float).reshape(4, 1)
        self.x_true = x0.copy()
        self.x_dr = x0.copy()
        self.v = cfg.v  # [m/s]
        self.yaw_rate = cfg.yaw_rate  # [rad/s]
        self.u = np.array([[self.v, self.yaw_rate]]).T

        self.dt = cfg.dt
        self.max_range = cfg.max_range
        self.fov = cfg.fov
        self.n_rays = cfg.n_rays  # 0 이면 고정 레퍼런스, 1 이상이면 시야각 광선으로 관측점 생성
        self.world = world

        # noise (표준편차)
        self.sigma_range = cfg.sigma_range  # [필터] 거리 관측 노이즈 (update_weights 의 σ)
        self.sigma_v = cfg.sigma_v  # [필터] 입력 속도 노이즈 (predict_particles 의 퍼짐)
        self.sigma_yaw_rate = cfg.sigma_yaw_rate  # [필터] 입력 각속도 노이즈 (predict_particles 의 퍼짐)
        self.sim_noise = sim_noise  # [시뮬레이션] 실제 센서/입력 노이즈 (observation 에서만 씀)

        # filter properties
        self.particle = cfg.particle  # 파티클 개수
        self.n_threshold = cfg.particle * cfg.n_threshold_ratio  # N_eff 가 이 값보다 작으면 리샘플링
        self.px = np.repeat(x0, self.particle, axis=1)  # 파티클들 (처음엔 전부 초기 상태에서 시작)
        self.pw = np.zeros((1, self.particle)) + 1.0 / self.particle  # 파티클 가중치 (균등)

    def calc_input(self, v, yaw_rate):
        """로봇에 주는 제어 입력. 일정한 속도 + 일정한 각속도 -> 원을 그리며 움직임"""
        self.u = np.array([[v, yaw_rate]]).T

    def predice_state(self, u):
        """Dead Reckoning 상태(x_dr)를 입력 u로 이동"""
        self.x_dr = self.motion_model(self.x_dr, u)
        return self.x_dr

    def predict_particles(self, u):
        """
        파티클 필터 예측(Predict) 단계.
        입력 u에 노이즈(sigma_v, sigma_yaw_rate)를 랜덤하게 섞어서 파티클마다 다르게 이동
        -> 입력의 불확실성을 파티클의 퍼짐으로 표현
        """
        for ip in range(self.particle):
            x = self.px[:, ip:ip + 1]
            ud1 = u[0, 0] + np.random.randn() * self.sigma_v
            ud2 = u[1, 0] + np.random.randn() * self.sigma_yaw_rate
            ud = np.array([[ud1, ud2]]).T
            self.px[:, ip] = self.motion_model(x, ud)[:, 0]
        return self.px

    def calc_delta(self, z):
        """
        파티클마다, 관측된 레퍼런스마다 delta(= 예상 거리 - 실제 관측 거리)를 구한다.
        이 파티클이 "진짜 위치"라면 레퍼런스까지 거리가 얼마여야 하는지(pre_z)를
        계산해서 실제 관측(z[i, 0])과 비교. 가중치는 건드리지 않는다.
          z      : 관측값들 [거리, 레퍼런스 x, 레퍼런스 y]
          return : delta (파티클 수 x 관측 수)
        """
        delta = np.zeros((self.particle, len(z[:, 0])))
        for ip in range(self.particle):
            for i in range(len(z[:, 0])):
                dx = self.px[0, ip] - z[i, 1]
                dy = self.px[1, ip] - z[i, 2]
                pre_z = math.hypot(dx, dy)  # 파티클 기준 예상 거리
                delta[ip, i] = pre_z - z[i, 0]
        return delta

    def update_weights(self, delta):
        """
        delta로 가중치(가우시안 우도)를 갱신하고 정규화한다.
        오차가 작을수록 우도가 크고, 관측이 여러 개면 곱해서 반영 (독립 가정)

        우도를 그대로 곱하면 파티클이 진짜 위치에서 멀 때 모든 가중치가 0으로 언더플로우해서
        pw.sum() == 0 -> NaN이 된다. 그래서 로그 영역에서 더하고, 최댓값을 빼서 정규화한다.
        (수학적으로는 곱하는 것과 같고, 가장 나은 파티클의 가중치가 항상 1이 되어 0으로 안 죽음)
        """
        sigma = self.sigma_range
        log_w = np.log(np.maximum(self.pw[0], 1e-300))  # 이전 가중치 (0이면 로그가 -inf라서 바닥값)
        for ip in range(self.particle):
            for i in range(delta.shape[1]):
                log_w[ip] += gauss_log_likelihood(delta[ip, i], sigma)

        w = np.exp(log_w - log_w.max())  # 최댓값을 빼서 언더플로우 방지
        self.pw = (w / w.sum())[np.newaxis, :]  # 가중치 합이 1이 되도록 정규화
        return self.pw

    def observation(self, u, rf_id):
        """
        시뮬레이션 한 step 진행. x_true, x_dr를 갱신하고 (z, ud)를 반환한다.
          z  : 관측값. 각 행이 [노이즈 낀 거리, 레퍼런스 x, 레퍼런스 y, 빔 각도]
               빔 각도: 로봇이 보는 방향(실제 yaw) 기준으로 잰 상대 각도 [rad] (-pi ~ pi, 왼쪽이 +)
          ud : 노이즈 낀 입력 (파티클 필터의 예측 단계에 사용)
        """
        x_next = self.motion_model(self.x_true, u)
        # 벽 충돌: 이동할 곳이 벽이면 위치는 그대로 두고 방향(yaw)만 바꾼다
        if self.world is not None and not self.world.is_free(x_next[0, 0], x_next[1, 0]):
            x_next[0, 0], x_next[1, 0] = self.x_true[0, 0], self.x_true[1, 0]
            x_next[3, 0] = 0.0
        self.x_true = x_next

        # 광선 모드: 지금 시야각 안에서 실제로 보이는 벽 위의 점들을 관측 대상(레퍼런스)으로 쓴다
        if self.n_rays > 0 and hasattr(self.world, "raycast"):
            rf_id = self.world.raycast(self.x_true[0, 0], self.x_true[1, 0], self.x_true[2, 0],
                                       self.fov, self.max_range, self.n_rays)

        z = np.zeros((0, 4))

        for i in range(len(rf_id[:, 0])):

            # 레퍼런스 포인트와 "실제" 로봇 사이의 거리 계산
            dx = self.x_true[0, 0] - rf_id[i, 0]
            dy = self.x_true[1, 0] - rf_id[i, 1]
            d = math.hypot(dx, dy)

            # 로봇이 보는 방향 기준 각도 차이 (-pi ~ pi로 정규화)
            bearing = math.atan2(-dy, -dx) - self.x_true[2, 0]
            bearing = math.atan2(math.sin(bearing), math.cos(bearing))

            # 시야각 안에 있고, 벽에 가려지지 않은 점만 관측됨
            if d <= self.max_range and abs(bearing) <= self.fov / 2.0 and (
                    self.world is None or
                    self.world.is_visible(self.x_true[0, 0], self.x_true[1, 0],
                                          rf_id[i, 0], rf_id[i, 1])):
                # 관측치에도 노이즈가 있다고 가정, 
                # 즉 Rfid-to-robot 거리 측정 센서가 오차를 갖는다고 가정
                dn = d + np.random.randn() * self.sim_noise.sigma_range
                zi = np.array([[dn, rf_id[i, 0], rf_id[i, 1], bearing]])
                z = np.vstack((z, zi))


        # 속도와 각속도에 노이즈 추가 (엔코더/IMU 오차 흉내)
        ud1 = u[0, 0] + np.random.randn() * self.sim_noise.sigma_v
        ud2 = u[1, 0] + np.random.randn() * self.sim_noise.sigma_yaw_rate
        ud = np.array([[ud1, ud2]]).T

        # 노이즈 낀 입력으로 Dead Reckoning -> 시간이 갈수록 오차가 누적됨
        self.x_dr = self.motion_model(self.x_dr, ud)

        return z, ud

    def calc_covariance(self, x_est):
        """
        가중 파티클 분포의 공분산 행렬 계산 (추정 불확실성)
        see ipynb doc
        """
        cov = np.zeros((4, 4))
        for i in range(self.particle):
            dx = (self.px[:, i:i + 1] - x_est)  # 추정값으로부터의 편차
            cov += self.pw[0, i] * dx @ dx.T  # 가중치를 곱해서 누적
        cov *= 1.0 / (1.0 - self.pw @ self.pw.T)  # 가중 샘플 공분산의 불편추정 보정

        return cov

    def pf_localization(self):
        """
        파티클 필터 위치 추정 (한 step의 마무리). 예측은 predict_particles,
        delta 계산은 calc_delta, 가중치 갱신은 update_weights에서 이미 끝났다고 보고,
        여기서는 추정값/공분산 계산 + 필요시 리샘플링만 한다.
          return : x_est (4 x 1), p_est (4 x 4)
        """
        # 추정 위치 = 가중치 평균 (가중치 큰 파티클에 더 비중)
        x_est = self.px.dot(self.pw.T)
        p_est = self.calc_covariance(x_est)

        # 유효 파티클 수: 가중치가 한쪽에 쏠릴수록 작아짐 (퇴화 degeneracy 지표)
        N_eff = 1.0 / (self.pw.dot(self.pw.T))[0, 0]
        if N_eff < self.n_threshold:
            self.re_sampling()
        return x_est, p_est

    def re_sampling(self):
        """
        Low variance 리샘플링 (systematic resampling)
        가중치가 큰 파티클은 여러 번 복제, 작은 파티클은 사라지게 하고
        리샘플링 후 모든 가중치를 1/NP로 초기화한다.
        """
        n = self.particle
        w_cum = np.cumsum(self.pw)  # 가중치 누적합 (0~1)
        base = np.arange(0.0, 1.0, 1 / n)  # 1/n 간격의 균등 구간
        re_sample_id = base + np.random.uniform(0, 1 / n)  # 랜덤 오프셋 하나만 뽑아 모두에 더함
        indexes = []
        ind = 0
        for ip in range(n):
            # 누적합 구간을 따라가며 해당 위치에 걸리는 파티클 인덱스 선택
            while re_sample_id[ip] > w_cum[ind]:
                ind += 1
            indexes.append(ind)

        self.px = self.px[:, indexes]
        self.pw = np.zeros((1, n)) + 1.0 / n  # 가중치 균등하게 초기화

    def motion_model(self, x, u):
        """
        운동 모델 (상태 x = [x, y, yaw, v]^T, 입력 u = [v, yaw_rate]^T)
          x_{t+1}   = x_t   + v * cos(yaw) * DT
          y_{t+1}   = y_t   + v * sin(yaw) * DT
          yaw_{t+1} = yaw_t + yaw_rate * DT
          v_{t+1}   = v  (입력 속도로 덮어씀)
        x_{t+1} = F x_t + B u 형태의 선형식으로 표현 (B는 현재 yaw에 의존)
        """
        F = np.array([[1.0, 0, 0, 0],
                      [0, 1.0, 0, 0],
                      [0, 0, 1.0, 0],
                      [0, 0, 0, 0]])  # 마지막 행이 0: 이전 v는 버리고 입력 v로 대체

        B = np.array([[self.dt * math.cos(x[2, 0]), 0],
                      [self.dt * math.sin(x[2, 0]), 0],
                      [0.0, self.dt],
                      [1.0, 0.0]])

        x = F.dot(x) + B.dot(u)

        return x


def gauss_likelihood(x, sigma):
    """평균 0, 표준편차 sigma인 가우시안의 확률밀도값. x는 (예측 거리 - 관측 거리) 오차"""
    p = 1.0 / math.sqrt(2.0 * math.pi * sigma ** 2) * \
        math.exp(-x ** 2 / (2 * sigma ** 2))

    return p


def gauss_log_likelihood(x, sigma):
    """gauss_likelihood의 로그값. 오차 x가 아무리 커도 0으로 언더플로우하지 않는다"""
    return -0.5 * math.log(2.0 * math.pi * sigma ** 2) - x ** 2 / (2 * sigma ** 2)


def plot_covariance_ellipse(x_est, p_est):  # pragma: no cover
    """추정 위치 주변에 공분산(x, y)을 타원으로 그림. 타원이 클수록 불확실"""
    p_xy = p_est[0:2, 0:2]
    # 고유값 = 타원 축 길이의 제곱, 고유벡터 = 타원 축 방향
    eig_val, eig_vec = np.linalg.eigh(p_xy)  # 공분산은 대칭행렬이라 eigh (eig는 복소수 경고가 남)

    # 큰 고유값/작은 고유값의 인덱스 구분
    if eig_val[0] >= eig_val[1]:
        big_ind = 0
        small_ind = 1
    else:
        big_ind = 1
        small_ind = 0

    t = np.arange(0, 2 * math.pi + 0.1, 0.1)

    # eig_val[big_ind] or eiq_val[small_ind] were occasionally negative
    # numbers extremely close to 0 (~10^-20), catch these cases and set the
    # respective variable to 0
    try:
        a = math.sqrt(eig_val[big_ind])  # 장축 반지름
    except ValueError:
        a = 0

    try:
        b = math.sqrt(eig_val[small_ind])  # 단축 반지름
    except ValueError:
        b = 0

    # 축 정렬된 타원 -> 장축 방향으로 회전 -> 추정 위치로 평행이동
    x = [a * math.cos(it) for it in t]
    y = [b * math.sin(it) for it in t]
    angle = math.atan2(eig_vec[1, big_ind], eig_vec[0, big_ind])
    fx = rot_mat_2d(angle) @ np.array([[x, y]])
    px = np.array(fx[:, 0] + x_est[0, 0]).flatten()
    py = np.array(fx[:, 1] + x_est[1, 0]).flatten()
    plt.plot(px, py, "--r")


class KeyboardController:
    """
    게임처럼 방향키를 "누르고 있는 동안" 입력이 결정되는 컨트롤러.
      위/아래 (누름) : 속도에 +/- accel * dt 만큼 가속 (최대 +/-v_max)
      위/아래 (뗌)   : brake 로 속도가 0으로 복귀
      좌/우 (누름)   : 각속도가 +/-yaw_max 를 향해 yaw_accel 로 변함 (왼쪽 = 반시계)
      좌/우 (뗌)     : 각속도가 0으로 복귀
      space : 즉시 정지    esc : 종료
    """

    def __init__(self, cfg: KeyboardConfig, dt):
        self.accel, self.brake, self.v_max = cfg.accel, cfg.brake, cfg.v_max
        self.yaw_max, self.yaw_accel = cfg.yaw_max, cfg.yaw_accel
        self.dt = dt
        self.keys = set()  # 현재 눌려있는 키
        self.v = 0.0
        self.yaw_rate = 0.0
        self.quit = False  # esc 를 누르거나 창을 닫으면 True -> main 루프가 끝나고 지도가 저장됨

    def on_press(self, event):  # pragma: no cover
        if event.key == 'escape':
            self.quit = True  # exit(0) 으로 바로 죽으면 저장 코드가 안 돌아서, 플래그만 세우고 루프가 끝나게 한다
        elif event.key == ' ':
            self.v, self.yaw_rate = 0.0, 0.0
        else:
            self.keys.add(event.key)

    def on_release(self, event):  # pragma: no cover
        self.keys.discard(event.key)

    @staticmethod
    def _move_toward(value, target, max_delta):
        """value를 target 쪽으로 최대 max_delta 만큼만 이동"""
        return value + float(np.clip(target - value, -max_delta, max_delta))

    def step(self):
        """한 time step 만큼 v, yaw_rate 를 갱신하고 (v, yaw_rate) 반환"""
        up, down = 'up' in self.keys, 'down' in self.keys
        left, right = 'left' in self.keys, 'right' in self.keys

        # 속도: 누르면 가속, 안 누르거나 동시에 누르면 0으로 감속
        if up != down:
            self.v += (self.accel if up else -self.accel) * self.dt
        else:
            self.v = self._move_toward(self.v, 0.0, self.brake * self.dt)
        self.v = float(np.clip(self.v, -self.v_max, self.v_max))

        # 각속도: 눌린 방향이 목표값, 떼면 목표 0
        target = self.yaw_max * (int(left) - int(right))
        self.yaw_rate = self._move_toward(self.yaw_rate, target, self.yaw_accel * self.dt)
        return self.v, self.yaw_rate


def draw_map(ax, builder, max_grid_cells=300):
    """
    ax 에 격자 지도를 그린다: 칸 경계선을 긋고, 알게 된 칸만 색을 칠한다.
    (광선이 지나간 빈 칸 = 초록, 막힌 칸 = 빨강, 아직 모르는 칸 = 흰색)
    (칸이 한 축에 max_grid_cells 개를 넘으면 선이 화면을 덮으니 경계선은 생략)
    return : imshow 객체 (매 step 마다 .set_data(builder.explored_view()) 로 갱신)
    """
    cmap = plt.get_cmap("RdYlGn_r").copy()
    cmap.set_bad("white")
    img = ax.imshow(builder.explored_view(), cmap=cmap, vmin=0, vmax=1, origin="upper",
                    extent=builder.extent, interpolation="nearest")  # nearest: 칸 사이를 섞지 않고 네모로
    if max(builder.shape) <= max_grid_cells:
        ax.set_xticks(np.linspace(builder.x_min, builder.x_max, builder.width + 1), minor=True)
        ax.set_yticks(np.linspace(builder.y_min, builder.y_max, builder.height + 1), minor=True)
        ax.grid(True, which="minor", color="gray", linewidth=0.3, alpha=0.6)
        ax.tick_params(which="minor", length=0)
    return img


def save_map_image(builder, path):
    """격자 지도를 이미지 파일(png 등)로 저장 (칸 경계선 + 막힌 칸 색칠 + colorbar)"""
    fig, ax = plt.subplots(figsize=(6, 6))
    img = draw_map(ax, builder)
    fig.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Grid map (resolution %.3f m/px)" % builder.resolution)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(config=None):
    print(__file__ + " start!!")

    cfg = Config() if config is None else config
    viz, kb, mapping = cfg.viz, cfg.keyboard, cfg.mapping
    time = 0.0
    room, x0, world = build_map(cfg)  # 지도 (벽 + 레퍼런스 포인트) 생성: map.py
    robot = Robot(cfg.robot, cfg.sim_noise, x0=x0, world=world)  # 실제 로봇 + Dead Reckoning
    builder = MapBuilder.from_map(  # 탐색한 정보로 지도 그리기
        room, mapping.resolution, mapping.use_flood_fill,
        l_free=mapping.l_free, l_occ=mapping.l_occ, l_min=mapping.l_min, l_max=mapping.l_max,
        occ_thresh=mapping.occ_thresh, free_thresh=mapping.free_thresh) if mapping.enabled else None

    def save_map():
        #탐색한 격자 지도를 이미지로 저장 (mapping.save_path)
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), mapping.save_path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        save_map_image(builder, path)
        print("map saved ->", path)

    # 상태 벡터 [x y yaw v]'
    x_est = robot.x_true.copy()  # 필터 추정값

    # 궤적 기록용 history
    h_x_est = x_est
    h_x_true = robot.x_true
    h_x_dr = robot.x_true

    # 키보드 조작: 방향키로 속도/각속도 지령을 바꾼다 (정지 상태에서 시작)
    cmd = {"v": 0.0, "yaw_rate": 0.0}
    ctrl = None

    # 화면: 왼쪽 = 탐색한 정보로 그린 지도, 오른쪽 = 시뮬레이션 (실제 로봇, 파티클, 추정)
    ax_map = ax_sim = None
    if viz.show_animation:
        if builder is not None:
            fig, (ax_map, ax_sim) = plt.subplots(1, 2, figsize=(2 * viz.figsize[0], viz.figsize[1]),
                                                 gridspec_kw={"wspace": 0.4})  # 좌우 사이를 벌려서 글자끼리 안 겹치게
            # 격자 지도: 칸 경계선을 긋고, 광선이 지나간 칸(초록)과 막힌 칸(빨강)에 색을 칠한다 (draw_map)
            map_img = draw_map(ax_map, builder)
            # colorbar(범례)는 지도 오른쪽 바로 옆에 붙인다. 지도 축 좌표 기준이라 지도 크기가 바뀌어도 따라가고, 지도의 공간을 떼지 않는다
            ax_cb = ax_map.inset_axes([1.02, 0.0, 0.035, 1.0])
            fig.colorbar(map_img, cax=ax_cb, label="occupancy probability")
            map_true, = ax_map.plot([], [], "o", color="royalblue", ms=7)  # 실제 로봇
            map_est, = ax_map.plot([], [], "o", color="red", ms=4)  # 필터 추정 위치
            ax_map.set_title("explored grid map (%s pose)\ngreen=free  red=occupied  white=unknown"
                             % ("estimated" if mapping.use_estimated_pose else "true"), fontsize=9)
        else:
            fig, ax_sim = plt.subplots(figsize=viz.figsize)

    if kb.enabled and viz.show_animation:
        for k in ("keymap.back", "keymap.forward"):  # 좌/우 방향키가 matplotlib 단축키와 겹치지 않게
            plt.rcParams[k] = [key for key in plt.rcParams[k] if key not in ("left", "right")]
        ctrl = KeyboardController(kb, dt=robot.dt)
        fig.canvas.mpl_connect('key_press_event', ctrl.on_press)
        fig.canvas.mpl_connect('key_release_event', ctrl.on_release)
        fig.canvas.mpl_connect('close_event', lambda e: setattr(ctrl, "quit", True))  # 창을 닫아도 정상 종료
        if builder is not None and mapping.save_path:
            plt.rcParams["keymap.save"] = []  # matplotlib 기본 저장창 단축키(s) 해제
            fig.canvas.mpl_connect('key_press_event', lambda e: e.key == 's' and save_map())
    else:
        cmd = {"v": 1.0, "yaw_rate": 0.1}  # 키보드를 안 쓰면 기존처럼 원을 그리며 이동

    while (kb.enabled and viz.show_animation) or cfg.sim_time >= time:
        if ctrl is not None and ctrl.quit:
            break
        time += robot.dt

        # 1. 로봇의 현재 입력(속도, 각속도) 가져오기 (키보드: 누르고 있는 키로 가속/복귀 계산)
        if ctrl is not None:
            cmd["v"], cmd["yaw_rate"] = ctrl.step()
        robot.calc_input(cmd["v"], cmd["yaw_rate"])
        u = robot.u

        # 2. 실제 로봇 이동 + 관측값 생성 (+ Dead Reckoning 계산)
        z, ud = robot.observation(u, room.rf_id)  # n_rays > 0 이면 내부에서 광선 관측점으로 대체됨
        x_true, x_dr = robot.x_true, robot.x_dr

        # 3. 파티클 예측 (노이즈 낀 입력 ud 사용)
        robot.predict_particles(ud)

        # 4. delta(예상 거리 - 관측 거리) 계산 -> 가우시안 우도로 가중치 갱신
        delta = robot.calc_delta(z)
        robot.update_weights(delta)

        # 5. 추정 위치/공분산 계산 + 필요시 리샘플링
        x_est, PEst = robot.pf_localization()
        px = robot.px

        # 6. 이번에 관측한 점들로 지도 갱신 (위치는 실제 위치 또는 필터 추정 위치)
        #    끝점은 정답 벽 좌표가 아니라 (그 위치 + 측정한 거리/각도) 로 만든 "로봇이 믿는 끝점"
        map_pose = x_est if mapping.use_estimated_pose else x_true
        seen = scan_to_world(map_pose, z[:, 0], z[:, 3])  # (n, 2) 로봇이 믿는 끝점의 월드 좌표
        if builder is not None:
            builder.update(map_pose, seen)

        # 기록 저장
        h_x_est = np.hstack((h_x_est, x_est))
        h_x_dr = np.hstack((h_x_dr, x_dr))
        h_x_true = np.hstack((h_x_true, x_true))
        if viz.trail_time is not None:  # 오래된 궤적 삭제: 최근 trail_time 초만 유지
            n_keep = int(viz.trail_time / robot.dt) + 1
            h_x_est, h_x_dr, h_x_true = h_x_est[:, -n_keep:], h_x_dr[:, -n_keep:], h_x_true[:, -n_keep:]

        if viz.show_animation:
            plt.sca(ax_sim)
            plt.cla()

            ray_mode = robot.n_rays > 0 and hasattr(world, "raycast")
            room.plot(show_ref=not ray_mode)  # 벽 (회색) + 레퍼런스 포인트 (검은 별). 광선 모드에선 고정 별 대신 지금 보이는 점만 그림
            if ray_mode:
                plt.plot(seen[:, 0], seen[:, 1], "*k")  # 로봇이 믿는 끝점 (위치 오차와 거리 노이즈가 반영됨)
            # 지도에 쓰는 위치 -> 로봇이 믿는 끝점을 잇는 검은 점선
            for i in range(len(seen)):
                plt.plot([map_pose[0, 0], seen[i, 0]], [map_pose[1, 0], seen[i, 1]], "--k", linewidth=0.8)
            # 시야각 영역: 로봇이 보는 방향 +- fov/2 부채꼴을 옅은 색으로 칠함
            yaw_deg = math.degrees(x_true[2, 0])
            half_deg = math.degrees(robot.fov) / 2.0
            plt.gca().add_patch(Wedge((x_true[0, 0], x_true[1, 0]), min(robot.max_range, viz.fov_draw_radius),
                                      yaw_deg - half_deg, yaw_deg + half_deg,
                                      color="m", alpha=0.12, lw=0))
            # 파티클 (초록 점) + 진행 방향 (얇은 선): 전체의 particle_draw_ratio 비율만큼 랜덤으로 뽑아서 그림
            n_draw = int(np.clip(round(robot.particle * viz.particle_draw_ratio),
                                 min(viz.min_particles_draw, robot.particle), robot.particle))
            idx = np.random.choice(robot.particle, n_draw, replace=False)
            sx, sy, syaw = px[0, idx], px[1, idx], px[2, idx]
            plt.plot(sx, sy, ".g")
            plt.plot(np.vstack((sx, sx + viz.particle_dir_len * np.cos(syaw))),
                     np.vstack((sy, sy + viz.particle_dir_len * np.sin(syaw))), "-g", linewidth=0.7)
            plt.plot(np.array(h_x_true[0, :]).flatten(),
                     np.array(h_x_true[1, :]).flatten(), "--b", linewidth=2.0)  # 실제 궤적 (파랑 점선)
            plt.plot(np.array(h_x_est[0, :]).flatten(),
                     np.array(h_x_est[1, :]).flatten(), "-r")  # PF 추정 궤적 (빨강)
            # 로봇: 진행 방향(yaw)을 향하는 삼각형 (뾰족한 쪽이 앞)
            c, s = math.cos(x_true[2, 0]), math.sin(x_true[2, 0])
            body = np.array([[1.0, 0.0], [-0.6, 0.5], [-0.6, -0.5]]) * viz.robot_size  # 앞, 뒤왼쪽, 뒤오른쪽
            body = body @ np.array([[c, s], [-s, c]]) + x_true[:2, 0]
            plt.gca().add_patch(Polygon(body, closed=True, facecolor="royalblue", edgecolor="k", zorder=5))
            plot_covariance_ellipse(x_est, PEst)  # 불확실성 타원
            # 보이는 범위. 왼쪽 지도는 항상 지도 전체.
            # 오른쪽 시뮬레이션은 view_radius 가 있으면 게임 카메라처럼 로봇을 따라다니는 고정 크기 창만 보여준다
            # (창은 지도 가장자리 밖으로 나가지 않고, 지도와 같은 가로세로 비율이라 좌우 패널의 모양이 같다)
            world_lim = (room.x_min, room.x_max, room.y_min, room.y_max)
            sim_lim = world_lim
            if viz.view_radius is not None:
                half_w = viz.view_radius
                half_h = viz.view_radius * (room.y_max - room.y_min) / (room.x_max - room.x_min)
                cx = float(np.clip(x_true[0, 0], room.x_min + half_w, room.x_max - half_w)) \
                    if 2 * half_w < room.x_max - room.x_min else (room.x_min + room.x_max) / 2.0
                cy = float(np.clip(x_true[1, 0], room.y_min + half_h, room.y_max - half_h)) \
                    if 2 * half_h < room.y_max - room.y_min else (room.y_min + room.y_max) / 2.0
                sim_lim = (cx - half_w, cx + half_w, cy - half_h, cy + half_h)
            for ax, (x_lo, x_hi, y_lo, y_hi) in ((ax_sim, sim_lim), (ax_map, world_lim)):
                if ax is not None:
                    ax.set_xlim(x_lo, x_hi)
                    ax.set_ylim(y_lo, y_hi)
                    ax.set_aspect("equal", adjustable="box")
            plt.grid(True)
            plt.title("v=%.1f m/s  yaw_rate=%.1f rad/s\n[hold arrows, space: stop, esc: quit]"
                      % (cmd["v"], cmd["yaw_rate"]), fontsize=9)
            if ax_map is not None:  # 왼쪽 지도 갱신
                map_img.set_data(builder.explored_view())
                map_true.set_data([x_true[0, 0]], [x_true[1, 0]])
                map_est.set_data([x_est[0, 0]], [x_est[1, 0]])
            plt.pause(0.001)

    if builder is not None and mapping.save_path:  # 시뮬레이션이 끝나면(esc / 창 닫기 포함) 저장
        save_map()


if __name__ == '__main__':
    main()

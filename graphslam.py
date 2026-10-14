"""
그래프 SLAM (pose graph) 예제

PythonRobotics 의 SLAM/GraphBasedSLAM/graph_based_slam.py (Atsushi Sakai) 를 이 프로젝트의
로봇 / 지도 / 설정에 맞게 옮긴 것.

[파티클 필터(particle_filter.py)와 다른 점]
  - 랜드마크(레퍼런스 포인트)의 정답 좌표를 쓰지 않는다. 로봇 기준의 거리, 각도, 그리고 "몇 번 랜드마크인지"만 쓴다.
  - 매 step 현재 위치만 고치는 대신, 지금까지의 키프레임 위치 전체를 한꺼번에 최적화한다.
  - 최적화된 위치로 지도(MapBuilder)를 처음부터 다시 그린다.

[그래프]
  노드 : 키프레임마다 로봇 위치 [x, y, yaw]
  엣지 1 (오도메트리) : 이웃한 두 키프레임 사이의 이동량 = Dead Reckoning 으로 잰 값
  엣지 2 (랜드마크)   : 같은 랜드마크를 본 두 키프레임.
                        "위치1 + 관측1 로 계산한 랜드마크 위치" 와 "위치2 + 관측2 로 계산한 위치" 가 같아야 한다.
  최적화 : Gauss-Newton (첫 노드는 고정. 원본은 H 에 단위행렬을 더해 고정하지만 그러면 첫 노드가 조금씩 밀려서,
           여기서는 첫 노드를 빼고 나머지만 푼다)

[원본 예제와 다른 점]
  - 원본은 랜드마크 엣지를 모든 키프레임 쌍으로 만든다 (키프레임 수의 제곱만큼 늘어남).
    여기서는 랜드마크마다 "바로 이전에 본 키프레임" 과 "처음 본 키프레임" 에만 잇는다.
    처음 본 키프레임과의 엣지가 오래전 위치와 지금 위치를 직접 묶어서 루프를 닫는 역할을 한다.
  - 원본에 없는 오도메트리 엣지를 넣었다. 랜드마크가 잘 안 보이는 구간에서도 그래프가 끊기지 않는다.

실행:
  python graphslam.py
  화면을 왼쪽 클릭하면 체크포인트가 추가되고, 로봇이 찍은 순서대로 따라간다. 오른쪽 클릭하면 전부 지운다.
  esc 또는 창 닫기로 종료. (체크포인트는 직선으로 따라가니, 벽이 있으면 모퉁이마다 찍어야 한다.
  가는 길에 벽에 막혀 GraphSlamConfig.stuck_time 동안 못 움직이면 그 체크포인트는 도착한 것으로 보고 다음으로 넘어간다)
  화면 없이 돌릴 때는 GraphSlamConfig.waypoints 에 체크포인트를 미리 넣고 sim_time 동안 돈다.
"""
import dataclasses
import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon, Wedge
from scipy import sparse
from scipy.sparse.linalg import spsolve

from config import Config
from lidar_to_grid_map import MapBuilder, scan_to_world
from map import build_map
from particle_filter import Robot, draw_map


def pi_2_pi(angle):
    """각도를 -pi ~ pi 로 정규화"""
    return math.atan2(math.sin(angle), math.cos(angle))


def relative_pose(x1, x2):
    """x1 위치에서 본 x2 의 상대 위치 (dx, dy, dyaw). x1, x2 = [x, y, yaw]"""
    c, s = math.cos(x1[2]), math.sin(x1[2])
    dx, dy = x2[0] - x1[0], x2[1] - x1[1]
    return np.array([c * dx + s * dy, -s * dx + c * dy, pi_2_pi(x2[2] - x1[2])])


def compose(x1, d):
    """x1 위치에서 상대 이동량 d = (dx, dy, dyaw) 만큼 움직인 위치"""
    c, s = math.cos(x1[2]), math.sin(x1[2])
    return np.array([x1[0] + c * d[0] - s * d[1], x1[1] + s * d[0] + c * d[1], pi_2_pi(x1[2] + d[2])])


class PoseGraph:
    """
    키프레임 위치(노드)와 제약(엣지)을 들고 있고, Gauss-Newton 으로 위치를 최적화한다.

    Attributes:
        poses: (N, 3) 배열. 키프레임 위치의 현재 추정값 [x, y, yaw].
        odo: 오도메트리 엣지 리스트. (i, j, 측정한 상대 이동량 (3,))
        sightings: 랜드마크 번호 -> [(키프레임 번호, 거리, 각도), ...] (본 순서대로)
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.poses = np.zeros((0, 3))
        self.odo = []
        self.sightings = {}
        self.omega_odo = np.diag([1.0 / cfg.odo_sigma_xy ** 2, 1.0 / cfg.odo_sigma_xy ** 2,
                                  1.0 / cfg.odo_sigma_yaw ** 2])

    def add_node(self, odo_meas, observations, first_pose=None):
        """
        키프레임을 하나 추가한다.

        Args:
            odo_meas: 이전 키프레임에서 이번 키프레임까지의 상대 이동량 (dx, dy, dyaw). 첫 노드면 무시.
            observations: (n, 3) 배열. 각 행이 [거리, 로봇 기준 각도, 랜드마크 번호].
            first_pose: 첫 노드의 위치 [x, y, yaw]. 첫 노드는 최적화에서 고정된다.
        """
        k = len(self.poses)
        if k == 0:
            guess = np.asarray(first_pose, dtype=float)
        else:
            guess = compose(self.poses[-1], odo_meas)  # 초기값: 직전 추정 위치 + 오도메트리
            self.odo.append((k - 1, k, np.asarray(odo_meas, dtype=float)))
        self.poses = np.vstack((self.poses, guess))
        for d, a, lid in observations:
            self.sightings.setdefault(int(lid), []).append((k, d, a))

    def landmark_pairs(self):
        """랜드마크 엣지를 만들 (키프레임 i, 거리, 각도), (키프레임 j, 거리, 각도) 쌍들"""
        pairs = []
        for seen in self.sightings.values():
            for n in range(1, len(seen)):
                pairs.append((seen[n - 1], seen[n]))  # 바로 이전에 본 키프레임과
                if n >= 2:
                    pairs.append((seen[0], seen[n]))  # 처음 본 키프레임과 (루프를 닫음)
        return pairs

    def _landmark_cov(self, d, c, s):
        """
        각 관측이 랜드마크 위치에 주는 불확실성 (m, 2, 2). 빔 방향은 거리 노이즈, 옆 방향은 거리 x 각도 노이즈.
          Sigma = R(t) diag(sigma_r^2, (d sigma_b)^2) R(t)^T,  c = cos(t), s = sin(t)
        """
        sr2 = self.cfg.sigma_range ** 2
        sb2 = (d * self.cfg.sigma_bearing) ** 2
        cov = np.empty((len(d), 2, 2))
        cov[:, 0, 0] = sr2 * c * c + sb2 * s * s
        cov[:, 1, 1] = sr2 * s * s + sb2 * c * c
        cov[:, 0, 1] = cov[:, 1, 0] = (sr2 - sb2) * c * s
        return cov

    def optimize(self):
        """
        Gauss-Newton 으로 모든 키프레임 위치를 고친다. 반환: (반복 횟수, 마지막 비용)

        속도를 위해 두 가지를 한다.
          - 엣지마다 파이썬 반복문을 돌지 않고, 같은 종류의 엣지를 numpy 배열로 한꺼번에 계산한다.
          - H 는 대부분이 0 인 희소 행렬이라 scipy.sparse 로 만들고 희소 해법(spsolve)으로 푼다.
            (꽉 찬 행렬은 노드 수의 제곱만큼 메모리를 쓰고, 풀이는 거의 세제곱으로 느려짐)
        """
        n = len(self.poses)
        if n < 2:
            return 0, 0.0
        x = self.poses.copy()

        # 엣지를 배열로 정리 (측정값은 반복 중에 바뀌지 않으므로 한 번만)
        oi = np.array([i for i, _, _ in self.odo])
        oj = np.array([j for _, j, _ in self.odo])
        oz = np.array([z for _, _, z in self.odo])
        pairs = self.landmark_pairs()
        li = np.array([p[0][0] for p in pairs], dtype=int)
        lj = np.array([p[1][0] for p in pairs], dtype=int)
        d1 = np.array([p[0][1] for p in pairs])
        a1 = np.array([p[0][2] for p in pairs])
        d2 = np.array([p[1][1] for p in pairs])
        a2 = np.array([p[1][2] for p in pairs])

        cost = 0.0
        for itr in range(1, self.cfg.max_iter + 1):
            blocks = []  # (노드 i, 노드 j, A, B, Omega, e) 묶음 (엣지 종류별로 배열)

            # 오도메트리 엣지: e = (추정 위치로 계산한 상대 이동량) - (오도메트리로 잰 상대 이동량)
            xi, xj = x[oi], x[oj]
            c, s = np.cos(xi[:, 2]), np.sin(xi[:, 2])
            dx, dy = xj[:, 0] - xi[:, 0], xj[:, 1] - xi[:, 1]
            e = np.column_stack((c * dx + s * dy - oz[:, 0],
                                 -s * dx + c * dy - oz[:, 1],
                                 np.arctan2(np.sin(xj[:, 2] - xi[:, 2] - oz[:, 2]),
                                            np.cos(xj[:, 2] - xi[:, 2] - oz[:, 2]))))
            m = len(oi)
            A = np.zeros((m, 3, 3))
            A[:, 0, 0], A[:, 0, 1], A[:, 0, 2] = -c, -s, -s * dx + c * dy
            A[:, 1, 0], A[:, 1, 1], A[:, 1, 2] = s, -c, -c * dx - s * dy
            A[:, 2, 2] = -1.0
            B = np.zeros((m, 3, 3))
            B[:, 0, 0], B[:, 0, 1] = c, s
            B[:, 1, 0], B[:, 1, 1] = -s, c
            B[:, 2, 2] = 1.0
            blocks.append((oi, oj, A, B, np.broadcast_to(self.omega_odo, (m, 3, 3)), e))

            # 랜드마크 엣지: e = (위치 j + 관측 j 로 계산한 랜드마크) - (위치 i + 관측 i 로 계산한 랜드마크)
            if len(li):
                t1, t2 = x[li, 2] + a1, x[lj, 2] + a2
                c1, s1, c2, s2 = np.cos(t1), np.sin(t1), np.cos(t2), np.sin(t2)
                p1 = x[li, :2] + (d1 * np.array([c1, s1])).T
                p2 = x[lj, :2] + (d2 * np.array([c2, s2])).T
                e = p2 - p1
                m = len(li)
                A = np.zeros((m, 2, 3))
                A[:, 0, 0], A[:, 0, 2] = -1.0, d1 * s1
                A[:, 1, 1], A[:, 1, 2] = -1.0, -d1 * c1
                B = np.zeros((m, 2, 3))
                B[:, 0, 0], B[:, 0, 2] = 1.0, -d2 * s2
                B[:, 1, 1], B[:, 1, 2] = 1.0, d2 * c2
                omega = np.linalg.inv(self._landmark_cov(d1, c1, s1) + self._landmark_cov(d2, c2, s2))
                blocks.append((li, lj, A, B, omega, e))

            # H (희소) 와 b 를 한꺼번에 채움
            rows, cols, vals = [], [], []
            b = np.zeros(3 * n)
            cost = 0.0
            for i, j, A, B, om, e in blocks:
                for r, cc, M in ((i, i, np.einsum('mki,mkl,mlj->mij', A, om, A)),
                                 (i, j, np.einsum('mki,mkl,mlj->mij', A, om, B)),
                                 (j, i, np.einsum('mki,mkl,mlj->mij', B, om, A)),
                                 (j, j, np.einsum('mki,mkl,mlj->mij', B, om, B))):
                    rows.append((3 * r[:, None, None] + np.arange(3)[None, :, None]).repeat(3, axis=2).ravel())
                    cols.append((3 * cc[:, None, None] + np.arange(3)[None, None, :]).repeat(3, axis=1).ravel())
                    vals.append(M.ravel())
                np.add.at(b, (3 * i[:, None] + np.arange(3)).ravel(), np.einsum('mki,mkl,ml->mi', A, om, e).ravel())
                np.add.at(b, (3 * j[:, None] + np.arange(3)).ravel(), np.einsum('mki,mkl,ml->mi', B, om, e).ravel())
                cost += float(np.einsum('mi,mij,mj->', e, om, e))
            H = sparse.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                                  shape=(3 * n, 3 * n)).tocsc()  # 같은 칸에 들어간 값은 자동으로 더해짐

            # 첫 노드는 좌표계의 기준이라 고정: 첫 노드의 행/열을 빼고 나머지만 푼다
            step = np.zeros(3 * n)
            step[3:] = spsolve(H[3:, 3:], -b[3:])
            x += step.reshape(n, 3)
            x[:, 2] = np.arctan2(np.sin(x[:, 2]), np.cos(x[:, 2]))
            if step @ step < 1e-8:
                break
        self.poses = x
        return itr, cost


def landmark_observations(z, rf_index, bearing_noise):
    """
    Robot.observation 의 z 를 그래프 SLAM 용 관측 [거리, 로봇 기준 각도, 랜드마크 번호] 로 바꾼다.
    랜드마크 번호는 레퍼런스 좌표로 찾는다 (= 원본 예제처럼 "어느 랜드마크인지는 안다" 는 가정).
    정답 좌표 자체는 그래프에 넘기지 않는다.
    """
    obs = []
    for d, rx, ry, bearing in z:
        lid = rf_index[(rx, ry)]
        obs.append((d, pi_2_pi(bearing + np.random.randn() * bearing_noise), lid))
    return np.array(obs).reshape(-1, 3)


def lidar_scan(world, x_true, robot, n_rays, range_sigma):
    """지도용 라이다 스캔 (거리, 로봇 기준 각도). 정답 지도에 광선을 쏴서 만들고 거리 노이즈를 섞는다"""
    if world is None or not hasattr(world, "raycast"):
        return np.zeros(0), np.zeros(0)
    pts = world.raycast(x_true[0], x_true[1], x_true[2], robot.fov, robot.max_range, n_rays)
    d = np.hypot(pts[:, 0] - x_true[0], pts[:, 1] - x_true[1])
    bearings = np.arctan2(pts[:, 1] - x_true[1], pts[:, 0] - x_true[0]) - x_true[2]
    bearings = np.arctan2(np.sin(bearings), np.cos(bearings))
    return d + np.random.randn(len(d)) * range_sigma, bearings


def build_grid(room, mapping, poses, scans):
    """키프레임 위치와 그때의 스캔으로 지도를 처음부터 다시 만든다 (log-odds 는 되돌릴 수 없어서 매번 새로)"""
    builder = MapBuilder.from_map(
        room, mapping.resolution, mapping.use_flood_fill,
        l_free=mapping.l_free, l_occ=mapping.l_occ, l_min=mapping.l_min, l_max=mapping.l_max,
        occ_thresh=mapping.occ_thresh, free_thresh=mapping.free_thresh)
    for pose, (ranges, bearings) in zip(poses, scans):
        if len(ranges):
            builder.update(pose, scan_to_world(pose, ranges, bearings))
    return builder


def robot_triangle(pose, size):
    """진행 방향(yaw)을 향하는 로봇 삼각형의 꼭짓점 (뾰족한 쪽이 앞). particle_filter.py 와 같은 모양"""
    c, s = math.cos(pose[2]), math.sin(pose[2])
    body = np.array([[1.0, 0.0], [-0.6, 0.5], [-0.6, -0.5]]) * size  # 앞, 뒤왼쪽, 뒤오른쪽
    return body @ np.array([[c, s], [-s, c]]) + pose[:2]


def follow_view(room, x, y, view_radius):
    """
    로봇을 따라다니는 화면 범위 (x_lo, x_hi, y_lo, y_hi). view_radius 가 None 이면 지도 전체.
    창은 지도 가장자리 밖으로 나가지 않고, 지도와 같은 가로세로 비율이다. (particle_filter.py 와 같은 규칙)
    """
    if view_radius is None:
        return room.x_min, room.x_max, room.y_min, room.y_max
    half_w = view_radius
    half_h = view_radius * (room.y_max - room.y_min) / (room.x_max - room.x_min)
    cx = float(np.clip(x, room.x_min + half_w, room.x_max - half_w)) \
        if 2 * half_w < room.x_max - room.x_min else (room.x_min + room.x_max) / 2.0
    cy = float(np.clip(y, room.y_min + half_h, room.y_max - half_h)) \
        if 2 * half_h < room.y_max - room.y_min else (room.y_min + room.y_max) / 2.0
    return cx - half_w, cx + half_w, cy - half_h, cy + half_h


def go_to_goal(pose, goal, gs):
    """
    체크포인트로 가는 속도/각속도. 도착했으면 None.
      각속도 = k_w * (체크포인트 방향 - 지금 방향)
      속도   = k_v * 남은 거리 * cos(방향 오차)  -> 방향이 크게 틀리면 거의 제자리에서 돌고, 맞으면 앞으로 감
    """
    dx, dy = goal[0] - pose[0], goal[1] - pose[1]
    dist = math.hypot(dx, dy)
    if dist < gs.goal_tolerance:
        return None
    err = pi_2_pi(math.atan2(dy, dx) - pose[2])
    yaw_rate = float(np.clip(gs.k_w * err, -gs.yaw_rate_max, gs.yaw_rate_max))
    v = min(gs.k_v * dist, gs.v_max) * max(math.cos(err), 0.0)
    return v, yaw_rate


def mean_error(est, true):
    """키프레임 위치 오차의 평균 [m]"""
    return float(np.mean(np.hypot(est[:, 0] - true[:, 0], est[:, 1] - true[:, 1]))) if len(est) else 0.0


def main(config=None):
    print(__file__ + " start!!")
    cfg = Config() if config is None else config
    gs, viz, mapping = cfg.graphslam, cfg.viz, cfg.mapping

    room, x0, world = build_map(cfg)
    # 랜드마크는 고정 레퍼런스 포인트로 관측 (광선 모드는 매번 다른 벽 점이라 같은 대상을 다시 알아볼 수 없음)
    robot = Robot(dataclasses.replace(cfg.robot, n_rays=0), cfg.sim_noise, x0=x0, world=world)
    rf_index = {(float(x), float(y)): i for i, (x, y) in enumerate(room.rf_id[:, :2])}
    range_sigma = cfg.sim_noise.sigma_range

    graph = PoseGraph(gs)
    scans = []  # 키프레임마다 지도용 스캔 (거리, 각도)
    kf_true, kf_dr = [], []  # 키프레임 시점의 정답 위치 / Dead Reckoning 위치 (비교용)
    h_true, h_dr = [], []  # 매 step 궤적 (그리기용)
    builder = None

    time, next_kf, next_opt = 0.0, 0.0, gs.optimize_every
    cmd = {"v": 0.0, "yaw_rate": 0.0}
    waypoints = [list(map(float, w)) for w in gs.waypoints]  # 남은 체크포인트 (앞에서부터 차례로)
    state = {"quit": False}
    stuck_steps = 0  # 벽에 막혀 못 움직인 연속 step 수

    fig = ax_map = ax_sim = None
    map_dirty = False  # 지도를 새로 만들었으면 왼쪽 패널을 다시 그림
    map_robot = None  # 왼쪽 지도 위의 실제 로봇 위치 점
    sim = None  # 오른쪽 화면의 그림 객체들 (처음 그릴 때 만듦)
    step_count = 0
    if viz.show_animation:
        fig, (ax_map, ax_sim) = plt.subplots(1, 2, figsize=(2 * viz.figsize[0], viz.figsize[1]),
                                             gridspec_kw={"wspace": 0.3})

        def on_click(event):
            """왼쪽 클릭: 체크포인트 추가 (두 화면 어디든 월드 좌표로 찍힘), 오른쪽 클릭: 전부 지움"""
            if event.inaxes not in (ax_sim, ax_map) or event.xdata is None:
                return
            if event.button == 1:
                waypoints.append([event.xdata, event.ydata])
            elif event.button == 3:
                waypoints.clear()

        fig.canvas.mpl_connect('button_press_event', on_click)
        fig.canvas.mpl_connect('key_press_event', lambda e: e.key == 'escape' and state.update(quit=True))
        fig.canvas.mpl_connect('close_event', lambda e: state.update(quit=True))

    # 화면이 있으면 창을 닫거나 esc 를 누를 때까지, 화면이 없으면 sim_time 동안 돈다
    while (viz.show_animation or cfg.sim_time >= time) and not state["quit"]:
        time += robot.dt

        # 1. 체크포인트로 가는 입력 계산 -> 이동 + 관측 (z 의 각 행: [노이즈 낀 거리, 레퍼런스 x, 레퍼런스 y, 빔 각도])
        #    로봇이 어디 있는지는 그래프가 추정하지만, 여기서는 운전이 목적이라 실제 위치로 따라간다
        cmd["v"], cmd["yaw_rate"] = 0.0, 0.0
        while waypoints:
            u_goal = go_to_goal(robot.x_true[:3, 0], waypoints[0], gs)
            if u_goal is not None:
                cmd["v"], cmd["yaw_rate"] = u_goal
                break
            waypoints.pop(0)  # 도착 -> 다음 체크포인트
        robot.calc_input(cmd["v"], cmd["yaw_rate"])
        prev_xy = robot.x_true[:2, 0].copy()
        z, _ = robot.observation(robot.u, room.rf_id)
        x_true, x_dr = robot.x_true[:3, 0].copy(), robot.x_dr[:3, 0].copy()

        # 벽에 막힘: 앞으로 가라고 했는데 위치가 그대로면 막힌 것. stuck_time 동안 계속 막히면 그 체크포인트는 도착 처리
        blocked = cmd["v"] > 1e-3 and np.allclose(x_true[:2], prev_xy)
        stuck_steps = stuck_steps + 1 if blocked else 0
        if waypoints and stuck_steps * robot.dt >= gs.stuck_time:
            waypoints.pop(0)
            stuck_steps = 0
        h_true.append(x_true)
        h_dr.append(x_dr)

        # 2. 키프레임 추가: 오도메트리 이동량 + 랜드마크 관측 + 지도용 스캔
        if time >= next_kf:
            next_kf += gs.keyframe_dt
            odo = relative_pose(kf_dr[-1], x_dr) if kf_dr else None
            graph.add_node(odo, landmark_observations(z, rf_index, cfg.sim_noise.sigma_bearing), first_pose=x_dr)
            scans.append(lidar_scan(world, x_true, robot, gs.n_scan_rays, range_sigma))
            kf_true.append(x_true)
            kf_dr.append(x_dr)

        # 3. 그래프 최적화 -> 최적화된 위치로 지도를 다시 그림
        if time >= next_opt:
            next_opt += gs.optimize_every
            itr, cost = graph.optimize()
            builder = build_grid(room, mapping, graph.poses, scans) if world is not None else None
            map_dirty = True
            print("t=%.1fs  nodes=%d  odo edges=%d  landmark edges=%d  iter=%d  cost=%.1f  "
                  "mean error: DR %.2f m -> graph %.2f m"
                  % (time, len(graph.poses), len(graph.odo), len(graph.landmark_pairs()), itr, cost,
                     mean_error(np.array(kf_dr), np.array(kf_true)), mean_error(graph.poses, np.array(kf_true))))

        # 4. 그리기 (draw_every step 마다 한 번. matplotlib 은 매번 그림 전체를 다시 그려서 계산보다 훨씬 느림)
        step_count += 1
        if viz.show_animation and (step_count % gs.draw_every == 0 or map_dirty):
            ht, hd, kp = np.array(h_true), np.array(h_dr), graph.poses

            # 오른쪽: 시뮬레이션 (로봇을 따라다니는 화면)
            # 벽, 랜드마크, 선, 삼각형은 처음 한 번만 만들고 매 step 데이터만 바꾼다 (매번 지우고 다시 그리면 매우 느림)
            if sim is None:
                plt.sca(ax_sim)
                room.plot(show_ref=True)  # 벽 (회색) + 랜드마크 (검은 별)
                sim = {
                    "obs": ax_sim.plot([], [], "--k", linewidth=0.8)[0],  # 실제 로봇 -> 이번에 관측한 랜드마크
                    "fov": ax_sim.add_patch(Wedge((0, 0), min(robot.max_range, viz.fov_draw_radius), 0, 0,
                                                  color="m", alpha=0.12, lw=0)),  # 시야각 부채꼴
                    "true": ax_sim.plot([], [], "--b", linewidth=2.0, label="true")[0],  # 실제 궤적
                    "dr": ax_sim.plot([], [], "-k", linewidth=1.0, label="dead reckoning")[0],  # 오도메트리만
                    "graph": ax_sim.plot([], [], ".-r", label="graph SLAM (keyframes)")[0],  # 최적화된 키프레임
                    "route": ax_sim.plot([], [], ":", color="darkorange", linewidth=1.5)[0],  # 로봇 -> 남은 체크포인트
                    "goals": ax_sim.plot([], [], "X", color="darkorange", ms=9, label="checkpoints")[0],
                    "robot": ax_sim.add_patch(Polygon(np.zeros((3, 2)), closed=True, facecolor="royalblue",
                                                      edgecolor="k", zorder=5)),  # 실제 로봇 (삼각형)
                }
                ax_sim.set_aspect("equal", adjustable="box")
                ax_sim.grid(True)
                ax_sim.legend(loc="upper right", fontsize=8)
            # 관측 선: 선분 사이를 NaN 으로 끊어서 선 객체 하나로 그림
            seg = np.full((3 * len(z), 2), np.nan)
            seg[0::3] = x_true[:2]
            seg[1::3] = z[:, 1:3]
            sim["obs"].set_data(seg[:, 0], seg[:, 1])
            yaw_deg, half_deg = math.degrees(x_true[2]), math.degrees(robot.fov) / 2.0
            sim["fov"].set_center((x_true[0], x_true[1]))
            sim["fov"].set_theta1(yaw_deg - half_deg)
            sim["fov"].set_theta2(yaw_deg + half_deg)
            sim["true"].set_data(ht[:, 0], ht[:, 1])
            sim["dr"].set_data(hd[:, 0], hd[:, 1])
            sim["graph"].set_data(kp[:, 0], kp[:, 1])
            wp = np.array(waypoints).reshape(-1, 2)
            sim["goals"].set_data(wp[:, 0], wp[:, 1])
            route = np.vstack((x_true[:2], wp))
            sim["route"].set_data(route[:, 0], route[:, 1])
            sim["robot"].set_xy(robot_triangle(x_true, viz.robot_size))
            x_lo, x_hi, y_lo, y_hi = follow_view(room, x_true[0], x_true[1], viz.view_radius)
            ax_sim.set_xlim(x_lo, x_hi)
            ax_sim.set_ylim(y_lo, y_hi)
            ax_sim.set_title("v=%.1f m/s  yaw_rate=%.1f rad/s  t=%.1fs  keyframes=%d\n"
                             "[left click: add checkpoint, right click: clear, esc: quit]"
                             % (cmd["v"], cmd["yaw_rate"], time, len(kp)), fontsize=9)

            # 왼쪽: 최적화된 위치로 다시 만든 지도 (지도가 바뀔 때만 다시 그리고, 로봇 점만 매 step 옮김)
            if map_dirty or map_robot is None:
                map_dirty = False
                ax_map.cla()
                if builder is not None:
                    draw_map(ax_map, builder, max_grid_cells=0)  # 칸 경계선은 생략 (눈금이 수백 개라 매 프레임 그리기가 매우 느려짐)
                ax_map.plot(kp[:, 0], kp[:, 1], ".-r", ms=3)
                map_robot, = ax_map.plot([], [], "o", color="royalblue", ms=6)
                ax_map.set_xlim(room.x_min, room.x_max)
                ax_map.set_ylim(room.y_min, room.y_max)
                ax_map.set_aspect("equal", adjustable="box")
                ax_map.set_title("grid map rebuilt from graph SLAM poses\n(updated every %.0fs)"
                                 % gs.optimize_every, fontsize=9)
            map_robot.set_data([x_true[0]], [x_true[1]])
            plt.pause(0.001)

    return graph, np.array(kf_true), np.array(kf_dr), builder


if __name__ == '__main__':
    main()

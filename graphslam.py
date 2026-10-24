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

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from config import Config
from lidar_to_grid_map import MapBuilder, grid_match, scan_to_world
from map import build_map
from particle_filter import Robot
from utils import cv_viewer as cvv


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


def empty_grid(room, mapping):
    """설정(MappingConfig)대로 room 과 같은 범위의 빈 격자 지도를 만든다"""
    return MapBuilder.from_map(
        room, mapping.resolution, mapping.use_flood_fill,
        l_free=mapping.l_free, l_occ=mapping.l_occ, l_min=mapping.l_min, l_max=mapping.l_max,
        occ_thresh=mapping.occ_thresh, free_thresh=mapping.free_thresh)


def build_grid(room, mapping, poses, scans):
    """키프레임 위치와 그때의 스캔으로 지도를 처음부터 다시 만든다 (log-odds 는 되돌릴 수 없어서 매번 새로)"""
    builder = empty_grid(room, mapping)
    for pose, (ranges, bearings) in zip(poses, scans):
        if len(ranges):
            builder.update(pose, scan_to_world(pose, ranges, bearings))
    return builder


class MapEval:
    """
    지도 평가 (VizConfig.show_map_eval). 정답 지도(입력 이미지)의 벽과 방의 경계선(map.GridMap.boundary_segments)을 기준으로 두 가지를 본다.

    [가운데 칸: 라이다 끝점] SLAM 위치가 맞나
      끝점 = 키프레임 위치 + 그때 잰 거리, 각도 (scan_to_world). 경계선까지 거리를 격자 없이 정확히 잰다 (방 쪽 +, 벽 안쪽 -).
      라이다(raycast)도 같은 경계선을 맞히기 때문에, 위치가 맞고 센서 노이즈가 없으면 거리는 정확히 0 이다.
      같은 스캔을 정답 위치(센서 노이즈만 남은 바닥) / Dead Reckoning 위치(보정 없음)로 찍은 값과 같이 본다.

    [왼쪽 칸: 점유 격자 지도] 최종 지도가 실제 지도와 맞나 (lidar_to_grid_map.grid_match)
      지도에서 정해진 칸(빈 칸 / 벽 칸) 중 실제 지도와 맞는 칸의 비율. 빈 칸은 실제로도 바닥이면, 벽 칸은 실제 벽에 붙어 있으면 맞음.
      같은 스캔을 정답 위치로 그린 지도(설정은 같음)의 비율도 같이 낸다 = 위치가 완벽할 때의 값

    정답 / Dead Reckoning 위치는 나중에 바뀌지 않아서 키프레임마다 한 번만 계산해 쌓는다 (정답 위치 지도도 이어서 그림).
    SLAM 위치는 최적화마다 바뀌어서 전부 다시 계산한다.
    """

    def __init__(self, room, mapping, world):
        self.world = world
        self.tol = mapping.eval_tol
        self.color_max = mapping.eval_color_max
        self.cell = mapping.resolution if mapping.resolution is not None else world.resolution
        self.walls = cvv.wall_image(world)  # 가운데 칸 배경 (정답 벽)
        self.true_map = empty_grid(room, mapping)  # 정답 위치로 그린 점유 지도 (왼쪽 칸 비교용)
        self.ref = {"true": [], "dr": []}  # 키프레임마다 기준 위치로 찍은 끝점의 경계선까지 거리
        self.scores = None
        self.points = np.zeros((0, 2))  # SLAM 위치로 찍은 끝점 (그리기용, 오차가 작은 것부터)
        self.colors = np.zeros((0, 3), np.uint8)
        self.wrong = None  # SLAM 지도에서 실제 지도와 안 맞는 칸 (왼쪽 칸에 파랑)
        self._line_mask = None  # 왼쪽 칸에 겹칠 정답 경계선 (패널 크기, 처음 그릴 때 만듦)

    def add_scan(self, pose_true, pose_dr, ranges, bearings):
        """키프레임 스캔 하나를 기준 위치 두 개로 찍어서 쌓는다 (정답 위치 지도에도 더함)"""
        if len(ranges):
            pts_true = scan_to_world(pose_true, ranges, bearings)
            self.true_map.update(pose_true, pts_true)  # build_grid 와 같은 순서, 같은 갱신 -> 정답 위치로 다시 만든 지도와 똑같음
            self.ref["true"].append(self.world.distance_to_boundary(pts_true))
            self.ref["dr"].append(self.world.distance_to_boundary(scan_to_world(pose_dr, ranges, bearings)))

    @staticmethod
    def stats(d):
        """(|거리| 평균, |거리| 95%, 거리 평균) [m]. 마지막은 부호 있는 평균 (+ 면 끝점이 방 쪽으로 떠 있음)"""
        if len(d) == 0:
            return math.nan, math.nan, math.nan
        return float(np.mean(np.abs(d))), float(np.percentile(np.abs(d), 95)), float(np.mean(d))

    def evaluate(self, builder, poses, scans):
        """SLAM 위치(키프레임마다)와 스캔, SLAM 점유 지도로 점수를 낸다 (최적화할 때마다)"""
        pts = [scan_to_world(p, r, b) for p, (r, b) in zip(poses, scans) if len(r)]
        pts = np.vstack(pts) if pts else np.zeros((0, 2))
        d = self.world.distance_to_boundary(pts)
        order = np.argsort(np.abs(d))  # 오차가 큰 점이 위에 그려지게
        self.points = pts[order]
        self.colors = cvv.value_colors(np.abs(d[order]), self.color_max)

        grid = grid_match(builder, self.world, self.tol)
        self.wrong = grid["wrong"]
        self.scores = {"slam": self.stats(d),
                       "true": self.stats(np.concatenate(self.ref["true"]) if self.ref["true"] else []),
                       "dr": self.stats(np.concatenate(self.ref["dr"]) if self.ref["dr"] else []),
                       "grid": grid,
                       "grid_true": grid_match(self.true_map, self.world, self.tol)}

    def endpoint_text(self, name):
        """
        가운데 칸 점수 (세 줄). 거리는 끝점에서 실제 벽 경계선까지, 평균 / 95% 는 크기만 본 값이고
        bias 는 부호를 살린 평균: + 면 끝점이 벽 앞(방 쪽)에 떠 있고, - 면 벽 안쪽으로 파고든 쪽으로 치우쳤다는 뜻
        """
        if self.scores is None:
            return "(waiting for the first optimization)\n\n"
        slam, true, dr = (100 * np.array(self.scores[key]) for key in ("slam", "true", "dr"))  # [cm]
        side = "in front of the wall" if slam[2] > 0 else "inside the wall"
        return ("%s: points are %.1f cm off on average, 95%% are within %.1f cm\n"
                "same scans on true pose %.1f / %.1f cm, on odometry only (no SLAM) %.1f / %.1f cm\n"
                "bias %+.1f cm: points lean %s (+ in front, - inside)"
                % (name, *slam[:2], *true[:2], *dr[:2], slam[2], side))

    def grid_text(self, name):
        """왼쪽 칸 점수 (두 줄): 지도에서 정해진 칸 중 실제 지도와 맞는 칸의 비율 (SLAM 지도 / 정답 위치로 그린 지도)"""
        if self.scores is None:
            return "(waiting for the first optimization)\n"
        g, t = self.scores["grid"], self.scores["grid_true"]
        return ("%s map: %.1f%% of cells match (free cells %.1f%%, wall cells %.1f%%)\n"
                "true pose map: %.1f%% (free cells %.1f%%, wall cells %.1f%%)"
                % (name, 100 * g["all"], 100 * g["free"], 100 * g["wall"],
                   100 * t["all"], 100 * t["free"], 100 * t["wall"]))

    def draw(self, panel):
        """가운데 칸: 정답 벽 (회색) 위에 SLAM 끝점을 경계선까지 거리에 따라 초록(0) ~ 빨강(color_max 이상) 으로 찍는다"""
        panel.paint(*self.walls)
        # 화면 한 픽셀이 클수록(멀리 볼수록) 점을 키워서 보이게. 확대해서 보면 1 픽셀
        size = 1 if panel.meters_to_px(0.1) >= 2 else 2
        panel.pixels(self.points, self.colors, size)

    def draw_grid(self, panel, map_img, extent):
        """왼쪽 칸: 점유 지도에서 실제 지도와 안 맞는 칸을 파랑으로 칠하고, 실제 벽 경계선을 어둡게 겹친다.
        왼쪽 칸은 항상 지도 전체라 경계선 그림은 한 번만 만듦"""
        img = map_img
        if self.wrong is not None and self.wrong.shape == map_img.shape[:2]:
            img = map_img.copy()
            img[self.wrong] = cvv.BLUE
        panel.paint(img, extent)
        if self._line_mask is None or self._line_mask.shape != (panel.h, panel.w):
            self._line_mask = panel.line_mask(self.world.boundary_segments())
        panel.img[self._line_mask] = (panel.img[self._line_mask] * 0.3).astype(np.uint8)  # 밑의 색이 비치게 어둡게만


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


def make_window(name, room, viz, map_eval, waypoints):
    """
    SLAM 화면 (cv2): [격자 지도 | 지도 평가 (map_eval 이 있을 때) | 시뮬레이션].
    어느 패널이든 왼쪽 클릭하면 그 자리에 체크포인트 추가, 오른쪽 클릭하면 전부 지움.
      return : (Window, {"map": Panel, "eval": Panel, "sim": Panel})
    """
    h = viz.panel_px
    w = h * (room.x_max - room.x_min) / (room.y_max - room.y_min)  # 지도와 같은 가로세로 비율
    full = (room.x_min, room.x_max, room.y_min, room.y_max)
    panels = {"map": cvv.Panel(w, h, full)}
    if map_eval is not None:
        panels["eval"] = cvv.Panel(w, h, full)
    panels["sim"] = cvv.Panel(w, h, full)

    def on_click(left, x, y):
        if left:
            waypoints.append([x, y])
        else:
            waypoints.clear()

    return cvv.Window(name, list(panels.values()), title_lines=4, on_click=on_click), panels


def draw_map_panels(panels, map_img, map_eval, room, viz, kp, x_true):
    """
    격자 지도 패널과 지도 평가 패널: 지도 그림 + 최적화된 키프레임 (빨강) + 실제 로봇 위치 (파랑).
    지도 평가 패널은 VizConfig.eval_view_radius 가 있으면 로봇을 따라다니며 확대해서 보여 준다.
    """
    full = (room.x_min, room.x_max, room.y_min, room.y_max)
    for key in ("map", "eval"):
        if key not in panels:
            continue
        p = panels[key]
        if key == "map" and map_eval is not None:
            map_eval.draw_grid(p, map_img, full)
        elif key == "map":
            p.paint(map_img, full)
        else:
            p.view = follow_view(room, x_true[0], x_true[1], viz.eval_view_radius)
            map_eval.draw(p)
        p.polyline(kp[:, :2], cvv.RED)
        p.dots(kp[:, :2], cvv.RED, 2)
        p.dots([x_true[:2]], cvv.BLUE, 5, edge=cvv.BLACK)


def map_titles(name, gs, map_eval):
    """격자 지도 / 지도 평가 패널의 제목"""
    if map_eval is None:
        return ["grid map from %s poses (rebuilt every %.0fs)\ngreen=free  red=occupied  white=unknown"
                % (name, gs.optimize_every)]
    return ["grid map cells that match the real map  (black = real wall, blue = wrong cell)\n%s"
            % map_eval.grid_text(name),
            "how far lidar points are from the real wall (green = on it, red = %.0fcm+ off)\n%s"
            % (100 * map_eval.color_max, map_eval.endpoint_text(name))]


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
    stuck_steps = 0  # 벽에 막혀 못 움직인 연속 step 수

    win = panels = map_eval = None
    map_dirty = False  # 지도를 새로 만들었으면 지도 그림을 새로 만듦
    step_count = 0
    if viz.show_animation:
        # 지도 평가는 정답 격자(이미지 지도)가 있을 때만
        map_eval = MapEval(room, mapping, world) if viz.show_map_eval and hasattr(room, "grid") else None
        win, panels = make_window("graph SLAM", room, viz, map_eval, waypoints)
        walls = cvv.wall_image(room)  # 시뮬레이션 패널 배경 (정답 벽)
        map_img = np.full((1, 1, 3), 255, np.uint8)  # 격자 지도 그림 (첫 최적화 전에는 빈 그림)
        frame_time = robot.dt * gs.draw_every / viz.sim_speed if viz.sim_speed else 0.0

    # 화면이 있으면 창을 닫거나 esc 를 누를 때까지, 화면이 없으면 sim_time 동안 돈다
    while (viz.show_animation or cfg.sim_time >= time) and not (win is not None and win.closed):
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
            if map_eval is not None:
                map_eval.add_scan(x_true, x_dr, *scans[-1])

        # 3. 그래프 최적화 -> 최적화된 위치로 지도를 다시 그림
        if time >= next_opt:
            next_opt += gs.optimize_every
            itr, cost = graph.optimize()
            builder = build_grid(room, mapping, graph.poses, scans) if world is not None else None
            map_dirty = True
            print("t=%.1fs  nodes=%d  odo edges=%d  landmark edges=%d  iter=%d  cost=%.1f"
                  % (time, len(graph.poses), len(graph.odo), len(graph.landmark_pairs()), itr, cost))
            if map_eval is not None and builder is not None:
                map_eval.evaluate(builder, graph.poses, scans)
                print("  endpoints: " + "   ".join(map_eval.endpoint_text("graph SLAM").split("\n")[:2]))  # 세 번째 줄(bias 설명)은 화면에만
                print("  grid map:  " + map_eval.grid_text("graph SLAM").replace("\n", "   "))

        # 4. 그리기 (cv2: 지도는 그림으로 들고 있다가 보이는 부분만 붙이고, 선/점/글자만 덧그림)
        step_count += 1
        if win is not None and (step_count % gs.draw_every == 0 or map_dirty):
            if map_dirty:
                map_dirty = False
                if builder is not None:
                    map_img = cvv.occupancy_image(builder)
            ht, hd, kp = np.array(h_true), np.array(h_dr), graph.poses
            wp = np.array(waypoints).reshape(-1, 2)

            # 시뮬레이션 (로봇을 따라다니는 화면)
            sim = panels["sim"]
            sim.view = follow_view(room, x_true[0], x_true[1], viz.view_radius)
            sim.paint(*walls)  # 벽 (회색)
            if not hasattr(room, "grid"):
                sim.rect(room.x_min, room.x_max, room.y_min, room.y_max, cvv.GRAY)  # 사각형 방의 벽
            sim.markers(room.rf_id, cvv.BLACK)  # 랜드마크 (검은 별)
            sim.sector(x_true[:2], min(robot.max_range, viz.fov_draw_radius),
                       x_true[2] - robot.fov / 2.0, x_true[2] + robot.fov / 2.0, cvv.MAGENTA)  # 시야각
            sim.segments(np.repeat(x_true[None, :2], len(z), axis=0), z[:, 1:3], cvv.BLACK)  # 이번에 관측한 랜드마크
            sim.polyline(hd[:, :2], cvv.BLACK)  # Dead Reckoning 궤적
            sim.polyline(ht[:, :2], cvv.BLUE, 2)  # 실제 궤적
            sim.polyline(kp[:, :2], cvv.RED)  # 최적화된 키프레임
            sim.dots(kp[:, :2], cvv.RED, 2)
            sim.polyline(np.vstack((x_true[:2], wp)), cvv.ORANGE)  # 로봇 -> 남은 체크포인트
            sim.markers(wp, cvv.ORANGE, kind="x", size=11, thickness=2)
            sim.polygon(robot_triangle(x_true, viz.robot_size), cvv.BLUE)  # 실제 로봇
            sim.legend([("true", cvv.BLUE, "line"), ("dead reckoning", cvv.BLACK, "line"),
                        ("graph SLAM (keyframes)", cvv.RED, "line"), ("landmark", cvv.BLACK, "star"),
                        ("checkpoints", cvv.ORANGE, "x")])

            draw_map_panels(panels, map_img, map_eval, room, viz, kp, x_true)
            win.show(map_titles("graph SLAM", gs, map_eval) +
                     ["v=%.1f m/s  yaw_rate=%.1f rad/s  t=%.1fs  keyframes=%d\n"
                      "left click: add checkpoint  right click: clear  esc: quit"
                      % (cmd["v"], cmd["yaw_rate"], time, len(kp))], frame_time)

    return graph, np.array(kf_true), np.array(kf_dr), builder


if __name__ == '__main__':
    main()

"""
ICP SLAM (scan matching + pose graph) 예제

graphslam.py 와 같은 구조(키프레임 그래프 + Gauss-Newton)인데, 랜드마크를 전혀 쓰지 않고 라이다 스캔만 쓴다.

[graphslam.py 와 다른 점]
  - graphslam 은 "몇 번 랜드마크인지" 를 알려주는 관측을 쓴다. 여기서는 그런 정보가 없다.
    대신 두 키프레임의 스캔(벽 위의 점 구름)을 ICP 로 겹쳐 보고, 겹치게 만드는 이동량을 "두 위치 사이의 상대 위치" 로 쓴다.
  - 그래프의 엣지가 전부 "상대 위치" 한 종류다 (오도메트리 엣지와 같은 모양). 어디서 왔느냐만 다르다.
      이웃 엣지 : 바로 이전 키프레임과 ICP (초기값 = 오도메트리). ICP 가 실패하면 오도메트리 값을 그대로 씀
      루프 엣지 : 추정 위치가 예전 키프레임과 가까우면 그 스캔과 ICP (초기값 = 두 추정 위치의 차이)
    루프 엣지가 오래전 위치와 지금 위치를 직접 묶어서, 그동안 쌓인 오차를 그래프 전체에 나눠 없앤다.

[ICP (Iterative Closest Point)]
  점 구름 A 를 점 구름 B 위에 겹치는 이동량(dx, dy, dyaw)을 찾는다.
    1. 지금 이동량으로 A 를 옮긴다
    2. A 의 각 점에 가장 가까운 B 의 점을 짝으로 잡는다 (KD-tree). 너무 먼 짝은 버린다 (겹치지 않는 부분)
    3. 짝지어진 점들을 가장 잘 겹치는 회전+이동을 닫힌 형태(SVD)로 푼다
    4. 이동량을 갱신하고, 변화가 작아질 때까지 1 로
  초기값이 정답에서 많이 벗어나면 엉뚱한 짝을 잡고 틀린 곳에 수렴한다 (local minimum). 그래서 초기값으로
  오도메트리를 쓰고, 수렴 후 평균 대응 거리가 크면 실패로 본다.

[한계]
  - 긴 복도처럼 한 방향으로 특징이 없는 곳에서는 ICP 가 그 방향으로 미끄러진다 (점-점 ICP 의 알려진 약점).
  - 루프를 잘못 닫으면(다른 곳을 같은 곳으로 착각) 지도가 크게 망가진다. loop_max_err 를 엄격하게 두는 이유.

실행:
  python icpslam.py
  조작은 graphslam.py 와 같다 (왼쪽 클릭: 체크포인트 추가, 오른쪽 클릭: 전부 지움, esc: 종료).
  화면 없이 돌릴 때는 IcpSlamConfig.waypoints 에 체크포인트를 미리 넣고 sim_time 동안 돈다.
"""
import dataclasses
import math

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree

from config import Config
from graphslam import (MapEval, build_grid, compose, draw_map_panels, follow_view, go_to_goal, lidar_scan,
                       make_window, map_titles, pi_2_pi, relative_pose, robot_triangle)
from map import build_map
from particle_filter import Robot
from utils import cv_viewer as cvv


def scan_points(ranges, bearings):
    """스캔 (거리, 로봇 기준 각도) 를 로봇 좌표계의 점 (n, 2) 로"""
    return np.column_stack((ranges * np.cos(bearings), ranges * np.sin(bearings)))


def transform_points(points, pose):
    """로봇 좌표계의 점들을 pose = [x, y, yaw] 좌표계 밖으로 옮긴다 (회전 후 이동)"""
    c, s = math.cos(pose[2]), math.sin(pose[2])
    return points @ np.array([[c, s], [-s, c]]) + pose[:2]


def best_fit_transform(a, b):
    """
    짝지어진 점 a[i] <-> b[i] 를 가장 잘 겹치는 회전+이동 (Kabsch / Procrustes).
    두 점 집합의 무게중심을 맞춘 뒤, 남은 회전을 SVD 로 한 번에 푼다.
      return : 이동량 [dx, dy, dyaw]  (a 를 이만큼 옮기면 b 에 겹침)
    """
    ca, cb = a.mean(axis=0), b.mean(axis=0)
    U, _, Vt = np.linalg.svd((a - ca).T @ (b - cb))
    d = np.sign(np.linalg.det(Vt.T @ U.T))  # SVD 가 회전 대신 거울 뒤집기를 내놓으면 바로잡음
    R = Vt.T @ np.diag([1.0, d]) @ U.T
    t = cb - R @ ca
    return np.array([t[0], t[1], math.atan2(R[1, 0], R[0, 0])])


def icp(src, dst, init, cfg):
    """
    src 점 구름을 dst 점 구름 위에 겹치는 이동량을 찾는다.
      src, dst : (n, 2) 각각 자기 로봇 좌표계의 점
      init     : 초기 이동량 [dx, dy, dyaw]. "dst 를 찍은 위치에서 본 src 를 찍은 위치" (= relative_pose 와 같은 뜻)
      return   : (성공 여부, (이동량 (3,), 대응점 평균 거리 [m], 대응점 개수))
    """
    T = np.asarray(init, dtype=float).copy()

    # 1. 점이 애초에 너무 적으면 맞춰 볼 수 없음 -> 바로 실패
    if len(src) < cfg.icp_min_inliers or len(dst) < cfg.icp_min_inliers:
        return False, (T, math.inf, 0)

    tree = cKDTree(dst)  # dst 에서 가장 가까운 점을 빨리 찾기 위한 트리
    err, n_matched = math.inf, 0

    for _ in range(cfg.icp_max_iter):
        # 2. 지금 이동량으로 src 를 dst 좌표계로 옮긴다
        moved = transform_points(src, T)

        # 3. 짝 찾기: 옮긴 점마다 dst 의 가장 가까운 점. 너무 먼 짝은 겹치지 않는 부분이라 버림
        dist, idx = tree.query(moved)
        matched = dist < cfg.icp_max_corr_dist
        n_matched = int(matched.sum())
        if n_matched < cfg.icp_min_inliers:
            return False, (T, math.inf, n_matched)  # 두 스캔이 거의 안 겹침 -> 실패
        err = float(dist[matched].mean())

        # 4. 짝들을 가장 잘 겹치는 보정량을 구해서 지금 이동량 뒤에 덧붙임
        delta = best_fit_transform(moved[matched], dst[idx[matched]])
        T = compose(delta, T)

        # 5. 보정량이 충분히 작으면 수렴
        if np.linalg.norm(delta) < cfg.icp_tol:
            break

    # 6. 다 겹친 뒤 대응점 평균 거리가 기준 안이면 성공
    return err <= cfg.icp_max_err, (T, err, n_matched)


class PoseGraph:
    """
    키프레임 위치(노드)와 상대 위치 제약(엣지)만 있는 그래프. graphslam.PoseGraph 의 오도메트리 엣지 부분과 같고,
    엣지마다 가중치가 다를 수 있다는 점만 다르다.

    Attributes:
        poses: (N, 3) 배열. 키프레임 위치의 현재 추정값 [x, y, yaw].
        edges: (i, j, 측정한 상대 위치 (3,), 정보 행렬 (3, 3), 루프 엣지 여부)
    """

    def __init__(self, max_iter):
        self.max_iter = max_iter
        self.poses = np.zeros((0, 3))
        self.edges = []

    def add_node(self, pose):
        """키프레임을 하나 추가하고 번호를 돌려준다"""
        self.poses = np.vstack((self.poses, np.asarray(pose, dtype=float)))
        return len(self.poses) - 1

    def add_edge(self, i, j, z, sigma_xy, sigma_yaw, loop=False):
        """노드 i 에서 본 노드 j 의 상대 위치 z = (dx, dy, dyaw) 제약을 추가"""
        omega = np.diag([1.0 / sigma_xy ** 2, 1.0 / sigma_xy ** 2, 1.0 / sigma_yaw ** 2])
        self.edges.append((i, j, np.asarray(z, dtype=float), omega, loop))

    def optimize(self):
        """Gauss-Newton 으로 모든 키프레임 위치를 고친다 (첫 노드 고정). 반환: (반복 횟수, 마지막 비용)"""
        n = len(self.poses)
        if n < 2 or not self.edges:
            return 0, 0.0
        x = self.poses.copy()
        ei = np.array([e[0] for e in self.edges])
        ej = np.array([e[1] for e in self.edges])
        ez = np.array([e[2] for e in self.edges])
        om = np.array([e[3] for e in self.edges])
        m = len(ei)
        ar3 = np.arange(3)
        rows = (3 * np.concatenate([ei, ei, ej, ej])[:, None, None] + ar3[None, :, None]).repeat(3, axis=2).ravel()
        cols = (3 * np.concatenate([ei, ej, ei, ej])[:, None, None] + ar3[None, None, :]).repeat(3, axis=1).ravel()

        cost, itr = 0.0, 0
        for itr in range(1, self.max_iter + 1):
            # e = (추정 위치로 계산한 상대 위치) - (측정한 상대 위치). 야코비안 A (노드 i 에 대해), B (노드 j 에 대해)
            xi, xj = x[ei], x[ej]
            c, s = np.cos(xi[:, 2]), np.sin(xi[:, 2])
            dx, dy = xj[:, 0] - xi[:, 0], xj[:, 1] - xi[:, 1]
            dyaw = xj[:, 2] - xi[:, 2] - ez[:, 2]
            e = np.column_stack((c * dx + s * dy - ez[:, 0], -s * dx + c * dy - ez[:, 1],
                                 np.arctan2(np.sin(dyaw), np.cos(dyaw))))
            A = np.zeros((m, 3, 3))
            A[:, 0, 0], A[:, 0, 1], A[:, 0, 2] = -c, -s, -s * dx + c * dy
            A[:, 1, 0], A[:, 1, 1], A[:, 1, 2] = s, -c, -c * dx - s * dy
            A[:, 2, 2] = -1.0
            B = np.zeros((m, 3, 3))
            B[:, 0, 0], B[:, 0, 1] = c, s
            B[:, 1, 0], B[:, 1, 1] = -s, c
            B[:, 2, 2] = 1.0

            vals = np.concatenate([np.einsum('mki,mkl,mlj->mij', A, om, A).ravel(),
                                   np.einsum('mki,mkl,mlj->mij', A, om, B).ravel(),
                                   np.einsum('mki,mkl,mlj->mij', B, om, A).ravel(),
                                   np.einsum('mki,mkl,mlj->mij', B, om, B).ravel()])
            H = sparse.coo_matrix((vals, (rows, cols)), shape=(3 * n, 3 * n)).tocsc()
            b = np.zeros(3 * n)
            np.add.at(b, (3 * ei[:, None] + ar3).ravel(), np.einsum('mki,mkl,ml->mi', A, om, e).ravel())
            np.add.at(b, (3 * ej[:, None] + ar3).ravel(), np.einsum('mki,mkl,ml->mi', B, om, e).ravel())
            cost = float(np.einsum('mi,mij,mj->', e, om, e))

            step = np.zeros(3 * n)
            step[3:] = spsolve(H[3:, 3:], -b[3:])  # 첫 노드는 좌표계의 기준이라 고정
            x += step.reshape(n, 3)
            x[:, 2] = np.arctan2(np.sin(x[:, 2]), np.cos(x[:, 2]))
            if step @ step < 1e-8:
                break
        self.poses = x
        return itr, cost


def find_loop_candidate(poses, k, cfg):
    """키프레임 k 의 추정 위치에서 loop_dist 안에 있고 번호가 loop_min_gap 이상 차이 나는 가장 가까운 예전 키프레임 (없으면 None)"""
    if k < cfg.loop_min_gap:
        return None
    old = poses[:k - cfg.loop_min_gap + 1]
    d = np.hypot(old[:, 0] - poses[k, 0], old[:, 1] - poses[k, 1])
    i = int(np.argmin(d))
    return i if d[i] < cfg.loop_dist else None


def main(config=None):
    print(__file__ + " start!!")
    cfg = Config() if config is None else config
    ic, viz, mapping = cfg.icpslam, cfg.viz, cfg.mapping

    room, x0, world = build_map(cfg)
    if world is None or not hasattr(world, "raycast"):
        raise RuntimeError("ICP SLAM 은 라이다 스캔이 필요해서 이미지 지도(ImageMapConfig.use_image_map=True)에서만 돈다")
    robot = Robot(dataclasses.replace(cfg.robot, n_rays=0), cfg.sim_noise, x0=x0, world=world)  # 레퍼런스 관측은 안 씀
    range_sigma = cfg.sim_noise.sigma_range

    graph = PoseGraph(ic.max_iter)
    scans, clouds = [], []  # 키프레임마다 스캔 (거리, 각도) 와 로봇 좌표계 점 구름
    kf_true, kf_dr = [], []  # 키프레임 시점의 정답 위치 / Dead Reckoning 위치 (비교용)
    h_true, h_dr = [], []  # 매 step 궤적 (그리기용)
    n_icp_fail = 0
    builder = None

    time, next_opt = 0.0, ic.optimize_every
    cmd = {"v": 0.0, "yaw_rate": 0.0}
    waypoints = [list(map(float, w)) for w in ic.waypoints]
    stuck_steps = 0

    win = panels = map_eval = None
    map_dirty = False
    step_count = 0
    if viz.show_animation:
        map_eval = MapEval(room, mapping, world) if viz.show_map_eval else None
        win, panels = make_window("ICP SLAM", room, viz, map_eval, waypoints)  # graphslam.py 와 같은 화면
        walls = cvv.wall_image(room)
        map_img = np.full((1, 1, 3), 255, np.uint8)
        frame_time = robot.dt * ic.draw_every / viz.sim_speed if viz.sim_speed else 0.0

    while (viz.show_animation or cfg.sim_time >= time) and not (win is not None and win.closed):
        time += robot.dt

        # 1. 체크포인트로 가는 입력 -> 이동. 관측(z)은 안 쓰고 x_true / x_dr 갱신만 필요하다
        cmd["v"], cmd["yaw_rate"] = 0.0, 0.0
        while waypoints:
            u_goal = go_to_goal(robot.x_true[:3, 0], waypoints[0], ic)
            if u_goal is not None:
                cmd["v"], cmd["yaw_rate"] = u_goal
                break
            waypoints.pop(0)
        robot.calc_input(cmd["v"], cmd["yaw_rate"])
        prev_xy = robot.x_true[:2, 0].copy()
        robot.observation(robot.u, np.zeros((0, 2)))
        x_true, x_dr = robot.x_true[:3, 0].copy(), robot.x_dr[:3, 0].copy()

        blocked = cmd["v"] > 1e-3 and np.allclose(x_true[:2], prev_xy)
        stuck_steps = stuck_steps + 1 if blocked else 0
        if waypoints and stuck_steps * robot.dt >= ic.stuck_time:
            waypoints.pop(0)
            stuck_steps = 0
        h_true.append(x_true)
        h_dr.append(x_dr)

        # 2. 키프레임 추가: 마지막 키프레임에서 오도메트리로 keyframe_dist 만큼 움직였거나 keyframe_angle 만큼 돌았으면
        #    스캔을 찍고, 이전 키프레임과 ICP 로 맞춰 이웃 엣지, 예전 키프레임과 맞으면 루프 엣지
        if kf_dr:
            moved = relative_pose(kf_dr[-1], x_dr)
            new_kf = math.hypot(moved[0], moved[1]) >= ic.keyframe_dist or abs(moved[2]) >= ic.keyframe_angle
        else:
            new_kf = True  # 첫 키프레임은 바로
        if new_kf:
            ranges, bearings = lidar_scan(world, x_true, robot, ic.n_scan_rays, range_sigma)
            cloud = scan_points(ranges, bearings)
            if not kf_dr:
                k = graph.add_node(x_dr)  # 첫 노드: 좌표계의 기준
            else:
                odo = relative_pose(kf_dr[-1], x_dr)
                ok, (rel, err, n_in) = icp(cloud, clouds[-1], odo, ic)  # 이번 스캔을 이전 스캔 위에 겹침
                if ok:
                    k = graph.add_node(compose(graph.poses[-1], rel))
                    graph.add_edge(k - 1, k, rel, ic.icp_sigma_xy, ic.icp_sigma_yaw)
                else:
                    n_icp_fail += 1
                    k = graph.add_node(compose(graph.poses[-1], odo))
                    graph.add_edge(k - 1, k, odo, ic.odo_sigma_xy, ic.odo_sigma_yaw)
                # 루프 닫기: 추정 위치 기준으로 가까운 예전 키프레임과 ICP. 초기값은 두 추정 위치의 차이
                i = find_loop_candidate(graph.poses, k, ic)
                if i is not None:
                    guess = relative_pose(graph.poses[i], graph.poses[k])
                    ok, (rel, err, n_in) = icp(cloud, clouds[i], guess, ic)
                    if ok and err <= ic.loop_max_err:
                        graph.add_edge(i, k, rel, ic.loop_sigma_xy, ic.loop_sigma_yaw, loop=True)
            scans.append((ranges, bearings))
            clouds.append(cloud)
            kf_true.append(x_true)
            kf_dr.append(x_dr)
            if map_eval is not None:
                map_eval.add_scan(x_true, x_dr, ranges, bearings)

        # 3. 그래프 최적화 -> 최적화된 위치로 지도를 다시 그림
        if time >= next_opt:
            next_opt += ic.optimize_every
            itr, cost = graph.optimize()
            builder = build_grid(room, mapping, graph.poses, scans)
            map_dirty = True
            n_loop = sum(1 for e in graph.edges if e[4])
            print("t=%.1fs  nodes=%d  edges=%d (loop %d, icp fail %d)  iter=%d  cost=%.1f"
                  % (time, len(graph.poses), len(graph.edges), n_loop, n_icp_fail, itr, cost))
            if map_eval is not None:
                map_eval.evaluate(builder, graph.poses, scans)
                print("  endpoints: " + "   ".join(map_eval.endpoint_text("ICP SLAM").split("\n")[:2]))  # 세 번째 줄(bias 설명)은 화면에만
                print("  grid map:  " + map_eval.grid_text("ICP SLAM").replace("\n", "   "))

        # 4. 그리기 (graphslam.py 와 같은 cv2 화면)
        step_count += 1
        if win is not None and (step_count % ic.draw_every == 0 or map_dirty):
            if map_dirty:
                map_dirty = False
                map_img = cvv.occupancy_image(builder)
            ht, hd, kp = np.array(h_true), np.array(h_dr), graph.poses
            wp = np.array(waypoints).reshape(-1, 2)
            loops = [(i, j) for i, j, _, _, lp in graph.edges if lp]
            # 루프로 닫힌 구간의 키프레임: 루프 엣지 i <-> j 가 생기면 i ~ j 사이 노드가 전부 닫힌 고리가 됨 (땅따먹기처럼)
            #   최적화 때 누적 오차를 나눠 갖는 게 이 구간 전체라서, 양 끝만이 아니라 구간을 통째로 초록 점으로 덮어 그림
            in_loop = np.zeros(len(kp), dtype=bool)
            for i, j in loops:
                in_loop[min(i, j):max(i, j) + 1] = True
            loop_ids = np.flatnonzero(in_loop)

            # 시뮬레이션 (로봇을 따라다니는 화면)
            sim = panels["sim"]
            sim.view = follow_view(room, x_true[0], x_true[1], viz.view_radius)
            sim.paint(*walls)  # 벽만 (랜드마크는 안 씀)
            if clouds:
                sim.dots(transform_points(clouds[-1], kf_true[-1]), cvv.MAGENTA, 1)  # 마지막 스캔 (찍었을 때의 실제 위치에)
            sim.polyline(hd[:, :2], cvv.BLACK)  # Dead Reckoning 궤적
            sim.polyline(ht[:, :2], cvv.BLUE, 2)  # 실제 궤적
            sim.polyline(kp[:, :2], cvv.RED)  # 최적화된 키프레임
            sim.dots(kp[:, :2], cvv.RED, 2)
            if loops:
                sim.segments(kp[[i for i, _ in loops], :2], kp[[j for _, j in loops], :2], cvv.GREEN, 2)  # 루프 엣지
            sim.dots(kp[loop_ids, :2], cvv.GREEN, 4, edge=cvv.DARK_GREEN)
            sim.polyline(np.vstack((x_true[:2], wp)), cvv.ORANGE)  # 로봇 -> 남은 체크포인트
            sim.markers(wp, cvv.ORANGE, kind="x", size=11, thickness=2)
            sim.polygon(robot_triangle(x_true, viz.robot_size), cvv.BLUE)  # 실제 로봇
            sim.legend([("lidar scan (last keyframe)", cvv.MAGENTA, "dot"), ("true", cvv.BLUE, "line"),
                        ("dead reckoning", cvv.BLACK, "line"), ("ICP SLAM (keyframes)", cvv.RED, "line"),
                        ("loop closure", cvv.GREEN, "line"), ("keyframe in a closed loop", cvv.GREEN, "dot"),
                        ("checkpoints", cvv.ORANGE, "x")])

            draw_map_panels(panels, map_img, map_eval, room, viz, kp, x_true)
            panels["map"].dots(kp[loop_ids, :2], cvv.GREEN, 3, edge=cvv.DARK_GREEN)
            win.show(map_titles("ICP SLAM", ic, map_eval) +
                     ["v=%.1f m/s  yaw_rate=%.1f rad/s  t=%.1fs  keyframes=%d  loops=%d\n"
                      "left click: add checkpoint  right click: clear  esc: quit"
                      % (cmd["v"], cmd["yaw_rate"], time, len(kp), len(loops))], frame_time)

    return graph, np.array(kf_true), np.array(kf_dr), builder


if __name__ == '__main__':
    main()

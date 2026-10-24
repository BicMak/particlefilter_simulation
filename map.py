"""
map: 파티클 필터 시뮬레이션에서 쓰는 지도(world)를 만드는 모듈

[지도 종류]
  Map      : 사각형 실내 공간. 벽 둘레에서 레퍼런스 포인트를 샘플링
  GridMap  : 손으로 그린 평면도 이미지에서 만든 occupancy grid. 벽 픽셀에서 레퍼런스 포인트를 샘플링
  두 지도는 같은 rf_id / plot() 인터페이스를 가진다 (GridMap 만 grid / is_visible 등 벽 정보를 추가로 가짐)

[이미지 -> GridMap 흐름]
  1. 이미지를 흑백으로 읽어서 어두운 픽셀 = 벽(occupied), 밝은 픽셀 = 빈 공간(free)
  2. 선이 살짝 끊긴 곳이 있어도 되도록 벽을 약간 두껍게(dilate) 만든다
  3. 픽셀 좌표 -> 월드 좌표(m)로 변환 (이미지 아래쪽이 y=0, resolution [m/px])
  4. 벽 픽셀에서 레퍼런스 포인트(rf_id)를 샘플링

사용:
    python map.py                      # input_image/Untitled.png -> output_image/map.png
    python map.py some.png --res 0.05  # 해상도 지정 [m/px]
코드에서:
    room, x0, world = build_map(cfg)   # config 에 맞는 지도 + 로봇 시작 상태 + 충돌/가림용 world
    gm = GridMap.from_image("input_image/Untitled.png", resolution=0.03)
    gm.rf_id      # (n_ref, 2) 벽 위 레퍼런스 포인트 [x, y]
    gm.is_free(x, y)
"""
import argparse
import math
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

from config import Config, RoomConfig

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INPUT = os.path.join(HERE, "input_image", "Untitled.png")
DEFAULT_OUTPUT = os.path.join(HERE, "output_image", "map.png")


class Map:
    """사각형 실내 공간(벽)과, 벽 위에서 샘플링한 레퍼런스 포인트(landmark)를 가진 지도"""

    def __init__(self, cfg: RoomConfig, rng=None):
        self.x_min, self.x_max = cfg.x_min, cfg.x_max
        self.y_min, self.y_max = cfg.y_min, cfg.y_max
        # 벽 경계에서 랜덤 샘플링한 레퍼런스 포인트들 [x, y]
        self.rf_id = self.sample_wall_points(cfg.n_ref, rng)

    def sample_wall_points(self, n, rng=None):
        """
        사각형 방의 경계(벽) 위에서 n개의 점을 랜덤 샘플링한다.
        둘레 길이(arc length) 기준으로 균등하게 뽑는다.
        -> 긴 벽에 점이 더 많이 생기고, 모든 위치가 같은 확률로 뽑힘
        """
        rng = np.random.default_rng() if rng is None else rng
        w = self.x_max - self.x_min  # 가로 길이
        h = self.y_max - self.y_min  # 세로 길이
        # 둘레 위의 위치 s를 [0, 둘레) 에서 균등 샘플링 (둘레 = 2(w+h))
        s = rng.uniform(0.0, 2.0 * (w + h), n)

        # s를 실제 (x, y) 좌표로 변환: 아래벽 -> 오른벽 -> 위벽 -> 왼벽 순으로 반시계 방향
        pts = np.zeros((n, 2))
        for i, si in enumerate(s):
            if si < w:  # 아래쪽 벽
                pts[i] = [self.x_min + si, self.y_min]
            elif si < w + h:  # 오른쪽 벽
                pts[i] = [self.x_max, self.y_min + (si - w)]
            elif si < 2 * w + h:  # 위쪽 벽
                pts[i] = [self.x_max - (si - w - h), self.y_max]
            else:  # 왼쪽 벽
                pts[i] = [self.x_min, self.y_max - (si - 2 * w - h)]
        return pts

    def plot(self, show_ref=True):  # pragma: no cover
        """방의 벽을 회색 굵은 선으로 그리고, 레퍼런스 포인트를 검은 별로 그린다 (show_ref=False 면 벽만)"""
        xs = [self.x_min, self.x_max, self.x_max, self.x_min, self.x_min]
        ys = [self.y_min, self.y_min, self.y_max, self.y_max, self.y_min]
        plt.plot(xs, ys, "-", color="gray", linewidth=3)
        if show_ref:
            plt.plot(self.rf_id[:, 0], self.rf_id[:, 1], "*k")


class GridMap:
    """이미지에서 만든 occupancy grid 지도. particle_filter.Map 과 같은 rf_id / plot() 인터페이스를 가진다."""

    def __init__(self, grid, resolution=0.03, n_ref=60, rng=None):
        """
          grid       : (H, W) bool 배열. True = 벽. 행 0 이 이미지의 맨 위
          resolution : 한 픽셀의 실제 길이 [m/px]
        """
        self.grid = grid
        self.resolution = resolution
        self.height, self.width = grid.shape
        self.x_min, self.x_max = 0.0, self.width * resolution
        self.y_min, self.y_max = 0.0, self.height * resolution
        self.rf_id = self.sample_wall_points(n_ref, rng)
        self._segments = None  # 벽과 방의 경계선 (boundary_segments 에서 처음 쓸 때 만듦)
        self._segment_tree = None

    @classmethod
    def from_image(cls, path, resolution=0.03, wall_thresh=200, close_px=5, n_ref=60, rng=None):
        """
          wall_thresh : 밝기(0~255)가 이 값보다 어두우면 벽
          close_px    : 벽을 이 픽셀 수만큼 두껍게 해서 선이 살짝 끊긴 틈을 메움
        """
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(path)
        if img.ndim == 3 and img.shape[2] == 4:  # 투명 배경은 흰색으로
            alpha = img[:, :, 3:4] / 255.0
            img = (img[:, :, :3] * alpha + 255 * (1 - alpha)).astype(np.uint8)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img

        wall = (gray < wall_thresh).astype(np.uint8)
        if close_px > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_px, close_px))
            wall = cv2.dilate(wall, kernel)
        return cls(wall.astype(bool), resolution, n_ref, rng)

    # ---------------- 좌표 변환 ----------------
    def world_to_pixel(self, x, y):
        """월드 (x, y)[m] -> (row, col). y 는 이미지 아래쪽이 0"""
        col = int(x / self.resolution)
        row = self.height - 1 - int(y / self.resolution)
        return row, col

    def is_free(self, x, y):
        """(x, y) 가 지도 안이고 벽이 아니면 True"""
        row, col = self.world_to_pixel(x, y)
        if not (0 <= row < self.height and 0 <= col < self.width):
            return False
        return not self.grid[row, col]

    def is_wall(self, points):
        """점들 (n, 2) 이 벽 픽셀 안이거나 지도 밖이면 True (n,)"""
        pts = np.asarray(points, dtype=float).reshape(-1, 2)
        cols = np.floor(pts[:, 0] / self.resolution).astype(int)
        rows = self.height - 1 - np.floor(pts[:, 1] / self.resolution).astype(int)
        inside = (rows >= 0) & (rows < self.height) & (cols >= 0) & (cols < self.width)
        wall = np.ones(len(pts), dtype=bool)
        wall[inside] = self.grid[rows[inside], cols[inside]]
        return wall

    # ---------------- 벽과 방의 경계선 ----------------
    def boundary_segments(self):
        """
        벽과 방의 경계선. (n, 4) 배열이고 각 행이 선분 [x1, y1, x2, y2] [m].

        픽셀 (row, col) 은 월드의 정사각형 x ∈ [col * res, (col + 1) * res], y ∈ [(H-1-row) * res, (H-row) * res] 이다.
        이웃한 두 픽셀이 하나는 벽, 하나는 빈 공간이면 그 사이 변(길이 res)이 경계선이다. 지도 밖은 벽으로 본다.
        raycast 도 같은 규칙이라, 노이즈 없는 라이다 끝점은 정확히 이 선분 위에 떨어진다.
        """
        if self._segments is None:
            res, H = self.resolution, self.height
            g = np.pad(self.grid, 1, constant_values=True)  # 지도 밖 = 벽
            # 세로 경계선: 가로로 붙은 두 픽셀이 다름. 원래 픽셀 (r, c-1) 과 (r, c) 사이 = x 가 c * res 인 선
            r, c = np.nonzero(g[1:-1, :-1] != g[1:-1, 1:])
            vertical = np.column_stack((c * res, (H - 1 - r) * res, c * res, (H - r) * res))
            # 가로 경계선: 세로로 붙은 두 픽셀이 다름. 원래 픽셀 (r-1, c) 와 (r, c) 사이 = y 가 (H - r) * res 인 선
            r, c = np.nonzero(g[:-1, 1:-1] != g[1:, 1:-1])
            horizontal = np.column_stack((c * res, (H - r) * res, (c + 1) * res, (H - r) * res))
            self._segments = np.vstack((vertical, horizontal)).astype(float)
            # 가까운 선분을 빨리 찾기 위한 KD-tree (선분 중점으로)
            self._segment_tree = cKDTree((self._segments[:, :2] + self._segments[:, 2:]) / 2.0)
        return self._segments

    def distance_to_boundary(self, points):
        """
        점들 (n, 2) 에서 가장 가까운 경계선까지의 거리 (n,) [m]. 방 쪽이면 +, 벽 안(또는 지도 밖)이면 -.

        선분이 만 개가 넘어서 전부 계산하지 않고, 중점이 가까운 후보 k 개만 정확히 계산한다.
        선분 길이가 res 라서 진짜 가장 가까운 선분의 중점은 (가장 가까운 거리 + res/2) 안에 있다.
        후보 중 가장 먼 중점이 그보다 가까우면 놓친 선분이 있을 수 있어서, 그 점만 후보를 두 배로 늘려 다시 찾는다.
        """
        seg = self.boundary_segments()
        pts = np.asarray(points, dtype=float).reshape(-1, 2)
        dist = np.zeros(len(pts))
        todo, k = np.arange(len(pts)), 8
        while len(todo):
            k_eff = min(k, len(seg))
            mid_d, idx = self._segment_tree.query(pts[todo], k=k_eff)
            mid_d, idx = mid_d.reshape(len(todo), -1), idx.reshape(len(todo), -1)
            s = seg[idx]  # (m, k, 4) 후보 선분
            px, py = pts[todo, 0:1], pts[todo, 1:2]
            # 축에 평행한 선분이라, x 와 y 각각 선분 범위를 벗어난 만큼이 그 방향 거리
            dx = np.maximum(np.maximum(np.minimum(s[..., 0], s[..., 2]) - px, px - np.maximum(s[..., 0], s[..., 2])), 0.0)
            dy = np.maximum(np.maximum(np.minimum(s[..., 1], s[..., 3]) - py, py - np.maximum(s[..., 1], s[..., 3])), 0.0)
            d = np.hypot(dx, dy).min(axis=1)
            dist[todo] = d
            sure = (mid_d[:, -1] >= d + self.resolution / 2.0) | (k_eff == len(seg))
            todo, k = todo[~sure], k * 2
        return np.where(self.is_wall(pts), -dist, dist)

    def is_visible(self, x0, y0, x1, y1, tol=0.3):
        """
        (x0, y0) 에서 (x1, y1) 의 벽 위 점이 보이는지 (사이에 다른 벽이 가리지 않는지).
        광선을 따라가며 처음 만나는 벽 픽셀을 찾고, 그게 목표점에서 tol[m] 안이면 보이는 것.
        (목표점 자체가 두께가 있는 벽 안에 있어서, 목표 바로 앞의 벽은 허용해야 함)
        """
        dist = math.hypot(x1 - x0, y1 - y0)
        n = max(int(dist / (self.resolution * 0.5)), 2)
        t = np.linspace(0.0, 1.0, n)
        cols = ((x0 + (x1 - x0) * t) / self.resolution).astype(int)
        rows = self.height - 1 - ((y0 + (y1 - y0) * t) / self.resolution).astype(int)
        inside = (rows >= 0) & (rows < self.height) & (cols >= 0) & (cols < self.width)
        hit = np.zeros(n, dtype=bool)
        hit[inside] = self.grid[rows[inside], cols[inside]]
        if not hit.any():
            return True
        first = np.argmax(hit)  # 처음 벽을 만난 지점
        return dist * (1.0 - t[first]) <= tol

    def raycast(self, x, y, yaw, fov, max_range, n_rays):
        """
        (x, y) 에서 yaw 방향 +- fov/2 안을 같은 간격으로 나눈 방향으로 광선 n_rays 개를 한꺼번에 쏴서,
        각 광선이 처음 만나는 벽과 방의 경계선 위의 점을 정확히 돌려준다. (진짜 라이다처럼 시야각을 왼쪽 끝 -> 오른쪽 끝으로 훑음)
        미리 뽑아둔 레퍼런스 대신 "로봇이 지금 시야각 안에서 실제로 보는 벽" 을 관측점으로 쓰기 위한 함수.

        [DDA (Amanatides & Woo)]
          광선이 지나가는 픽셀을 경계를 넘을 때마다 하나씩 따라간다. 광선이 다음 세로 격자선 / 가로 격자선을 만나는
          거리(t_max_x, t_max_y) 중 가까운 쪽으로 한 칸 넘어가고, 넘어간 칸이 벽이면 그 경계를 넘은 지점이 끝점이다.
          일정 간격으로 찍어 보는 방식과 달리 벽 모서리를 건너뛰지 않고, 끝점이 경계선(boundary_segments)에 정확히 놓인다.
        지도 밖은 벽으로 본다. max_range 안에서 벽을 못 만난 광선은 버린다.
          return : (n, 2) 월드 좌표 [x, y]  (각도가 작은 광선부터 순서대로)
        """
        res = self.resolution
        limit = min(max_range, math.hypot(self.x_max, self.y_max))  # 거리 제한이 inf 여도 지도 크기까지만
        # 시야각 안을 균등하게 n_rays 등분 (가장자리가 시야각 밖으로 새지 않게 살짝 안쪽). 1개면 정면만
        if n_rays == 1:
            angles = np.array([yaw])
        else:
            angles = yaw + np.linspace(-fov / 2.0, fov / 2.0, n_rays) * 0.999  # (A,)
        n = len(angles)
        dx, dy = np.cos(angles), np.sin(angles)
        ix = np.full(n, math.floor(x / res))  # 지금 칸 (열). y 쪽 칸 번호 iy 는 아래에서부터 셈 (행 = height - 1 - iy)
        iy = np.full(n, math.floor(y / res))
        step_x, step_y = np.sign(dx).astype(int), np.sign(dy).astype(int)
        with np.errstate(divide="ignore", invalid="ignore"):
            # 다음 세로/가로 격자선까지의 거리, 격자선 사이 간격 (광선과 평행하면 무한대)
            t_max_x = np.where(dx > 0, ((ix + 1) * res - x) / dx, np.where(dx < 0, (ix * res - x) / dx, np.inf))
            t_max_y = np.where(dy > 0, ((iy + 1) * res - y) / dy, np.where(dy < 0, (iy * res - y) / dy, np.inf))
            t_delta_x = np.where(dx != 0, res / np.abs(dx), np.inf)
            t_delta_y = np.where(dy != 0, res / np.abs(dy), np.inf)

        t_hit = np.full(n, np.nan)
        active = np.arange(n)  # 아직 벽을 못 만난 광선
        while len(active):
            a = active
            cross_x = t_max_x[a] < t_max_y[a]  # True: 세로 격자선을 먼저 넘음 (옆 칸으로), False: 가로 격자선 (위/아래 칸으로)
            t = np.where(cross_x, t_max_x[a], t_max_y[a])  # 이번에 넘는 경계까지의 거리
            ix[a] += np.where(cross_x, step_x[a], 0)
            iy[a] += np.where(cross_x, 0, step_y[a])
            t_max_x[a] += np.where(cross_x, t_delta_x[a], 0.0)
            t_max_y[a] += np.where(cross_x, 0.0, t_delta_y[a])
            rows, cols = self.height - 1 - iy[a], ix[a]
            inside = (rows >= 0) & (rows < self.height) & (cols >= 0) & (cols < self.width)
            wall = np.ones(len(a), dtype=bool)  # 지도 밖 = 벽
            wall[inside] = self.grid[rows[inside], cols[inside]]
            in_range = t <= limit
            hit = wall & in_range
            t_hit[a[hit]] = t[hit]
            active = a[~wall & in_range]
        got = ~np.isnan(t_hit)
        return np.column_stack((x + t_hit[got] * dx[got], y + t_hit[got] * dy[got])).reshape(-1, 2)

    def wall_pixels_world(self):
        """모든 벽 픽셀의 월드 좌표 (N, 2)"""
        rows, cols = np.nonzero(self.grid)
        x = (cols + 0.5) * self.resolution
        y = (self.height - 1 - rows + 0.5) * self.resolution
        return np.column_stack((x, y))

    def sample_wall_points(self, n, rng=None):
        """벽 픽셀 중에서 n개를 랜덤으로 뽑아 레퍼런스 포인트로 사용"""
        rng = np.random.default_rng() if rng is None else rng
        pts = self.wall_pixels_world()
        idx = rng.choice(len(pts), size=min(n, len(pts)), replace=False)
        return pts[idx]

    # ---------------- 시각화 / 저장 ----------------
    def plot(self, show_ref=True):  # pragma: no cover
        """벽(회색) + 레퍼런스 포인트(검은 별, show_ref=False 면 생략). particle_filter.Map.plot() 과 동일한 용도"""
        extent = (self.x_min, self.x_max, self.y_min, self.y_max)
        plt.imshow(np.where(self.grid, 0.5, 1.0), cmap="gray", vmin=0, vmax=1,
                   extent=extent, origin="upper")
        if show_ref:
            plt.plot(self.rf_id[:, 0], self.rf_id[:, 1], "*k")

    def save_image(self, path):
        """정리된 맵 이미지 저장 (벽=검정, 빈 공간=흰색, 레퍼런스=빨간 점)"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        out = np.where(self.grid, 0, 255).astype(np.uint8)
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
        for x, y in self.rf_id:
            row, col = self.world_to_pixel(x, y)
            cv2.circle(out, (col, row), 4, (0, 0, 255), -1)
        cv2.imwrite(path, out)


def build_map(cfg: Config):
    """
    config 에 맞는 지도를 만든다.
      return (room, x0, world)
        room  : 지도 (Map 또는 GridMap). rf_id / plot() 를 가짐
        x0    : 로봇 초기 상태 [x, y, yaw, v] (사각형 방이면 None = 원점)
        world : 벽 충돌/가림 판정에 쓸 GridMap (사각형 방이면 None)
    """
    if cfg.image_map.use_image_map:
        im = cfg.image_map
        room = GridMap.from_image(im.image, resolution=im.resolution, n_ref=im.n_ref)  # 손그림 지도
        return room, im.robot_start, room
    return Map(cfg.room), None, None  # 사각형 방


def main():
    ap = argparse.ArgumentParser(description="손그림 평면도 -> 맵 변환")
    ap.add_argument("image", nargs="?", default=DEFAULT_INPUT)
    ap.add_argument("--out", default=DEFAULT_OUTPUT)
    ap.add_argument("--res", type=float, default=0.03, help="해상도 [m/px]")
    ap.add_argument("--n-ref", type=int, default=60, help="레퍼런스 포인트 개수")
    ap.add_argument("--show", action="store_true", help="결과를 창으로 보기")
    args = ap.parse_args()

    gm = GridMap.from_image(args.image, resolution=args.res, n_ref=args.n_ref)
    gm.save_image(args.out)
    print("map size: %.1f x %.1f m, 벽 픽셀 %d개, 레퍼런스 %d개 -> %s"
          % (gm.x_max, gm.y_max, gm.grid.sum(), len(gm.rf_id), args.out))

    if args.show:
        gm.plot()
        plt.axis("equal")
        plt.show()


if __name__ == "__main__":
    main()

"""
maptograph: 손으로 그린 평면도 이미지 -> 파티클 필터에서 쓸 수 있는 맵

[흐름]
  1. 이미지를 흑백으로 읽어서 어두운 픽셀 = 벽(occupied), 밝은 픽셀 = 빈 공간(free)
  2. 선이 살짝 끊긴 곳이 있어도 되도록 벽을 약간 두껍게(dilate) 만든다
  3. 픽셀 좌표 -> 월드 좌표(m)로 변환 (이미지 아래쪽이 y=0, resolution [m/px])
  4. 벽 픽셀에서 레퍼런스 포인트(rf_id)를 샘플링 -> particle_filter.Map.rf_id 와 같은 형식

사용:
    python maptograph.py                      # input_image/Untitled.png -> output_image/map.png
    python maptograph.py some.png --res 0.05  # 해상도 지정 [m/px]
코드에서:
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

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INPUT = os.path.join(HERE, "input_image", "Untitled.png")
DEFAULT_OUTPUT = os.path.join(HERE, "output_image", "map.png")


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
    def plot(self):  # pragma: no cover
        """벽(회색) + 레퍼런스 포인트(검은 별). particle_filter.Map.plot() 과 동일한 용도"""
        extent = (self.x_min, self.x_max, self.y_min, self.y_max)
        plt.imshow(np.where(self.grid, 0.5, 1.0), cmap="gray", vmin=0, vmax=1,
                   extent=extent, origin="upper")
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

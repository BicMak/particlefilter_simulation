"""
cv2 로 그리는 화면 (graphslam.py, icpslam.py 에서 사용)

matplotlib 은 매 프레임 그림 전체(축, 눈금, 글자, 이미지 확대)를 다시 그려서 한 프레임에 100 ms 넘게 걸린다.
여기서는 지도를 numpy 이미지로 들고 있다가 지금 보이는 부분만 잘라 패널 크기로 늘리고(warpAffine),
그 위에 선 / 점 / 글자만 cv2 로 덧그린다. 한 프레임이 몇 ms 면 끝난다.

[구성]
  Panel  : 월드 좌표 [m] 의 사각형 영역 하나를 그리는 판. 월드 <-> 픽셀 변환과 그리기 함수를 가짐
  Window : 패널 여러 개를 가로로 붙이고, 패널마다 위에 제목 줄을 달아 창 하나로 띄움. 마우스 클릭, esc, 창 닫기 처리

색은 전부 cv2 순서인 BGR 이다.
"""
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (150, 150, 150)
BLUE = (225, 105, 65)  # royalblue
RED = (40, 40, 220)
ORANGE = (0, 140, 255)  # darkorange
GREEN = (50, 205, 50)  # limegreen
DARK_GREEN = (0, 100, 0)
MAGENTA = (200, 0, 200)

FONT = cv2.FONT_HERSHEY_SIMPLEX


def rgb_to_bgr(rgb):
    """0~1 실수 RGB 그림 (H, W, 3) -> 0~255 BGR 그림"""
    return np.ascontiguousarray((np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)[:, :, ::-1])


def wall_image(room):
    """
    정답 지도의 배경 그림 (벽 = 회색, 빈 공간 = 흰색) 과 그 월드 범위 (x_min, x_max, y_min, y_max).
    사각형 방(Map)은 격자가 없어서 흰 바탕만 돌려주고, 벽 테두리는 Panel.rect 로 따로 그린다.
    """
    extent = (room.x_min, room.x_max, room.y_min, room.y_max)
    if not hasattr(room, "grid"):
        return np.full((1, 1, 3), 255, np.uint8), extent
    img = np.where(room.grid[:, :, None], np.array(GRAY, np.uint8), np.array(WHITE, np.uint8))
    return img.astype(np.uint8), extent


# 점유 확률 -> 색 (particle_filter.draw_map 과 같은 RdYlGn_r: 빈 칸 초록, 막힌 칸 빨강)
_OCC_LUT = rgb_to_bgr(plt.get_cmap("RdYlGn_r")(np.linspace(0.0, 1.0, 256))[None, :, :3])[0]


def value_colors(values, vmax):
    """0 ~ vmax 값을 초록 -> 노랑 -> 빨강 색 (n, 3) 으로. vmax 이상은 빨강"""
    return _OCC_LUT[np.clip((np.asarray(values) / vmax * 255).astype(int), 0, 255)]


def occupancy_image(builder):
    """MapBuilder 의 점유 확률을 색 그림으로. 모르는 칸(free_thresh ~ occ_thresh)은 흰색"""
    occ = builder.occupancy
    img = _OCC_LUT[np.clip((occ * 255).astype(int), 0, 255)]
    img[(occ >= builder.FREE_THRESH) & (occ <= builder.OCC_THRESH)] = WHITE
    return img


class Panel:
    """
    월드 좌표 [m] 의 사각형 영역(view) 하나를 width x height 픽셀 그림(img)에 그리는 판.
    매 프레임 paint() 로 배경을 깔고, 그 위에 그리기 함수들로 덧그린다.
    """

    def __init__(self, width, height, view):
        self.w, self.h = int(width), int(height)
        self.view = tuple(view)  # (x_lo, x_hi, y_lo, y_hi) [m]
        self.img = np.full((self.h, self.w, 3), 255, np.uint8)

    # ---------------- 좌표 변환 ----------------
    def to_px(self, xy):
        """월드 좌표 (n, 2) -> 픽셀 (n, 2) int32. 픽셀은 (열, 행) 순서이고 행 0 이 위"""
        x_lo, x_hi, y_lo, y_hi = self.view
        xy = np.asarray(xy, dtype=float).reshape(-1, 2)
        u = (xy[:, 0] - x_lo) / (x_hi - x_lo) * self.w
        v = (y_hi - xy[:, 1]) / (y_hi - y_lo) * self.h
        # 화면 밖으로 아주 먼 점도 cv2 가 받을 수 있는 범위로 자름 (선은 화면 경계에서 알아서 잘림)
        return np.clip(np.column_stack((u, v)), -1e5, 1e5).astype(np.int32)

    def to_world(self, u, v):
        """픽셀 (열 u, 행 v) -> 월드 (x, y)"""
        x_lo, x_hi, y_lo, y_hi = self.view
        return x_lo + (u + 0.5) / self.w * (x_hi - x_lo), y_hi - (v + 0.5) / self.h * (y_hi - y_lo)

    def meters_to_px(self, d):
        return d / (self.view[1] - self.view[0]) * self.w

    # ---------------- 배경 ----------------
    def paint(self, image, extent):
        """
        월드 범위 extent 를 덮는 그림 image 에서 지금 view 에 보이는 부분을 잘라 패널 크기로 늘려 깐다.
        자르기 + 늘리기를 warpAffine 한 번으로 한다 (패널 픽셀 -> 원본 픽셀 대응을 직접 넘김). 지도 밖은 흰색.
        """
        x0, x1, y0, y1 = extent
        H, W = image.shape[:2]
        sx, sy = (x1 - x0) / W, (y1 - y0) / H  # 원본 한 픽셀의 크기 [m]
        x_lo, x_hi, y_lo, y_hi = self.view
        a = (x_hi - x_lo) / (self.w * sx)  # 패널 한 픽셀 = 원본 몇 픽셀 (가로)
        b = (y_hi - y_lo) / (self.h * sy)  # (세로)
        M = np.float32([[a, 0.0, (x_lo - x0) / sx + 0.5 * a - 0.5],  # 픽셀 중심끼리 맞추려고 0.5 를 더하고 뺌
                        [0.0, b, (y1 - y_hi) / sy + 0.5 * b - 0.5]])
        self.img = cv2.warpAffine(image, M, (self.w, self.h), flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=WHITE)

    # ---------------- 그리기 ----------------
    def polyline(self, xy, color, thickness=1):
        """점들을 차례로 이은 꺾은선"""
        if len(xy) >= 2:
            cv2.polylines(self.img, [self.to_px(xy)], False, color, thickness, cv2.LINE_AA)

    def segments(self, a, b, color, thickness=1):
        """a[i] -> b[i] 선분들"""
        for p, q in zip(self.to_px(a), self.to_px(b)):
            cv2.line(self.img, tuple(map(int, p)), tuple(map(int, q)), color, thickness, cv2.LINE_AA)

    def dots(self, xy, color, radius=2, edge=None):
        """채운 동그라미. edge 를 주면 테두리를 그 색으로"""
        for p in self.to_px(xy):
            c = (int(p[0]), int(p[1]))
            cv2.circle(self.img, c, radius, color, -1, cv2.LINE_AA)
            if edge is not None:
                cv2.circle(self.img, c, radius, edge, 1, cv2.LINE_AA)

    def pixels(self, xy, colors, size=1):
        """
        점마다 다른 색 colors (n, 3) 으로 size x size 픽셀을 찍는다.
        반복문 없이 배열에 바로 써서 점이 수만 개여도 빠르다. 같은 픽셀이면 뒤에 오는 점이 덮는다.
        """
        p = self.to_px(xy)
        colors = np.asarray(colors, np.uint8).reshape(-1, 3)
        for du in range(size):
            for dv in range(size):
                u, v = p[:, 0] + du - size // 2, p[:, 1] + dv - size // 2
                ok = (u >= 0) & (u < self.w) & (v >= 0) & (v < self.h)
                self.img[v[ok], u[ok]] = colors[ok]

    def line_mask(self, segments):
        """선분들 (n, 4) [x1, y1, x2, y2] [m] 을 지금 view 기준 1 픽셀 두께로 그린 마스크 (h, w) bool. 선분이 수만 개여도 cv2 한 번에 그림"""
        segments = np.asarray(segments, dtype=float).reshape(-1, 4)
        ends = np.stack((self.to_px(segments[:, :2]), self.to_px(segments[:, 2:])), axis=1)  # (n, 2, 2)
        mask = np.zeros((self.h, self.w), np.uint8)
        cv2.polylines(mask, list(ends), False, 1, 1)
        return mask.astype(bool)

    def markers(self, xy, color, kind="star", size=7, thickness=1):
        """마커. kind = "star" (별) 또는 "x" (X). 화면 밖 점은 건너뜀"""
        marker = cv2.MARKER_STAR if kind == "star" else cv2.MARKER_TILTED_CROSS
        for u, v in self.to_px(xy):
            if 0 <= u < self.w and 0 <= v < self.h:
                cv2.drawMarker(self.img, (int(u), int(v)), color, marker, size, thickness, cv2.LINE_AA)

    def polygon(self, xy, color, edge=BLACK):
        """채운 다각형 (로봇 삼각형 등)"""
        pts = self.to_px(xy)
        cv2.fillPoly(self.img, [pts], color, cv2.LINE_AA)
        cv2.polylines(self.img, [pts], True, edge, 1, cv2.LINE_AA)

    def rect(self, x_min, x_max, y_min, y_max, color, thickness=3):
        """사각형 테두리 (사각형 방의 벽)"""
        p = self.to_px([[x_min, y_max], [x_max, y_min]])
        cv2.rectangle(self.img, tuple(map(int, p[0])), tuple(map(int, p[1])), color, thickness)

    def sector(self, center, radius, a0, a1, color, alpha=0.12):
        """반투명 부채꼴 (시야각). a0 -> a1 [rad], 반시계가 + (월드 기준)"""
        overlay = self.img.copy()
        c = tuple(map(int, self.to_px([center])[0]))
        r = max(int(self.meters_to_px(radius)), 1)
        # 이미지는 y 가 아래로 커서 각도 부호가 반대
        cv2.ellipse(overlay, c, (r, r), 0.0, -np.degrees(a1), -np.degrees(a0), color, -1, cv2.LINE_AA)
        cv2.addWeighted(overlay, alpha, self.img, 1.0 - alpha, 0.0, dst=self.img)

    def legend(self, items):
        """
        오른쪽 위 범례. items = [(label, color, kind), ...]
          kind: "line" (선), "dot" (점), "x" (X 마커), "star" (별)
        """
        if not items:
            return
        row_h, pad = 16, 5
        w = 30 + max(cv2.getTextSize(label, FONT, 0.4, 1)[0][0] for label, _, _ in items) + pad
        h = row_h * len(items) + pad
        x0, y0 = self.w - w - 6, 6
        cv2.rectangle(self.img, (x0, y0), (x0 + w, y0 + h), WHITE, -1)
        cv2.rectangle(self.img, (x0, y0), (x0 + w, y0 + h), GRAY, 1)
        for n, (label, color, kind) in enumerate(items):
            y = y0 + pad + row_h * n + row_h // 2
            if kind == "line":
                cv2.line(self.img, (x0 + 5, y), (x0 + 25, y), color, 2, cv2.LINE_AA)
            elif kind == "dot":
                cv2.circle(self.img, (x0 + 15, y), 4, color, -1, cv2.LINE_AA)
            elif kind == "x":
                cv2.drawMarker(self.img, (x0 + 15, y), color, cv2.MARKER_TILTED_CROSS, 9, 2)
            else:
                cv2.drawMarker(self.img, (x0 + 15, y), color, cv2.MARKER_STAR, 9, 1)
            cv2.putText(self.img, label, (x0 + 30, y + 4), FONT, 0.4, BLACK, 1, cv2.LINE_AA)


class Window:
    """
    패널 여러 개를 가로로 붙여 창 하나에 띄운다. 패널마다 위에 제목 줄(title_lines 줄)을 둔다.
    창은 마우스로 크기를 바꿀 수 있다 (그림이 비율을 유지하며 늘어남).

      on_click(left, x, y) : 패널 위를 클릭하면 불림. left = 왼쪽 버튼 여부, (x, y) = 그 패널의 월드 좌표
      closed               : esc 를 누르거나 창을 닫으면 True
    """
    LINE_H = 18  # 제목 한 줄 높이 [px]
    GAP = 12  # 패널 사이 간격 [px]

    def __init__(self, name, panels, title_lines=3, on_click=None):
        self.name = name
        self.panels = panels
        self.top = self.LINE_H * title_lines + 8
        self.offsets = np.cumsum([0] + [p.w + self.GAP for p in panels[:-1]])  # 패널마다 왼쪽 끝 x [px]
        self.on_click = on_click
        self.closed = False
        self._last_show = None
        cv2.namedWindow(name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_NORMAL)
        width = int(self.offsets[-1] + panels[-1].w)
        cv2.resizeWindow(name, width, self.top + max(p.h for p in panels))
        cv2.setMouseCallback(name, self._mouse)

    def _mouse(self, event, x, y, flags, param):
        if self.on_click is None or event not in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN):
            return
        for p, x0 in zip(self.panels, self.offsets):
            if x0 <= x < x0 + p.w and self.top <= y < self.top + p.h:
                wx, wy = p.to_world(x - x0, y - self.top)
                self.on_click(event == cv2.EVENT_LBUTTONDOWN, wx, wy)
                return

    def show(self, titles, min_interval=0.0):
        """
        패널들을 붙여 창에 띄운다. titles = 패널마다 제목 문자열 (줄바꿈 \\n 으로 여러 줄).
        이전 show 에서 min_interval [s] 가 안 지났으면 남은 시간만큼 기다린다 (시뮬레이션 속도 맞추기).
        """
        canvas = np.full((self.top + max(p.h for p in self.panels), int(self.offsets[-1] + self.panels[-1].w), 3),
                         255, np.uint8)
        for p, x0, title in zip(self.panels, self.offsets, titles):
            canvas[self.top:self.top + p.h, x0:x0 + p.w] = p.img
            cv2.rectangle(canvas, (int(x0), self.top), (int(x0 + p.w - 1), self.top + p.h - 1), GRAY, 1)
            for n, line in enumerate(title.split("\n")):
                cv2.putText(canvas, line, (int(x0) + 2, self.LINE_H * (n + 1)), FONT, 0.45, BLACK, 1, cv2.LINE_AA)
        cv2.imshow(self.name, canvas)

        wait = 0.0
        if self._last_show is not None:
            wait = self._last_show + min_interval - time.perf_counter()
        key = cv2.waitKey(max(int(wait * 1000), 1))
        self._last_show = time.perf_counter()
        if key == 27 or cv2.getWindowProperty(self.name, cv2.WND_PROP_VISIBLE) < 1:
            self.closed = True
            cv2.destroyWindow(self.name)

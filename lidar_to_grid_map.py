"""

LIDAR to 2D grid map example

author: Erno Horvath, Csaba Hajdu based on Atsushi Sakai's scripts

[이 파일 설명 - 공부용 주석]
라이다(각도, 거리) 데이터를 "격자 지도(occupancy grid map)"로 바꾸는 코드.
격자 지도 = 공간을 일정한 크기의 칸(픽셀)으로 쪼개서, 칸마다 "비었나/막혔나/모르나"를 적어둔 지도.

  칸 값    0.0 : 빈 공간(free)    0.5 : 아직 모름(unknown)    1.0 : 장애물(occupied)

핵심 아이디어
  1. 라이다는 로봇에서 장애물까지 레이저를 쏜다.
     -> 로봇~장애물 사이는 "비어 있다", 레이저가 닿은 끝점은 "막혀 있다".
  2. 두 칸 사이의 직선이 지나가는 칸을 구하는 알고리즘이 Bresenham.
  3. 아래 MapBuilder 는 이걸 매 step 누적해서 지도를 계속 키워가는 객체.
"""

import math
from collections import deque

import cv2
import matplotlib.pyplot as plt
import numpy as np


def bresenham(start, end):
    """두 칸 사이를 잇는 직선이 지나가는 칸 목록을 구한다 (브레젠험 직선 알고리즘).

    정수 덧셈, 뺄셈, 비교만으로 계산한다. 참고: en.wikipedia.org/wiki/Bresenham's_line_algorithm
    (원본: roguebasin.com)

    Args:
        start: 시작 칸 (a, b). 정수여야 한다. 격자에서는 (row, col) 순서로 넘긴다.
        end: 끝 칸 (a, b). 정수여야 한다. start 와 같은 순서로 넘긴다.

    Returns:
        np.ndarray: shape (N, 2) 의 칸 목록. 각 행은 넘긴 순서 그대로의 [a, b] 이고,
        start 에서 end 방향 순서로 정렬되어 있다. 첫 행은 start, 마지막 행은 end 이다.

    Example:
        >>> bresenham((4, 4), (6, 10))
        array([[ 4,  4], [ 4,  5], [ 5,  6], [ 5,  7], [ 5,  8], [ 6,  9], [ 6, 10]])
    """
    # 아이디어: 두 점 사이를 칸 단위로 이으려면, 더 길게 변하는 축(x 또는 y)을 한 칸씩 전진시키면서
    # 다른 축은 "오차(error)"가 쌓일 때만 한 칸 움직이면 된다. (실수 나눗셈 없이 정수 연산만 사용)
    # 초기 조건 설정
    x1, y1 = start
    x2, y2 = end
    dx = x2 - x1
    dy = y2 - y1
    is_steep = abs(dy) > abs(dx)  # 선이 얼마나 가파른지 판단 (y가 더 많이 변하면 가파른 선)

    # 1. 기울기가 1보다 큰 선이면 x, y 를 바꿔서 "가파른 선"을 "완만한 선"으로 바꾼다.
    if is_steep:  
        x1, y1 = y1, x1
        x2, y2 = y2, x2

    if x1 <= x2:
        swapped = False
    elif x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        swapped = True


    dx = x2 - x1  # 변화량 다시 계산
    dy = y2 - y1  # 변화량 다시 계산
    error = int(dx / 2.0)  # 오차 초기화 (이상적인 직선에서 y가 얼마나 벗어났는지 누적하는 값)
    y_step = 1 if y1 < y2 else -1  # y 가 올라가는 선이면 +1, 내려가는 선이면 -1
    # 시작점~끝점 사이 범위를 따라가며 칸 좌표를 생성
    y = y1
    points = []
    
    for x in range(x1, x2 + 1):
        coord = [y, x] if is_steep else (x, y)  # 아까 바꿨던 x, y 를 원래대로 되돌려서 저장
        points.append(coord)
        error -= abs(dy)  # x 로 한 칸 갈 때마다 y 쪽 오차가 |dy| 만큼 쌓임
        if error < 0:  # 오차가 한 칸치 넘으면 y 도 한 칸 이동
            y += y_step
            error += dx
    if swapped:  # 시작/끝을 맞바꿨었다면 목록을 뒤집어 원래 방향(시작→끝)으로 되돌림
        points.reverse()
    points = np.array(points)
    return points


def atan_zero_to_twopi(y, x):
    """atan2 결과(-pi ~ pi)를 0 ~ 2pi 범위로 바꿔서 돌려준다.

    Args:
        y: 벡터의 y 성분.
        x: 벡터의 x 성분.

    Returns:
        float: 0 이상 2pi 미만의 각도 [rad].
    """
    angle = math.atan2(y, x)
    if angle < 0.0:
        angle += math.pi * 2.0
    return angle


def init_flood_fill(center_point, obstacle_points, xy_points, min_coord,
                    xy_resolution):
    """flood fill 방식의 1단계: 장애물 점들을 이어서 "벽의 윤곽선"을 free(0)로 먼저 그려둔다.

    윤곽선 안쪽을 채우는 건 다음 단계인 flood_fill 이 한다.

    Args:
        center_point: 로봇이 있는 칸 (x, y) 인덱스. 윤곽선의 시작점이 된다.
        obstacle_points: 관측된 장애물의 월드 좌표 (ox, oy). 각각 길이 n 의 배열.
        xy_points: 지도 크기 (xw, yw) [칸 수].
        min_coord: 지도의 월드 좌표 시작점 (min_x, min_y) [m].
        xy_resolution: 한 칸의 한 변 길이 [m/px].

    Returns:
        np.ndarray: shape (xw, yw) 의 지도. 윤곽선 칸은 0.0, 나머지는 0.5(모름).
    """
    center_x, center_y = center_point
    prev_ix, prev_iy = center_x - 1, center_y
    ox, oy = obstacle_points
    xw, yw = xy_points
    min_x, min_y = min_coord
    occupancy_map = (np.ones((xw, yw))) * 0.5  # 처음엔 전부 "모름(0.5)"
    for (x, y) in zip(ox, oy):
        # 월드 좌표 -> 칸 인덱스: (좌표 - 지도 시작점) / 한 칸 크기
        # x coordinate of the the occupied area
        ix = int(round((x - min_x) / xy_resolution))
        # y coordinate of the the occupied area
        iy = int(round((y - min_y) / xy_resolution))
        free_area = bresenham((prev_ix, prev_iy), (ix, iy))  # 이전 점 ~ 이번 점 사이를 직선으로 이음
        for fa in free_area:
            occupancy_map[fa[0]][fa[1]] = 0  # free area 0.0
        prev_ix = ix
        prev_iy = iy
    return occupancy_map


def flood_fill(center_point, occupancy_map):
    """시작점에서 동서남북으로 퍼져 나가며 "모름(0.5)" 칸을 빈 칸(0.0)으로 채운다.

    이미 0.5 가 아닌 칸(윤곽선 등)에 막히면 더 퍼지지 않는다. (그림판의 "페인트 통"과 같은 원리)
    occupancy_map 을 제자리에서 수정한다.

    Args:
        center_point: 채우기를 시작할 칸 (x, y) 인덱스.
        occupancy_map: 윤곽선이 그려진 지도 (2차원 배열). 0.5 인 칸만 채워진다.
    """
    # Fill empty areas with queue method
    # 시작점(로봇 위치)에서 동서남북으로 퍼져나가며 "모름(0.5)" 칸을 free(0)로 채운다.
    # 윤곽선(이미 0 으로 칠해진 칸)에 막혀서 바깥으로는 안 퍼진다. (그림판의 "페인트 통"과 같은 원리)
    sx, sy = occupancy_map.shape
    fringe = deque()
    fringe.appendleft(center_point)
    while fringe:
        n = fringe.pop()
        nx, ny = n
        # West
        if nx > 0:
            if occupancy_map[nx - 1, ny] == 0.5:
                occupancy_map[nx - 1, ny] = 0.0
                fringe.appendleft((nx - 1, ny))
        # East
        if nx < sx - 1:
            if occupancy_map[nx + 1, ny] == 0.5:
                occupancy_map[nx + 1, ny] = 0.0
                fringe.appendleft((nx + 1, ny))
        # North
        if ny > 0:
            if occupancy_map[nx, ny - 1] == 0.5:
                occupancy_map[nx, ny - 1] = 0.0
                fringe.appendleft((nx, ny - 1))
        # South
        if ny < sy - 1:
            if occupancy_map[nx, ny + 1] == 0.5:
                occupancy_map[nx, ny + 1] = 0.0
                fringe.appendleft((nx, ny + 1))



def scan_to_world(pose, ranges, bearings):
    """로봇 기준의 거리와 각도 측정을 월드 좌표의 끝점으로 바꾼다.

    정답 벽 좌표가 아니라 pose 와 측정값으로 끝점을 만들기 때문에, pose 오차와 거리 노이즈가
    그대로 끝점에 반영된다. 즉 "로봇이 믿는 끝점"이다.

        끝점 = (pose x, pose y) + 거리 * (cos(yaw + 빔 각도), sin(yaw + 빔 각도))

    Args:
        pose: 로봇 상태 [x, y, yaw, ...] (예: x_est, x_true 처럼 4x1 배열도 가능).
            앞의 세 값 (x, y, yaw[rad]) 을 쓴다.
        ranges: 측정된 거리 (n,) [m].
        bearings: 로봇이 보는 방향 기준의 상대 빔 각도 (n,) [rad]. 왼쪽(반시계)이 +.

    Returns:
        np.ndarray: shape (n, 2) 의 월드 좌표 [x, y] [m]. MapBuilder.update 의 points 로 바로 넘길 수 있다.
    """
    pose = np.ravel(pose)
    ranges = np.ravel(ranges)
    angles = pose[2] + np.ravel(bearings)  # 월드 기준 빔 각도
    return np.column_stack((pose[0] + ranges * np.cos(angles),
                            pose[1] + ranges * np.sin(angles)))


class MapBuilder:
    """
    로봇이 돌아다니며 관측한 점들로 점유 격자 지도(픽셀맵)를 누적해서 만들고 들고 있는 객체.
    시각화는 이 객체의 occupancy / extent 를 가져다 그리기만 하면 된다.

    격자는 log-odds 배열로 들고 있고, occupancy 로 확률(0~1)을 꺼낸다.
      0.0 ~ 0.5 : 빈 공간 쪽,  0.5 : 아직 모름,  0.5 ~ 1.0 : 장애물 쪽

    매 update 마다 로봇 위치에서 관측된 점까지 Bresenham 직선을 긋고
      직선이 지나간 칸 -> 빈 공간 쪽으로 (L_FREE)
      끝점 칸          -> 장애물 쪽으로 (L_OCC)
    를 더한다. log-odds 라서 여러 번 관측될수록 확신이 쌓이고, 한두 번 틀려도 뒤집히지 않는다.

    [log-odds 란?]
    확률 p 를 그대로 더하면 곱셈 법칙이 안 맞는다. 대신 log(p / (1-p)) 로 바꾸면
    "관측이 들어올 때마다 그냥 더하기"만 하면 베이즈 갱신이 된다.
      log-odds 0  <-> 확률 0.5 (모름),  양수 <-> 장애물 쪽,  음수 <-> 빈 공간 쪽
    확률로 되돌릴 때는 p = 1 - 1 / (1 + e^L)  (아래 occupancy 프로퍼티)

    resolution: 한 칸(픽셀)의 한 변 길이 [m/px]
    좌표 규칙: 행 0 = 이미지 맨 위, y 는 아래쪽이 y_min.
    위치 pose 에 x_true 를 주면 깨끗한 지도, x_est 를 주면 위치 오차가 번진 지도가 나온다.

    Attributes:
        x_min, x_max, y_min, y_max: 지도의 월드 범위 [m].
        resolution: 한 칸의 한 변 길이 [m/px].
        height, width: 격자의 행, 열 개수.
        log_odds: shape (height, width) 의 log-odds 배열. 0 = 모름.
        explored: shape (height, width) 의 bool 배열. 한 번이라도 관측된 칸이면 True.
            log_odds 가 0 근처인 칸이 "관측했지만 증거가 상쇄된 칸"인지 "한 번도 못 본 칸"인지 구분하는 데 쓴다.
        use_flood_fill: True 면 update 에서 부채꼴 안쪽도 flood fill 로 빈 공간 처리.
        L_FREE, L_OCC: 관측 한 번당 log-odds 변화량 (지나간 칸 / 끝점 칸).
        L_MIN, L_MAX: log-odds 범위.
        OCC_THRESH, FREE_THRESH: 점유 확률이 OCC 초과면 "막힌 칸", FREE 미만이면 "빈 칸", 그 사이는 "모름".
    """
    def __init__(self, x_min, x_max, y_min, y_max, resolution=0.3, use_flood_fill=False,
                 l_free=-0.85, l_occ=0.4, l_min=-4.0, l_max=4.0, occ_thresh=0.6, free_thresh=0.4):
        """모든 칸이 "모름"인 빈 지도를 만든다.

        Args:
            x_min: 지도의 왼쪽 끝 월드 x [m].
            x_max: 지도의 오른쪽 끝 월드 x [m].
            y_min: 지도의 아래쪽 끝 월드 y [m].
            y_max: 지도의 위쪽 끝 월드 y [m].
            resolution: 한 칸의 한 변 길이 [m/px].
            use_flood_fill: True 면 update 에서 광선 사이 빈틈도 flood fill 로 빈 공간 처리한다.
            l_free: 광선이 지나간 칸이 받는 log-odds 변화량 (음수 = 빈 공간 쪽).
            l_occ: 끝점 칸이 받는 log-odds 변화량 (양수 = 장애물 쪽).
            l_min: log-odds 하한. 너무 확신하면 나중에 고치기 힘들어서 제한한다.
            l_max: log-odds 상한.
            occ_thresh: 점유 확률이 이 값을 넘으면 "막힌 칸".
            free_thresh: 점유 확률이 이 값 미만이면 "빈 칸". 두 값 사이는 "모름".
        """
        self.x_min, self.x_max = x_min, x_max
        self.y_min, self.y_max = y_min, y_max
        self.resolution = resolution
        self.use_flood_fill = use_flood_fill  # True: 광선 사이 빈틈도 flood fill 로 빈 공간 처리 (update 참고)
        self.L_FREE, self.L_OCC = l_free, l_occ  # 관측 한 번당 log-odds 변화량
        self.L_MIN, self.L_MAX = l_min, l_max  # log-odds 범위
        self.OCC_THRESH, self.FREE_THRESH = occ_thresh, free_thresh  # "막힌 칸" / "빈 칸" 을 가르는 확률

        # 칸 개수 = 범위 길이 / 한 칸 크기 (나머지가 남으면 올림해서 지도 끝이 잘리지 않게)
        self.height = int(math.ceil((y_max - y_min) / resolution))
        self.width = int(math.ceil((x_max - x_min) / resolution))

        #실제 점유 확률은 occupancy 프로퍼티에서 계산하고, 여기서는 log-odds 배열만 들고 있다.
        self.log_odds = np.zeros((self.height, self.width))  # 0 = 모름 (확률 0.5)
        self.explored = np.zeros((self.height, self.width), dtype=bool)  # 한 번이라도 관측된 칸


    @classmethod
    def from_map(cls, world, resolution=None, use_flood_fill=False, **params):
        """기존 지도(GridMap / Map)와 같은 범위로 빈 지도를 만든다.

        Args:
            world: x_min, x_max, y_min, y_max 속성을 가진 지도 객체.
            resolution: 한 칸의 한 변 길이 [m/px]. None 이면 world.resolution, 그것도 없으면 0.03.
            use_flood_fill: __init__ 의 use_flood_fill 과 같다.
            **params: l_free, l_occ, l_min, l_max, occ_thresh, free_thresh (__init__ 참고).

        Returns:
            MapBuilder: 모든 칸이 "모름"인 새 지도.
        """
        res = resolution if resolution is not None else getattr(world, "resolution", 0.03)
        return cls(world.x_min, world.x_max, world.y_min, world.y_max, res, use_flood_fill, **params)

    @property
    def extent(self):
        """월드 범위. plt.imshow(extent=...) 에 바로 넘길 수 있다.

        Returns:
            tuple: (x_min, x_max, y_min, y_max) [m].
        """
        return self.x_min, self.x_max, self.y_min, self.y_max

    @property
    def shape(self):
        """격자 크기.

        Returns:
            tuple: (height, width) [칸 수].
        """
        return self.log_odds.shape

    def world_to_pixel(self, x, y):
        """월드 좌표를 칸 번호로 바꾼다. 소수점은 내림(floor)한다.

        Args:
            x: 월드 x [m].
            y: 월드 y [m].

        Returns:
            tuple: (row, col). 행 0 이 이미지 맨 위이다. 지도 밖의 값도 범위 검사 없이 나올 수 있으니
            in_bounds 로 확인해야 한다.
        """

        # x -> 열(col): 왼쪽 끝에서 얼마나 떨어졌나 / 칸 크기
        # int() 는 음수에서 0 쪽으로 잘라서 지도 왼쪽 밖의 점이 0번 칸에 들어오므로 floor 를 쓴다. (칸의 왼쪽 모서리 기준)
        col =math.floor((x - self.x_min) / self.resolution)
        # y -> 행(row): 이미지는 행 0 이 "맨 위"인데 월드 y 는 위로 갈수록 커지므로 뒤집어 준다
        row = self.height - 1 - math.floor((y - self.y_min) / self.resolution)
        return row, col

    def pixel_to_world(self, row, col):
        """칸 번호를 그 칸 중심의 월드 좌표로 바꾼다.

        Args:
            row: 행 번호 (0 이 이미지 맨 위).
            col: 열 번호.

        Returns:
            tuple: 그 칸 중심의 월드 (x, y) [m].
        """
        # +0.5 는 칸의 "모서리"가 아니라 "중심"을 가리키게 하려는 것
        x = self.x_min + (col + 0.5) * self.resolution
        y = self.y_min + (self.height - 1 - row + 0.5) * self.resolution
        return x, y

    def in_bounds(self, row, col):
        """칸 번호가 지도 안인지 확인한다.

        Args:
            row: 행 번호.
            col: 열 번호.

        Returns:
            bool: 지도 안이면 True.
        """
        return 0 <= row < self.height and 0 <= col < self.width

    def get(self, x, y):
        """월드 좌표가 속한 칸의 점유 확률을 구한다.

        Args:
            x: 월드 x [m].
            y: 월드 y [m].

        Returns:
            float | None: 점유 확률 (0 ~ 1, 0.5 = 모름). 지도 밖이면 None.
        """
        row, col = self.world_to_pixel(x, y)

        if self.in_bounds(row, col):
            # log-odds -> 확률 변환: p = 1 - 1 / (1 + e^L)
            L = self.log_odds[row, col]
            return 1.0 - 1.0 / (1.0 + math.exp(L))
        else:
            return None 

    def update(self, pose, points):
        """이번 step 의 관측으로 지도를 갱신한다 (log_odds 를 제자리에서 누적).

        use_flood_fill 가 꺼져 있으면 점마다 Bresenham 직선을 긋고, 지나간 칸에 L_FREE,
        끝점 칸에 L_OCC 를 더한다.
        켜져 있으면 위 직선들에 더해 로봇과 관측점들이 이루는 부채꼴 안쪽도 flood fill 로
        빈 공간 처리한다. 이때 한 칸은 step 당 L_FREE 를 한 번만 받고, 이웃한 점 사이는 벽이
        이어져 있다고 가정한다.

        Args:
            pose: 로봇 위치. [x, y, ...] 형태이면 앞의 두 값만 쓴다
                (예: robot.x_true 처럼 4x1 배열도 가능).
            points: 이번 step 에 관측된 끝점들의 월드 좌표 (n, 2) [m]. 예: z[:, 1:3].
                지도 밖의 점은 무시한다.

        Note:
            로봇이 지도 밖에 있으면 아무것도 하지 않고 돌아간다.
        """
        pose = np.ravel(pose)  # 4x1 열벡터든 리스트든 1차원 [x, y, ...] 로 통일
        r0, c0 = self.world_to_pixel(pose[0], pose[1])
        if not self.in_bounds(r0, c0):
            return  # 로봇이 지도 밖이면 시작점을 칸으로 못 바꾸니 건너뜀

        ends = [self.world_to_pixel(x, y) for x, y in np.asarray(points).reshape(-1, 2)]  # 관측점 -> 칸
        ends = [(r, c) for r, c in ends if self.in_bounds(r, c)]  # 지도 밖 점은 버림

        if self.use_flood_fill and len(ends) >= 2:
            free_mask = self._fan_free_mask(r0, c0, ends)
            self.log_odds[free_mask] += self.L_FREE  # 빈 칸으로 정해진 칸: step 당 한 번만
            self.explored[free_mask] = True
            for r1, c1 in ends:
                self.log_odds[r1, c1] += self.L_OCC  # 끝점 칸: 장애물 쪽으로
                self.explored[r1, c1] = True
        else:
            for r1, c1 in ends:
                ray = bresenham((r0, c0), (r1, c1))  # 로봇 칸 -> 관측점 칸을 잇는 칸 목록
                free = ray[:-1]  # 마지막 칸(끝점)은 제외
                self.log_odds[free[:, 0], free[:, 1]] += self.L_FREE  # 지나간 칸: 빈 공간 쪽으로 (음수라 값이 줄어듦)
                self.log_odds[r1, c1] += self.L_OCC  # 끝점 칸: 장애물 쪽으로
                self.explored[free[:, 0], free[:, 1]] = True  # 광선이 지나간 칸과 끝점 칸은 "봤다"
                self.explored[r1, c1] = True

        # 값이 무한정 커지면 나중에 반대 증거가 와도 못 뒤집으니 범위를 제한
        np.clip(self.log_odds, self.L_MIN, self.L_MAX, out=self.log_odds)

    def _fan_free_mask(self, r0, c0, ends):
        """로봇과 관측점들이 이루는 부채꼴 다각형 안쪽을 빈 칸으로 표시한 마스크를 만든다.

        순서는 다음과 같다.
          1) 관측점을 각도 순으로 정렬한다.
          2) 로봇 -> 첫 점 -> ... -> 마지막 점 -> 로봇 순서의 다각형을 경계 칸까지 포함해서 채운다.
             (로봇에서 양 끝 점으로 가는 변과 로봇 칸도 같이 채워진다)
          3) 점들끼리 이은 선(벽이 이어졌다고 가정한 선)과 끝점 칸은 다시 비운다.
             벽일 수 있어서 빈 칸으로 치지 않고, 끝점은 update 에서 막힌 칸 증거를 따로 받는다.

        Args:
            r0: 로봇 칸의 행 번호.
            c0: 로봇 칸의 열 번호.
            ends: 지도 안에 있는 관측점 칸 (row, col) 의 리스트. 2개 이상이어야 의미가 있다.

        Returns:
            np.ndarray: shape (H, W) 의 bool 배열. True 인 칸이 이번 step 에 빈 칸으로 정해진 칸이다.
            끝점 칸은 포함하지 않는다.
        """
        ends = np.array(ends)
        # 각도 순 정렬: 점들의 평균 방향을 기준으로 한 상대 각도로 정렬 
        #(±180° 경계에서 순서가 꼬이지 않게)
        ang = np.arctan2(r0 - ends[:, 0], ends[:, 1] - c0)  # 행은 아래로 커지므로 부호를 뒤집음
        mean = np.arctan2(np.sin(ang).sum(), np.cos(ang).sum())
        rel = np.arctan2(np.sin(ang - mean), np.cos(ang - mean))
        ends = ends[np.argsort(rel)]

        canvas = np.zeros(self.log_odds.shape, dtype=np.uint8)
        poly = np.vstack(([r0, c0], ends))  # 꼭짓점: 로봇, 첫 점, ..., 마지막 점
        cv2.fillPoly(canvas, [poly[:, ::-1].astype(np.int32)], 1)  # OpenCV 는 (x=열, y=행) 순서
        cv2.polylines(canvas, [ends[:, ::-1].astype(np.int32)], False, 0)  # 점들끼리 이은 선은 다시 비움
        mask = canvas.astype(bool)
        mask[ends[:, 0], ends[:, 1]] = False  # 끝점 칸은 빈 칸이 아님
        return mask

    @property
    def occupancy(self):
        """log-odds 를 점유 확률로 바꾼 값 (호출할 때마다 새로 계산한다).

        Returns:
            np.ndarray: shape (H, W) 의 확률. 0.5 = 모름, 1 에 가까울수록 장애물.
        """
        return 1.0 - 1.0 / (1.0 + np.exp(self.log_odds))

    def explored_view(self):
        """관측으로 알게 된 칸만 값을 갖고, 모르는 칸은 가려진(masked) 배열을 만든다.

        FREE_THRESH ~ OCC_THRESH 사이(0.4 ~ 0.6)의 확률은 "모름"으로 보고 가린다.

        Returns:
            np.ma.MaskedArray: shape (H, W) 의 점유 확률. 알게 된 칸(막힌 칸 + 광선이 지나간 빈 칸)만 값이 있다.
        """
        occ = self.occupancy
        return np.ma.masked_where((occ >= self.FREE_THRESH) & (occ <= self.OCC_THRESH), occ)

    def save(self, path):
        """log-odds, 탐험 여부(explored), 범위/해상도를 한 파일(.npz)에 저장한다.

        Args:
            path: 저장할 파일 경로. np.savez 규칙에 따라 .npz 확장자가 붙는다.
        """
        np.savez(path, log_odds=self.log_odds, explored=self.explored,
                 resolution=self.resolution, bounds=np.array(self.extent))

    @classmethod
    def load(cls, path):
        """save 로 저장한 파일에서 지도를 복원한다.

        Args:
            path: save 로 만든 .npz 파일 경로.

        Returns:
            MapBuilder: 저장 당시의 범위, 해상도, log-odds, 탐험 여부를 가진 지도.
                use_flood_fill 와 log-odds 변화량, 임곗값은 저장하지 않으므로 기본값이 된다.
                explored 가 없는 예전 파일이면 log_odds 가 0 이 아닌 칸을 탐험한 칸으로 본다.
        """
        f = np.load(path)
        mb = cls(*map(float, f["bounds"]), resolution=float(f["resolution"]))
        mb.log_odds = f["log_odds"]
        mb.explored = f["explored"] if "explored" in f.files else mb.log_odds != 0
        return mb



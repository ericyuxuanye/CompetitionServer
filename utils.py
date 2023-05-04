import numpy as np
from math import atan2, pi, sin, cos, sqrt
from numba import jit
from time import time, sleep


class Encoder:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.data = [width, height]

    def clear(self):
        self.data = [self.width, self.height]

    def image(
        self, name: str, x: int, y: int, width: int, height: int, rotation: int = 0
    ):
        """
        Draws image on the canvas

        name: name of image
        x: horizontal distance from center to left of canvas
        y: vertical distance from center to top of canvas
        width: width of draw image
        height: height of drawn image
        rotation: clockwise rotation in degrees
        """
        self.data.append(
            {
                "type": "image",
                "name": name,
                "x": x,
                "y": y,
                "width": width,
                "height": height,
                "rotation": rotation,
            }
        )

    def background(self, color):
        self.data.append(
            {
                "type": "background",
                "color": color,
            }
        )

    def background_image(self, image):
        """
        Just draws an image that fills the canvas
        """
        self.image(image, self.width // 2, self.height // 2, self.width, self.height)

    def get_commands(self):
        return self.data


class Timer:
    def __init__(self, framerate):
        self.wait = 1 / framerate
        self.start = time()

    def tick(self):
        sleep_time = self.wait - ((time() - self.start) % self.wait)
        sleep(sleep_time)


@jit(nopython=True)
def line_angle(line):
    angle_radians = -atan2(line[1], line[0])
    # car's front is 90 degrees away
    angle_radians -= pi / 2
    # convert radians to degrees
    return angle_radians * 180 / pi


@jit(nopython=True)
def line_dist_sq(point, line):
    vec_const = (line[2] - point[0], line[3] - point[1])
    t_comp = line[0] * line[0] + line[1] * line[1]
    c = vec_const[0] * line[0] + vec_const[1] * line[1]
    t = -c / t_comp
    if t > 1:
        t = 1
    elif t < 0:
        t = 0
    x = t * line[0] + line[2]
    y = t * line[1] + line[3]
    return t, (point[0] - x) ** 2 + (point[1] - y) ** 2


@jit(nopython=True)
def calc_distance_from_start(point, lines):
    res, min_dist = line_dist_sq(point, lines[0])
    for i in range(1, len(lines)):
        t, dist = line_dist_sq(point, lines[i])
        if dist < min_dist:
            min_dist = dist
            res = i + t
    return res


@jit(nopython=True)
def get_distances(point, rotation, border_lines):
    theta = rotation / 180 * pi
    vertical_line = (cos(theta + pi / 2), -sin(theta + pi / 2), point[0], point[1])
    horizontal_line = (cos(theta), -sin(theta), point[0], point[1])
    forward_diagonal = (cos(theta + pi / 3), -sin(theta + pi / 3), point[0], point[1])
    backward_diagonal = (
        cos(theta + 2 * pi / 3),
        -sin(theta + 2 * pi / 3),
        point[0],
        point[1],
    )
    min_ft = min_rt = min_frt = min_flt = 1000
    min_bt = min_lt = min_brt = min_blt = -1000
    for line in border_lines:
        t1, f1 = line_intersect(vertical_line, line)
        if t1 >= 0 and 0 <= f1 <= 1:
            min_ft = min(min_ft, t1)
        if t1 <= 0 and 0 <= f1 <= 1:
            min_bt = max(min_bt, t1)

        t2, f2 = line_intersect(horizontal_line, line)
        if t2 >= 0 and 0 <= f2 <= 1:
            min_rt = min(min_rt, t2)
        if t2 <= 0 and 0 <= f2 <= 1:
            min_lt = max(min_lt, t2)

        t3, f3 = line_intersect(forward_diagonal, line)
        if t3 >= 0 and 0 <= f3 <= 1:
            min_frt = min(min_frt, t3)
        if t3 <= 0 and 0 <= f3 <= 1:
            min_blt = max(min_blt, t3)

        t4, f4 = line_intersect(backward_diagonal, line)
        if t4 >= 0 and 0 <= f4 <= 1:
            min_flt = min(min_flt, t4)
        if t4 <= 0 and 0 <= f4 <= 1:
            min_brt = max(min_brt, t4)

    return (
        min_ft,
        -min_bt,
        -min_lt,
        min_rt,
        min_frt,
        -min_blt,
        min_flt,
        -min_brt,
    )


@jit(nopython=True)
def relative_car_velocities(velocity, rotation):
    theta = atan2(-velocity[1], velocity[0])
    r = sqrt(velocity[0] ** 2 + velocity[1] ** 2)
    car_theta = rotation / 180 * pi + pi / 2
    vertical_velocity = r * cos(theta - car_theta)
    horizontal_velocity = r * sin(theta - car_theta)
    return (vertical_velocity, horizontal_velocity)


@jit(nopython=True)
def line_intersect(line1, line2):
    a, c, b, d = line1
    f, h, g, j = line2
    denom = c * f - a * h
    if denom == 0 or h == 0:
        return np.nan, np.nan
    t = -(d * f - b * h + g * h - f * j) / denom
    s = (c * t + d - j) / h
    return t, s


@jit(nopython=True)
def create_line(p1, p2):
    return (p2[0] - p1[0], p2[1] - p1[1], p1[0], p1[1])


@jit(nopython=True)
def car_touching_line(center_x, center_y, width, height, rotation, lines, border_lines):
    theta = rotation / 180 * pi + pi / 2
    hw = width / 2
    hh = height / 2
    hwx = hw * cos(theta + pi / 2)
    hwy = hw * sin(theta + pi / 2)
    hhx = hh * cos(theta)
    hhy = hh * sin(theta)
    points = np.empty((4, 2))
    points[0] = (center_x + hwx + hhx, center_y + hwy + hhy)
    points[1] = (center_x - hwx + hhx, center_y - hwy + hhy)
    points[2] = (center_x + hwx - hhx, center_y + hwy - hhy)
    points[3] = (center_x - hwx - hhx, center_y - hwy - hhy)
    lines = np.empty((4, 4))
    lines[0] = create_line(points[0], points[1])
    lines[1] = create_line(points[1], points[3])
    lines[2] = create_line(points[3], points[2])
    lines[3] = create_line(points[2], points[0])
    for l1 in border_lines:
        for l2 in lines:
            t, s = line_intersect(l1, l2)
            if 0 <= t <= 1 and 0 <= s <= 1:
                return True
    return False

def read_file(filename):
    with open(filename, "rt") as f:
        n = int(f.readline())
        lines = np.empty((n, 4))
        for i in range(n):
            lines[i] = tuple(map(int, f.readline().split()))
        border_lines = np.empty((n*2, 4))
        for i in range(n * 2):
            border_lines[i] = tuple(map(float, f.readline().split()))

    return lines, border_lines

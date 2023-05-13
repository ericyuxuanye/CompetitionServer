import numpy as np
import random
from math import pi, sin, cos, atan2, sqrt
from utils import (
    line_angle,
    relative_car_velocities,
    car_touching_line,
    calc_distance_from_start,
    get_distances,
)
from data import num_lines, lines, border_lines


class Car:
    def __init__(self, acceleration, friction, rot_speed):
        self.width = 24
        self.height = 47
        self.rot_speed = rot_speed
        self.accel = acceleration
        self.friction = friction
        self.just_hit = False
        self.velocity = np.empty(2)
        self.reset()

    def reset(self):
        idx = random.randint(0, len(lines) - 1)
        line = lines[idx]
        self.x = line[2]
        self.y = line[3]
        self.velocity[:] = [0.0, 0.0]
        self.prev_distance = idx
        self.rotation = line_angle(line)

    def update(self, left, right, forward, backward):
        if left:
            self.rotation += self.rot_speed
        if right:
            self.rotation -= self.rot_speed

        radians = self.rotation / 180 * pi + pi / 2
        if forward:
            self.velocity[0] += self.accel * cos(radians)
            # subtract because y is flipped
            self.velocity[1] -= self.accel * sin(radians)
        if backward:
            self.velocity[0] -= self.accel * cos(radians)
            self.velocity[1] += self.accel * sin(radians)
        # friction calculation
        r = sqrt(self.velocity[0] ** 2 + self.velocity[1] ** 2)
        theta = atan2(self.velocity[1], self.velocity[0])
        r = max(r - self.friction, 0)
        self.velocity[0] = r * cos(theta)
        self.velocity[1] = r * sin(theta)

        self.x += self.velocity[0]
        self.y += self.velocity[1]
        if car_touching_line(
            self.x, self.y, self.width, self.height, self.rotation, lines, border_lines
        ):
            self.just_hit = True
            self.reset()
            return

    def get_state(self):
        state = np.empty((10,), dtype=np.float32)
        distances = get_distances((self.x, self.y), self.rotation, border_lines)
        state[:8] = distances
        velocities = relative_car_velocities(self.velocity, self.rotation)
        state[8:10] = velocities
        return state

    def get_reward(self):
        curr_distance = calc_distance_from_start((self.x, self.y), lines)
        reward = curr_distance - self.prev_distance
        if reward > num_lines / 2:
            reward = curr_distance - num_lines - self.prev_distance
        elif reward < -num_lines / 2:
            reward = (num_lines - self.prev_distance) + curr_distance
        self.prev_distance = curr_distance
        reward *= 50
        # if reward <= 0:
        #     # So that the agent does not stay in place
        #     reward = -0.1
        # So that agent does not stay in place
        # reward -= 0.1
        if self.just_hit:
            self.just_hit = False
        return reward

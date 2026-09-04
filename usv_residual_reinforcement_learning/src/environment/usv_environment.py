import numpy as np


def wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


class USVEnvironment:
    """
    Simplified planar autonomous-surface-vehicle environment.

    State vector:
        [x, y, psi, u, r]

    where:
        x, y : position [m]
        psi  : heading [rad]
        u    : surge speed [m/s]
        r    : yaw rate [rad/s]

    Action:
        [throttle, rudder] normalized to [-1, 1].

    This is intentionally a lightweight educational model.
    """

    def __init__(
        self,
        dt=0.1,
        goal=(40.0, 30.0),
        obstacles=None,
        seed=None,
    ):
        self.dt = float(dt)
        self.goal = np.asarray(goal, dtype=np.float32)
        self.rng = np.random.default_rng(seed)

        if obstacles is None:
            obstacles = [
                (15.0, 10.0, 3.0),
                (25.0, 20.0, 3.5),
            ]
        self.obstacles = obstacles

        self.max_speed = 4.0
        self.max_yaw_rate = np.deg2rad(35.0)
        self.goal_radius = 2.0
        self.collision_margin = 0.8
        self.max_steps = 800

        self.state = None
        self.steps = 0

    @property
    def observation_dim(self):
        return 8

    @property
    def action_dim(self):
        return 2

    def reset(self):
        self.state = np.array([0.0, 0.0, 0.0, 0.5, 0.0], dtype=np.float32)
        self.steps = 0
        return self._observation()

    def _observation(self):
        x, y, psi, u, r = self.state
        dx, dy = self.goal - self.state[:2]
        distance = np.hypot(dx, dy)
        bearing = np.arctan2(dy, dx)
        heading_error = wrap_angle(bearing - psi)

        min_clearance = self.minimum_clearance()

        return np.array(
            [
                x / 50.0,
                y / 50.0,
                np.sin(psi),
                np.cos(psi),
                u / self.max_speed,
                r / self.max_yaw_rate,
                distance / 60.0,
                heading_error / np.pi,
            ],
            dtype=np.float32,
        )

    def minimum_clearance(self):
        x, y = self.state[:2]
        if not self.obstacles:
            return 100.0

        clearances = []
        for ox, oy, radius in self.obstacles:
            d = np.hypot(x - ox, y - oy) - radius
            clearances.append(d)
        return float(min(clearances))

    def _collision(self):
        return self.minimum_clearance() <= self.collision_margin

    def step(self, action):
        throttle, rudder = np.clip(action, -1.0, 1.0)

        x, y, psi, u, r = self.state

        # Generic low-order surge and yaw dynamics.
        u_dot = 1.2 * throttle - 0.35 * u
        r_dot = 1.8 * rudder - 1.1 * r

        u = np.clip(u + self.dt * u_dot, 0.0, self.max_speed)
        r = np.clip(r + self.dt * r_dot, -self.max_yaw_rate, self.max_yaw_rate)

        psi = wrap_angle(psi + self.dt * r)
        x = x + self.dt * u * np.cos(psi)
        y = y + self.dt * u * np.sin(psi)

        self.state = np.array([x, y, psi, u, r], dtype=np.float32)
        self.steps += 1

        distance = np.linalg.norm(self.goal - self.state[:2])
        collision = self._collision()
        success = distance <= self.goal_radius
        timeout = self.steps >= self.max_steps

        # Generic shaped reward.
        reward = -0.02
        reward -= 0.015 * distance
        reward -= 0.01 * float(throttle**2 + rudder**2)

        if self.minimum_clearance() < 4.0:
            reward -= 0.3 * max(0.0, 4.0 - self.minimum_clearance())

        if success:
            reward += 100.0
        if collision:
            reward -= 100.0

        done = success or collision or timeout

        info = {
            "distance_to_goal": float(distance),
            "minimum_clearance": float(self.minimum_clearance()),
            "success": bool(success),
            "collision": bool(collision),
            "timeout": bool(timeout),
        }

        return self._observation(), float(reward), done, info

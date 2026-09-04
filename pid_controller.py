import numpy as np


def wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


class PIDController:
    """Generic PID controller with integral clamping."""

    def __init__(self, kp, ki=0.0, kd=0.0, integral_limit=10.0):
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.integral_limit = float(integral_limit)
        self.integral = 0.0
        self.prev_error = 0.0

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0

    def update(self, error, dt):
        self.integral += error * dt
        self.integral = np.clip(
            self.integral,
            -self.integral_limit,
            self.integral_limit,
        )

        derivative = (error - self.prev_error) / max(dt, 1e-6)
        self.prev_error = error

        return (
            self.kp * error
            + self.ki * self.integral
            + self.kd * derivative
        )


class NominalUSVController:
    """
    Nominal speed and heading controller.

    Output:
        [throttle, rudder] in [-1, 1]
    """

    def __init__(self):
        self.speed_pid = PIDController(0.8, 0.05, 0.02)
        self.heading_pid = PIDController(1.6, 0.02, 0.15)

    def reset(self):
        self.speed_pid.reset()
        self.heading_pid.reset()

    def compute(self, speed_ref, speed, heading_ref, heading, dt):
        speed_error = speed_ref - speed
        heading_error = wrap_angle(heading_ref - heading)

        throttle = self.speed_pid.update(speed_error, dt)
        rudder = self.heading_pid.update(heading_error, dt)

        return np.clip(np.array([throttle, rudder]), -1.0, 1.0)

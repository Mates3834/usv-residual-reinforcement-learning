import numpy as np

from src.control.pid_controller import NominalUSVController
from src.environment.usv_environment import USVEnvironment
from src.guidance.los_guidance import los_guidance
from src.planning.hybrid_astar import hybrid_astar


def evaluate_pid():
    env = USVEnvironment()
    controller = NominalUSVController()

    path = hybrid_astar(
        start=(0.0, 0.0, 0.0),
        goal=tuple(env.goal),
        obstacles=env.obstacles,
    )

    obs = env.reset()
    controller.reset()
    waypoint_idx = 0
    trajectory = [env.state[:2].copy()]

    for _ in range(env.max_steps):
        x, y, psi, speed, _ = env.state
        psi_ref, waypoint_idx = los_guidance(
            (x, y), path, waypoint_idx
        )

        action = controller.compute(
            2.5,
            speed,
            psi_ref,
            psi,
            env.dt,
        )

        obs, reward, done, info = env.step(action)
        trajectory.append(env.state[:2].copy())

        if done:
            break

    return np.asarray(trajectory), path, info


if __name__ == "__main__":
    trajectory, path, info = evaluate_pid()
    print(info)

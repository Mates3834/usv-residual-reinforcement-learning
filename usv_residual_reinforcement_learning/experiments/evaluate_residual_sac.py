import numpy as np

from src.control.pid_controller import NominalUSVController
from src.guidance.los_guidance import los_guidance


def evaluate_residual_sac(env, agent, path, speed_ref=2.5):
    """
    Evaluate a trained residual-SAC policy on top of the nominal PID.
    """
    controller = NominalUSVController()
    controller.reset()

    obs = env.reset()
    waypoint_idx = 0
    trajectory = [env.state[:2].copy()]

    for _ in range(env.max_steps):
        x, y, psi, speed, _ = env.state
        psi_ref, waypoint_idx = los_guidance(
            (x, y), path, waypoint_idx
        )

        nominal = controller.compute(
            speed_ref,
            speed,
            psi_ref,
            psi,
            env.dt,
        )

        residual = agent.residual_action(obs, deterministic=True)
        action = np.clip(nominal + residual, -1.0, 1.0)

        obs, _, done, info = env.step(action)
        trajectory.append(env.state[:2].copy())

        if done:
            break

    return np.asarray(trajectory), info

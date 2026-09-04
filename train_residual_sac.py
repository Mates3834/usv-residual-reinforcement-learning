import numpy as np

from src.agents.residual_sac import ReplayBuffer, ResidualSAC
from src.control.pid_controller import NominalUSVController
from src.guidance.los_guidance import los_guidance
from src.planning.hybrid_astar import hybrid_astar


def train_residual_sac(
    env,
    episodes=100,
    warmup_steps=1000,
    batch_size=128,
    speed_ref=2.5,
):
    """
    Generic residual-SAC training loop.

    Residual action is added to the nominal PID command:
        u_total = clip(u_pid + delta_u_sac, -1, 1)
    """

    controller = NominalUSVController()
    agent = ResidualSAC(
        state_dim=env.observation_dim,
        action_dim=env.action_dim,
    )
    replay = ReplayBuffer()

    path = hybrid_astar(
        start=(0.0, 0.0, 0.0),
        goal=tuple(env.goal),
        obstacles=env.obstacles,
    )

    total_steps = 0
    history = []

    for episode in range(episodes):
        obs = env.reset()
        controller.reset()
        waypoint_idx = 0
        episode_reward = 0.0

        for _ in range(env.max_steps):
            x, y, psi, u, _ = env.state
            psi_ref, waypoint_idx = los_guidance(
                (x, y),
                path,
                waypoint_idx,
            )

            nominal = controller.compute(
                speed_ref,
                u,
                psi_ref,
                psi,
                env.dt,
            )

            if total_steps < warmup_steps:
                residual = np.random.uniform(-0.1, 0.1, size=2)
            else:
                residual = agent.residual_action(obs)

            total_action = np.clip(nominal + residual, -1.0, 1.0)
            next_obs, reward, done, info = env.step(total_action)

            replay.add(obs, residual, reward, next_obs, done)
            obs = next_obs
            episode_reward += reward
            total_steps += 1

            if len(replay) >= batch_size and total_steps >= warmup_steps:
                agent.update(replay, batch_size)

            if done:
                break

        history.append(
            {
                "episode": episode + 1,
                "reward": float(episode_reward),
                "success": info["success"],
                "collision": info["collision"],
                "timeout": info["timeout"],
            }
        )

        print(
            f"Episode {episode+1:03d} | "
            f"Reward {episode_reward:8.2f} | "
            f"Success {info['success']} | "
            f"Collision {info['collision']}"
        )

    return agent, history

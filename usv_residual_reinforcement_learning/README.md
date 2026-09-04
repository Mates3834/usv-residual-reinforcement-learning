# USV Residual Reinforcement Learning

Generic and sanitized research-oriented implementation of residual reinforcement
learning for autonomous surface vehicle navigation.

The control architecture is:

Hybrid A* -> LOS Guidance -> Nominal PID -> Residual SAC -> USV

The residual policy augments, rather than replaces, the nominal controller:

    u_total = u_pid + delta_u_sac

Included modules:

- Simplified planar USV environment
- Static obstacle representation
- Hybrid-A*-style grid/heading planner
- Line-of-Sight guidance
- Nominal PID heading/speed control
- Soft Actor-Critic residual policy
- Replay buffer
- Residual SAC training loop
- PID baseline evaluation
- Residual SAC evaluation
- Generic navigation demo

> Note: This repository contains generic and sanitized implementations.
> Project-specific vessel parameters, operational scenarios, training data,
> tuned hyperparameters, and unpublished research results are intentionally omitted.

# USV Residual Reinforcement Learning

A research-oriented framework for autonomous surface vehicle (USV) navigation using **Hybrid A\*** path planning, **Line-of-Sight (LOS) guidance**, a nominal **PID controller**, and **Residual Soft Actor-Critic (Residual SAC)** reinforcement learning.

The main idea is to preserve the stability and interpretability of a conventional controller while allowing a reinforcement-learning policy to learn corrective control actions under navigation errors, disturbances, and environmental uncertainty.

---

## Overview

Autonomous marine navigation requires several components to operate together, including global path planning, guidance, low-level control, obstacle avoidance, and adaptation to environmental disturbances.

Pure model-based controllers can provide reliable nominal behavior but may experience performance degradation when the system operates under modeling uncertainty or changing environmental conditions. In contrast, reinforcement-learning controllers can learn nonlinear control strategies directly from interaction with the environment, but replacing the entire nominal controller with a learned policy may reduce interpretability and increase the training burden.

This project investigates a **Residual Reinforcement Learning** architecture in which a Soft Actor-Critic agent does not replace the conventional controller. Instead, the learned policy generates bounded corrective actions that augment the nominal PID command.

The resulting control architecture combines classical autonomous-navigation methods with modern deep reinforcement learning.

---

## System Architecture

The overall framework is organized as:

```text
Environment / Navigation Map
          │
          ▼
     Hybrid A*
   Path Planning
          │
          ▼
     Planned Path
          │
          ▼
    LOS Guidance
          │
          ▼
 Reference Heading
   & Speed Command
          │
          ▼
   Nominal PID Control ──────────┐
          │                      │
          │                      ▼
          │                Residual SAC
          │                Correction
          │                      │
          └──────────┬───────────┘
                     ▼
              Combined Control
                     │
                     ▼
                 USV Model
                     │
                     ▼
              State Feedback
```

The final control command is expressed as

\[
\mathbf{u}_t =
\mathbf{u}_{PID,t}
+
\Delta\mathbf{u}_{SAC,t}
\]

where

- \(\mathbf{u}_{PID,t}\) is the nominal control command,
- \(\Delta\mathbf{u}_{SAC,t}\) is the learned residual correction,
- \(\mathbf{u}_t\) is the final command applied to the vehicle.

The residual action is bounded before being combined with the nominal controller.

---

## 1. USV Navigation Environment

A simplified planar USV environment is provided for algorithm development and reinforcement-learning experiments.

The generic state vector is

\[
\mathbf{x}
=
[x,\ y,\ \psi,\ u,\ r]^T
\]

where

- \(x,y\) are the global position coordinates,
- \(\psi\) is the vessel heading,
- \(u\) is the surge velocity,
- \(r\) is the yaw rate.

The public implementation intentionally uses a lightweight dynamic model rather than a vessel-specific high-fidelity hydrodynamic model.

This keeps the repository suitable for algorithm demonstration while separating the public implementation from project-specific vehicle parameters.

---

## 2. Hybrid A* Path Planning

A Hybrid-A*-style planner is used to generate a collision-free global path from the initial USV position to the destination.

Unlike a conventional two-dimensional grid search, the planner includes a discretized heading state:

\[
q = (x,y,\psi)
\]

This introduces basic vehicle-orientation information into the search process.

The cost function follows the conventional structure

\[
f(n)=g(n)+h(n)
\]

where

- \(g(n)\) represents the accumulated path cost,
- \(h(n)\) represents the estimated cost from the current state to the goal.

A turning penalty is also introduced to discourage unnecessary heading changes.

The public implementation is intentionally compact and should be interpreted as a **Hybrid-A*-style educational planner**, rather than a complete production-grade Hybrid A* implementation.

---

## 3. Line-of-Sight Guidance

After the global path has been generated, the guidance layer converts the path into heading references for the USV controller.

For a selected waypoint

\[
P_w=(x_w,y_w)
\]

and current vessel position

\[
P=(x,y),
\]

the LOS heading command is calculated as

\[
\psi_{LOS}
=
\operatorname{atan2}(y_w-y,\ x_w-x).
\]

When the vessel enters a predefined switching radius around the active waypoint, the guidance system advances to the next waypoint.

The LOS layer therefore acts as the interface between global path planning and low-level control.

---

## 4. Nominal PID Controller

The nominal controller provides the baseline control behavior of the USV.

Two control channels are considered:

```text
Speed Error   → PID → Throttle Command
Heading Error → PID → Rudder Command
```

The generic PID law is

\[
u(t)
=
K_Pe(t)
+
K_I\int e(t)\,dt
+
K_D\frac{de(t)}{dt}.
\]

The heading error is wrapped to

\[
[-\pi,\pi]
\]

to avoid discontinuities around the angular boundary.

The nominal controller is responsible for maintaining reasonable navigation behavior even without reinforcement-learning corrections.

This is an important feature of the residual-learning architecture.

---

## 5. Residual Reinforcement Learning

Instead of learning the complete control law from scratch, the reinforcement-learning agent learns only a correction to the nominal controller.

Let

\[
\pi_{SAC}(s_t)
\]

represent the learned policy.

The residual command is generated as

\[
\Delta\mathbf{u}_{SAC,t}
=
S_r\pi_{SAC}(s_t),
\]

where \(S_r\) limits the authority of the learned controller.

The complete command becomes

\[
\mathbf{u}_t
=
\operatorname{clip}
\left(
\mathbf{u}_{PID,t}
+
\Delta\mathbf{u}_{SAC,t}
\right).
\]

This architecture allows the RL agent to focus on correcting deficiencies in the nominal controller rather than relearning the complete navigation task.

---

## 6. Soft Actor-Critic

Soft Actor-Critic is an off-policy actor-critic reinforcement-learning algorithm.

The implementation contains:

- stochastic Gaussian actor,
- twin Q-value critics,
- target critic networks,
- experience replay,
- entropy-regularized policy optimization,
- soft target-network updates.

The actor produces a stochastic residual action.

The two critic networks estimate

\[
Q_{\theta_1}(s,a)
\]

and

\[
Q_{\theta_2}(s,a).
\]

The minimum critic estimate is used during target computation to reduce positive value-estimation bias.

---

## 7. Entropy-Regularized Learning

SAC optimizes both expected return and policy entropy.

Conceptually, the objective can be represented as

\[
J(\pi)
=
\mathbb{E}
\left[
\sum_t
r(s_t,a_t)
+
\alpha
\mathcal{H}
\left(
\pi(\cdot|s_t)
\right)
\right].
\]

The entropy term encourages exploration during training.

The parameter \(\alpha\) controls the trade-off between reward maximization and stochastic exploration.

---

## 8. Experience Replay

Transitions generated during interaction with the environment are stored in a replay buffer:

\[
(s_t,a_t,r_t,s_{t+1},d_t).
\]

Mini-batches are randomly sampled from the replay memory during optimization.

This improves data reuse and reduces temporal correlation between successive training samples.

The public implementation includes a generic replay-buffer module as part of the Residual SAC agent.

---

## 9. Residual Action Space

The residual policy generates corrections for two generic control channels:

\[
\Delta\mathbf{u}
=
[\Delta u_{throttle},\Delta u_{rudder}]^T.
\]

These corrections are scaled before being added to the nominal controller.

Therefore, the learned controller operates with deliberately limited control authority.

Conceptually:

```text
Nominal PID
     │
     ├───────────────┐
     │               │
     ▼               ▼
 Base Action     SAC Residual
     │               │
     └───────┬───────┘
             ▼
       Final Action
```

This differs from direct reinforcement learning, where the neural policy generates the complete control command.

---

## 10. Reward Structure

The generic environment uses a shaped reward containing several components.

Conceptually,

\[
R_t
=
R_{progress}
+
R_{goal}
-
P_{collision}
-
P_{clearance}
-
P_{control}.
\]

The public example considers factors such as

- distance to the goal,
- successful goal arrival,
- collision,
- obstacle clearance,
- control effort.

The exact reward structure used in research experiments can be modified depending on the navigation objective and training curriculum.

---

## 11. Obstacle Representation

Static circular obstacles are included in the generic navigation environment.

Each obstacle is represented as

\[
O_i=(x_i,y_i,r_i)
\]

where \(r_i\) represents its radius.

The environment continuously evaluates the minimum obstacle clearance:

\[
d_{min}
=
\min_i
\left(
\|P-P_{O_i}\|-r_i
\right).
\]

This quantity can be incorporated into collision detection and reward shaping.

---

## 12. Training Framework

The generic training pipeline follows:

```text
Initialize Environment
        │
        ▼
Generate Global Path
        │
        ▼
Initialize PID + SAC
        │
        ▼
     Reset USV
        │
        ▼
     LOS Guidance
        │
        ▼
   PID Base Action
        │
        ▼
 SAC Residual Action
        │
        ▼
   Combine Actions
        │
        ▼
 Environment Step
        │
        ▼
 Store Transition
        │
        ▼
 Sample Replay Buffer
        │
        ▼
 Update Critics
        │
        ▼
 Update Actor
        │
        ▼
 Soft Target Update
```

A short random-action warm-up period can be used before policy updates begin in order to populate the replay buffer.

---

## 13. Baseline Evaluation

The repository separates nominal and learning-based evaluation.

### PID Baseline

The conventional navigation pipeline can be evaluated using

```text
Hybrid A* → LOS → PID → USV
```

without reinforcement learning.

### Residual SAC

The complete framework can then be evaluated using

```text
Hybrid A* → LOS → PID + Residual SAC → USV
```

This structure allows the effect of the learned residual controller to be studied relative to the conventional baseline.

---

## 14. Evaluation Metrics

The framework is suitable for evaluating metrics such as:

- navigation success rate,
- collision rate,
- timeout rate,
- cross-track error,
- heading error,
- minimum obstacle clearance,
- completion time,
- control effort,
- cumulative reward.

These metrics provide a basis for comparing conventional and learning-assisted navigation architectures.

No unpublished experimental performance values are included in the public version of this repository.

---

## Source Code

The repository contains generic implementations of the main modules used to demonstrate the proposed architecture.

```text
src/
├── environment/
│   ├── __init__.py
│   └── usv_environment.py
│
├── planning/
│   ├── __init__.py
│   └── hybrid_astar.py
│
├── guidance/
│   ├── __init__.py
│   └── los_guidance.py
│
├── control/
│   ├── __init__.py
│   └── pid_controller.py
│
├── agents/
│   ├── __init__.py
│   └── residual_sac.py
│
└── training/
    ├── __init__.py
    └── train_residual_sac.py
```

### Module Descriptions

| Module | Description |
|---|---|
| `usv_environment.py` | Generic planar USV simulation environment |
| `hybrid_astar.py` | Heading-aware Hybrid-A*-style global planner |
| `los_guidance.py` | LOS waypoint guidance |
| `pid_controller.py` | Nominal speed and heading PID controllers |
| `residual_sac.py` | SAC actor, twin critics, replay buffer and residual policy |
| `train_residual_sac.py` | Residual SAC training pipeline |
| `evaluate_pid.py` | Conventional PID baseline evaluation |
| `evaluate_residual_sac.py` | Residual SAC evaluation |
| `run_navigation_demo.py` | Generic navigation demonstration |
| `train_demo.py` | Example training script |

The `__init__.py` files define the corresponding directories as Python packages and contain short package-level descriptions.

---

## Repository Structure

```text
usv-residual-reinforcement-learning/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── src/
│   ├── environment/
│   │   ├── __init__.py
│   │   └── usv_environment.py
│   │
│   ├── planning/
│   │   ├── __init__.py
│   │   └── hybrid_astar.py
│   │
│   ├── guidance/
│   │   ├── __init__.py
│   │   └── los_guidance.py
│   │
│   ├── control/
│   │   ├── __init__.py
│   │   └── pid_controller.py
│   │
│   ├── agents/
│   │   ├── __init__.py
│   │   └── residual_sac.py
│   │
│   └── training/
│       ├── __init__.py
│       └── train_residual_sac.py
│
├── experiments/
│   ├── evaluate_pid.py
│   └── evaluate_residual_sac.py
│
└── examples/
    ├── run_navigation_demo.py
    └── train_demo.py
```

---

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd usv-residual-reinforcement-learning
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Main dependencies:

- Python
- NumPy
- PyTorch
- Matplotlib

---

## Running the Navigation Demo

The conventional PID navigation example can be executed with:

```bash
python examples/run_navigation_demo.py
```

The demo performs:

```text
Path Planning
     ↓
LOS Guidance
     ↓
PID Control
     ↓
USV Simulation
```

and visualizes the planned path and simulated vessel trajectory.

---

## Running the Residual SAC Training Example

A generic training example is provided in:

```bash
python examples/train_demo.py
```

The example initializes the USV environment, nominal PID controller, replay buffer, and Residual SAC agent before executing the training loop.

The provided example is intended to demonstrate the software architecture rather than reproduce final research results.

---

## Technologies

- Python
- PyTorch
- NumPy
- Matplotlib
- Deep Reinforcement Learning
- Soft Actor-Critic
- Residual Reinforcement Learning
- Autonomous Navigation
- Marine Robotics
- Path Planning
- Guidance and Control

---

## Research Areas

This project is related to:

- Autonomous Surface Vehicles
- Marine Robotics
- Guidance, Navigation and Control
- Autonomous Navigation
- Deep Reinforcement Learning
- Residual Reinforcement Learning
- Soft Actor-Critic
- Path Planning
- Adaptive and Intelligent Control
- Learning-Based Control

---

## Project Motivation

The central research question behind this framework is whether a learning-based controller can improve autonomous navigation performance **without discarding the structure of an existing model-based control system**.

Rather than asking a neural policy to learn the entire control problem from scratch, Residual Reinforcement Learning introduces a structured relationship between classical control and machine learning:

\[
\boxed{
\text{Model-Based Control}
+
\text{Learned Residual}
=
\text{Hybrid Intelligent Control}
}
\]

This makes the architecture particularly relevant to autonomous systems where conventional guidance and control methods already provide a strong nominal solution.

---

## Public Implementation Notice

The source code provided in this repository contains **generic and sanitized implementations** of the main algorithms used to demonstrate the research framework.

The public implementation does **not** contain:

- vessel-specific operational parameters,
- platform-specific hydrodynamic coefficients,
- operational mission configurations,
- restricted navigation data,
- unpublished experimental datasets,
- final research tuning parameters,
- unpublished performance results.

The repository is intended to demonstrate the **software architecture, algorithmic concepts, and research methodology** without exposing project-specific implementation details.

---

## Status

**Research project / active development**

The public repository currently provides the core architecture and generic demonstration modules. Additional experimental results, benchmark comparisons, training analyses, and publication information may be added as the research progresses.

---

## Author

**Mehmet Ateş**

Research interests: autonomous systems, marine robotics, guidance and control, path planning, and reinforcement learning.

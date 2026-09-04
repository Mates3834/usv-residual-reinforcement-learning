# Residual Reinforcement Learning for Autonomous USV Navigation

A hybrid model-based and learning-based control framework for autonomous
Unmanned Surface Vehicle (USV) navigation using **Hybrid A*** global path
planning, **Line-of-Sight (LOS)** guidance, nominal **PID control**, and
**Residual Soft Actor-Critic (SAC)** reinforcement learning.

The main objective is to improve the robustness and navigation performance
of a conventional autonomous control architecture without replacing the
baseline controller with a fully learning-based policy.

---

## System Architecture

The proposed autonomous navigation stack consists of:

**Hybrid A* → LOS Guidance → PID Control → Residual SAC → USV Dynamics**

The nominal controller provides a reliable baseline control action, while
the SAC policy learns an additional corrective action.

The final control command is defined as:

u_total = u_PID + Δu_SAC

This allows reinforcement learning to focus on compensating for behavior
that is not adequately handled by the nominal controller.

---

## Global Path Planning

A **Hybrid A*** based planner generates a feasible global reference path
between the initial and target positions.

The generated path is subsequently used by the guidance layer to determine
the desired navigation behavior.

---

## LOS Guidance

A **Line-of-Sight (LOS)** guidance algorithm converts the planned path into
reference commands for the low-level control system.

This provides a hierarchical architecture:

Hybrid A*
    ↓
Reference Path
    ↓
LOS Guidance
    ↓
Reference Command
    ↓
PID + Residual SAC
    ↓
USV

---

## Nominal PID Controller

The PID controller acts as the baseline controller.

Instead of requiring the reinforcement learning agent to learn the complete
control problem from scratch, the PID controller provides nominal closed-loop
behavior.

The baseline action is represented as:

u_PID

This creates a stable model-based foundation on top of which the RL policy
can learn corrective behavior.

---

## Residual Soft Actor-Critic

**Soft Actor-Critic (SAC)** is used as the reinforcement learning component.

Unlike a direct RL controller, the SAC policy does not replace the nominal
controller.

Instead, it learns a residual action:

Δu_SAC

which is combined with the PID command:

u_total = u_PID + Δu_SAC

Conceptually:

State / Tracking Information
        │
        ├───────────────► PID ─────────► u_PID
        │
        └───────────────► SAC ─────────► Δu_SAC
                                            │
                           u_PID + Δu_SAC ◄──┘
                                  │
                                  ▼
                                 USV
                                  │
                                  └──── Feedback

---

## Curriculum Learning

Training is progressively performed across increasingly difficult
navigation environments.

### Stage 1 — Easy
Basic navigation scenarios are used to establish the initial policy.

### Stage 2 — Medium
Environmental complexity and disturbances are increased.

### Stage 3 — Hard
More challenging navigation conditions are introduced.

### Stage 4 — Full
The learned policy is evaluated under the complete range of operating
conditions.

This curriculum-based strategy allows previously learned navigation
behavior to be transferred to increasingly difficult environments.

---

## Benchmark Controllers

The proposed Residual SAC architecture is designed to be compared against
multiple control strategies under identical scenarios:

| Controller | Description |
|---|---|
| PID | Classical baseline controller |
| PPO | Reinforcement learning baseline |
| Direct SAC | SAC controlling the system directly |
| Residual SAC | PID augmented by SAC residual actions |

The comparison is designed to determine whether residual learning provides
advantages over both conventional and fully learning-based approaches.

---

## Evaluation Metrics

Performance is evaluated using:

- Navigation success rate
- Collision rate
- Cross-track error
- Heading error
- Minimum obstacle clearance
- Travel / completion time
- Control effort
- Episode reward
- Computational performance

---

## Training Environments

The simulation framework investigates autonomous navigation under varying
conditions including:

- Static obstacles
- Dynamic obstacles
- Environmental disturbances
- Randomized initial conditions
- Increasing environment difficulty

The objective is to evaluate both learning performance and policy
generalization.

---

## Technologies

- Python
- Reinforcement Learning
- Soft Actor-Critic (SAC)
- Residual Reinforcement Learning
- Hybrid A*
- LOS Guidance
- PID Control
- Autonomous Navigation
- USV Simulation

---

## Research Areas

- Autonomous Surface Vehicles
- Marine Robotics
- Reinforcement Learning for Control
- Residual Reinforcement Learning
- Autonomous Navigation
- Path Planning
- Guidance and Control
- Learning-Based Control
- Intelligent Autonomous Systems

---

## Repository Structure

```text
usv-residual-reinforcement-learning/
│
├── README.md
│
├── planning/
│   └── hybrid_astar/
│
├── guidance/
│   └── los/
│
├── control/
│   ├── pid/
│   └── residual_sac/
│
├── environment/
│
├── training/
│   ├── easy/
│   ├── medium/
│   ├── hard/
│   └── full/
│
├── evaluation/
│   ├── pid/
│   ├── ppo/
│   ├── direct_sac/
│   └── residual_sac/
│
├── results/
│
└── docs/

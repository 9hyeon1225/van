VAN-Flow

Official implementation of VAN-Flow, proposed in:

Variance-Averse n-Step Offline Reinforcement Learning for Sparse Long-Horizon Environments
ICML 2026

Overview

VAN-Flow is an offline reinforcement learning algorithm designed for sparse, long-horizon environments.

The method combines a distributional critic with variance-averse objectives and a flow-matching actor to enable stable n-step learning under high return variance.

Key Ideas

Categorical (distributional) critic for modeling return uncertainty

Variance-averse expectation to stabilize long-horizon backups

Flow-matching actor trained with Euler discretization

Best-of-N action selection guided by variance-aware Q-values

Action chunking for joint multi-step control

Code Structure
.
├── agents/
│   └── van_flow_agent.py
├── utils/
│   ├── encoders.py
│   ├── networks.py
│   └── flax_utils.py
├── train.py
└── README.md

Installation
pip install -r requirements.txt


Required packages include jax, flax, optax, and ml_collections.

Running Experiments
Offline Training
MUJOCO_GL=egl python main.py python main.py  --horizon_length=3 --env_name=antmaze-large-explore-singletask-task4-v0 --agent.lmbda=3



Configuration

Important hyperparameters:

num_atoms: number of categorical atoms

horizon_length: n-step return horizon

delta: variance-aversion strength

actor_num_samples: best-of-N samples

flow_steps: Euler steps for the flow actor

risk: enable variance-averse Q guidance

See get_config() for the full list.

Notes

This code is intended for research and reproducibility.
Hyperparameters are task-dependent and should be tuned per environment.

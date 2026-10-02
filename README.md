# VAN-Flow  
**Variance-Averse n-Step Offline Reinforcement Learning for Sparse Long-Horizon Environments**

This repository contains the official implementation of **VAN-Flow**, a variance-aware offline reinforcement learning framework proposed for sparse, long-horizon environments.  
VAN-Flow addresses the instability of long-horizon $n$-step returns in offline RL by explicitly modeling and penalizing return variance using categorical distributional critics and variance-averse expectations.

📄 **Paper**: *Variance-Averse n-Step Offline Reinforcement Learning for Sparse Long-Horizon Environments* (NeurIPS 2026 accepted)  
🌐 **Project Page**: https://9hyeon1225.github.io/van/

---

## 🚀 Key Idea

While $n$-step returns reduce long-horizon bootstrapping bias, they amplify **return variance**, which severely degrades performance in offline RL—especially under heterogeneous or noisy datasets.

**VAN-Flow** introduces:
- **Variance-Averse Expectation** for categorical return distributions
- **Distributional n-step critic** to explicitly capture return variance
- **Flow-matching policy** with variance-aware $Q$-guidance
- **Rejection sampling** to avoid high-variance, out-of-distribution actions

This enables stable and effective long-horizon learning even under high-variance offline datasets.

---

## ✨ Main Contributions

- Empirical analysis showing the failure of naive $n$-step returns under high-variance offline data
- A **variance-averse expectation operator** with theoretical guarantees under convex order
- **VAN-Flow**, a unified offline RL framework combining:
  - $n$-step returns
  - categorical distributional critics
  - flow-based policies with $Q$ guidance
- Strong performance on **D4RL AntMaze** and **OGBench** long-horizon benchmarks
- Robust offline-to-online fine-tuning behavior

---

## 🧠 Method Overview

**Critic**
- Categorical distributional critic (C51-style)
- $n$-step Bellman backup
- Variance-aware aggregation using variance-averse expectation

**Actor**
- Flow-matching policy (ODE-inspired)
- Best-of-$N$ rejection sampling
- Variance-averse $Q$-guided optimization

---

## 🧪 Experimental Results

VAN-Flow consistently outperforms strong baselines including:
- IQL, ReBRAC, HIQL
- LEQ, QC, TD3BC+MS
- FQL, BFN, D4PG

across:
- Sparse-reward tasks
- Long-horizon environments
- High-variance offline datasets
- Offline-to-online adaptation settings

---

## ⚙️ Installation

# Install dependencies
pip install -r requirements.txt

```bash
#humanoidmaze-giant-navigate
MUJOCO_GL=egl python main.py \
  --env_name humanoidmaze-giant-navigate-singletask-v0 \
  --horizon_length 4 \
  --agent.lmbda 10 \
  --agent.discount=0.999 \
  --agent.v_min=-1000 \

#humanoidmaze-large-navigate
MUJOCO_GL=egl python main.py \
  --env_name humanoidmaze-large-navigate-singletask-v0 \
  --horizon_length 4 \
  --agent.lmbda 10 \
  --agent.discount=0.999 \
  --agent.v_min=-1000 \

#antmaze-giant-navigate
MUJOCO_GL=egl python main.py \
  --env_name antmaze-giant-navigate-singletask-v0 \
  --horizon_length 8 \
  --agent.lmbda 10 \
  --agent.discount=0.999 \
  --agent.v_min=-1000 \

#antmaze-large-navigate
MUJOCO_GL=egl python main.py \
  --env_name antmaze-large-navigate-singletask-v0 \
  --horizon_length 4 \
  --agent.lmbda 3

#antmaze-large-explore
MUJOCO_GL=egl python main.py \
  --env_name antmaze-large-explore-singletask-v0 \
  --horizon_length 3 \
  --agent.lmbda 3

#antmaze-teleport-navigate
MUJOCO_GL=egl python main.py \
  --env_name antmaze-giant-navigate-singletask-v0 \
  --horizon_length 2 \
  --agent.lmbda 300
  --agent.discount=0.999 \
  --agent.v_min=-1000
  --agent.delta=7
#scene-play
MUJOCO_GL=egl python main.py \
  --env_name scene-play-singletask-v0 \
  --horizon_length 3 \
  --agent.lmbda 3

#puzzle-3x3-play
MUJOCO_GL=egl python main.py \
  --env_name puzzle-3x3-play-singletask-v0 \
  --horizon_length 5 \
  --agent.lmbda 3

#puzzle-3x3-noisy
MUJOCO_GL=egl python main.py \
  --env_name puzzle-3x3-noisy-singletask-v0 \
  --horizon_length 5 \
  --agent.lmbda 3

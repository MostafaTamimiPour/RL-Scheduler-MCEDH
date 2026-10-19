# Hybrid Rule/RL Scheduler for Energy-Harvesting Mixed-Criticality Systems

## 📖 Project Overview
This repository presents a Python implementation of a scheduling framework for **Energy-Harvesting Mixed-Criticality Real-Time Systems**. It combines the strict safety guarantees of the ED-H (Earliest Deadline First with Harvesting) algorithm with **Reinforcement Learning (RL)**. 

The core philosophy is a **Hybrid Rule + RL Design**. The RL agent is never allowed to make unsafe decisions. Strict safety rules are enforced structurally as a "Hard Mask" in the environment, and the RL agent only operates within the genuine discretion zone (ED-H's Rule 5) to decide which safe task to run or when to idle.

## ✨ Key Features
- **Multiple RL Algorithms**: Implements Tabular **Q-Learning**, **SARSA**, and **SARSA(λ)** from scratch.
- **Energy Harvesting Models**: Supports `constant`, `random`, and `sinusoidal` energy harvesting profiles, along with static energy leakage.
- **Realistic Overheads**: Simulates the time and energy cost of Preemption (task switching) and Mode-Switches (LO to HI transitions).
- **Hard Safety Mask**: Guarantees that HI tasks never miss deadlines by filtering out unsafe actions before the RL agent is consulted.
- **Rich Visualizations**: Generates exact Gantt charts (task execution, energy traces) and RL training plots.

## 🧠 Architecture & Safety Design
### Why a Hard Mask instead of Reward Penalties?
In safety-critical systems, an exploratory RL agent might accidentally choose a dangerous action if safety relies solely on negative rewards (penalties). To prevent this:
1.  **Rules 3/4 (Forced Actions)**: If the battery is depleted or a HI task has zero slack, the environment forces the safe action, bypassing the RL agent entirely.
2.  **Hard Safety Filter**: A LO task is never offered as a legal choice in HI mode unless it is mathematically proven safe.
3.  **Result**: The agent only chooses from a list of provably safe actions.

### Reward Philosophy
- **HI Tasks**: Carry zero positive reward. They are a mandatory duty guaranteed by the hard safety layer.
- **LO Tasks**: All positive rewards are concentrated here. Maximizing LO task completion while staying in the cheaper LO mode is the actual metric of system performance.

## 🤖 Algorithms Implemented
You can switch between algorithms by changing the `ALGORITHM` variable in `main.py`:
1. **Q-Learning (Off-policy)**: Learns the optimal policy using the maximum Q-value of the next state (`max Q(s', a')`). Highly effective here because the hard mask ensures the `max` function only evaluates safe actions.
2. **SARSA (On-policy)**: Updates Q-values based on the actual action taken (`Q(s', a')`). Learns a more conservative, safer policy during exploration.
3. **SARSA(λ) (On-policy with Eligibility Traces)**: Speeds up learning by propagating rewards backward through time using eligibility traces (controlled by the `LAMBDA` parameter).

## 📁 Repository Structure
```text
├── environment.py   # Core simulation: Rules 1-5, energy models, hard safety masks.
├── agent.py         # RL algorithms: Q-Learning, SARSA, SARSA(λ) with state masking.
├── utils.py         # Visualization: Gantt chart, energy trace, and RL training plots.
├── main.py          # Entry point: Configures parameters, trains the agent, and evaluates.
└── README.md        # Project documentation.
```

## How to Run
Install dependencies (if needed):

bash
pip install numpy matplotlib
Run the scheduler:

bash
python main.py
(You can configure the algorithm, harvesting mode, and rewards directly inside main.py)

📊 Sample Results
Here are sample outputs from the simulation after training the agent:
<img width="1536" height="760" alt="training_results" src="https://github.com/user-attachments/assets/5a74eaaf-4d5d-4be6-9ead-5b1049be9496" />

1. Training Progress
The agent successfully learns to maximize the reward by prioritizing LO tasks safely. The red line shows the 20-episode moving average.
 

2. Task Schedule & Energy (Gantt Chart)
Visualization of task execution, battery levels, preemption overheads (orange), and LO-to-HI mode switches (purple).
<img width="1536" height="760" alt="gantt_chart" src="https://github.com/user-attachments/assets/4c553a03-629a-491a-9f45-8b21f866d6c9" />


⚠️ Note
Due to the stochastic nature of the energy harvesting profiles (e.g., random or sinusoidal) and the epsilon-greedy exploration during RL training, the reproduced plots may vary slightly between different runs. The implementation and core logic remain fully reproducible.

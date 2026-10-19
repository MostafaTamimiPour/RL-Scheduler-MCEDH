"""
main.py - Main execution file for Task Scheduler with Q-Learning
"""

import random
import numpy as np
from environment import SchedulerEnv
from agent import RLAgent
from utils import plot_gantt_chart, plot_training_results

if __name__ == "__main__":
    print("="*60)
    print("MIXED-CRITICALITY TASK SCHEDULER with Q-Learning")
    print("="*60)
    
    # Set random seeds for reproducibility
    random.seed(101)
    np.random.seed(101)
    
    # ============================================================
    ALGORITHM = 'sarsa'  # 'q_learning' or 'sarsa' or 'sarsa_lambda'
    # ============================================================

    # ----- Agent Parameters -----
    NUM_EPISODES = 5000
    ALPHA = 0.7
    GAMMA = 0.9
    LAMBDA = 0.5  #only for SARSA(λ)
    EPSILON_START = 1.0
    EPSILON_MIN = 0.001
    DECAY_TYPE = 'exponential' # 'exponential'  'linear'  'sqrt'

    # ----- Harvesting Mode -----
    HARVEST_MODE = 'sinusoidal'    # 'constant' / 'random' / 'sinusoidal'
    HARVEST_MEAN = 10              # mean
    HARVEST_AMPLITUDE = 5          # range of fluctuating energy 
    HARVEST_PERIOD = 5            # period of energy 

    # For backward compat, still pass HARVEST_RATE for random mode
    HARVEST_RATE = (10, 10)
    LEAKAGE_RATE = 0.1
    
    # ----- Environment Parameters -----
    MAX_ENERGY = 50
    SWITCH_PROBABILITY = 0.10
    MAX_TIME = None
    HI_MODE_FACTOR = 8/5  # ← controlling (1.0, 1.5, 2.0)

    # ----- Reward Settings -----
    # Change only these values to change what the RL agent learns.
    # Positive values are rewards; negative values are penalties.
    #
    # NOTE ON THE HYBRID RULE+RL DESIGN: HI-task safety is now enforced
    # structurally by environment.get_valid_actions() (Rules 3/4 + the
    # LO-in-HI hard filter), not by reward shaping. CATASTROPHIC_HI_MISS_
    # PENALTY and UNSAFE_LO_IN_HI_PENALTY are kept only as a defensive net
    # (env.step() rejects any action outside get_valid_actions()) -- they
    # should essentially never fire once every caller masks properly.
    #
    # HI_EXECUTION_REWARD and HI_COMPLETION_BONUS are deliberately ZERO.
    # HI tasks are a duty, not a scored achievement: they are already
    # guaranteed by the hard safety layer (forced execution the instant
    # they become mandatory) and by CATASTROPHIC_HI_MISS_PENALTY, so
    # rewarding them further would just bias the agent toward running them
    # early/voluntarily. That's undesirable here because completing a HI
    # task's LO-mode budget while still in LO mode is what triggers the
    # LO->HI switch -- so deferring optional HI work in favor of LO work,
    # whenever it's safe to do so, keeps the system in the cheaper LO mode
    # longer. All positive reward is therefore concentrated on LO execution
    # and completion, since that's what actually reflects system quality
    # once HI safety is already guaranteed structurally.

    REWARDS = {
        # ===== Penalties =====
        "IDLE_PENALTY": -15.0,                     # idling is costly
        "UNAVAILABLE_TASK_PENALTY": -5.0,          # defensive-only: masked out by get_valid_actions()
        "UNSAFE_LO_IN_HI_PENALTY": -50.0,          # defensive-only: masked out by get_valid_actions()
        "NOT_ENOUGH_ENERGY_PENALTY": -2.0,         # mild energy shortfall
        "PREEMPTION_OVERHEAD_REWARD": -15.0,       # preemption has a cost
        "PREEMPTION_OVERHEAD_FAILURE_PENALTY": -10.0,
        "MODE_SWITCH_OVERHEAD_PENALTY": -5.0,
        "LO_DEADLINE_MISS_PENALTY": -200.0,         # losing a LO task is costly
        "CATASTROPHIC_HI_MISS_PENALTY": -1500.0,    

        # ===== Execution rewards =====
        "HI_EXECUTION_REWARD": 0.0,                # duty, not an achievement -- see note above
        "LO_EXECUTION_REWARD": 100.0,               # LO work is the real performance signal

        # ===== Completion rewards =====
        "HI_COMPLETION_BONUS": 0.0,                # duty, not an achievement -- see note above
        "LO_COMPLETION_BONUS": 100.0,               # finishing LO tasks is what "performance" means here

        # ===== Special rewards =====
        "OPTIONAL_LO_IN_HI_REWARD": 100.0,         # safe LO execution in HI-mode (the real Rule-5 win)
        "ALL_TASKS_FINISHED_BONUS": 600.0,         # full success!
        "CONTINUE_SAME_TASK_BONUS": 20.0,

    }

    
    # ===== Create Environment =====
    # Keep this scenario deterministic for reproducible experiments.  Pass
    # ``tasks=None`` to SchedulerEnv when you want randomly generated tasks.

    TASKS = [
        (0, 4, 18, 4, False, 5, "HI"),
        (1, 4, 26, 4, False, 5, "HI"),
        (2, 2,  6, 2, False, 3, "LO"),   
        (3, 2, 10, 2, False, 3, "LO"),   
        (4, 3, 24, 3, False, 4, "LO"),
    ]
    # In HI mode, HI execution and energy/unit are both doubled.
    env = SchedulerEnv(
        tasks=TASKS,
        max_energy=MAX_ENERGY,
        harvest_rate=HARVEST_RATE,
        max_time=MAX_TIME,
        switch_probability=SWITCH_PROBABILITY,
        hi_mode_factor=HI_MODE_FACTOR,  
        reward_settings=REWARDS,
        harvest_mode=HARVEST_MODE,
        harvest_mean=HARVEST_MEAN,
        harvest_amplitude=HARVEST_AMPLITUDE,
        harvest_period=HARVEST_PERIOD,
        leakage_rate=LEAKAGE_RATE
    )
    
    print("\nTasks:")
    for task in env.tasks:
        print(f"Task {task['id']} [{task['criticality']}]: exec_time={task['exec_time']} ms, "
              f"deadline={task['deadline']} ms, energy_cost={task['energy_cost']}")
    
    env.render()
    
    print(f"\nSettings:")
    print(f"  Episodes: {NUM_EPISODES}")
    print(f"  Alpha: {ALPHA}")
    print(f"  Gamma: {GAMMA}")
    if ALGORITHM == 'sarsa_lambda':
        print(f"  Lambda: {LAMBDA}")
    print(f"  HI Mode Factor: {HI_MODE_FACTOR}")  
    print(f"  Max Energy: {MAX_ENERGY}")
    print(f"  Harvest Rate: {HARVEST_RATE}")
    print(f"  Mode switch probability: {SWITCH_PROBABILITY:.0%}")
    print("\nReward settings:")
    for reward_name, reward_value in REWARDS.items():
        print(f"  {reward_name}: {reward_value}")
    
    # ===== Create Agent and Train =====
    agent = RLAgent(
        env,
        algorithm=ALGORITHM,
        alpha=ALPHA,
        gamma=GAMMA,
        lam=LAMBDA,
        epsilon_start=EPSILON_START,
        epsilon_min=EPSILON_MIN,
        decay_type=DECAY_TYPE
    )
    
    Q, rewards, eps_history = agent.train(episodes=NUM_EPISODES)
    
    # ===== Plot Training Results =====
    plot_training_results(rewards, eps_history, EPSILON_MIN)
    
    # ===== Test Learned Policy =====
    print("\n" + "="*60)
    print("Testing Learned Policy:")
    print("="*60)
    
    state = env.reset()
    done = False
    total_reward = 0
    steps = 0
    
    while not done and steps < 50:
        # Mask here too: evaluation must respect the same Rules 3/4/5 safety
        # boundary as training, so the greedy policy is never even offered
        # an unsafe or HI-mandatory action to (mis-)choose from.
        valid_actions, forced = env.get_valid_actions()
        if len(valid_actions) == 1:
            action = valid_actions[0]
        elif state in Q:
            q_vals = {a: Q[state][a] for a in valid_actions}
            action = max(q_vals, key=q_vals.get)
        else:
            action = random.choice(valid_actions)
        
        state, reward, done = env.step(action)
        total_reward += reward
        steps += 1
        env.render()
    
    print(f"\nTotal Reward: {total_reward}")
    print(f"Steps: {steps}")
    
    # ===== Plot Gantt Chart with Energy =====
    plot_gantt_chart(
        schedule_log=env.schedule_log,
        tasks=env.tasks,
        energy_log=env.energy_log,  
        title="Task Schedule with Energy",
        max_energy=env.max_energy,
        mode_log=env.mode_log,
        max_time=env.max_time,
    )
    
    # ===== Print Q-Table =====
    print("\n" + "="*60)
    print("Q-Table (Sample):")
    print("="*60)
    for i, (state, q_vals) in enumerate(Q.items()):
        if i < 10:  # Show first 10 states
            print(f"State {state}: {q_vals}")
    if len(Q) > 10:
        print(f"... and {len(Q)-10} more states")

"""Plots and summaries for the scheduler experiments."""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def plot_gantt_chart(schedule_log, tasks, energy_log=None, title="Task Schedule", max_energy=None,
                     mode_log=None, max_time=None):
    """Plot interval-accurate execution and battery traces.

    Each schedule record represents [start_time, end_time), so a completion
    ending at ``deadline`` is correctly shown as on time.
    """
    if not schedule_log and not energy_log:
        print("No execution or energy history to display.")
        return

    fig, axes = plt.subplots(3 if energy_log else 2, 1, figsize=(14, 12), sharex=True)
    ax_gantt, ax_timeline = axes[:2]
    ax_energy = axes[2] if energy_log else None
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    task_colors = {task["id"]: colors[i % len(colors)] for i, task in enumerate(tasks)}
    last_logged_time = max(
        [task["deadline"] for task in tasks]
        + [entry["end_time"] for entry in schedule_log]
        + [entry["time"] for entry in (energy_log or [])]
    )
    # ``max_time`` comes from the environment.  It makes all integer time
    # boundaries visible even when the episode finishes early.
    display_max_time = max(last_logged_time, max_time if max_time is not None else 0)
    overhead_row = max(task["id"] for task in tasks) + 1
    has_overhead_event = any(entry.get("event") in ("mode_switch_overhead", "preemption_overhead")
                             for entry in schedule_log)

    for entry in schedule_log:
        task_id = entry["task_id"]
        start, end = entry["start_time"], entry["end_time"]
        if entry.get("event") in ("mode_switch_overhead", "preemption_overhead"):
            # Each overhead is a real CPU interval, separate from every task.
            if entry["event"] == "mode_switch_overhead":
                color = "purple"
                label = "LO->HI"
            else:
                color = "darkorange"
                label = f"T{entry['from_task_id']}->T{entry['to_task_id']}"
            ax_gantt.barh(overhead_row, end - start, left=start, color=color, edgecolor="black")
            ax_gantt.text((start + end) / 2, overhead_row, label, ha="center", va="center",
                          color="white", weight="bold", fontsize=8)
            ax_timeline.barh(0, end - start, left=start, color=color, edgecolor="black")
            ax_timeline.text((start + end) / 2, 0, label, ha="center", va="center",
                              color="white", weight="bold", fontsize=8)
        else:
            ax_gantt.barh(task_id, end - start, left=start, color=task_colors[task_id], edgecolor="black")
            ax_gantt.text((start + end) / 2, task_id, f"T{task_id}", ha="center", va="center", color="white", weight="bold")
            ax_timeline.barh(0, end - start, left=start, color=task_colors[task_id], edgecolor="black")
            ax_timeline.text((start + end) / 2, 0, f"T{task_id}", ha="center", va="center", color="white", weight="bold")

    for task in tasks:
        for axis in (ax_gantt, ax_timeline):
            axis.axvline(task["deadline"], color="crimson", linestyle="--", alpha=.75)
        ax_gantt.text(task["deadline"] + .05, task["id"] + .28, f"D{task['id']}={task['deadline']}", color="crimson", fontsize=9)

    # A vertical purple line makes the LO -> HI transition visible in every plot.
    for event in mode_log or []:
        for axis in axes:
            axis.axvline(event["time"], color="purple", linestyle="-.", linewidth=2)
        ax_gantt.text(event["time"] + .05, max(task["id"] for task in tasks) + .45,
                      "HI mode", color="purple", weight="bold")

    gantt_rows = [task["id"] for task in tasks]
    gantt_labels = [f"Task {task['id']}" for task in tasks]
    if has_overhead_event:
        gantt_rows.append(overhead_row)
        gantt_labels.append("System overhead")
    ax_gantt.set_yticks(gantt_rows, gantt_labels)
    ax_gantt.set_ylabel("Task")
    ax_gantt.set_title(title)
    ax_gantt.legend(handles=[mpatches.Patch(color="crimson", label="Deadline"),
                             mpatches.Patch(color="gray", label="Executed task unit"),
                             mpatches.Patch(color="purple", label="LO -> HI overhead (1 ms, 1 energy)"),
                             mpatches.Patch(color="darkorange", label="Task-switch overhead (1 ms, 1 energy)")], loc="upper right")
    ax_timeline.set_yticks([])
    ax_timeline.set_ylabel("CPU")
    ax_timeline.set_title("Exact processor timeline (each box is one time slot)")

    # Draw every time slot first. Task bars drawn above show exactly which
    # task occupied [0, 1), [1, 2), [2, 3), and so on. Empty boxes mean idle.
    for time_slot in range(display_max_time):
        ax_timeline.barh(0, 1, left=time_slot, color="white", edgecolor="black", linewidth=1, zorder=0)

    if ax_energy is not None:
        # The terminal calls render() after every step.  Therefore the printed
        # value "Time: t | Energy: e" is exactly energy_after at x=t.
        # Include the initial terminal state at time 0 as the first point.
        end_times = [entry["time"] for entry in energy_log]
        initial_energy = energy_log[0]["energy_before"]
        terminal_energy = [entry["energy_after"] for entry in energy_log]
        plot_times = [0] + end_times
        plot_energy = [initial_energy] + terminal_energy
        ax_energy.plot(plot_times, plot_energy, marker="o", color="forestgreen",
                       label="Terminal energy", linewidth=2)
        executed = [entry for entry in energy_log if entry["task_executed"] is not None]
        if executed:
            ax_energy.scatter([entry["time"] for entry in executed], [entry["energy_after"] for entry in executed], color="tab:red", label="Task executed", zorder=4)
        insufficient = [entry for entry in energy_log if entry["reason"] == "not_enough_energy"]
        if insufficient:
            ax_energy.scatter([entry["time"] for entry in insufficient], [entry["energy_after"] for entry in insufficient], color="orange", marker="x", s=70, label="Insufficient energy")
        switch_energy = [entry for entry in energy_log if entry["reason"] == "mode_switch_overhead"]
        if switch_energy:
            ax_energy.scatter([entry["time"] for entry in switch_energy],
                              [entry["energy_after"] for entry in switch_energy],
                              color="purple", marker="s", s=80, label="LO->HI overhead: -1 energy", zorder=5)
            for entry in switch_energy:
                ax_energy.axvspan(entry["start_time"], entry["time"], color="purple", alpha=.12)
        preemption_energy = [entry for entry in energy_log if entry["reason"] == "preemption_overhead"]
        if preemption_energy:
            ax_energy.scatter([entry["time"] for entry in preemption_energy],
                              [entry["energy_after"] for entry in preemption_energy],
                              color="darkorange", marker="D", s=70,
                              label="Task-switch overhead: -1 energy", zorder=5)
            for entry in preemption_energy:
                ax_energy.axvspan(entry["start_time"], entry["time"], color="darkorange", alpha=.12)
        capacity = max_energy if max_energy is not None else max(plot_energy)
        ax_energy.axhline(capacity, color="purple", linestyle=":", label=f"Battery capacity ({capacity})")
        ax_energy.set_ylabel("Energy")
        ax_energy.set_title("Energy exactly as printed in the terminal")
        ax_energy.legend()

    for axis in axes:
        axis.set_xlim(0, display_max_time)
        axis.set_xticks(range(display_max_time + 1))
        axis.tick_params(axis="x", labelbottom=True)
        axis.grid(axis="x", alpha=.3)
    axes[-1].set_xlabel("Time (ms; one slot = 0.001 s)")
    plt.tight_layout()
    plt.show()
    print_schedule_summary(schedule_log, tasks, energy_log)


def print_schedule_summary(schedule_log, tasks, energy_log=None):
    by_task = {task["id"]: [] for task in tasks}
    for entry in schedule_log:
        if "event" not in entry:
            by_task[entry["task_id"]].append(entry)
    print("\nSCHEDULE SUMMARY")
    for task in tasks:
        entries = by_task[task["id"]]
        completion = max((entry["end_time"] for entry in entries), default=None)
        on_time = task["done"] and completion is not None and completion <= task["deadline"]
        status = "done on time" if on_time else ("done late" if task["done"] else "not completed")
        print(f"Task {task['id']} [{task.get('criticality', 'LO')}]: "
              f"units executed={len(entries)}, normal budget={task['exec_time']}, completion={completion}, "
              f"deadline={task['deadline']} -> {status}")
    if energy_log:
        total_harvested = sum(entry["harvested"] for entry in energy_log)
        accepted = sum(entry["harvested_accepted"] for entry in energy_log)
        consumed = sum(entry["energy_available"] - entry["energy_after"] for entry in energy_log)
        print(f"Energy: harvested={total_harvested} (stored={accepted}), consumed={consumed}, final={energy_log[-1]['energy_after']}")


def plot_training_results(rewards, eps_history, epsilon_min, title="Training Results"):
    """Plot reward statistics without off-by-one moving-average alignment."""
    if not rewards:
        return
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    rewards_array = np.asarray(rewards, dtype=float)
    window = min(20, len(rewards_array))
    moving_average = np.convolve(rewards_array, np.ones(window) / window, mode="valid")
    x_ma = np.arange(window - 1, len(rewards_array))
    recent = rewards_array[-min(100, len(rewards_array)):]

    axes[0, 0].plot(rewards_array, alpha=.4, label="Episode reward")
    axes[0, 0].plot(x_ma, moving_average, color="crimson", linewidth=2, label=f"{window}-episode mean")
    axes[0, 0].set(title="Training rewards", xlabel="Episode", ylabel="Total reward")
    axes[0, 0].legend()
    axes[0, 1].plot(eps_history, color="tab:blue")
    axes[0, 1].axhline(epsilon_min, color="crimson", linestyle="--", label=f"minimum ({epsilon_min})")
    axes[0, 1].set(title="Epsilon decay", xlabel="Episode", ylabel="Epsilon")
    axes[0, 1].legend()
    axes[1, 0].hist(recent, bins=min(20, len(recent)), color="tab:blue", alpha=.7)
    axes[1, 0].axvline(recent.mean(), color="crimson", linestyle="--", label=f"mean ({recent.mean():.2f})")
    axes[1, 0].set(title="Reward distribution (last 100 episodes)", xlabel="Total reward", ylabel="Frequency")
    axes[1, 0].legend()
    axes[1, 1].plot(np.cumsum(rewards_array) / np.arange(1, len(rewards_array) + 1), color="forestgreen")
    axes[1, 1].set(title="Cumulative mean reward", xlabel="Episode", ylabel="Mean reward")
    for axis in axes.flat:
        axis.grid(alpha=.3)
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()

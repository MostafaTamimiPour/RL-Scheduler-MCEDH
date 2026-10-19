"""Hybrid rule/RL environment for energy-harvesting mixed-criticality scheduling.

There are two modes.  In LO (normal) mode all jobs can run.  When a HI job
uses its normal execution budget, the system changes to HI mode with the given
probability.  HI mode means a HI job needs twice its normal execution time and
each of its future execution units costs twice as much energy.  LO jobs become
optional: they may run only when a simple worst-case energy/time check keeps
all unfinished HI jobs safe.

Time is stored as an integer number of ticks. One tick is exactly 0.001
seconds (1 ms), so ``exec_time=1000`` and ``deadline=1000`` mean one second.

SAFETY DESIGN (hybrid rule + RL, mirrors ED-H's Rules 1-5)
------------------------------------------------------------------
The RL agent never decides everything. At every tick, :meth:`get_valid_actions`
computes the *legal* action set deterministically, before the agent is asked
to choose anything:

  * Rule 3 (energy exhausted)      -> forced idle. RL is not consulted.
  * Rule 4a (a HI task has zero    -> forced execution of that HI task
    slack left; it must run now      (earliest-deadline-first if several
    or it will miss its deadline)     are simultaneously critical). RL is
                                       not consulted.
  * Hard safety filter              -> a LO task is never offered as a
    (LO task in HI mode)              legal choice unless ``_can_run_lo_
                                       in_hi_mode`` proves it cannot cause
                                       a future HI miss. This used to be a
                                       reward *penalty*; it is now a hard
                                       mask, so the agent cannot select it
                                       even by accident/exploration.
  * Rule 4b (battery full)          -> "idle" is removed from the menu so
                                       harvested energy is not wasted, but
                                       *which* safe task to run is still a
                                       real, RL-owned choice if more than
                                       one is available.
  * Rule 5 (everything else)        -> genuine discretion: RL freely
                                       chooses among every safe candidate
                                       task plus "idle". This is the ONLY
                                       zone where the agent's policy
                                       actually matters.

Because Rules 3/4 and the LO-in-HI safety filter are enforced structurally
(the unsafe/forced actions are simply never in the returned action list),
a HI-task deadline miss can only happen if a task set is infeasible to begin
with -- never because of an exploratory or badly-trained RL decision. This
mirrors what you'd want in the real MCED-H setting: RL only reorders/selects
among choices that ED-H's own Rule 5 already declares equally safe.
"""

import random
import math 


class SchedulerEnv:
    # Keep time as integers (ticks), rather than adding floating-point 0.001.
    # This avoids values such as 0.999999999 when a task takes one second.
    TIME_STEP_SECONDS = 0.001
    TIME_STEP_MILLISECONDS = 1
    MODE_SWITCH_ENERGY = 1

    # These are defaults only. main.py sends the values used by an experiment.
    #
    # Reward philosophy: HI tasks carry ZERO positive reward for execution or
    # completion. They are not optional work to be incentivized -- they are
    # a duty, already guaranteed by the hard safety layer in
    # get_valid_actions() (Rule 4a forces them the moment they become
    # mandatory) and backstopped by CATASTROPHIC_HI_MISS_PENALTY. Giving them
    # a reward on top of that would bias the free-choice (Rule 5) policy
    # toward running HI tasks early/voluntarily, which is actively
    # counter-productive here: finishing a HI task's LO-mode budget while
    # still in LO mode is exactly what triggers the LO->HI mode switch (see
    # `finished_hi_in_lo_mode` in step()). Every tick spent on LO work
    # instead, when it's safe to defer HI work, is a tick the system stays
    # in the cheaper LO mode. So all positive reward is concentrated on LO
    # execution/completion -- that's the metric that actually reflects
    # system performance and quality once HI safety is already guaranteed
    # structurally.
    DEFAULT_REWARDS = {
        "IDLE_PENALTY": -100.0,
        "UNAVAILABLE_TASK_PENALTY": -10.0,
        "UNSAFE_LO_IN_HI_PENALTY": -10.0,
        "NOT_ENOUGH_ENERGY_PENALTY": -2.0,
        "PREEMPTION_OVERHEAD_REWARD": -1.0,
        "PREEMPTION_OVERHEAD_FAILURE_PENALTY": -1.0,
        "MODE_SWITCH_OVERHEAD_PENALTY": -5.0,
        "LO_DEADLINE_MISS_PENALTY": -10.0,
        "CATASTROPHIC_HI_MISS_PENALTY": -1000.0,
        "HI_EXECUTION_REWARD": 0.0,
        "LO_EXECUTION_REWARD": 50.0,
        "OPTIONAL_LO_IN_HI_REWARD": 300.0,
        "HI_COMPLETION_BONUS": 0.0,
        "LO_COMPLETION_BONUS": 90.0,
        "ALL_TASKS_FINISHED_BONUS": 100.0,
        "CONTINUE_SAME_TASK_BONUS": 5.0,

    }

    def __init__(self, num_tasks=None, tasks=None, max_energy=None,
                 harvest_rate=None, max_time=None, switch_probability=None, hi_mode_factor=None,
                 reward_settings=None,
                harvest_mode=None,       # ← 'constant' یا 'random' یا 'sinusoidal'
                harvest_mean=None,               # ← میانگین
                harvest_amplitude=None,           # ← دامنه (2 یا 3)
                harvest_period=None,               # ← پریود نوسان (به تیک)
                leakage_rate=None):          



        """Create the scheduler.

        A task is ``(id, exec_time, deadline, remaining, done, energy_cost,
        criticality)``.  Criticality is ``"HI"`` or ``"LO"``.  The old
        six-item form is also accepted and is treated as an LO task.

        Execution times, deadlines, and max_time are integer 1-ms ticks.

        ``remaining`` and ``done`` are reset values, so their input values are
        ignored.  This makes every training episode start from the same jobs.
        """
        if max_energy <= 0:
            raise ValueError("max_energy must be positive")
        if harvest_rate[0] < 0 or harvest_rate[0] > harvest_rate[1]:
            raise ValueError("harvest_rate must be a non-negative (min, max) pair")
        if not 0 <= switch_probability <= 1:
            raise ValueError("switch_probability must be between 0 and 1")

        self.max_energy = max_energy
        self.harvest_rate = harvest_rate
        self.switch_probability = switch_probability
        self.hi_mode_factor = hi_mode_factor
        self.harvest_mode = harvest_mode
        self.harvest_mean = harvest_mean
        self.harvest_amplitude = harvest_amplitude
        self.harvest_period = harvest_period
        self.leakage_rate = leakage_rate

        # For sinusoidal mode, derive the (min, max) bounds so that
        # _can_run_lo_in_hi_mode() can still use harvest_rate[0] as its
        # pessimistic lower bound.
        if harvest_mode == 'sinusoidal':
            self.harvest_rate = (harvest_mean - harvest_amplitude,
                                harvest_mean + harvest_amplitude)
        elif harvest_mode == 'constant':
            self.harvest_rate = (harvest_mean, harvest_mean)
        # else: keep whatever was passed in (random mode)



        self.rewards = self.DEFAULT_REWARDS.copy()
        if reward_settings is not None:
            unknown_names = set(reward_settings) - set(self.rewards)
            if unknown_names:
                raise ValueError(f"unknown reward setting(s): {sorted(unknown_names)}")
            self.rewards.update(reward_settings)
        task_specs = []

        if tasks is None:
            horizon = max_time or 10
            for task_id in range(num_tasks):
                exec_time = random.randint(1, 3)
                deadline = random.randint(exec_time + 1, horizon)
                energy_cost = random.randint(1, min(3, max_energy))
                criticality = "HI" if task_id == 0 else "LO"
                task_specs.append((task_id, exec_time, deadline, energy_cost, criticality))
        else:
            for item in tasks:
                if len(item) == 6:
                    task_id, exec_time, deadline, _remaining, _done, energy_cost = item
                    criticality = "LO"
                elif len(item) == 7:
                    task_id, exec_time, deadline, _remaining, _done, energy_cost, criticality = item
                else:
                    raise ValueError("each task must have 6 or 7 values")
                criticality = str(criticality).upper()
                task_specs.append((task_id, exec_time, deadline, energy_cost, criticality))

        if not task_specs:
            raise ValueError("at least one task is required")
        if len({item[0] for item in task_specs}) != len(task_specs):
            raise ValueError("task IDs must be unique")
        for task_id, exec_time, deadline, energy_cost, criticality in task_specs:
            if exec_time <= 0 or deadline <= 0 or energy_cost <= 0:
                raise ValueError(f"task {task_id} needs positive execution time, deadline, and energy")
            if energy_cost > max_energy:
                raise ValueError(f"task {task_id} needs more energy than the battery holds")
            if criticality not in ("HI", "LO"):
                raise ValueError(f"task {task_id} criticality must be HI or LO")
            if criticality == "HI" and 2 * energy_cost > max_energy:
                raise ValueError(f"HI task {task_id} needs a larger battery for its HI-mode energy cost")

        self.tasks = []
        for task_id, exec_time, deadline, energy_cost, criticality in task_specs:
            self.tasks.append({
                "id": task_id,
                "exec_time": exec_time,
                "base_energy_cost": energy_cost,
                "energy_cost": energy_cost,
                "deadline": deadline,
                "criticality": criticality,
                "remaining": exec_time,
                "executed_units": 0,
                "done": False,
                "dropped": False,
            })

        self.num_tasks = len(self.tasks)
        largest_deadline = max(task["deadline"] for task in self.tasks)
        self.max_time = largest_deadline if max_time is None else max_time
        if self.max_time < largest_deadline:
            raise ValueError("max_time cannot be earlier than a deadline")
        self.actions = [f"run_task_{i}" for i in range(self.num_tasks)] + ["idle"]
        self.idle_action = self.num_tasks
        self.reset()

    def reset(self):
        self.time = 0
        self.energy = 0
        self.mode = "LO"
        self.mode_switch_pending = False
        self.last_task_index = None
        self.preemption_task_pending = None
        self.schedule_log = []
        self.energy_log = []
        self.mode_log = []
        self._tick_harvest = None  # (before, harvested, accepted) for the current tick, or None
        self.last_tick_harvest = None 
        for task in self.tasks:
            task["remaining"] = task["exec_time"]
            task["executed_units"] = 0
            task["energy_cost"] = task["base_energy_cost"]
            task["done"] = False
            task["dropped"] = False
        self.state = self._get_state()
        return self.state

    def _get_state(self):
        # ---- Global features ----
        # Bucket energy coarsely: 11 buckets (0-4, 5-9, ..., 45-50)
        energy_bucket = min(10, int(self.energy // 5))
        
        # 'urgency' = how close is the soonest HI task to forced execution?
        # min over HI tasks of (deadline - time - remaining)
        hi_slacks = [
            t["deadline"] - self.time - t["remaining"]
            for t in self.tasks
            if t["criticality"] == "HI" and not t["done"] and not t["dropped"]
        ]
        hi_urgency = min(hi_slacks) if hi_slacks else 99
        hi_urgency = min(10, max(0, hi_urgency))  # bucket 0..10
        
        state = [
            1 if self.mode == "HI" else 0,
            1 if self.mode_switch_pending else 0,
            energy_bucket,
            hi_urgency,
            -1 if self.last_task_index is None else self.last_task_index,
        ]
        
        # ---- Per-task features (relative, not absolute) ----
        for task in self.tasks:
            if task["done"]:
                state.extend((0, 0, 1))          # 0 remaining, 0 slack, done
            elif task["dropped"]:
                state.extend((0, 0, 2))          # dropped marker
            else:
                slack = task["deadline"] - self.time - task["remaining"]
                slack_bucket = min(10, max(-1, slack))
                state.extend((
                    task["remaining"],           # remaining units (small int)
                    slack_bucket,                # bucketed slack
                    0,                           # not done
                ))
        return tuple(state)




    def get_time_in_seconds(self):
        """Return the current integer tick time converted to seconds."""
        return self.time * self.TIME_STEP_SECONDS

    def _harvest_energy(self):
        if self.harvest_mode == 'sinusoidal':
                # P(t) = mean + A * sin(2*pi*t/T)
                value = self.harvest_mean + self.harvest_amplitude * math.sin(
                    2 * math.pi * self.time / self.harvest_period
                )
                harvested = max(0, int(round(value)))
        elif self.harvest_mode == 'constant':
            harvested = self.harvest_mean
        else:  # 'random'
            harvested = random.randint(self.harvest_rate[0], self.harvest_rate[1])
        
        accepted = min(harvested, self.max_energy - self.energy)
        self.energy += accepted
        return harvested, accepted

    def _harvest_for_this_tick(self):
        """Harvest energy for the current tick exactly once, on first use.

        Both ``get_valid_actions()`` (which the agent consults to decide
        what it's even allowed to choose) and ``step()`` (which executes the
        choice) need to see the *same*, already-harvested energy level for a
        given tick -- harvesting must happen before any rule check or
        execution, not after. Whichever of the two is called first for a new
        tick triggers the harvest; the other reuses the cached result so
        energy is never harvested twice for one tick.
        """
        if self._tick_harvest is None:
            # Static energy leakage: a fixed fraction of stored energy is lost
            # every tick, before any new energy is harvested.
            leaked = self.energy * self.leakage_rate
            self.energy = max(0, self.energy - leaked)
            before = self.energy  # energy after leakage, before harvesting

            harvested, accepted = self._harvest_energy()
            self._tick_harvest = (before, harvested, accepted, leaked)
        return self._tick_harvest

    def _clear_tick_harvest(self):
        # Save the harvest info for render() to display after step() returns.
        if self._tick_harvest is not None:
            self.last_tick_harvest = self._tick_harvest
        self._tick_harvest = None

    def _hi_deadline_missed(self):
        for task in self.tasks:
            if task["criticality"] == "HI" and not task["done"] and self.time >= task["deadline"]:
                return True
        return False

    def _slack_time(self, task):
        """Simplified per-task slack: ticks of breathing room before this
        task's own deadline, given only its own remaining work (not the full
        ED-H processor-demand window ``h(t, d_i)``). A slack of 0 means the
        task must run in this very tick or it will miss its deadline."""
        return task["deadline"] - self.time - task["remaining"]

    def get_valid_actions(self):
        """Return (valid_actions, forced) for the *current* internal state.

        ``valid_actions`` is the list of action indices the RL agent is
        allowed to pick from right now. ``forced`` is True when there is
        exactly one legal action, i.e. no real decision existed (Rules 3/4
        made the call, not RL). This is the enforcement point described in
        the module docstring: everything unsafe or mandatory is filtered out
        here, *before* the agent ever sees an action list.
        """
        # A pending system-owned slot (mode-switch or preemption overhead)
        # ignores whatever the agent picks anyway (see step()). Report a
        # trivial single-choice menu so callers don't special-case it.
        if self.mode_switch_pending or self.preemption_task_pending is not None:
            return [self.idle_action], True

        # Harvest THIS tick's energy before making or reporting any decision.
        # Everything below -- Rule 3's idle check, task affordability, Rule
        # 4b's full-battery check -- must see the post-harvest level, or the
        # agent would be judging idle-vs-busy against energy that's already
        # stale by the time this tick actually happens.
        self._harvest_for_this_tick()

        # ----- Rule 3: no energy at all -> forced idle. -----
        if self.energy <= 0:
            return [self.idle_action], True

        # ----- Build the safe candidate set. -----
        # A task is only ever offered to the agent if:
        #   (a) it still has work to do, and
        #   (b) when it's a LO task in HI mode, running it provably cannot
        #       cause any HI task to miss its deadline later (hard filter,
        #       not a reward penalty).
        candidates = []
        must_run = []  # (deadline, index) for HI tasks with zero effective slack left
        for idx, task in enumerate(self.tasks):
            if task["done"] or task["dropped"]:
                continue
            if self.mode == "HI" and task["criticality"] == "LO":
                if not self._can_run_lo_in_hi_mode(task):
                    continue  # never legal -> agent cannot pick this
            if self.energy < task["energy_cost"]:
                continue  # cannot actually afford it right now -> not offered
            candidates.append(idx)
            if task["criticality"] == "HI":
                slack = self._slack_time(task)
                # Switching onto this task from whatever ran last costs one
                # extra tick of preemption overhead before it can execute.
                # That tick must be "spent" before slack reaches 0, or the
                # forced choice below would burn the task's last tick on the
                # switch itself instead of on actual execution.
                switch_needed = (
                    self.last_task_index is not None
                    and self.last_task_index != idx
                    and not self.tasks[self.last_task_index]["done"]
                    and not self.tasks[self.last_task_index]["dropped"]
                )
                threshold = 1 if switch_needed else 0
                if slack <= threshold:
                    must_run.append((task["deadline"], idx))

        # ----- Rule 4a: a HI task must run now or it misses its deadline. -----
        # If several are simultaneously critical, EDF (earliest deadline
        # first) breaks the tie -- exactly Rule 1's priority order.
        if must_run:
            must_run.sort()
            return [must_run[0][1]], True

        # No task is currently eligible (all HI tasks done/dropped-N/A, all
        # LO tasks finished or unsafe to run in HI mode) -> idle is the only
        # legal choice.
        if not candidates:
            return [self.idle_action], True

        # ----- Rule 4b: battery full -> don't waste harvested energy. -----
        valid = list(candidates)
        if self.energy < self.max_energy:
            valid.append(self.idle_action)

        if len(valid) == 1:
            return valid, True

        # ----- Rule 5: genuine discretion -- this is RL's zone. -----
        return sorted(valid), False

    def _drop_late_lo_tasks(self):
        """LO deadline loss reduces quality, but is not a catastrophic failure."""
        dropped_count = 0
        for task in self.tasks:
            if task["criticality"] == "LO" and not task["done"] and not task["dropped"]:
                if self.time >= task["deadline"]:
                    task["dropped"] = True
                    dropped_count += 1
        return dropped_count

    def _enter_hi_mode(self):
        """Apply the user's HI-mode execution and energy rule once."""
        self.mode = "HI"
        self.mode_log.append({"time": self.time, "event": "Entered HI mode"})
        for task in self.tasks:
            if task["criticality"] == "HI":
                # Work already done in LO mode counts.  A task that appeared
                # complete in LO mode needs its extra HI budget after switch.
                hi_execution_time = int(self.hi_mode_factor * task["exec_time"])
                task["remaining"] = hi_execution_time - task["executed_units"]
                task["energy_cost"] = 1 * task["base_energy_cost"]
                task["done"] = task["remaining"] == 0

    def _run_mode_switch_overhead(self):
        """Record the mandatory 1-ms, 1-energy LO-to-HI switch operation."""
        start_time = self.time
        before, harvested, accepted, leaked = self._harvest_for_this_tick()
        enough_energy = self.energy >= self.MODE_SWITCH_ENERGY

        if enough_energy:
            self.energy -= self.MODE_SWITCH_ENERGY
            reason = "mode_switch_overhead"
        else:
            # The one-energy switch operation cannot finish.  Time is still
            # spent attempting it, and the caller ends the episode safely.
            reason = "mode_switch_not_enough_energy"

        # task_id=None makes this a separate system event, not a task unit.
        self.schedule_log.append({
            "task_id": None,
            "start_time": start_time,
            "end_time": start_time + 1,
            "mode": "HI",
            "criticality": "SYSTEM",
            "event": "mode_switch_overhead",
        })
        self._log_energy(before, harvested, accepted, None, reason, leaked)
        self.time += 1
        self._clear_tick_harvest()
        return enough_energy

    def _run_preemption_overhead(self, next_task_index):
        """Use one slot for switching from the previous task to another task."""
        start_time = self.time
        previous_task = self.tasks[self.last_task_index]
        next_task = self.tasks[next_task_index]
        before, harvested, accepted, leaked  = self._harvest_for_this_tick()
        enough_energy = self.energy >= self.MODE_SWITCH_ENERGY

        if enough_energy:
            self.energy -= self.MODE_SWITCH_ENERGY
            reason = "preemption_overhead"
            # The requested task runs on the next call to step().
            self.preemption_task_pending = next_task_index
            reward = self.rewards["PREEMPTION_OVERHEAD_REWARD"]
        else:
            reason = "preemption_not_enough_energy"
            reward = self.rewards["PREEMPTION_OVERHEAD_FAILURE_PENALTY"]

        self.schedule_log.append({
            "task_id": None,
            "start_time": start_time,
            "end_time": start_time + 1,
            "mode": self.mode,
            "criticality": "SYSTEM",
            "event": "preemption_overhead",
            "from_task_id": previous_task["id"],
            "to_task_id": next_task["id"],
        })
        self._log_energy(before, harvested, accepted, None, reason, leaked )
        self.time += 1
        self._clear_tick_harvest()
        return self._finish_step(reward, not enough_energy)



    def _finish_step(self, reward, switch_overhead_failed=False):
        """Apply common deadline and completion checks after one 1-ms slot."""
        dropped_now = self._drop_late_lo_tasks()
        if dropped_now > 0:
            reward += self.rewards["LO_DEADLINE_MISS_PENALTY"] * dropped_now

        hi_failed = self._hi_deadline_missed()

        # Separate the two ideas:
        #   * all_done    -> every task actually COMPLETED -> full-success bonus
        #   * all_settled -> no task left to run (done or dropped) -> episode
        #                    can end cleanly, but no bonus because quality was lost
        all_done = all(task["done"] for task in self.tasks)
        all_settled = all(task["done"] or task["dropped"] for task in self.tasks)

        if switch_overhead_failed or hi_failed:
            reward, done = self.rewards["CATASTROPHIC_HI_MISS_PENALTY"], True
        elif self.mode_switch_pending:
            # The next step must perform the mandatory switch overhead.
            done = False
        elif all_done:
            reward += self.rewards["ALL_TASKS_FINISHED_BONUS"]
            done = True
        elif all_settled:
            # Every task has finished or been dropped. Episode is over, but
            # no full-success bonus: at least one LO task was lost, which is
            # a real quality loss, not a "system did great" outcome.
            done = True
        elif self.time >= self.max_time:
            reward, done = self.rewards["CATASTROPHIC_HI_MISS_PENALTY"], True
        else:
            done = False

        self.state = self._get_state()
        return self.state, reward, done



    def _run_pending_mode_switch(self):
        """Use this call only for the mandatory 1-ms switch-overhead slot."""
        self.mode_switch_pending = False
        self._enter_hi_mode()
        overhead_failed = not self._run_mode_switch_overhead()
        return self._finish_step(self.rewards["MODE_SWITCH_OVERHEAD_PENALTY"], overhead_failed)




    def _can_run_lo_in_hi_mode(self, lo_task):
        """Return True only if EDF-style HI reserve remains after this LO unit.

        The check uses the minimum possible harvested energy.  It simulates
        running every remaining HI unit in earliest-deadline-first order using
        simple loops.  Therefore LO work cannot be selected when it would take
        time or energy needed by the critical jobs.
        """
        if self.energy < lo_task["energy_cost"]:
            return False

        # ===== سربار Preemption برای سوئیچ به LO =====
        future_time = self.time + 1
        future_energy = self.energy - lo_task["energy_cost"]
        
        # اگه last_task_index متفاوت از LO باشه، یه Preemption لازمه
        if (self.last_task_index is not None 
            and self.last_task_index != lo_task["id"]
            and not self.tasks[self.last_task_index]["done"]):
            future_time += 1   # Preemption قبل از LO
            future_energy -= self.MODE_SWITCH_ENERGY
        
        # ===== سربار Preemption برای سوئیچ به HI (بعد از LO) =====
        future_time += 1   # Preemption بعد از LO
        future_energy -= self.MODE_SWITCH_ENERGY

        # ===== شبیه‌سازی HI تسک‌ها =====
        hi_tasks = []
        for task in self.tasks:
            if task["criticality"] == "HI" and not task["done"]:
                hi_tasks.append(task)
        hi_tasks.sort(key=lambda task: task["deadline"])

        for task in hi_tasks:
            units = task["remaining"]
            for _ in range(units):
                if future_time + 1 > task["deadline"]:
                    return False
                future_energy = min(self.max_energy, future_energy + self.harvest_rate[0])
                if future_energy < task["energy_cost"]:
                    return False
                future_energy -= task["energy_cost"]
                future_time += 1

        return True

    def _log_energy(self, before, harvested, accepted, task_id, reason, leaked):
        self.energy_log.append({
            "start_time": self.time,
            "time": self.time + 1,
            "energy_before": before,
            "harvested": harvested,
            "harvested_accepted": accepted,
            "leaked": leaked,
            "energy_available": before + accepted,
            "energy_after": self.energy,
            "task_executed": task_id,
            "reason": reason,
            "mode": self.mode,
        })

    def step(self, action):
        """Run one 1-ms time slot. A HI deadline miss ends the episode."""
        # A triggered mode change owns this complete time slot.  The caller's
        # action is intentionally ignored: no task may run during overhead.
        if self.mode_switch_pending:
            return self._run_pending_mode_switch()

        # After the preemption slot, immediately execute the task that caused
        # the switch. The caller's action is ignored for this one slot.
        executing_pending_task = self.preemption_task_pending is not None
        if executing_pending_task:
            task_index = self.preemption_task_pending
            self.preemption_task_pending = None
        elif isinstance(action, str):
            task_index = self.idle_action if action == "idle" else int(action.split("_")[-1])
        else:
            task_index = action
        if not isinstance(task_index, int) or task_index < 0 or task_index > self.idle_action:
            raise ValueError(f"invalid action: {action}")

        # Harvest THIS tick's energy first, before any rule check or
        # execution happens -- see _harvest_for_this_tick(). If the agent
        # already called get_valid_actions() for this tick (the normal RL
        # flow), that call already harvested and this just reuses the result.
        before, harvested, accepted, leaked = self._harvest_for_this_tick()

        if self._hi_deadline_missed() or self.time >= self.max_time:
            return self.state, self.rewards["CATASTROPHIC_HI_MISS_PENALTY"], True

        # Defensive net: callers are expected to mask with get_valid_actions()
        # before ever choosing an action (see module docstring / agent.py).
        # If something outside that legal set slips through anyway, refuse it
        # here rather than silently executing something unsafe. This should
        # never fire during normal training/testing once masking is used.
        if not executing_pending_task:
            legal_actions, _forced = self.get_valid_actions()
            if task_index not in legal_actions:
                reward, reason = self.rewards["UNSAFE_LO_IN_HI_PENALTY"], "action_not_legal"
                self._log_energy(before, harvested, accepted, None, reason)
                self.time += 1
                self._clear_tick_harvest()
                return self._finish_step(reward)

        # Do not charge before the first task. Keep last_task_index through
        # idle periods, so A -> idle -> A has no overhead, while A -> idle -> B
        # does have the required overhead.
        if not executing_pending_task and task_index != self.idle_action:
            task = self.tasks[task_index]
            changing_task = self.last_task_index is not None and task_index != self.last_task_index
            if changing_task and not task["done"] and not task["dropped"]:
                return self._run_preemption_overhead(task_index)

        reward = self.rewards["IDLE_PENALTY"]
        reason = "idle"
        executed_task_id = None
        finished_hi_in_lo_mode = False

        if task_index != self.idle_action:
            task = self.tasks[task_index]
            # NOTE: task_index is guaranteed to be in get_valid_actions() at
            # this point (checked above), so "done"/"dropped"/unsafe-LO-in-HI
            # cases are structurally impossible here -- they are masked out
            # before the agent ever chooses, not punished after the fact.
            if self.mode == "HI" and task["criticality"] == "LO":
                self.energy -= task["energy_cost"]
                task["remaining"] -= 1
                task["executed_units"] += 1
                executed_task_id = task["id"]
                reason = "optional_lo_executed_in_hi"
                reward = self.rewards["OPTIONAL_LO_IN_HI_REWARD"]
            elif self.energy < task["energy_cost"]:
                reward, reason = self.rewards["NOT_ENOUGH_ENERGY_PENALTY"], "not_enough_energy"
            else:
                self.energy -= task["energy_cost"]
                task["remaining"] -= 1
                task["executed_units"] += 1
                executed_task_id = task["id"]
                reason = "executed"
                if task["criticality"] == "HI":
                    reward = self.rewards["HI_EXECUTION_REWARD"]
                else:
                    reward = self.rewards["LO_EXECUTION_REWARD"]


            if executed_task_id is not None:
                if task_index == self.last_task_index:
                    reward += self.rewards["CONTINUE_SAME_TASK_BONUS"]
                self.last_task_index = task_index
                self.schedule_log.append({
                    "task_id": task["id"], "start_time": self.time,
                    "end_time": self.time + 1, "mode": self.mode,
                    "criticality": task["criticality"],
                })
            if task["remaining"] == 0:
                task["done"] = True
                if task["criticality"] == "HI":
                    reward += self.rewards["HI_COMPLETION_BONUS"]
                else:
                    reward += self.rewards["LO_COMPLETION_BONUS"]
                if self.mode == "LO" and task["criticality"] == "HI":
                    finished_hi_in_lo_mode = True

        self._log_energy(before, harvested, accepted, executed_task_id, reason, leaked)
        # Advance exactly one tick = 0.001 seconds = 1 ms.
        self.time += 1
        self._clear_tick_harvest()

        # An LO miss only drops quality. A triggered switch consumes the next
        # call to step(), so this call still advances exactly 1 ms.
        if finished_hi_in_lo_mode and random.random() < self.switch_probability:
            self.mode_switch_pending = True
            reward += self.rewards["MODE_SWITCH_OVERHEAD_PENALTY"]
        return self._finish_step(reward)

    def render(self):
        mode_text = self.mode
        if self.mode_switch_pending:
            mode_text = "LO (LO->HI overhead next)"
        elif self.preemption_task_pending is not None:
            task_id = self.tasks[self.preemption_task_pending]["id"]
            mode_text = f"{self.mode} (task switch complete; T{task_id} runs next)"
        
        # Show harvested energy for this tick (if available).
        harvest_text = ""
        if self.last_tick_harvest is not None:
            before, harvested, accepted, leaked = self.last_tick_harvest
            harvest_text = f" | Harvested: {harvested} (stored: {accepted})"
            if leaked > 0:
                harvest_text += f" | Leaked: {leaked:.2f}"
        
        print(f"\nTime: {self.time} ms ({self.get_time_in_seconds():.3f} s) | "
            f"Mode: {mode_text} | Energy: {self.energy}/{self.max_energy}{harvest_text}")
        for task in self.tasks:
            if task["done"]:
                status = "done"
            elif task["dropped"]:
                status = "dropped (quality loss)"
            else:
                status = f"{task['remaining']} units left"
            print(f"Task {task['id']} [{task['criticality']}]: {status}; "
                f"deadline={task['deadline']} ms, energy/unit={task['energy_cost']}")
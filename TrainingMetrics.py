"""Write training progress and tabular exploration metrics."""

import csv
from pathlib import Path

from statistics import mean


class TrainingMetrics:
    def __init__(
        self,
        path,
        finite_state_count=None,
        finite_state_action_count=None,
        track_decisions=True,
    ):
        self.path = Path(path)
        self.finite_state_count = finite_state_count
        self.finite_state_action_count = finite_state_action_count
        self.track_decisions = track_decisions
        self.decisions = 0
        self.exploratory_actions = 0
        self.states = set()
        self.state_actions = set()
        self.rows = []

    def record_decision(self, decision_type, state, action, exploratory):
        if not self.track_decisions:
            return
        self.decisions += 1
        self.exploratory_actions += int(exploratory)
        state_key = tuple(state)
        self.states.add(state_key)
        self.state_actions.add((decision_type, state_key, action))

    def snapshot(self, episode, recent_scores, epsilon):
        if self.rows and self.rows[-1]["episode"] == episode:
            return

        row = {
            "episode": episode,
            "interval_average_score": mean(recent_scores),
            "interval_win_rate": sum(score > 0 for score in recent_scores) / len(recent_scores),
            "epsilon": epsilon,
        }
        if self.track_decisions:
            row.update({
                "cumulative_decisions": self.decisions,
                "cumulative_exploratory_actions": self.exploratory_actions,
                "exploratory_action_percentage": (
                    100.0 * self.exploratory_actions / self.decisions
                ),
                "unique_observed_states": len(self.states),
                "state_coverage_percentage": (
                    100.0 * len(self.states) / self.finite_state_count
                ),
                "unique_observed_state_actions": len(self.state_actions),
                "state_action_coverage_percentage": (
                    100.0 * len(self.state_actions) / self.finite_state_action_count
                ),
            })
        self.rows.append(row)
        self.write()

    def write(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)

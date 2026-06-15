# Scoundrel Reinforcement Learning

Source code for a bachelor thesis comparing human-designed heuristics, tabular
Q-learning, and Deep Q-learning on the single-player card game Scoundrel. The
DQN can also be trained with an optional Joker rule.

## Main files

- `Engine.py`: game rules and state transitions.
- `Agents/`: random and human-designed heuristic agents.
- `Simulate.py`: evaluates the heuristic agents.
- `ql_agent.py`: trains and evaluates the tabular Q-learning agent.
- `dqn_agent.py`: trains and evaluates the DQN.
- `TrainingMetrics.py`: writes periodic training statistics.
- `*QualitativeAnalysis.py`: analyzes recurring policy behavior.

## Requirements

Python 3 with NumPy and PyTorch installed.

"""Evaluate one saved DQN checkpoint."""

import argparse

from dqn_agent import evaluate, load_agent


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a saved DQN checkpoint.")
    parser.add_argument("checkpoint", help="Path to the .pt checkpoint file.")
    parser.add_argument("--episodes", type=int, default=10_000)
    parser.add_argument("--seed-start", type=int, default=3_000_000)
    parser.add_argument("--include-joker", action="store_true")
    args = parser.parse_args()

    agent = load_agent(args.checkpoint)
    evaluate(
        agent,
        n_episodes=args.episodes,
        seed_start=args.seed_start,
        include_joker=args.include_joker,
    )

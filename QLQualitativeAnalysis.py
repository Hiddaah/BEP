"""Summarize a saved tabular Q-learning policy."""

import argparse
from pathlib import Path

from Engine import (
    Player,
    avoid_room,
    create_dungeon,
    draw_room,
    lose_score,
    normal_card_count,
    resolve_card,
    win_score,
)
from QualitativePolicyUtils import (
    AVOID_ACTION_NAMES,
    CARD_ACTION_NAMES,
    episode_result,
    policy_event,
    write_analysis,
)
from ql_agent import (
    get_state,
    load_agent,
    resolve_pick_action,
    valid_pick_actions,
)


def trace_game(agent, seed):
    deck = create_dungeon(seed=seed)
    player = Player()
    carry_over = None
    can_avoid = True
    last_taken_card = None
    events = []

    while True:
        if not deck and carry_over is None:
            return episode_result(
                seed, win_score(player, last_taken_card), True, deck, [], events
            )

        room = draw_room(deck, carry_over)
        player.reset_turn()

        if len(room) == 4 and can_avoid:
            state = get_state(player, room, normal_card_count(deck))
            q_values = agent.Q_avoid[state]
            action = max([0, 1], key=lambda candidate: q_values[candidate])
            events.append(policy_event(
                "avoid", player, room, action, AVOID_ACTION_NAMES,
            ))
            if action == 1:
                avoid_room(deck, room)
                carry_over = None
                can_avoid = False
                continue

        can_avoid = True
        cards_to_take = 3 if len(room) == 4 else len(room)
        for _ in range(cards_to_take):
            state = get_state(player, room, normal_card_count(deck))
            valid = valid_pick_actions(player, room)
            q_values = agent.Q_card[state]
            action = max(valid, key=lambda candidate: q_values[candidate])
            events.append(policy_event(
                "card", player, room, action, CARD_ACTION_NAMES,
            ))
            idx, attack_action = resolve_pick_action(player, room, action)
            card = room.pop(idx)
            resolve_card(player, card, attack_action)
            last_taken_card = card

            if player.health <= 0:
                return episode_result(
                    seed, lose_score(player, deck, room), False, deck, room, events
                )
            if not deck and not room:
                return episode_result(
                    seed, win_score(player, last_taken_card), True, deck, room, events
                )

        carry_over = room[0] if room else None


def main():
    parser = argparse.ArgumentParser(
        description="Summarize a saved Q-learning policy."
    )
    parser.add_argument("--checkpoint", default="Checkpoints/ql_latest.pkl")
    parser.add_argument("--seed-start", type=int, default=3_000_000)
    parser.add_argument("--n-seeds", type=int, default=1_000)
    parser.add_argument("--output", default="ql_qualitative_results")
    args = parser.parse_args()

    agent = load_agent(args.checkpoint)
    traces = [
        trace_game(agent, seed)
        for seed in range(args.seed_start, args.seed_start + args.n_seeds)
    ]
    write_analysis(traces, Path(args.output))
    print(f"Wrote Q-learning qualitative analysis to {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()

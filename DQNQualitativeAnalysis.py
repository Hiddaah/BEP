"""Summarize an original-rule or Joker-rule DQN policy."""

import argparse
import random
from pathlib import Path

import dqn_agent as dqn
from Engine import (
    Player,
    avoid_room,
    create_dungeon,
    draw_room,
    lose_score,
    normal_card_count,
    remove_jokers_and_refill,
    resolve_card,
    shuffle_dungeon,
    win_score,
)
from QualitativePolicyUtils import (
    AVOID_ACTION_NAMES,
    CARD_ACTION_NAMES,
    episode_result,
    policy_event,
    write_analysis,
)
from ql_agent import resolve_pick_action, valid_pick_actions


JOKER_ACTION_NAMES = {0: "discard Joker", 1: "shuffle dungeon"}


def q_values(head, state):
    with dqn.torch.no_grad():
        tensor = dqn.torch.tensor(
            state, dtype=dqn.torch.float32, device=dqn.DEVICE
        ).unsqueeze(0)
        return head.online(tensor).squeeze(0).cpu().numpy()


def trace_game(agent, seed, include_joker):
    deck = create_dungeon(seed=seed, include_joker=include_joker)
    joker_rng = random.Random(seed + 10_000_000)
    player = Player()
    carry_over = None
    can_avoid = True
    last_taken_card = None
    resolved_cards = set()
    known_bottom_cards = []
    events = []
    joker_found = 0
    joker_used = 0

    while True:
        if not deck and carry_over is None:
            return episode_result(
                seed, win_score(player, last_taken_card), True, deck, [], events,
                joker_found=joker_found, joker_used=joker_used,
            )

        room = draw_room(deck, carry_over)
        player.reset_turn()
        room, encountered_jokers = remove_jokers_and_refill(deck, room)
        known_bottom_cards = dqn._known_cards_still_in_deck(known_bottom_cards, deck)

        if encountered_jokers:
            joker_found += encountered_jokers
            state = dqn.encode_state(
                player, room, normal_card_count(deck), resolved_cards, known_bottom_cards
            )
            values = q_values(agent.head_joker, state)
            action = max([0, 1], key=lambda candidate: values[candidate])
            events.append(policy_event(
                "joker", player, room, action, JOKER_ACTION_NAMES, known_bottom_cards,
            ))
            if action == 1:
                joker_used += 1
                shuffle_dungeon(deck, joker_rng)
                known_bottom_cards = []

        if len(room) == 4 and can_avoid:
            state = dqn.encode_state(
                player, room, normal_card_count(deck), resolved_cards, known_bottom_cards
            )
            values = q_values(agent.head_avoid, state)
            action = max([0, 1], key=lambda candidate: values[candidate])
            events.append(policy_event(
                "avoid", player, room, action, AVOID_ACTION_NAMES, known_bottom_cards,
            ))
            if action == 1:
                avoid_room(deck, room)
                known_bottom_cards = room + known_bottom_cards
                carry_over = None
                can_avoid = False
                continue

        can_avoid = True
        cards_to_take = 3 if len(room) == 4 else len(room)
        for _ in range(cards_to_take):
            state = dqn.encode_state(
                player, room, normal_card_count(deck), resolved_cards, known_bottom_cards
            )
            valid = valid_pick_actions(player, room)
            values = q_values(agent.head_card, state)
            action = max(valid, key=lambda candidate: values[candidate])
            events.append(policy_event(
                "card", player, room, action, CARD_ACTION_NAMES, known_bottom_cards,
            ))
            idx, attack_action = resolve_pick_action(player, room, action)
            card = room.pop(idx)
            resolved_cards.add(card)
            resolve_card(player, card, attack_action)
            last_taken_card = card

            if player.health <= 0:
                return episode_result(
                    seed, lose_score(player, deck, room), False, deck, room, events,
                    joker_found=joker_found, joker_used=joker_used,
                )
            if not deck and not room:
                return episode_result(
                    seed, win_score(player, last_taken_card), True, deck, room, events,
                    joker_found=joker_found, joker_used=joker_used,
                )

        carry_over = room[0] if room else None


def main():
    parser = argparse.ArgumentParser(
        description="Summarize a saved original-rule or Joker-rule DQN."
    )
    parser.add_argument("--include-joker", action="store_true")
    parser.add_argument("--checkpoint")
    parser.add_argument("--seed-start", type=int, default=3_000_000)
    parser.add_argument("--n-seeds", type=int, default=1_000)
    parser.add_argument("--output")
    args = parser.parse_args()

    checkpoint = args.checkpoint or (
        "Checkpoints/dqn_joker_best.pt"
        if args.include_joker
        else "Checkpoints/dqn_original_best.pt"
    )
    output = args.output or (
        "dqn_joker_qualitative_results"
        if args.include_joker
        else "dqn_original_qualitative_results"
    )
    agent = dqn.load_agent(checkpoint)
    agent.set_eval()
    traces = [
        trace_game(agent, seed, args.include_joker)
        for seed in range(args.seed_start, args.seed_start + args.n_seeds)
    ]
    write_analysis(traces, Path(output))
    print(f"Wrote DQN qualitative analysis to {Path(output).resolve()}")


if __name__ == "__main__":
    main()

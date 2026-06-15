"""Shared output helpers for Q-learning and DQN policy analysis."""

import csv
from collections import Counter
from statistics import mean

from Engine import is_monster, normal_card_count
from ql_agent import (
    PICK_MONSTER_ACE_BARE,
    PICK_MONSTER_ACE_WEAPON,
    PICK_MONSTER_FACE_BARE,
    PICK_MONSTER_FACE_WEAPON,
    PICK_MONSTER_LOW_BARE,
    PICK_MONSTER_LOW_WEAPON,
    PICK_MONSTER_MED_BARE,
    PICK_MONSTER_MED_WEAPON,
    PICK_POTION,
    PICK_WEAPON,
)


CARD_ACTION_NAMES = {
    PICK_WEAPON: "take weapon",
    PICK_POTION: "take potion",
    PICK_MONSTER_LOW_BARE: "fight low monster barehanded",
    PICK_MONSTER_LOW_WEAPON: "fight low monster with weapon",
    PICK_MONSTER_MED_BARE: "fight medium monster barehanded",
    PICK_MONSTER_MED_WEAPON: "fight medium monster with weapon",
    PICK_MONSTER_FACE_BARE: "fight face monster barehanded",
    PICK_MONSTER_FACE_WEAPON: "fight face monster with weapon",
    PICK_MONSTER_ACE_BARE: "fight ace barehanded",
    PICK_MONSTER_ACE_WEAPON: "fight ace with weapon",
}
AVOID_ACTION_NAMES = {0: "enter room", 1: "flee room"}


def expected_damage(player, room):
    damages = []
    for card in room:
        if not is_monster(card):
            damages.append(0)
        elif player.weapon_use(card):
            damages.append(max(0, card[1] - player.weapon[1]))
        else:
            damages.append(card[1])
    return sum(sorted(damages)[:3 if len(room) == 4 else len(room)])


def policy_event(
    decision_type,
    player,
    room,
    selected_action,
    action_names,
    known_bottom_cards=None,
):
    damage = expected_damage(player, room)
    return {
        "decision_type": decision_type,
        "selected_action": action_names[selected_action],
        "context": {
            "health_band": (
                "low" if player.health <= 7
                else "medium" if player.health <= 13
                else "high"
            ),
            "has_weapon": player.weapon is not None,
            "room_monsters": sum(is_monster(card) for card in room),
            "expected_damage_band": (
                "none" if damage == 0
                else "low" if damage <= 5
                else "medium" if damage <= 10
                else "high"
            ),
            "known_bottom_available": bool(known_bottom_cards),
        },
    }


def episode_result(seed, score, won, deck, room, events, **extra):
    return {
        "seed": seed,
        "score": score,
        "won": won,
        "cards_cleared": 44 - normal_card_count(deck) - normal_card_count(room or []),
        "events": events,
        **extra,
    }


def write_analysis(traces, output_dir):
    if not traces:
        raise ValueError("At least one seed is required.")
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_fields = [key for key in traces[0] if key != "events"]
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(
            {key: trace[key] for key in summary_fields}
            for trace in traces
        )

    events = [event for trace in traces for event in trace["events"]]
    context_fields = [
        "decision_type",
        "health_band",
        "has_weapon",
        "room_monsters",
        "expected_damage_band",
        "known_bottom_available",
    ]
    policy_counts = Counter()
    context_totals = Counter()
    action_counts = Counter()
    decision_totals = Counter()

    for event in events:
        context = event["context"]
        context_key = (
            event["decision_type"],
            *(context[field] for field in context_fields[1:]),
        )
        action = event["selected_action"]
        context_totals[context_key] += 1
        policy_counts[(*context_key, action)] += 1
        action_counts[(event["decision_type"], action)] += 1
        decision_totals[event["decision_type"]] += 1

    policy_fields = [
        *context_fields,
        "selected_action",
        "count",
        "percentage_within_context",
    ]
    with (output_dir / "policy_summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=policy_fields)
        writer.writeheader()
        for key, count in sorted(policy_counts.items()):
            writer.writerow(dict(zip(policy_fields, [
                *key,
                count,
                100.0 * count / context_totals[key[:-1]],
            ])))

    action_fields = [
        "decision_type",
        "selected_action",
        "count",
        "percentage_within_decision_type",
    ]
    with (output_dir / "action_summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=action_fields)
        writer.writeheader()
        for (decision_type, action), count in sorted(action_counts.items()):
            writer.writerow({
                "decision_type": decision_type,
                "selected_action": action,
                "count": count,
                "percentage_within_decision_type": (
                    100.0 * count / decision_totals[decision_type]
                ),
            })

    scores = [trace["score"] for trace in traces]
    cards_cleared = [trace["cards_cleared"] for trace in traces]
    with (output_dir / "aggregate.txt").open("w", encoding="utf-8") as file:
        file.write(f"Games: {len(traces)}\n")
        file.write(f"Average score: {mean(scores):+.2f}\n")
        file.write(f"Win rate: {sum(score > 0 for score in scores) / len(scores):.2%}\n")
        file.write(f"Average cards cleared: {mean(cards_cleared):.2f}\n")
        file.write(f"Recorded policy decisions: {len(events)}\n")

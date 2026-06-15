"""Compare recurring heuristic behaviors on matched seeds."""

import argparse
import csv
from collections import Counter
from pathlib import Path

from Engine import (
    Player,
    create_dungeon,
    draw_room,
    avoid_room,
    is_monster,
    is_potion,
    is_weapon,
    lose_score,
    resolve_card,
    win_score,
)
from Agents.Greedy import GreedyAgent
from Agents.Random import RandomAgent
from Agents.Survival import SurvivalAgent
from Agents.Weapon import WeaponAgent


AGENT_FACTORIES = {
    "random": lambda seed: RandomAgent(seed=seed),
    "greedy": lambda seed: GreedyAgent(),
    "survival": lambda seed: SurvivalAgent(),
    "weapon": lambda seed: WeaponAgent(),
}


def expected_room_damage(agent, player, room):
    if hasattr(agent, "room_expected_damage"):
        return agent.room_expected_damage(player, room)

    damages = []
    for card in room:
        if is_monster(card):
            if hasattr(agent, "immediate_damage"):
                damages.append(agent.immediate_damage(player, card))
            elif player.weapon_use(card):
                damages.append(max(0, card[1] - player.weapon[1]))
            else:
                damages.append(card[1])
        else:
            damages.append(0)
    damages.sort()
    return sum(damages[:min(3, len(damages))])


def classify_avoid(player, can_avoid, avoided, damage):
    flags = []
    if not can_avoid:
        return flags
    if avoided and damage <= 3:
        flags.append("possibly_wasteful_avoid")
    if not avoided and damage >= player.health:
        flags.append("dangerous_room_entered")
    return flags


def classify_pick(player, room, chosen_index, action):
    card = room[chosen_index]
    flags = []

    if is_potion(card):
        if player.potion_used_this_turn:
            flags.append("second_potion_wasted")
        elif player.health == 20:
            flags.append("potion_at_full_health")

    if is_weapon(card) and player.weapon is not None and card[1] < player.weapon[1]:
        flags.append("weaker_weapon_equipped")

    if is_monster(card):
        selected_damage = (
            max(0, card[1] - player.weapon[1])
            if action == "weapon" and player.weapon_use(card)
            else card[1]
        )
        other_damage = [
            0 if not is_monster(other) else (
                max(0, other[1] - player.weapon[1])
                if player.weapon_use(other)
                else other[1]
            )
            for i, other in enumerate(room)
            if i != chosen_index
        ]
        if selected_damage >= player.health and any(damage < player.health for damage in other_damage):
            flags.append("lethal_card_chosen_with_safer_option")
        if action == "weapon" and card[1] <= 5:
            flags.append("weapon_used_on_weak_monster")
        if (
            action == "bare"
            and player.weapon_use(card)
            and min(card[1], player.weapon[1]) >= 4
        ):
            flags.append("large_weapon_saving_declined")

    return flags


def trace_game(agent, seed):
    deck = create_dungeon(seed=seed)
    player = Player()
    carry_over = None
    can_avoid = True
    last_taken_card = None
    failure_counts = Counter()

    while True:
        if not deck and carry_over is None:
            score = win_score(player, last_taken_card)
            break

        room = draw_room(deck, carry_over)
        player.reset_turn()
        if len(room) == 4:
            damage = expected_room_damage(agent, player, room)
            avoided = agent.choose_avoid(player, room, can_avoid)
            flags = classify_avoid(player, can_avoid, avoided, damage)
            failure_counts.update(flags)

            if avoided:
                avoid_room(deck, room)
                carry_over = None
                can_avoid = False
                continue

        can_avoid = True
        cards_to_take = 3 if len(room) == 4 else len(room)

        for pick_number in range(cards_to_take):
            chosen_index = agent.choose_card(player, room, pick_number)
            chosen_card = room[chosen_index]
            action = agent.choose_action(player, chosen_card) if is_monster(chosen_card) else None
            flags = classify_pick(player, room, chosen_index, action)

            card = room.pop(chosen_index)
            resolve_card(player, card, action)
            last_taken_card = card
            failure_counts.update(flags)

            if player.health <= 0:
                score = lose_score(player, deck, room)
                return {
                    "agent": type(agent).__name__,
                    "seed": seed,
                    "score": score,
                    "won": False,
                    "cards_cleared": 44 - len(deck) - len(room),
                    "failures": dict(failure_counts),
                }

        carry_over = room[0] if room else None
        if carry_over is not None and is_monster(carry_over) and carry_over[1] >= 10:
            failure_counts["dangerous_monster_left_as_carry_over"] += 1

    return {
        "agent": type(agent).__name__,
        "seed": seed,
        "score": score,
        "won": True,
        "cards_cleared": 44,
        "failures": dict(failure_counts),
    }


def matched_seed_analysis(seed_start, n_seeds):
    traces = []
    for seed in range(seed_start, seed_start + n_seeds):
        for factory in AGENT_FACTORIES.values():
            traces.append(trace_game(factory(seed), seed))
    return traces


def write_outputs(traces, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    failure_names = sorted({
        failure
        for trace in traces
        for failure in trace["failures"]
    })
    fieldnames = ["agent", "seed", "score", "won", "cards_cleared", *failure_names]
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for trace in traces:
            row = {key: trace[key] for key in fieldnames[:5]}
            row.update({name: trace["failures"].get(name, 0) for name in failure_names})
            writer.writerow(row)

    by_seed = {}
    for trace in traces:
        by_seed.setdefault(trace["seed"], []).append(trace)
    with (output_dir / "interesting_seeds.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "seed",
                "score_spread",
                "best_agent",
                "best_score",
                "worst_agent",
                "worst_score",
            ],
        )
        writer.writeheader()
        ranked = []
        for seed, seed_traces in by_seed.items():
            best = max(seed_traces, key=lambda trace: trace["score"])
            worst = min(seed_traces, key=lambda trace: trace["score"])
            ranked.append({
                "seed": seed,
                "score_spread": best["score"] - worst["score"],
                "best_agent": best["agent"],
                "best_score": best["score"],
                "worst_agent": worst["agent"],
                "worst_score": worst["score"],
            })
        writer.writerows(sorted(ranked, key=lambda row: row["score_spread"], reverse=True))

    aggregate = {}
    for trace in traces:
        agent = trace["agent"]
        if agent not in aggregate:
            aggregate[agent] = {
                "games": 0,
                "total_score": 0,
                "total_cards_cleared": 0,
                "failures": Counter(),
            }
        aggregate[agent]["games"] += 1
        aggregate[agent]["total_score"] += trace["score"]
        aggregate[agent]["total_cards_cleared"] += trace["cards_cleared"]
        aggregate[agent]["failures"].update(trace["failures"])

    with (output_dir / "aggregate.txt").open("w", encoding="utf-8") as file:
        for agent, values in aggregate.items():
            games = values["games"]
            file.write(f"{agent}\n")
            file.write(f"  Average score: {values['total_score'] / games:+.2f}\n")
            file.write(
                f"  Average cards cleared: "
                f"{values['total_cards_cleared'] / games:.2f}\n"
            )
            file.write("  Recurring flagged decisions:\n")
            for name, count in values["failures"].most_common():
                file.write(f"    {name}: {count} ({count / games:.2f} per game)\n")
            file.write("\n")


def main():
    parser = argparse.ArgumentParser(
        description="Trace heuristic decisions on matched Scoundrel seeds."
    )
    parser.add_argument("--seed-start", type=int, default=136)
    parser.add_argument("--n-seeds", type=int, default=20)
    parser.add_argument("--output", default="qualitative_results")
    args = parser.parse_args()

    traces = matched_seed_analysis(args.seed_start, args.n_seeds)
    output_dir = Path(args.output)
    write_outputs(traces, output_dir)
    print(f"Wrote {len(traces)} matched-seed traces to {output_dir.resolve()}")
    print(f"Open {output_dir / 'aggregate.txt'} for the recurring-failure summary.")
    print(f"Open {output_dir / 'interesting_seeds.csv'} to select case-study seeds.")


if __name__ == "__main__":
    main()

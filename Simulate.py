import numpy as np


from Engine import play_scoundrel
from Agents.Random import RandomAgent
from Agents.Greedy import GreedyAgent
from Agents.Survival import SurvivalAgent
from Agents.Weapon import WeaponAgent

# Simulation and evaluation

def _summarize(results):
    scores = [result["score"] for result in results]
    score_array = np.asarray(scores, dtype=np.float64)
    cards_cleared = [result["cards_cleared"] for result in results]
    remaining_monsters = [result["remaining_monster_value"] for result in results]
    avoid_opportunities = [result["avoid_opportunities"] for result in results]
    avoided_rooms = [result["avoided_rooms"] for result in results]

    total_avoid_opportunities = sum(avoid_opportunities)
    total_avoided_rooms = sum(avoided_rooms)
    avoid_rate = (
        total_avoided_rooms / total_avoid_opportunities
        if total_avoid_opportunities
        else 0.0
    )
    wins = sum(result["won"] for result in results)
    win_rate = wins / len(scores)
    score_std = float(np.std(score_array, ddof=1)) if len(scores) > 1 else 0.0
    confidence_margin = 1.96 * score_std / np.sqrt(len(scores))
    reached_36 = sum(cleared >= 36 for cleared in cards_cleared)
    reached_40 = sum(cleared >= 40 for cleared in cards_cleared)
    reached_44 = sum(cleared >= 44 for cleared in cards_cleared)

    return {
        "n_games": len(scores),
        "wins": wins,
        "win_rate": win_rate,
        "avg_score": float(np.mean(score_array)),
        "score_ci_95": confidence_margin,
        "median_score": float(np.median(score_array)),
        "std_score": score_std,
        "score_p10": float(np.percentile(score_array, 10)),
        "score_p90": float(np.percentile(score_array, 90)),
        "min_score": min(scores),
        "max_score": max(scores),
        "best_score": max(scores),
        "worst_score": min(scores),
        "avg_cleared": float(np.mean(cards_cleared)),
        "max_cleared": max(cards_cleared),
        "reached_36": reached_36,
        "reached_40": reached_40,
        "reached_44": reached_44,
        "avg_remaining_monster_value": float(np.mean(remaining_monsters)),
        "avg_avoid_opportunities": float(np.mean(avoid_opportunities)),
        "avg_avoided_rooms": float(np.mean(avoided_rooms)),
        "avoid_rate": avoid_rate,
    }


def simulate(agent, n_games=1000, seed=0):
    results = [
        play_scoundrel(
            agent,
            seed=seed + n,
            verbose=False,
            return_details=True,
        )
        for n in range(n_games)
    ]
    return _summarize(results)


def evaluate_agent(agent, n_episodes=1000, seed_start=0):
    stats = simulate(
        agent,
        n_games=n_episodes,
        seed=seed_start,
    )

    print(f"Evaluation over {n_episodes} episodes:")
    print(f"  Wins      : {stats['wins']}/{n_episodes} ({stats['win_rate']:.1%})")
    print(f"  Avg score : {stats['avg_score']:+.2f}")
    print(f"  Score 95% CI: +/- {stats['score_ci_95']:.2f}")
    print(f"  Score std : {stats['std_score']:.2f}")
    print(f"  Median score: {stats['median_score']:+.2f}")
    print(f"  Score P10 / P90: {stats['score_p10']:+.2f} / {stats['score_p90']:+.2f}")
    print(f"  Min score : {stats['min_score']:+.2f}")
    print(f"  Max score : {stats['max_score']:+.2f}")
    print(f"  Avg cleared: {stats['avg_cleared']:.2f} cards")
    print(f"  Max cleared: {stats['max_cleared']} cards")
    print(
        f"  Reached 36 cards: {stats['reached_36']}/{n_episodes} "
        f"({stats['reached_36']/n_episodes:.1%})"
    )
    print(
        f"  Reached 40 cards: {stats['reached_40']}/{n_episodes} "
        f"({stats['reached_40']/n_episodes:.1%})"
    )
    print(
        f"  Reached 44 cards: {stats['reached_44']}/{n_episodes} "
        f"({stats['reached_44']/n_episodes:.1%})"
    )
    print(f"  Avg remaining monster value: {stats['avg_remaining_monster_value']:.2f}")
    print(f"  Avg avoid opportunities: {stats['avg_avoid_opportunities']:.2f}")
    print(f"  Avg avoided rooms: {stats['avg_avoided_rooms']:.2f}")
    print(f"  Avoid rate: {stats['avoid_rate']:.1%}")
    return stats


if __name__ == "__main__":
    SEED = 136
    N_EPISODES = 100_000

    agents = [
        ("Random", RandomAgent(seed=SEED)),
        ("Greedy", GreedyAgent()),
        ("Survival", SurvivalAgent()),
        ("Weapon", WeaponAgent()),
    ]

    print(f"Running {N_EPISODES} games per heuristic from seed {SEED}.")
    for name, agent in agents:
        print(f"\n=== {name} ===")
        evaluate_agent(
            agent,
            n_episodes=N_EPISODES,
            seed_start=SEED,
        )

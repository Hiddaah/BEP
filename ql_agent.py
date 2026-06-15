"""Tabular Q-learning agent for Scoundrel."""

import pickle
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

from Engine import (
    create_dungeon, draw_room, avoid_room, resolve_card,
    lose_score, win_score,
    is_monster, is_weapon, is_potion,
    Player, normal_card_count,
    episode_result,
)
from TrainingMetrics import TrainingMetrics


# State bins

MONSTER_LOW_MAX = 6
MONSTER_MED_MAX = 10

WEAPON_WEAK_MAX = 5
POTION_SMALL_MAX = 5

HEALTH_LOW_MAX = 7
HEALTH_MID_MAX = 13
HEALTH_HIGH_MAX = 19

WLIM_LOW_MAX = 6
WLIM_MED_MAX = 10

EXPECTED_DAMAGE_LOW_MAX = 5
EXPECTED_DAMAGE_MED_MAX = 10

# Shaped learning rewards; evaluation still uses the official Scoundrel score.
TERMINAL_WIN_REWARD = 100.0
TERMINAL_LOSS_REWARD = -100.0
CARD_PROGRESS_REWARD = 1.0
AVOID_REWARD = -1.0
LATE_GAME_CARDS_LEFT = 16
END_GAME_CARDS_LEFT = 8
LATE_GAME_REWARD = 2.0
END_GAME_REWARD = 4.0

PICK_WEAPON = 0
PICK_POTION = 1
PICK_MONSTER_LOW_BARE = 2
PICK_MONSTER_LOW_WEAPON = 3
PICK_MONSTER_MED_BARE = 4
PICK_MONSTER_MED_WEAPON = 5
PICK_MONSTER_FACE_BARE = 6
PICK_MONSTER_FACE_WEAPON = 7
PICK_MONSTER_ACE_BARE = 8
PICK_MONSTER_ACE_WEAPON = 9

MONSTER_ACTION_GROUPS = [
    (PICK_MONSTER_LOW_BARE, PICK_MONSTER_LOW_WEAPON, 2, MONSTER_LOW_MAX),
    (PICK_MONSTER_MED_BARE, PICK_MONSTER_MED_WEAPON, MONSTER_LOW_MAX + 1, MONSTER_MED_MAX),
    (PICK_MONSTER_FACE_BARE, PICK_MONSTER_FACE_WEAPON, MONSTER_MED_MAX + 1, 13),
    (PICK_MONSTER_ACE_BARE, PICK_MONSTER_ACE_WEAPON, 14, 14),
]


# Binning helpers

def monster_bin(value):
    if value <= MONSTER_LOW_MAX:
        return 1
    if value <= MONSTER_MED_MAX:
        return 2
    return 3

def weapon_bin(value):
    return 1 if value <= WEAPON_WEAK_MAX else 2

def potion_bin(value):
    return 1 if value <= POTION_SMALL_MAX else 2

def health_bin(hp):
    if hp <= HEALTH_LOW_MAX:
        return 0
    if hp <= HEALTH_MID_MAX:
        return 1
    if hp <= HEALTH_HIGH_MAX:
        return 2
    return 3

def weapon_limit_bin(limit):
    if limit is None:
        return 0
    if limit <= WLIM_LOW_MAX:
        return 1
    if limit <= WLIM_MED_MAX:
        return 2
    return 3

def deck_progress_bin(cards_remaining):
    if cards_remaining > LATE_GAME_CARDS_LEFT:
        return 0
    if cards_remaining > END_GAME_CARDS_LEFT:
        return 1
    return 2

def estimated_card_damage(player, card):
    if not is_monster(card):
        return 0
    if player.weapon_use(card):
        return max(0, card[1] - player.weapon[1])
    return card[1]

def expected_damage_bin(player, room):
    damages = sorted(estimated_card_damage(player, card) for card in room)
    cards_to_take = 3 if len(room) == 4 else len(room)
    expected_damage = sum(damages[:cards_to_take])

    if expected_damage == 0:
        return 0
    if expected_damage <= EXPECTED_DAMAGE_LOW_MAX:
        return 1
    if expected_damage <= EXPECTED_DAMAGE_MED_MAX:
        return 2
    return 3

def terminal_learning_reward(score):
    if score > 0:
        return TERMINAL_WIN_REWARD + score
    return max(TERMINAL_LOSS_REWARD, float(score))

def survival_depth_reward(cards_remaining):
    reward = 0.0
    if cards_remaining <= LATE_GAME_CARDS_LEFT:
        reward += LATE_GAME_REWARD
    if cards_remaining <= END_GAME_CARDS_LEFT:
        reward += END_GAME_REWARD
    return reward


# State encoder

def get_state(player, room, deck_size):
    monsters = [c for c in room if is_monster(c)]
    potions  = [c for c in room if is_potion(c)]

    nM_bin = min(len(monsters), 4)
    maxM_bin = max((monster_bin(c[1]) for c in monsters), default=0)
    P_bin = max((potion_bin(c[1]) for c in potions), default=0)

    h_bin = health_bin(player.health)
    wval_bin = 0 if player.weapon is None else weapon_bin(player.weapon[1])
    wlim_bin = 0 if player.weapon is None else weapon_limit_bin(player.weapon_limit)
    potion_used_bin = int(player.potion_used_this_turn)
    dmg_bin = expected_damage_bin(player, room)

    cards_remaining = deck_size + len(room)
    d_bin = deck_progress_bin(cards_remaining)

    return (nM_bin, maxM_bin, P_bin,
            h_bin, wval_bin, wlim_bin,
            potion_used_bin, dmg_bin, d_bin)


# Q-table factory and epsilon-greedy policy

def make_qtable(n_actions):
    return defaultdict(lambda: np.zeros(n_actions))

def epsilon_greedy(q_values, valid_actions, epsilon):
    if random.random() < epsilon:
        return random.choice(valid_actions)
    return max(valid_actions, key=lambda a: q_values[a])


# Agent

class QLearningAgent:
    def __init__(
        self,
        alpha=0.1,
        gamma=0.95,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.9995,
    ):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.Q_avoid = make_qtable(2)
        self.Q_card = make_qtable(10)

        self._deck_size = 44
        self._pending_action = None

    def choose_avoid(self, player, room, can_avoid):
        if not can_avoid or len(room) < 4:
            return False
        s = get_state(player, room, self._deck_size)
        a = epsilon_greedy(self.Q_avoid[s], [0, 1], self.epsilon)
        return bool(a)

    def choose_card(self, player, room, picks_done):
        s = get_state(player, room, self._deck_size)
        valid = valid_pick_actions(player, room)
        a = epsilon_greedy(self.Q_card[s], valid, self.epsilon)
        idx, action = resolve_pick_action(player, room, a)
        self._pending_action = action
        return idx

    def choose_action(self, player, card):
        if self._pending_action is not None:
            action = self._pending_action
            self._pending_action = None
            return action

        return "bare"

    def update(self, table_key, state, action, reward,
               next_table_key=None, next_state=None, next_valid_actions=None,
               done=False):
        Q = self.q_table_for(table_key)
        target = float(reward)

        if not done and next_table_key is not None and next_state is not None:
            valid = next_valid_actions or []
            if valid:
                next_Q = self.q_table_for(next_table_key)
                target += self.gamma * max(next_Q[next_state][a] for a in valid)

        Q[state][action] += self.alpha * (target - Q[state][action])

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def q_table_for(self, key):
        return {
            "avoid": self.Q_avoid,
            "card": self.Q_card,
        }[key]


# Episode logic

def valid_pick_actions(player, room):
    actions = []
    if any(is_weapon(card) for card in room):
        actions.append(PICK_WEAPON)
    if any(is_potion(card) for card in room):
        actions.append(PICK_POTION)

    for bare_action, weapon_action, low, high in MONSTER_ACTION_GROUPS:
        monsters = [
            card for card in room
            if is_monster(card) and low <= card[1] <= high
        ]
        if monsters:
            actions.append(bare_action)
            if any(player.weapon_use(card) for card in monsters):
                actions.append(weapon_action)
    return actions


def _best_index(candidates, key):
    return max(candidates, key=lambda item: key(item[1]))[0]


def resolve_pick_action(player, room, action):
    indexed = list(enumerate(room))

    if action == PICK_WEAPON:
        candidates = [(i, card) for i, card in indexed if is_weapon(card)]
        if candidates:
            return _best_index(candidates, lambda card: card[1]), None

    if action == PICK_POTION:
        candidates = [(i, card) for i, card in indexed if is_potion(card)]
        if candidates:
            return _best_index(
                candidates,
                lambda card: min(card[1], 20 - player.health),
            ), None

    group = next(
        (
            (bare_action, weapon_action, low, high)
            for bare_action, weapon_action, low, high in MONSTER_ACTION_GROUPS
            if action in (bare_action, weapon_action)
        ),
        None,
    )
    if group is None:
        return 0, None

    _, weapon_action, low, high = group
    candidates = [
        (i, card) for i, card in indexed
        if is_monster(card) and low <= card[1] <= high
    ]
    if action == weapon_action:
        candidates = [
            (i, card) for i, card in candidates
            if player.weapon_use(card)
        ]
        attack_action = "weapon"
    else:
        attack_action = "bare"

    if candidates:
        return min(
            candidates,
            key=lambda item: (estimated_card_damage(player, item[1]), item[1][1]),
        )[0], attack_action

    return 0, None


def _preview_next_room(deck, carry_over=None):
    room = []
    if carry_over is not None:
        room.append(carry_over)

    normal_cards = len(room)
    i = len(deck) - 1
    while normal_cards < 4 and i >= 0:
        card = deck[i]
        room.append(card)
        normal_cards += 1
        i -= 1

    return room


def _deck_size_after_preview(deck, preview_room, carry_over=None):
    drawn_from_deck = len(preview_room) - (1 if carry_over is not None else 0)
    return normal_card_count(deck[:len(deck) - drawn_from_deck])


def select_action(q_table, state, valid_actions, epsilon, metrics=None, decision_type=None):
    exploratory = random.random() < epsilon
    if exploratory:
        action = random.choice(valid_actions)
    else:
        action = max(valid_actions, key=lambda candidate: q_table[state][candidate])
    if metrics is not None:
        metrics.record_decision(decision_type, state, action, exploratory)
    return action


def _decision_for_visible_room(player, room, deck_size, can_avoid=True):
    if len(room) == 4 and can_avoid:
        return "avoid", get_state(player, room, deck_size), [0, 1]
    return "card", get_state(player, room, deck_size), valid_pick_actions(player, room)


def _next_decision_after_pick(player, room, deck, picks_left_in_turn):
    if picks_left_in_turn > 0:
        return "card", get_state(player, room, normal_card_count(deck)), valid_pick_actions(player, room)

    next_carry = room[0] if room else None
    preview_room = _preview_next_room(deck, next_carry)
    if not preview_room:
        return None, None, []

    preview_deck_size = _deck_size_after_preview(deck, preview_room, next_carry)
    saved = player.potion_used_this_turn
    player.potion_used_this_turn = False
    next_key, preview_state, valid = _decision_for_visible_room(
        player,
        preview_room,
        preview_deck_size,
    )
    player.potion_used_this_turn = saved
    return next_key, preview_state, valid


def run_episode(agent, seed=None, train=True, return_details=False, metrics=None):
    deck = create_dungeon(seed=seed)
    player = Player()
    carry_over = None
    can_avoid = True
    last_taken_card = None
    epsilon = agent.epsilon if train else 0.0
    avoid_opportunities = 0
    avoided_rooms = 0

    while True:
        if not deck and carry_over is None:
            score = win_score(player, last_taken_card)
            if return_details:
                return episode_result(
                    score,
                    deck,
                    None,
                    won=True,
                    avoid_opportunities=avoid_opportunities,
                    avoided_rooms=avoided_rooms,
                )
            return score

        room = draw_room(deck, carry_over)
        player.reset_turn()

        if len(room) == 4:
            deck_size = normal_card_count(deck)
            s_avoid = get_state(player, room, deck_size)

            if can_avoid:
                avoid_opportunities += 1
                a_avoid = select_action(
                    agent.Q_avoid, s_avoid, [0, 1], epsilon, metrics, "avoid"
                )
            else:
                a_avoid = 0

            if a_avoid == 1:
                avoided_rooms += 1
                avoid_room(deck, room)
                preview_room = _preview_next_room(deck)
                preview_deck_size = _deck_size_after_preview(deck, preview_room)
                next_key, s_next, next_valid = _decision_for_visible_room(
                    player,
                    preview_room,
                    preview_deck_size,
                    can_avoid=False,
                )

                if train:
                    agent.update(
                        "avoid", s_avoid, a_avoid, AVOID_REWARD,
                        next_key, s_next, next_valid,
                    )

                carry_over = None
                can_avoid = False
                continue

            if train and can_avoid:
                agent.update(
                    "avoid", s_avoid, a_avoid, 0.0,
                    "card", s_avoid, valid_pick_actions(player, room),
                )

        can_avoid = True
        cards_to_take = 3 if len(room) == 4 else len(room)

        for pick_number in range(cards_to_take):
            deck_size = normal_card_count(deck)
            s_card = get_state(player, room, deck_size)
            valid_cards = valid_pick_actions(player, room)
            a_card = select_action(
                agent.Q_card, s_card, valid_cards, epsilon, metrics, "card"
            )

            idx, action = resolve_pick_action(player, room, a_card)
            card = room.pop(idx)
            hp_before = player.health

            resolve_card(player, card, action)
            last_taken_card = card

            if player.health <= 0:
                score = lose_score(player, deck, room)
                learning_reward = terminal_learning_reward(score)
                if train:
                    agent.update(
                        "card", s_card, a_card,
                        learning_reward, done=True,
                    )
                if return_details:
                    return episode_result(
                        score,
                        deck,
                        room,
                        won=False,
                        avoid_opportunities=avoid_opportunities,
                        avoided_rooms=avoided_rooms,
                    )
                return score

            if not deck and not room:
                score = win_score(player, last_taken_card)
                learning_reward = terminal_learning_reward(score)
                if train:
                    agent.update(
                        "card", s_card, a_card,
                        learning_reward, done=True,
                    )
                if return_details:
                    return episode_result(
                        score,
                        deck,
                        room,
                        won=True,
                        avoid_opportunities=avoid_opportunities,
                        avoided_rooms=avoided_rooms,
                    )
                return score

            cards_remaining = normal_card_count(deck) + normal_card_count(room)
            reward = (
                (player.health - hp_before)
                + CARD_PROGRESS_REWARD
                + survival_depth_reward(cards_remaining)
            )
            if train:
                next_key, next_state, next_valid = _next_decision_after_pick(
                    player,
                    room,
                    deck,
                    cards_to_take - pick_number - 1,
                )

                if next_key is not None:
                    agent.update(
                        "card", s_card, a_card, reward,
                        next_key, next_state, next_valid,
                    )
                else:
                    agent.update("card", s_card, a_card, reward, done=True)

        carry_over = room[0] if room else None


# Training loop

def train(
    n_episodes=2_000_000,
    alpha=0.05,
    gamma=0.95,
    epsilon_start=1.0,
    epsilon_min=0.005,
    epsilon_decay=0.99999,
    seed_start=136,
    verbose_every=200_000,
    metrics_every=50_000,
    metrics_path=None,
    checkpoint_every=0,
    checkpoint_prefix="ql_checkpoint",
):
    if seed_start is not None:
        random.seed(seed_start)
        np.random.seed(seed_start)

    agent = QLearningAgent(alpha, gamma, epsilon_start, epsilon_min, epsilon_decay)
    scores, wins = [], 0
    metrics = (
        TrainingMetrics(
            metrics_path,
            finite_state_count=69_120,
            finite_state_action_count=69_120 * 12,
        )
        if metrics_path
        else None
    )

    for ep in range(n_episodes):
        seed = None if seed_start is None else seed_start + ep
        score = run_episode(agent, seed=seed, train=True, metrics=metrics)
        agent.decay_epsilon()

        scores.append(score)
        if score > 0:
            wins += 1

        if verbose_every and (ep + 1) % verbose_every == 0:
            recent = scores[-verbose_every:]
            avg    = np.mean(recent)
            win_r  = sum(s > 0 for s in recent) / verbose_every
            print(
                f"Episode {ep+1:>7d} | "
                f"avg score (last {verbose_every}): {avg:+.2f} | "
                f"win rate: {win_r:.1%} | "
                f"epsilon = {agent.epsilon:.4f}"
            )

        if metrics and metrics_every and (ep + 1) % metrics_every == 0:
            metrics.snapshot(ep + 1, scores[-metrics_every:], agent.epsilon)

        if checkpoint_every and (ep + 1) % checkpoint_every == 0:
            path = f"{checkpoint_prefix}_{ep+1}.pkl"
            save_agent(agent, path)
            print(f"   [checkpoint] saved {path}")

    if metrics and (not metrics.rows or metrics.rows[-1]["episode"] != n_episodes):
        interval = scores[-metrics_every:] if metrics_every else scores
        metrics.snapshot(n_episodes, interval, agent.epsilon)

    if verbose_every:
        print(f"\nTraining complete. Overall win rate: {wins/n_episodes:.1%}")

    return agent


# Persistence

def _qtable_to_dict(qtable):
    return {state: values.copy() for state, values in qtable.items()}


def _dict_to_qtable(data, n_actions):
    qtable = make_qtable(n_actions)
    for state, values in data.items():
        qtable[state] = np.array(values, dtype=float)
    return qtable


def save_agent(agent, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "alpha": agent.alpha,
        "gamma": agent.gamma,
        "epsilon": agent.epsilon,
        "epsilon_min": agent.epsilon_min,
        "epsilon_decay": agent.epsilon_decay,
        "Q_avoid": _qtable_to_dict(agent.Q_avoid),
        "Q_card": _qtable_to_dict(agent.Q_card),
    }
    with path.open("wb") as f:
        pickle.dump(payload, f)


def load_agent(path):
    with Path(path).open("rb") as f:
        payload = pickle.load(f)

    agent = QLearningAgent(
        alpha=payload["alpha"],
        gamma=payload["gamma"],
        epsilon=payload["epsilon"],
        epsilon_min=payload["epsilon_min"],
        epsilon_decay=payload["epsilon_decay"],
    )
    agent.Q_avoid = _dict_to_qtable(payload["Q_avoid"], 2)
    agent.Q_card = _dict_to_qtable(payload["Q_card"], 10)
    return agent


# Evaluation

def evaluate(agent, n_episodes=1_000, seed_start=0):
    """Run the trained agent with epsilon=0 and report statistics."""
    results = []
    for ep in range(n_episodes):
        result = run_episode(
            agent,
            seed=seed_start + ep,
            train=False,
            return_details=True,
        )
        results.append(result)

    scores = [result["score"] for result in results]
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
    score_array = np.asarray(scores, dtype=np.float64)
    wins = sum(result["won"] for result in results)
    win_rate = wins / n_episodes
    score_std = float(np.std(score_array, ddof=1)) if n_episodes > 1 else 0.0
    confidence_margin = 1.96 * score_std / np.sqrt(n_episodes)
    reached_36 = sum(cleared >= 36 for cleared in cards_cleared)
    reached_40 = sum(cleared >= 40 for cleared in cards_cleared)
    reached_44 = sum(cleared >= 44 for cleared in cards_cleared)
    print(f"Evaluation over {n_episodes} episodes:")
    print(f"  Wins      : {wins}/{n_episodes} ({win_rate:.1%})")
    print(f"  Avg score : {np.mean(score_array):+.2f}")
    print(f"  Score 95% CI: +/- {confidence_margin:.2f}")
    print(f"  Score std : {score_std:.2f}")
    print(f"  Median score: {np.median(score_array):+.2f}")
    print(
        f"  Score P10 / P90: "
        f"{np.percentile(score_array, 10):+.2f} / "
        f"{np.percentile(score_array, 90):+.2f}"
    )
    print(f"  Min score : {min(scores):+.2f}")
    print(f"  Max score : {max(scores):+.2f}")
    print(f"  Avg cleared: {np.mean(cards_cleared):.2f} cards")
    print(f"  Max cleared: {max(cards_cleared)} cards")
    print(f"  Reached 36 cards: {reached_36}/{n_episodes} ({reached_36/n_episodes:.1%})")
    print(f"  Reached 40 cards: {reached_40}/{n_episodes} ({reached_40/n_episodes:.1%})")
    print(f"  Reached 44 cards: {reached_44}/{n_episodes} ({reached_44/n_episodes:.1%})")
    print(f"  Avg remaining monster value: {np.mean(remaining_monsters):.2f}")
    print(f"  Avg avoid opportunities: {np.mean(avoid_opportunities):.2f}")
    print(f"  Avg avoided rooms: {np.mean(avoided_rooms):.2f}")
    print(f"  Avoid rate: {avoid_rate:.1%}")
    return scores


# Entry point

if __name__ == "__main__":
    print("=== Training ===")
    agent = train(
        seed_start=136,
        verbose_every=50_000,
        metrics_every=50_000,
        metrics_path="Checkpoints/ql_training_metrics.csv",
        checkpoint_every=50_000,
        checkpoint_prefix="Checkpoints/ql_checkpoint",
    )
    save_agent(agent, "Checkpoints/ql_latest.pkl")
    print("Saved agent to Checkpoints/ql_latest.pkl")

    print("\n=== Greedy evaluation ===")
    evaluate(agent, n_episodes=10_000, seed_start=1_000_000)

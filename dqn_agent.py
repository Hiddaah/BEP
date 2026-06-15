"""Deep Q-learning agent for Scoundrel."""

import argparse
import random
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from TrainingMetrics import TrainingMetrics

from Engine import (
    create_dungeon, draw_room, avoid_room, resolve_card,
    lose_score, win_score,
    is_monster, is_weapon, is_potion, is_joker, shuffle_dungeon,
    remove_jokers_and_refill, Player, normal_card_count,
    episode_result,
)
from ql_agent import (
    AVOID_REWARD, CARD_PROGRESS_REWARD,
    terminal_learning_reward, survival_depth_reward,
    valid_pick_actions, resolve_pick_action,
)


# Reproducibility

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


DEVICE = torch.device("cpu")


# Feature encoding

BASE_STATE_DIM = 35
MAX_EXACT_BOTTOM_CARDS = 8
BOTTOM_CARD_FEATURES = 4
BOTTOM_SUMMARY_FEATURES = 8
BOTTOM_FEATURE_START = BASE_STATE_DIM
BOTTOM_EXACT_START = BOTTOM_FEATURE_START + 1
BOTTOM_SUMMARY_START = BOTTOM_EXACT_START + MAX_EXACT_BOTTOM_CARDS * BOTTOM_CARD_FEATURES
STATE_DIM = BOTTOM_SUMMARY_START + BOTTOM_SUMMARY_FEATURES

COMPOSITION_TOTALS = np.array([10, 8, 6, 2, 9, 9], dtype=np.float32)
TOTAL_MONSTER_VALUE = 208.0
MAX_EXPECTED_DAMAGE = 42.0
JOKER_REWARD = 0.0


def encode_state(
    player,
    room,
    deck_size,
    resolved_cards=None,
    known_bottom_cards=None,
):
    feats = np.zeros(STATE_DIM, dtype=np.float32)

    feats[0] = player.health / 20.0
    if player.weapon is not None:
        feats[1] = 1.0
        feats[2] = player.weapon[1] / 14.0
        feats[3] = 1.0 if player.weapon_limit is None else player.weapon_limit / 14.0
    feats[4] = 1.0 if player.potion_used_this_turn else 0.0

    def card_sort_key(card):
        suit_rank = 0 if is_monster(card) else (1 if is_weapon(card) else 2)
        return (suit_rank, card[1])

    sorted_room = sorted(room, key=card_sort_key)
    for i, card in enumerate(sorted_room[:4]):
        base = 5 + i * 4
        feats[base + 0] = 1.0 if is_monster(card) else 0.0
        feats[base + 1] = 1.0 if is_weapon(card) else 0.0
        feats[base + 2] = 1.0 if is_potion(card) else 0.0
        feats[base + 3] = card[1] / 14.0

    cards_remaining = deck_size + len(room)
    feats[21] = cards_remaining / 44.0
    feats[22] = (20 - player.health) / 20.0
    feats[23] = sum(1 for card in room if is_monster(card)) / 4.0
    expected_damage = _expected_damage(player, room)
    feats[24] = expected_damage / MAX_EXPECTED_DAMAGE

    # Fled rooms return to the deck, so memory is based on resolved cards only.
    if resolved_cards is not None:
        feats[25:31] = _remaining_composition_features(resolved_cards)
        feats[31] = _remaining_monster_value_from_resolved(resolved_cards) / TOTAL_MONSTER_VALUE

    feats[32] = 1.0 if player.health > expected_damage else 0.0
    feats[33] = max(0.0, player.health - expected_damage) / 20.0
    feats[34] = 1.0 if cards_remaining <= 4 else 0.0

    known_bottom_cards = known_bottom_cards or []
    feats[BOTTOM_FEATURE_START] = len(known_bottom_cards) / 44.0
    upcoming_known_cards = list(reversed(known_bottom_cards))[:MAX_EXACT_BOTTOM_CARDS]
    for i, card in enumerate(upcoming_known_cards):
        base = BOTTOM_EXACT_START + i * BOTTOM_CARD_FEATURES
        feats[base + 0] = 1.0 if is_monster(card) else 0.0
        feats[base + 1] = 1.0 if is_weapon(card) else 0.0
        feats[base + 2] = 1.0 if is_potion(card) else 0.0
        feats[base + 3] = card[1] / 14.0

    feats[BOTTOM_SUMMARY_START:BOTTOM_SUMMARY_START + 6] = (
        _bottom_composition_features(known_bottom_cards)
    )
    known_monster_value = sum(
        card[1] for card in known_bottom_cards if is_monster(card)
    )
    feats[BOTTOM_SUMMARY_START + 6] = known_monster_value / TOTAL_MONSTER_VALUE
    feats[BOTTOM_SUMMARY_START + 7] = (
        sum(1 for card in known_bottom_cards if is_monster(card)) / 28.0
    )

    return feats


def _composition_counts(cards):
    counts = np.zeros(6, dtype=np.float32)
    for card in cards:
        if is_monster(card):
            if card[1] <= 6:
                counts[0] += 1
            elif card[1] <= 10:
                counts[1] += 1
            elif card[1] < 14:
                counts[2] += 1
            else:
                counts[3] += 1
        elif is_weapon(card):
            counts[4] += 1
        elif is_potion(card):
            counts[5] += 1
    return counts


def _bottom_composition_features(known_bottom_cards):
    counts = _composition_counts(known_bottom_cards)
    return counts / COMPOSITION_TOTALS


def _remaining_composition_features(resolved_cards):
    resolved_counts = _composition_counts(resolved_cards)
    remaining_counts = np.clip(COMPOSITION_TOTALS - resolved_counts, 0.0, COMPOSITION_TOTALS)
    return remaining_counts / COMPOSITION_TOTALS


def _remaining_monster_value_from_resolved(resolved_cards):
    resolved_monster_value = sum(
        card[1] for card in resolved_cards
        if is_monster(card)
    )
    return max(0.0, TOTAL_MONSTER_VALUE - resolved_monster_value)


def _card_damage(player, card):
    if not is_monster(card):
        return 0
    if player.weapon_use(card):
        return max(0, card[1] - player.weapon[1])
    return card[1]


def _expected_damage(player, room):
    damages = sorted(_card_damage(player, card) for card in room)
    cards_to_take = 3 if len(room) == 4 else len(room)
    return sum(damages[:cards_to_take])


# Networks and replay

class QNet(nn.Module):
    def __init__(self, in_dim, n_actions, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buf = deque(maxlen=capacity)

    def push(self, state, action, reward, next_key, next_state, done, valid_next):
        self.buf.append((
            state,
            action,
            reward,
            next_key,
            next_state,
            done,
            list(valid_next or []),
        ))

    def sample(self, batch_size):
        batch = random.sample(self.buf, batch_size)
        states, actions, rewards, next_keys, next_states, dones, valid_next = zip(*batch)

        states = torch.tensor(np.array(states), dtype=torch.float32, device=DEVICE)
        actions = torch.tensor(actions, dtype=torch.long, device=DEVICE).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=DEVICE).unsqueeze(1)
        dones = torch.tensor(dones, dtype=torch.float32, device=DEVICE).unsqueeze(1)
        next_arr = np.array([
            state if state is not None else np.zeros(STATE_DIM, dtype=np.float32)
            for state in next_states
        ])
        next_states = torch.tensor(next_arr, dtype=torch.float32, device=DEVICE)
        return (
            states,
            actions,
            rewards,
            list(next_keys),
            next_states,
            dones,
            list(valid_next),
        )

    def __len__(self):
        return len(self.buf)


class DQNHead:
    def __init__(self, n_actions, lr=1e-3, gamma=0.95,
                 buffer_size=100_000, batch_size=128, hidden=64):
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size

        self.online = QNet(STATE_DIM, n_actions, hidden).to(DEVICE)
        self.target = QNet(STATE_DIM, n_actions, hidden).to(DEVICE)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        self.opt = torch.optim.Adam(self.online.parameters(), lr=lr)
        self.replay = ReplayBuffer(buffer_size)

    def act(self, state_vec, valid_actions, epsilon, metrics=None, decision_type=None):
        exploratory = random.random() < epsilon
        if exploratory:
            action = random.choice(valid_actions)
        else:
            with torch.no_grad():
                state = torch.tensor(
                    state_vec, dtype=torch.float32, device=DEVICE
                ).unsqueeze(0)
                q_values = self.online(state).squeeze(0).cpu().numpy()
            action = max(valid_actions, key=lambda candidate: q_values[candidate])
        if metrics is not None:
            metrics.record_decision(decision_type, state_vec, action, exploratory)
        return action

    def learn(self, heads):
        if len(self.replay) < self.batch_size:
            return None

        (
            states,
            actions,
            rewards,
            next_keys,
            next_states,
            dones,
            valid_next,
        ) = self.replay.sample(self.batch_size)

        q_sa = self.online(states).gather(1, actions)

        with torch.no_grad():
            q_next_values = torch.zeros_like(rewards)
            for i, (next_key, valid) in enumerate(zip(next_keys, valid_next)):
                if dones[i].item() or next_key is None or not valid:
                    continue

                next_head = heads[next_key]
                next_state = next_states[i].unsqueeze(0)
                q_online = next_head.online(next_state).squeeze(0)
                best_action = max(valid, key=lambda action: q_online[action].item())
                q_target = next_head.target(next_state).squeeze(0)
                q_next_values[i, 0] = q_target[best_action]

            target = rewards + self.gamma * q_next_values * (1.0 - dones)

        loss = F.smooth_l1_loss(q_sa, target)

        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.opt.step()
        return float(loss.item())

    def sync_target(self):
        self.target.load_state_dict(self.online.state_dict())

    def set_eval(self):
        self.online.eval()

    def set_train(self):
        self.online.train()


class DQNAgent:
    def __init__(
        self,
        lr=1e-3,
        gamma=0.95,
        epsilon=1.0,
        epsilon_min=0.005,
        epsilon_decay=0.9999,
        buffer_size=100_000,
        batch_size=128,
        hidden=64,
        target_sync_every=1000,
    ):
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.target_sync_every = target_sync_every
        self.hidden = hidden

        self.head_avoid = DQNHead(2, lr, gamma, buffer_size, batch_size, hidden)
        self.head_joker = DQNHead(2, lr, gamma, buffer_size, batch_size, hidden)
        self.head_card = DQNHead(10, lr, gamma, buffer_size, batch_size, hidden)
        self._heads = {
            "avoid": self.head_avoid,
            "joker": self.head_joker,
            "card": self.head_card,
        }
        self._learn_steps = 0

    def heads(self):
        return self._heads

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def learn_batch(self, n_steps=8):
        heads = self.heads()
        for _ in range(n_steps):
            for head in heads.values():
                head.learn(heads)
            self._learn_steps += 1
            if self._learn_steps % self.target_sync_every == 0:
                self.sync_targets()

    def sync_targets(self):
        for head in self.heads().values():
            head.sync_target()

    def set_eval(self):
        for head in self.heads().values():
            head.set_eval()

    def set_train(self):
        for head in self.heads().values():
            head.set_train()


# Episode mechanics

def _preview_next_room(deck, carry_over=None):
    room = []
    if carry_over is not None:
        room.append(carry_over)

    normal_cards = sum(1 for card in room if not is_joker(card))
    i = len(deck) - 1
    while normal_cards < 4 and i >= 0:
        card = deck[i]
        room.append(card)
        if not is_joker(card):
            normal_cards += 1
        i -= 1
    return room


def _deck_size_after_preview(deck, preview_room, carry_over=None):
    drawn_from_deck = len(preview_room) - (1 if carry_over is not None else 0)
    return normal_card_count(deck[:len(deck) - drawn_from_deck])


def _known_cards_still_in_deck(known_bottom_cards, deck):
    deck_cards = set(deck)
    return [card for card in known_bottom_cards if card in deck_cards]


def _known_cards_after_preview(known_bottom_cards, deck, preview_room, carry_over=None):
    drawn_from_deck = len(preview_room) - (1 if carry_over is not None else 0)
    preview_deck = deck[:len(deck) - drawn_from_deck]
    return _known_cards_still_in_deck(known_bottom_cards, preview_deck)


def _decision_for_visible_room(
    player,
    room,
    deck_size,
    resolved_cards,
    known_bottom_cards,
    can_avoid=True,
):
    room_without_joker = [card for card in room if not is_joker(card)]
    if len(room_without_joker) != len(room):
        return (
            "joker",
            encode_state(
                player,
                room_without_joker,
                deck_size,
                resolved_cards,
                known_bottom_cards,
            ),
            [0, 1],
        )
    if len(room_without_joker) == 4 and can_avoid:
        return (
            "avoid",
            encode_state(
                player,
                room_without_joker,
                deck_size,
                resolved_cards,
                known_bottom_cards,
            ),
            [0, 1],
        )
    return (
        "card",
        encode_state(
            player,
            room_without_joker,
            deck_size,
            resolved_cards,
            known_bottom_cards,
        ),
        valid_pick_actions(player, room_without_joker),
    )


def _next_decision_after_pick(
    player,
    room,
    deck,
    picks_left_in_turn,
    resolved_cards,
    known_bottom_cards,
):
    if picks_left_in_turn > 0:
        return (
            "card",
            encode_state(
                player,
                room,
                normal_card_count(deck),
                resolved_cards,
                known_bottom_cards,
            ),
            valid_pick_actions(player, room),
        )

    next_carry = room[0] if room else None
    preview_room = _preview_next_room(deck, next_carry)
    if not preview_room:
        return None, None, []

    preview_deck_size = _deck_size_after_preview(deck, preview_room, next_carry)
    preview_known_bottom = _known_cards_after_preview(
        known_bottom_cards,
        deck,
        preview_room,
        next_carry,
    )
    saved = player.potion_used_this_turn
    player.potion_used_this_turn = False
    next_key, preview_state, valid = _decision_for_visible_room(
        player,
        preview_room,
        preview_deck_size,
        resolved_cards,
        preview_known_bottom,
    )
    player.potion_used_this_turn = saved
    return next_key, preview_state, valid


def store_transition(agent, head_key, state, action, reward,
                     next_key=None, next_state=None, next_valid=None, done=False):
    agent.heads()[head_key].replay.push(
        state,
        action,
        reward,
        next_key,
        next_state,
        done,
        next_valid or [],
    )


def run_episode(
    agent,
    seed=None,
    train=True,
    return_details=False,
    include_joker=False,
    metrics=None,
):
    deck = create_dungeon(seed=seed, include_joker=include_joker)
    joker_rng = random.Random(None if seed is None else seed + 10_000_000)
    player = Player()
    carry_over = None
    can_avoid = True
    last_taken_card = None
    epsilon = agent.epsilon if train else 0.0
    avoid_opportunities = 0
    avoided_rooms = 0
    joker_found = 0
    joker_used = 0
    resolved_cards = set()
    known_bottom_cards = []

    while True:
        if not deck and carry_over is None:
            score = win_score(player, last_taken_card)
            if return_details:
                return episode_result(
                    score, deck, None, True, avoid_opportunities, avoided_rooms,
                    joker_found, joker_used
                )
            return score

        room = draw_room(deck, carry_over)
        player.reset_turn()

        room, encountered_jokers = remove_jokers_and_refill(deck, room)
        known_bottom_cards = _known_cards_still_in_deck(known_bottom_cards, deck)
        if encountered_jokers:
            joker_found += encountered_jokers
            s_joker = encode_state(
                player,
                room,
                normal_card_count(deck),
                resolved_cards,
                known_bottom_cards,
            )
            a_joker = agent.head_joker.act(
                s_joker, [0, 1], epsilon, metrics, "joker"
            )
            if a_joker == 1:
                joker_used += 1
                shuffle_dungeon(deck, joker_rng)
                known_bottom_cards = []

            if not deck and not room:
                score = win_score(player, last_taken_card)
                reward = terminal_learning_reward(score)
                if train:
                    store_transition(agent, "joker", s_joker, a_joker, reward, done=True)
                if return_details:
                    return episode_result(
                        score, deck, None, True, avoid_opportunities, avoided_rooms,
                        joker_found, joker_used
                    )
                return score

            if train:
                if len(room) == 4 and can_avoid:
                    next_key, next_valid = "avoid", [0, 1]
                else:
                    next_key, next_valid = "card", valid_pick_actions(player, room)
                next_state = encode_state(
                    player,
                    room,
                    normal_card_count(deck),
                    resolved_cards,
                    known_bottom_cards,
                )
                store_transition(
                    agent, "joker", s_joker, a_joker, JOKER_REWARD,
                    next_key, next_state, next_valid,
                )

        if len(room) == 4:
            deck_size = normal_card_count(deck)
            s_avoid = encode_state(
                player,
                room,
                deck_size,
                resolved_cards,
                known_bottom_cards,
            )

            if can_avoid:
                avoid_opportunities += 1
                a_avoid = agent.head_avoid.act(
                    s_avoid, [0, 1], epsilon, metrics, "avoid"
                )
            else:
                a_avoid = 0

            if a_avoid == 1:
                avoided_rooms += 1
                avoid_room(deck, room)
                known_bottom_cards = room + known_bottom_cards
                preview_room = _preview_next_room(deck)
                preview_deck_size = _deck_size_after_preview(deck, preview_room)
                preview_known_bottom = _known_cards_after_preview(
                    known_bottom_cards,
                    deck,
                    preview_room,
                )
                next_key, s_next, next_valid = _decision_for_visible_room(
                    player,
                    preview_room,
                    preview_deck_size,
                    resolved_cards,
                    preview_known_bottom,
                    can_avoid=False,
                )

                if train:
                    store_transition(
                        agent, "avoid", s_avoid, a_avoid, AVOID_REWARD,
                        next_key, s_next, next_valid,
                    )

                carry_over = None
                can_avoid = False
                continue

            if train and can_avoid:
                store_transition(
                    agent, "avoid", s_avoid, a_avoid, 0.0,
                    "card", s_avoid, valid_pick_actions(player, room),
                )

        can_avoid = True
        cards_to_take = 3 if len(room) == 4 else len(room)

        for pick_number in range(cards_to_take):
            deck_size = normal_card_count(deck)
            s_card = encode_state(
                player,
                room,
                deck_size,
                resolved_cards,
                known_bottom_cards,
            )
            valid_cards = valid_pick_actions(player, room)
            a_card = agent.head_card.act(
                s_card, valid_cards, epsilon, metrics, "card"
            )

            idx, action = resolve_pick_action(player, room, a_card)
            card = room.pop(idx)
            resolved_cards.add(card)
            hp_before = player.health

            resolve_card(player, card, action)
            last_taken_card = card

            if player.health <= 0:
                score = lose_score(player, deck, room)
                reward = terminal_learning_reward(score)
                if train:
                    store_transition(agent, "card", s_card, a_card, reward, done=True)
                if return_details:
                    return episode_result(
                        score, deck, room, False, avoid_opportunities, avoided_rooms,
                        joker_found, joker_used
                    )
                return score

            if not deck and not room:
                score = win_score(player, last_taken_card)
                reward = terminal_learning_reward(score)
                if train:
                    store_transition(agent, "card", s_card, a_card, reward, done=True)
                if return_details:
                    return episode_result(
                        score, deck, room, True, avoid_opportunities, avoided_rooms,
                        joker_found, joker_used
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
                    resolved_cards,
                    known_bottom_cards,
                )
                if next_key is not None:
                    store_transition(
                        agent, "card", s_card, a_card, reward,
                        next_key, next_state, next_valid,
                    )
                else:
                    store_transition(agent, "card", s_card, a_card, reward, done=True)

        carry_over = room[0] if room else None


# Training and evaluation

def train(
    n_episodes=50_000,
    lr=1e-3,
    gamma=0.95,
    epsilon_start=1.0,
    epsilon_min=0.005,
    epsilon_decay=0.99995,
    hidden=64,
    batch_size=32,
    buffer_size=100_000,
    target_sync_every=2000,
    learn_steps_per_episode=1,
    seed_start=None,
    verbose_every=5000,
    eval_every=0,
    eval_n_episodes=2_000,
    eval_seed_start=2_000_000,
    best_checkpoint_path=None,
    checkpoint_every=50_000,
    checkpoint_prefix="dqn_agent_checkpoint",
    include_joker=False,
    metrics_every=50_000,
    metrics_path=None,
):
    if seed_start is not None:
        set_seed(seed_start)

    agent = DQNAgent(
        lr=lr,
        gamma=gamma,
        epsilon=epsilon_start,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
        buffer_size=buffer_size,
        batch_size=batch_size,
        hidden=hidden,
        target_sync_every=target_sync_every,
    )
    scores = []
    metrics = (
        TrainingMetrics(metrics_path, track_decisions=False)
        if metrics_path
        else None
    )
    best_eval_score = float("-inf")
    for ep in range(n_episodes):
        seed = None if seed_start is None else seed_start + ep
        score = run_episode(
            agent,
            seed=seed,
            train=True,
            include_joker=include_joker,
        )
        agent.learn_batch(n_steps=learn_steps_per_episode)
        agent.decay_epsilon()
        scores.append(score)

        if verbose_every and (ep + 1) % verbose_every == 0:
            recent = scores[-verbose_every:]
            wins = sum(score > 0 for score in recent)
            win_r = wins / len(recent)
            print(
                f"Episode {ep+1:>7d} | avg score (last {len(recent)}): "
                f"{np.mean(recent):+.2f} | wins: {wins}/{len(recent)} "
                f"({win_r:.1%}) | "
                f"epsilon = {agent.epsilon:.4f}"
            )

        if eval_every and (ep + 1) % eval_every == 0:
            ev = evaluate(
                agent,
                n_episodes=eval_n_episodes,
                seed_start=eval_seed_start,
                verbose=False,
                include_joker=include_joker,
            )
            eval_avg = float(np.mean(ev))
            eval_wins = sum(score > 0 for score in ev)
            print(
                f"   [eval @ {ep+1}] avg {eval_avg:+.2f} "
                f"median {np.median(ev):+.2f} max {max(ev):+d} "
                f"wins {eval_wins}/{len(ev)}"
            )
            if best_checkpoint_path and eval_avg > best_eval_score:
                best_eval_score = eval_avg
                save_agent(agent, best_checkpoint_path)
                print(f"   [best] saved {best_checkpoint_path}")

        if checkpoint_every and (ep + 1) % checkpoint_every == 0:
            path = f"{checkpoint_prefix}_{ep+1}.pt"
            save_agent(agent, path)
            print(f"   [checkpoint] saved {path}")

        if metrics and metrics_every and (ep + 1) % metrics_every == 0:
            metrics.snapshot(ep + 1, scores[-metrics_every:], agent.epsilon)

    if metrics and (not metrics.rows or metrics.rows[-1]["episode"] != n_episodes):
        interval = scores[-metrics_every:] if metrics_every else scores
        metrics.snapshot(n_episodes, interval, agent.epsilon)

    return agent


def evaluate(agent, n_episodes=2000, seed_start=0, verbose=True, include_joker=False):
    agent.set_eval()
    results = [
        run_episode(
            agent,
            seed=seed_start + i,
            train=False,
            return_details=True,
            include_joker=include_joker,
        )
        for i in range(n_episodes)
    ]
    agent.set_train()

    scores = [result["score"] for result in results]
    cards_cleared = [result["cards_cleared"] for result in results]
    remaining_monsters = [result["remaining_monster_value"] for result in results]
    avoid_opportunities = [result["avoid_opportunities"] for result in results]
    avoided_rooms = [result["avoided_rooms"] for result in results]
    joker_found = [result["joker_found"] for result in results]
    joker_used = [result["joker_used"] for result in results]
    total_avoid_opportunities = sum(avoid_opportunities)
    avoid_rate = (
        sum(avoided_rooms) / total_avoid_opportunities
        if total_avoid_opportunities
        else 0.0
    )

    if verbose:
        score_array = np.asarray(scores, dtype=np.float64)
        wins = sum(result["won"] for result in results)
        win = wins / n_episodes
        score_std = float(np.std(score_array, ddof=1)) if n_episodes > 1 else 0.0
        confidence_margin = 1.96 * score_std / np.sqrt(n_episodes)
        reached_36 = sum(cleared >= 36 for cleared in cards_cleared)
        reached_40 = sum(cleared >= 40 for cleared in cards_cleared)
        reached_44 = sum(cleared >= 44 for cleared in cards_cleared)
        print(f"Evaluation over {n_episodes} episodes:")
        print(f"  Wins      : {wins}/{n_episodes} ({win:.1%})")
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
        if sum(joker_found):
            joker_use_rate = sum(joker_used) / sum(joker_found)
            print(f"  Avg jokers found: {np.mean(joker_found):.2f}")
            print(f"  Avg jokers used: {np.mean(joker_used):.2f}")
            print(f"  Joker use rate: {joker_use_rate:.1%}")
    return scores


# Persistence

def save_agent(agent, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dim": STATE_DIM,
        "avoid": agent.head_avoid.online.state_dict(),
        "joker": agent.head_joker.online.state_dict(),
        "card": agent.head_card.online.state_dict(),
        "epsilon": agent.epsilon,
        "hidden": agent.hidden,
    }, path)


def load_agent(path, hidden=None):
    state = torch.load(path, map_location=DEVICE, weights_only=True)
    checkpoint_state_dim = state.get("state_dim")
    if checkpoint_state_dim is None and "avoid" in state:
        first_layer = state["avoid"].get("net.0.weight")
        if first_layer is not None:
            checkpoint_state_dim = first_layer.shape[1]
    if checkpoint_state_dim is not None and checkpoint_state_dim != STATE_DIM:
        raise ValueError(
            f"{path} was saved with STATE_DIM={checkpoint_state_dim}, "
            f"but this code expects STATE_DIM={STATE_DIM}. "
            "Start a fresh run after changing the DQN state representation."
        )
    if hidden is None:
        hidden = state.get("hidden", 64)
    agent = DQNAgent(hidden=hidden)
    for key, head in agent.heads().items():
        if key in state:
            head.online.load_state_dict(state[key])
            head.target.load_state_dict(state[key])
    agent.epsilon = state.get("epsilon", 0.0)
    return agent


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Scoundrel DQN.")
    parser.add_argument(
        "--include-joker",
        action="store_true",
        help="Train the optional Joker-rule DQN instead of the original-rule DQN.",
    )
    args = parser.parse_args()

    USE_JOKER_RULE = args.include_joker
    run_name = "Checkpoints/dqn_joker" if USE_JOKER_RULE else "Checkpoints/dqn_original"
    set_seed(0)
    print("=== Training DQN ===")
    print(f"Device: {DEVICE}")
    print(f"Joker rule: {'on' if USE_JOKER_RULE else 'off'}")
    agent = train(
        n_episodes=1_500_000,
        seed_start=136,
        verbose_every=50_000,
        hidden=64,
        batch_size=32,
        learn_steps_per_episode=1,
        epsilon_min=0.005,
        epsilon_decay=0.99999,
        target_sync_every=2_000,
        checkpoint_every=50_000,
        eval_every=50_000,
        eval_n_episodes=2_000,
        eval_seed_start=2_000_000,
        best_checkpoint_path=f"{run_name}_best.pt",
        checkpoint_prefix=f"{run_name}_checkpoint",
        include_joker=USE_JOKER_RULE,
        metrics_every=50_000,
        metrics_path=f"{run_name}_training_metrics.csv",
    )
    latest_path = f"{run_name}_latest.pt"
    save_agent(agent, latest_path)
    print(f"Saved agent to {latest_path}")

    print("\n=== Final evaluation: latest checkpoint ===")
    evaluate(agent, n_episodes=10_000, seed_start=3_000_000, include_joker=USE_JOKER_RULE)

    print("\n=== Final evaluation: best held-out checkpoint ===")
    best_agent = load_agent(f"{run_name}_best.pt")
    evaluate(best_agent, n_episodes=10_000, seed_start=3_000_000, include_joker=USE_JOKER_RULE)

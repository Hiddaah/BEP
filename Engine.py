import random

# Card definitions

CLUBS = 0
SPADES = 1
DIAMONDS = 2
HEARTS = 3
JOKER = 4

SUIT_NAMES = {
    CLUBS: "clubs",
    SPADES: "spades",
    DIAMONDS: "diamonds",
    HEARTS: "hearts",
    JOKER: "joker",
}

VALUE_NAMES = {
    11: "Jack",
    12: "Queen",
    13: "King",
    14: "Ace",
}

def card_str(card):
    suit, value = card
    if suit == JOKER:
        return "Joker"
    value_str = VALUE_NAMES.get(value, str(value))
    return f"{value_str} of {SUIT_NAMES[suit]}"

def is_monster(card):
    return card[0] in (CLUBS, SPADES)

def is_weapon(card):
    return card[0] == DIAMONDS

def is_potion(card):
    return card[0] == HEARTS

def is_joker(card):
    return card[0] == JOKER


# Deck creation

def create_dungeon(seed=None, include_joker=False):
   
    rng = random.Random(seed)
    deck = []

    for suit in (CLUBS, SPADES, DIAMONDS, HEARTS):
        for value in range(2, 15):
            if suit in (DIAMONDS, HEARTS) and value >= 11:
                continue
            deck.append((suit, value))

    rng.shuffle(deck)
    if include_joker:
        # Insert the Joker after shuffling so matched seeds keep the same
        # ordering of the 44 ordinary dungeon cards.
        deck.insert(rng.randrange(len(deck) + 1), (JOKER, 0))
    return deck


# Player state

class Player:
    def __init__(self):
        self.health = 20
        self.weapon = None
        self.weapon_limit = None
        self.potion_used_this_turn = False

    def reset_turn(self):
        self.potion_used_this_turn = False

    def equip_weapon(self, weapon_card):
        self.weapon = weapon_card
        self.weapon_limit = None

    def weapon_use(self, monster_card):
        if self.weapon is None:
            return False
        monster_value = monster_card[1]
        return self.weapon_limit is None or monster_value <= self.weapon_limit

    def __str__(self):
        weapon_str = card_str(self.weapon) if self.weapon else "None"
        return (
            f"Health={self.health}, "
            f"Weapon={weapon_str}, "
            f"WeaponLimit={self.weapon_limit}"
        )
    

# Main game

def draw_room(deck, carry_over):

    room = []
    if carry_over is not None:
        room.append(carry_over)

    while len(room) < 4 and deck:
        room.append(deck.pop())

    return room

def remove_jokers_and_refill(deck, room):
    """Remove encountered Jokers and refill the room with ordinary cards."""
    joker_count = sum(1 for card in room if is_joker(card))
    room = [card for card in room if not is_joker(card)]

    while len(room) < 4 and deck:
        card = deck.pop()
        if is_joker(card):
            joker_count += 1
        else:
            room.append(card)

    return room, joker_count

def avoid_room(deck, room):
    deck[:0] = room

def shuffle_dungeon(deck, rng=None):
    if rng is None:
        random.shuffle(deck)
    else:
        rng.shuffle(deck)

def apply_weapon(player, weapon_card):
    player.equip_weapon(weapon_card)

def apply_potion(player, potion_card):

    if not player.potion_used_this_turn:
        player.health = min(20, player.health + potion_card[1])
        player.potion_used_this_turn = True

def fight_barehanded(player, monster_card):
    player.health -= monster_card[1]

def fight_with_weapon(player, monster_card):
    weapon_value = player.weapon[1]
    monster_value = monster_card[1]
    damage = max(0, monster_value - weapon_value)
    player.health -= damage
    player.weapon_limit = monster_value

def resolve_card(player, card, action):
    if is_weapon(card):
        apply_weapon(player, card)
    elif is_potion(card):
        apply_potion(player, card)
    elif is_monster(card):
        if action == "weapon" and player.weapon_use(card):
            fight_with_weapon(player, card)
        else:
            fight_barehanded(player, card)
    elif is_joker(card):
        pass
    else:
        raise ValueError("Unknown card type")

def lose_score(player, deck, remaining_room=None):

    score = player.health

    for card in deck:
        if is_monster(card):
            score -= card[1]

    if remaining_room is None:
        return score

    if isinstance(remaining_room, tuple):
        remaining_cards = [remaining_room]
    else:
        remaining_cards = remaining_room

    for card in remaining_cards:
        if is_monster(card):
            score -= card[1]
    return score

def win_score(player, last_taken_card):
    return player.health

def attack_actions(player, monster_card):
    actions = ["bare"]
    if player.weapon_use(monster_card):
        actions.append("weapon")
    return actions

def describe_room(room):
    return [f"{i}: {card_str(card)}" for i, card in enumerate(room)]


def remaining_monster_value(deck, room=None):
    cards = list(deck)
    if room:
        cards.extend(room)
    return sum(card[1] for card in cards if is_monster(card))


def normal_card_count(cards):
    return sum(1 for card in cards if not is_joker(card))


def episode_result(
    score,
    deck,
    room=None,
    won=False,
    avoid_opportunities=0,
    avoided_rooms=0,
    joker_found=0,
    joker_used=0,
):
    remaining_cards = normal_card_count(deck)
    if room:
        remaining_cards += normal_card_count(room)

    return {
        "score": score,
        "won": won,
        "cards_cleared": 44 - remaining_cards,
        "remaining_monster_value": remaining_monster_value(deck, room),
        "avoid_opportunities": avoid_opportunities,
        "avoided_rooms": avoided_rooms,
        "joker_found": joker_found,
        "joker_used": joker_used,
    }


# Scoundrel game loop

def play_scoundrel(
    agent,
    seed=None,
    verbose=False,
    include_joker=False,
    return_details=False,
):
    deck = create_dungeon(seed=seed, include_joker=include_joker)
    joker_rng = random.Random(None if seed is None else seed + 10_000_000)
    player = Player()
    carry_over = None
    can_avoid = True
    last_taken_card = None
    avoid_opportunities = 0
    avoided_rooms = 0
    joker_found = 0
    joker_used = 0

    while True:
        if not deck and carry_over is None:
            score = win_score(player, last_taken_card)
            if verbose:
                print("\nDungeon cleared.")
                print("Final score:", score)
            if return_details:
                return episode_result(
                    score,
                    deck,
                    None,
                    won=True,
                    avoid_opportunities=avoid_opportunities,
                    avoided_rooms=avoided_rooms,
                    joker_found=joker_found,
                    joker_used=joker_used,
                )
            return score

        room = draw_room(deck, carry_over)

        room, encountered_jokers = remove_jokers_and_refill(deck, room)
        if encountered_jokers:
            joker_found += encountered_jokers
            use_joker = (
                agent.choose_joker(player, room, normal_card_count(deck))
                if hasattr(agent, "choose_joker")
                else False
            )
            if use_joker:
                joker_used += 1
                shuffle_dungeon(deck, joker_rng)
                if verbose:
                    print("Joker used: remaining dungeon shuffled.")
            elif verbose:
                print("Joker discarded unused.")

        if verbose:
            print("\n" + "=" * 50)
            cards_left = normal_card_count(deck) + normal_card_count(room)
            print(f"New turn. Cards left in dungeon: {cards_left}")
            print("Room:", [card_str(c) for c in room])

        player.reset_turn()

        if len(room) == 4:
            if can_avoid:
                avoid_opportunities += 1
            avoided = agent.choose_avoid(player, room, can_avoid)
            if avoided:
                avoided_rooms += 1
                avoid_room(deck, room)
                carry_over = None
                can_avoid = False
                if verbose:
                    print("Room avoided.")
                continue

        can_avoid = True
        cards_to_take = 3 if len(room) == 4 else len(room)

        for picks_done in range(cards_to_take):
            idx = agent.choose_card(player, room, picks_done)
            card = room.pop(idx)

            if verbose:
                print("Chosen:", card_str(card))

            if is_monster(card):
                action = agent.choose_action(player, card)
            else:
                action = None

            resolve_card(player, card, action)
            last_taken_card = card

            if verbose:
                print("After action:", player)

            if player.health <= 0:
                score = lose_score(player, deck, room)
                if verbose:
                    print("\nYou died.")
                    print("Final score:", score)
                if return_details:
                    return episode_result(
                        score,
                        deck,
                        room,
                        won=False,
                        avoid_opportunities=avoid_opportunities,
                        avoided_rooms=avoided_rooms,
                        joker_found=joker_found,
                        joker_used=joker_used,
                    )
                return score

        carry_over = room[0] if room else None


if __name__ == "__main__":
    from Agents.Manual import ManualAgent

    seed_text = input("Seed (press Enter for random): ").strip()
    seed = int(seed_text) if seed_text else None

    joker_text = input("Use Joker rule? (y/n, default n): ").strip().lower()
    include_joker = joker_text in ("y", "yes")

    score = play_scoundrel(
        ManualAgent(),
        seed=seed,
        verbose=True,
        include_joker=include_joker,
    )
    print("\nFinal score:", score)

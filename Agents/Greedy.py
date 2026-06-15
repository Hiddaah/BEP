from Engine import attack_actions, is_potion, is_weapon, is_monster

class GreedyAgent:
    def __init__(
        self,
        avoid_threshold=12,
        damage_weight=80,
        potion_value=-80,
        weapon_value=50,
        weapon_use_threshold=0
    ):
        self.avoid_threshold = avoid_threshold
        self.damage_weight = damage_weight
        self.potion_value = potion_value
        self.weapon_value = weapon_value
        self.weapon_use_threshold = weapon_use_threshold

    def choose_avoid(self, player, room, can_avoid):
        if not can_avoid:
            return False

        damages = []
        for card in room:
            if is_potion(card) or is_weapon(card):
                damages.append(0)
            else:
                damages.append(self.immediate_damage(player, card))

        damages.sort()
        expected_damage = sum(damages[:3])

        return expected_damage >= self.avoid_threshold

    def choose_card(self, player, room, picks_done):
        best_idx = 0
        best_value = float("-inf")

        for idx, card in enumerate(room):
            value = self.card_utility(player, card)

            if value > best_value:
                best_value = value
                best_idx = idx

        return best_idx

    def choose_action(self, player, monster_card):
        actions = attack_actions(player, monster_card)

        if "weapon" in actions:
            bare = monster_card[1]
            weapon = max(0, monster_card[1] - player.weapon[1])

            if bare - weapon >= self.weapon_use_threshold:
                return "weapon"

        return "bare"

    def immediate_damage(self, player, card):
        if is_potion(card) or is_weapon(card):
            return 0

        if player.weapon_use(card):
            return min(card[1], max(0, card[1] - player.weapon[1]))

        return card[1]

    def card_utility(self, player, card):
        if is_potion(card):
            heal = 0 if player.potion_used_this_turn else min(card[1], 20 - player.health)
            return self.potion_value + heal

        if is_weapon(card):
            current = player.weapon[1] if player.weapon else 0
            return self.weapon_value + (card[1] - current)

        if is_monster(card):
            dmg = self.immediate_damage(player, card)
            return -self.damage_weight * dmg + card[1]

        return -9999

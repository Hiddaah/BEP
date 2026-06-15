from Engine import is_monster, is_weapon, is_potion, attack_actions


class SurvivalAgent:
    def __init__(
        self,
        avoid_health_buffer=2,
        low_health_threshold=6,
        potion_priority=140,
        weapon_priority=40,
        damage_weight=100,
        weapon_use_threshold=0,
    ):
        self.avoid_health_buffer = avoid_health_buffer
        self.low_health_threshold = low_health_threshold
        self.potion_priority = potion_priority
        self.weapon_priority = weapon_priority
        self.damage_weight = damage_weight
        self.weapon_use_threshold = weapon_use_threshold

    def choose_avoid(self, player, room, can_avoid):
        if not can_avoid:
            return False

        expected_damage = self.room_expected_damage(player, room)

        # Avoid if expected damage would leave too little health
        if player.health - expected_damage <= self.avoid_health_buffer:
            return True

        # Extra caution at low health
        if player.health <= self.low_health_threshold and expected_damage >= 6:
            return True

        return False

    def choose_card(self, player, room, picks_done):
        best_idx = 0
        best_utility = float("-inf")

        for idx, card in enumerate(room):
            utility = self.card_utility(player, card)
            if utility > best_utility:
                best_utility = utility
                best_idx = idx

        return best_idx

    def choose_action(self, player, monster_card):
        actions = attack_actions(player, monster_card)

        if "weapon" not in actions:
            return "bare"

        bare_damage = monster_card[1]
        weapon_damage = max(0, monster_card[1] - player.weapon[1])

        # Survival heuristic uses weapon more readily
        if bare_damage - weapon_damage >= self.weapon_use_threshold:
            return "weapon"

        return "bare"

    def immediate_damage(self, player, card):
        if is_potion(card) or is_weapon(card):
            return 0

        if player.weapon_use(card):
            return min(card[1], max(0, card[1] - player.weapon[1]))

        return card[1]

    def room_expected_damage(self, player, room):
        """
        Simple estimate of the damage from the three cards likely to be taken.
        Potions and weapons count as 0 immediate damage.
        """
        damages = [self.immediate_damage(player, card) for card in room]
        damages.sort()
        return sum(damages[:3])

    def card_utility(self, player, card):
        if is_potion(card):
            if player.potion_used_this_turn:
                return 0

            heal = min(card[1], 20 - player.health)

            # Potions become much more valuable at low health
            bonus = 40 if player.health <= self.low_health_threshold else 0
            return self.potion_priority + 2 * heal + bonus

        if is_weapon(card):
            current = player.weapon[1] if player.weapon else 0
            upgrade = card[1] - current

            # Weapon is useful, but less important than immediate survival
            return self.weapon_priority + upgrade

        if is_monster(card):
            damage = self.immediate_damage(player, card)

            # Large penalty on damage
            utility = -self.damage_weight * damage

            # Extra penalty if this hit is dangerous relative to current health
            if player.health - damage <= self.avoid_health_buffer:
                utility -= 200

            # Slight preference for smaller monsters when damage ties
            utility -= card[1]

            return utility

        return -9999

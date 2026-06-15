from Engine import is_monster, is_weapon, is_potion, attack_actions


class WeaponAgent:
    def __init__(
        self,
        avoid_threshold=16,
        potion_priority=70,
        weapon_priority=120,
        damage_weight=80,
        weapon_use_threshold=4,
        small_monster_penalty=10,
        weapon_chain_bonus=80,
    ):
        self.avoid_threshold = avoid_threshold
        self.potion_priority = potion_priority
        self.weapon_priority = weapon_priority
        self.damage_weight = damage_weight
        self.weapon_use_threshold = weapon_use_threshold
        self.small_monster_penalty = small_monster_penalty
        self.weapon_chain_bonus = weapon_chain_bonus

    def choose_avoid(self, player, room, can_avoid):
        if not can_avoid:
            return False

        room_risk = self.room_risk(player, room)
        return room_risk >= self.avoid_threshold

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
        damage_saved = bare_damage - weapon_damage

        # Use weapon if it saves enough damage
        if damage_saved >= self.weapon_use_threshold:
            return "weapon"

        # Also use weapon on large monsters, even if threshold is not met
        if monster_card[1] >= 10:
            return "weapon"

        # Otherwise preserve the weapon chain
        return "bare"

    def immediate_damage(self, player, card):
        if is_potion(card) or is_weapon(card):
            return 0

        if player.weapon_use(card):
            return min(card[1], max(0, card[1] - player.weapon[1]))

        return card[1]

    def room_risk(self, player, room):
        """
        Estimate room danger from a weapon-preservation perspective.
        Higher = more dangerous.
        """
        risk = 0

        monster_values = [card[1] for card in room if is_monster(card)]
        n_monsters = len(monster_values)

        risk += n_monsters * 2
        risk += sum(v >= 10 for v in monster_values) * 3

        if player.weapon is None:
            risk += 3
        else:
            if player.weapon[1] <= 4:
                risk += 2

        # If weapon limit is already restrictive, strong monsters are more dangerous
        if player.weapon_limit is not None:
            for v in monster_values:
                if v > player.weapon_limit:
                    risk += 2

        return risk

    def card_utility(self, player, card):
        if is_weapon(card):
            current = player.weapon[1] if player.weapon else 0
            upgrade = card[1] - current
            return self.weapon_priority + 10 * upgrade

        if is_potion(card):
            if player.potion_used_this_turn:
                return 0

            heal = min(card[1], 20 - player.health)
            return self.potion_priority + 2 * heal

        if is_monster(card):
            return self.monster_utility(player, card)

        return -9999

    def monster_utility(self, player, monster_card):
        monster_value = monster_card[1]

        # Case 1: no weapon
        if player.weapon is None:
            return -self.damage_weight * monster_value

        # Case 2: weapon cannot be used
        if not player.weapon_use(monster_card):
            return -self.damage_weight * monster_value - 40

        # Case 3: weapon can be used
        weapon_damage = max(0, monster_value - player.weapon[1])
        damage_saved = monster_value - weapon_damage

        utility = -self.damage_weight * weapon_damage

        # Reward monsters that are high enough to justify using the weapon
        utility += self.weapon_chain_bonus * (monster_value / 14)

        # Penalize using weapon on very small monsters
        if monster_value <= 5:
            utility -= self.small_monster_penalty

        # Slight preference for larger legal monsters:
        # better to "spend" the weapon on stronger monsters than tiny ones
        utility += monster_value

        return utility

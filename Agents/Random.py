from Engine import attack_actions
import random

# Randomized agent
class RandomAgent:
    def __init__(self, seed=None):
        self.rng = random.Random(seed)

    def choose_avoid(self, player, room, can_avoid):
        if not can_avoid:
            return False
        return self.rng.choice([True, False])

    def choose_card(self, player, room, picks_done):
        return self.rng.randrange(len(room))

    def choose_action(self, player, monster_card):
        return self.rng.choice(attack_actions(player, monster_card))

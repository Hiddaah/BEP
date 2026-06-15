from Engine import describe_room, card_str, attack_actions

# Manual agent

class ManualAgent:
    def choose_avoid(self, player, room, can_avoid):
        print("\nCurrent room:")
        for line in describe_room(room):
            print(" ", line)
        if not can_avoid:
            return False
        while True:
            ans = input("Avoid room? (y/n): ").strip().lower()
            if ans in ("y", "yes"):
                return True
            if ans in ("n", "no"):
                return False

    def choose_card(self, player, room, picks_done):
        print(f"\nPick {picks_done + 1} of 3")
        print(player)
        for line in describe_room(room):
            print(" ", line)

        while True:
            try:
                idx = int(input("Choose card index: ").strip())
                if 0 <= idx < len(room):
                    return idx
            except ValueError:
                pass
            print("Invalid index.")

    def choose_action(self, player, monster_card):
        actions = attack_actions(player, monster_card)
        if actions == ["bare"]:
            print(f"Must fight {card_str(monster_card)} barehanded.")
            return "bare"

        while True:
            ans = input(
                f"Fight {card_str(monster_card)} with bare hands or weapon? (bare/weapon): "
            ).strip().lower()
            if ans in actions:
                return ans
            print(f"Invalid action. Allowed: {actions}")

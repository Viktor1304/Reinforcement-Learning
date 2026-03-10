from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import random


class BlackjackGame:
    @staticmethod
    def draw_card() -> int:
        card = random.randint(1, 13)
        # Face cards (11, 12, 13) are worth 10
        return min(card, 10)

    @staticmethod
    def sum_hand(hand: list[int]) -> tuple[int, bool]:
        """Returns the optimal sum of the hand and whether it has a usable ace."""
        usable_ace = 1 in hand and sum(hand) + 10 <= 21
        total = sum(hand) + 10 if usable_ace else sum(hand)
        return total, usable_ace

    @staticmethod
    def is_bust(hand: list[int]) -> bool:
        total, _ = BlackjackGame.sum_hand(hand)
        return total > 21

    @staticmethod
    def dealer_turn(dealer_hand: list[int]) -> list[int]:
        while True:
            total, _ = BlackjackGame.sum_hand(dealer_hand)
            if total >= 17:
                break
            dealer_hand.append(BlackjackGame.draw_card())
        return dealer_hand


def get_all_states():
    states = []
    # Player sum: 12 to 21. Dealer showing: 1 (Ace) to 10. Usable Ace: True/False
    for player_sum in range(4, 22):
        for dealer_card in range(1, 11):
            for usable_ace in [True, False]:
                states.append((player_sum, dealer_card, usable_ace))
    return states


def generate_episode(policy):
    """Generates a single episode using Exploring Starts."""
    # 1. Exploring Starts: Randomly pick initial state and action
    player_sum = random.randint(12, 21)
    dealer_card = random.randint(1, 10)
    usable_ace = random.choice([True, False])

    first_action = random.choice(["hit", "stick"])

    # Setup hands to match the exploring start
    player_hand = [player_sum - 11, 11] if usable_ace else [player_sum]
    dealer_hand = [dealer_card, BlackjackGame.draw_card()]

    episode = []

    # 2. Player's turn
    action = first_action
    while True:
        state = (player_sum, dealer_card, usable_ace)
        episode.append((state, action))

        if action == "stick":
            break

        # Player hits
        player_hand.append(BlackjackGame.draw_card())
        player_sum, usable_ace = BlackjackGame.sum_hand(player_hand)

        if player_sum > 21:
            break  # Bust

        action = policy[state]

    # 3. Determine Reward
    player_total, _ = BlackjackGame.sum_hand(player_hand)

    if player_total > 21:
        reward = -1  # Player busted
    else:
        # Dealer's turn
        dealer_hand = BlackjackGame.dealer_turn(dealer_hand)
        dealer_total, _ = BlackjackGame.sum_hand(dealer_hand)

        if dealer_total > 21:
            reward = 1  # Dealer busted
        elif player_total > dealer_total:
            reward = 1  # Player wins
        elif player_total < dealer_total:
            reward = -1  # Dealer wins
        else:
            reward = 0  # Draw

    return episode, reward


def train_mc_exploring_starts(num_episodes=1_000_000):
    all_states = get_all_states()

    # Initialize Q(s, a), Returns, and Policy
    Q = defaultdict(lambda: {"hit": 0.0, "stick": 0.0})
    returns_sum = defaultdict(lambda: {"hit": 0.0, "stick": 0.0})
    returns_count = defaultdict(lambda: {"hit": 0.0, "stick": 0.0})

    # Initial policy: stick if player_sum >= 20, else hit
    policy = {state: "stick" if state[0] >= 20 else "hit" for state in all_states}

    for i in range(1, num_episodes + 1):
        if i % 50000 == 0:
            print(f"Training episode {i}/{num_episodes}...")

        episode, reward = generate_episode(policy)

        # First-visit MC
        visited_state_actions = set()
        for state, action in episode:
            if (state, action) not in visited_state_actions:
                visited_state_actions.add((state, action))

                # Update Returns
                returns_sum[state][action] += reward
                returns_count[state][action] += 1

                # Update Q
                Q[state][action] = (
                    returns_sum[state][action] / returns_count[state][action]
                )

                # Improve Policy (Greedy update)
                if Q[state]["hit"] > Q[state]["stick"]:
                    policy[state] = "hit"
                else:
                    policy[state] = "stick"

    return Q, policy


def plot_value_function(Q, usable_ace=True):
    player_sums = np.arange(12, 22)
    dealer_cards = np.arange(1, 11)

    X, Y = np.meshgrid(dealer_cards, player_sums)
    Z = np.zeros_like(X, dtype=float)

    for i, ps in enumerate(player_sums):
        for j, dc in enumerate(dealer_cards):
            state = (ps, dc, usable_ace)
            # Value of a state is the max Q value for that state
            Z[i, j] = max(Q[state]["hit"], Q[state]["stick"]) if state in Q else 0

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor="none")

    ax.set_xlabel("Dealer Showing")
    ax.set_ylabel("Player Sum")
    ax.set_zlabel("State Value")
    ax.set_title(f"Optimal Value Function (Usable Ace = {usable_ace})")

    plt.show()


def main():
    print("Starting Monte Carlo Exploring Starts for Blackjack...")
    Q, policy = train_mc_exploring_starts()

    print("Training complete! Plotting results...")
    plot_value_function(Q, usable_ace=True)
    plot_value_function(Q, usable_ace=False)


if __name__ == "__main__":
    main()

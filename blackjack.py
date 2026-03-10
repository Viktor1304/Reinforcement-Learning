# Infinite deck of cards, ace can count as 1 or 11
# Usable ace - ace counts as 11 doesn't go bust


from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import random


class State:
    def __init__(
        self,
        player_sum: int = 0,
        dealer_card: int | None = None,
        usable_ace: bool = False,
    ) -> None:
        self.player_sum = player_sum
        self.dealer_card = dealer_card
        self.usable_ace = usable_ace

    @staticmethod
    def draw_card() -> int:
        card = random.randint(1, 13)
        return min(card, 10)

    @staticmethod
    def draw_hand() -> list[int]:
        return [State.draw_card(), State.draw_card()]

    @staticmethod
    def usable_ace(hand):
        return 1 in hand and sum(hand) + 10 <= 21

    @staticmethod
    def sum_hand(hand):
        if State.usable_ace(hand):
            return sum(hand) + 10
        return sum(hand)

    @staticmethod
    def is_bust(hand):
        return State.sum_hand(hand) > 21

    @staticmethod
    def score(hand):
        return 0 if State.is_bust(hand) else State.sum_hand(hand)

    @staticmethod
    def dealer_policy(dealer_hand):
        while State.sum_hand(dealer_hand) < 17:
            dealer_hand.append(State.draw_card())
        return dealer_hand

    def __eq__(  # type: ignore
        self, other: object
    ) -> bool:
        if not isinstance(other, State):
            return NotImplemented
        return (
            self.player_sum == other.player_sum
            and self.dealer_card == other.dealer_card
            and self.usable_ace == other.usable_ace
        )


def player_policy(player_sum):
    # Fixed policy: stick on 20 or 21
    return "stick" if player_sum >= 20 else "hit"


def generate_episode():
    episode = []
    rewards = []

    player_hand = State.draw_hand()
    dealer_hand = State.draw_hand()

    while True:
        player_sum = State.sum_hand(player_hand)
        dealer_card = dealer_hand[0]
        usable = State.usable_ace(player_hand)

        state = (player_sum, dealer_card, usable)
        episode.append(state)

        action = player_policy(player_sum)

        if action == "stick":
            break

        player_hand.append(State.draw_card())

        if State.is_bust(player_hand):
            rewards.append(-1)
            return episode, rewards

        rewards.append(0)

    dealer_hand = State.dealer_policy(dealer_hand)

    player_score = State.score(player_hand)
    dealer_score = State.score(dealer_hand)

    if dealer_score > 21 or player_score > dealer_score:
        rewards.append(1)
    elif player_score < dealer_score:
        rewards.append(-1)
    else:
        rewards.append(0)

    return episode, rewards


def monte_carlo_value_estimation(episodes=500000):
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)
    V = defaultdict(float)

    rewards_dict: dict[int, int] = {}

    for _ in range(episodes):
        episode, rewards = generate_episode()
        sum_rewards = sum(rewards)
        if sum_rewards not in rewards_dict:
            rewards_dict[sum_rewards] = 0
        rewards_dict[sum_rewards] += 1

        G = 0.0
        for t in range(len(episode) - 1, -1, -1):
            state = episode[t]
            reward = rewards[t]
            G += reward

            if state not in episode[:t]:  # First-visit MC
                returns_sum[state] += G
                returns_count[state] += 1
                V[state] = returns_sum[state] / returns_count[state]

    print("Reward distribution:", rewards_dict)

    return V


def plot_value_function(V, usable_ace=True):
    player_sums = np.arange(12, 22)
    dealer_cards = np.arange(1, 11)

    X, Y = np.meshgrid(dealer_cards, player_sums)
    Z = np.zeros_like(X, dtype=float)

    for i, ps in enumerate(player_sums):
        for j, dc in enumerate(dealer_cards):
            Z[i, j] = V.get((ps, dc, usable_ace), 0)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(X, Y, Z)

    ax.set_xlabel("Dealer Showing")
    ax.set_ylabel("Player Sum")
    ax.set_zlabel("State Value")
    ax.set_title(f"Blackjack State Value (Usable Ace = {usable_ace})")

    plt.show()


def main():
    V = monte_carlo_value_estimation(episodes=500000)

    plot_value_function(V, usable_ace=True)
    plot_value_function(V, usable_ace=False)


if __name__ == "__main__":
    main()

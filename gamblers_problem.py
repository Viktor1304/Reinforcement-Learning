import matplotlib.pyplot as plt

# Set a small threshold for convergence
theta = 1e-12


def value_iteration(p_h: float):
    # Initialize the value function for all states to zero
    V = [0.0] * 101

    sweep: int = 0

    while True:
        sweep += 1
        delta = 0.0
        for s in range(1, 100):
            v = V[s]
            action_values = []
            for bet in range(1, min(s, 100 - s) + 1):
                s_win = s + bet
                s_lose = s - bet

                reward_win = 1.0 if s_win == 100 else 0.0
                reward_lose = 0.0

                win_value = p_h * (reward_win + V[s_win])
                lose_value = (1 - p_h) * (reward_lose + V[s_lose])

                action_values.append(win_value + lose_value)
            V[s] = max(action_values) if action_values else 0.0
            delta = max(delta, abs(v - V[s]))
        if delta < theta:
            break

    print(f"Value iteration converged after {sweep} sweeps.")
    return V


def plot_state_value(V: list[float]):
    plt.figure(figsize=(10, 6))
    plt.plot(V)
    plt.title("State Value Function for Gambler's Problem")
    plt.xlabel("Capital")
    plt.ylabel("Value")
    plt.grid()
    plt.show()


def policy(V: list[float], p_h: float) -> list[int]:
    policy = [0] * 101
    for s in range(1, 100):
        action_values = []
        for bet in range(1, min(s, 100 - s) + 1):
            s_win = s + bet
            s_lose = s - bet

            reward_win = 1.0 if s_win == 100 else 0.0

            win_value = p_h * (reward_win + V[s_win])
            lose_value = (1 - p_h) * (V[s_lose])

            action_values.append((win_value + lose_value, bet))
        if action_values:
            policy[s] = max(action_values)[1]
    return policy


def plot_policy(policy: list[int]):
    plt.figure(figsize=(10, 6))
    plt.bar(range(101), policy)
    plt.title("Optimal Policy for Gambler's Problem")
    plt.xlabel("Capital")
    plt.ylabel("Optimal Bet")
    plt.grid()
    plt.show()


def main():
    p_h = 0.4  # Probability of winning
    V = value_iteration(p_h)
    plot_state_value(V[0:99])
    optimal_policy = policy(V, p_h)
    print("Optimal Policy:", optimal_policy)
    plot_policy(optimal_policy)


if __name__ == "__main__":
    main()

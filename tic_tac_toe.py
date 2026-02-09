import random
import numpy as np
import matplotlib.pyplot as plt

ALPHA = 0.9
EPSILON = 0.1
GAMMA = 0.9


def full_board(board_state: str) -> bool:
    return "0" not in board_state


def check_winner(board_state: str) -> int | None:
    for i in range(3):
        if (
            board_state[3 * i]
            == board_state[3 * i + 1]
            == board_state[3 * i + 2]
            != "0"
        ):
            return int(board_state[3 * i])
        if board_state[i] == board_state[i + 3] == board_state[i + 6] != "0":
            return int(board_state[i])

    if board_state[0] == board_state[4] == board_state[8] != "0":
        return int(board_state[0])
    if board_state[2] == board_state[4] == board_state[6] != "0":
        return int(board_state[2])

    if full_board(board_state):
        return -1  # draw

    return None  # game continues


def initialise_states() -> dict[str, float]:
    states = {}
    states["000000000"] = 0.5
    return states


def get_successive_positions(board: str, mark: str) -> list[str]:
    return [board[:i] + mark + board[i + 1 :] for i, c in enumerate(board) if c == "0"]


def simulate_move_opponent(board: str) -> str:
    empties = [i for i, c in enumerate(board) if c == "0"]
    if empties:
        i = random.choice(empties)
        board = board[:i] + "2" + board[i + 1 :]
    return board


def expected_value_after_opponent(
    board_after_x: str, states: dict[str, float]
) -> float:
    winner = check_winner(board_after_x)
    if winner == 1:
        return 1
    if winner == 2:
        return -1
    if winner == -1:
        return -1

    opp_positions = get_successive_positions(board_after_x, "2")
    values = []

    for b in opp_positions:
        winner = check_winner(b)
        if winner == 1:
            values.append(1)
        elif winner == 2:
            values.append(-1)
        elif winner == -1:
            values.append(-1)
        else:
            values.append(states.setdefault(b, 0.5))  # X turn again

    return sum(values) / len(values)


def simulate_move_agent(
    board: str, states: dict[str, float], alpha: float, epsilon: float
) -> str:
    actions = get_successive_positions(board, "1")

    if random.random() < epsilon:
        return random.choice(actions)

    best_val = -1
    best_board = board

    for a in actions:
        val = expected_value_after_opponent(a, states)
        if val > best_val:
            best_val = val
            best_board = a

    states.setdefault(board, 0.5)
    states[board] += alpha * (best_val - states[board])

    return best_board


def simulate_game(states: dict[str, float], alpha: float, epsilon: float) -> int:
    board = "000000000"

    while True:
        # X move
        board = simulate_move_agent(board, states, alpha, epsilon)
        w = check_winner(board)
        if w is not None:
            return w

        # O move (random)
        board = simulate_move_opponent(board)
        w = check_winner(board)
        if w is not None:
            return -1


def main():
    states = initialise_states()

    GAME_TEST = 1000
    GAME_COUNT = 100000

    alpha_arr: list[float] = np.linspace(0, ALPHA, GAME_COUNT)
    epsilon_arr: list[float] = np.linspace(0, EPSILON, GAME_COUNT)

    results = []
    wins_arr: list[int] = []
    losses_arr: list[int] = []

    for game in range(GAME_COUNT):
        alpha = alpha_arr[-game - 1]
        epsilon = epsilon_arr[-game - 1]
        simulate_game(states, alpha, epsilon)

        if game % GAME_TEST == 0:
            score = 0
            wins = 0
            losses = 0
            for _ in range(1000):
                result = simulate_game(states, alpha, epsilon)
                if result == 1:
                    wins += 1
                else:
                    losses += 1
                score += result
            results.append(score)
            wins_arr.append(wins)
            losses_arr.append(losses)

    plt.plot(range(0, GAME_COUNT, GAME_TEST), results)
    plt.xlabel("Number of Games")
    plt.ylabel("Score (X wins - O wins)")
    plt.title("Tic Tac Toe Learning Progress")
    plt.grid()
    plt.show()

    plt.plot(range(0, GAME_COUNT, GAME_TEST), wins_arr, label="X Wins")
    plt.plot(range(0, GAME_COUNT, GAME_TEST), losses_arr, label="O Wins")
    plt.xlabel("Number of Games")
    plt.ylabel("Number of Wins")
    plt.title("Tic Tac Toe Wins Over Time")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()

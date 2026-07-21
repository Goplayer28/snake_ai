"""Reinforcement-learning environment for the local SnakeGame backend."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from snake_game import SnakeGame


DIRECTIONS = ("UP", "RIGHT", "DOWN", "LEFT")
# SnakeGame's absolute action codes are UP=0, LEFT=1, RIGHT=2, DOWN=3.
ABSOLUTE_ACTIONS = {"UP": 0, "LEFT": 1, "RIGHT": 2, "DOWN": 3}
RELATIVE_ACTION_NAMES = ("STRAIGHT", "RIGHT", "LEFT")


@dataclass(frozen=True)
class RewardConfig:
    """A bounded lexicographic score-then-time reward.

    One extra food is worth 1.0. Across a whole episode, all time cost is less
    than 0.5 and the terminal cost is 0.25. Consequently, an episode with one
    more food always has a higher undiscounted return; for equal scores and the
    same terminal outcome, fewer steps always has a higher return.
    """

    food_reward: float = 1.0
    total_time_budget: float = 0.5
    terminal_cost: float = 0.25


class SnakeRLEnv:
    """Direct, pixel-free adapter over SnakeGame.reset/get_state/step."""

    def __init__(
        self,
        board_size: int = 12,
        seed: int = 0,
        max_episode_steps: int = 5000,
        max_steps_without_food: int | None = None,
        silent_mode: bool = True,
        reward_config: RewardConfig | None = None,
    ) -> None:
        if max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be positive")
        self.board_size = board_size
        self.max_episode_steps = max_episode_steps
        self.max_steps_without_food = (
            max_steps_without_food
            if max_steps_without_food is not None
            else board_size * board_size * 4
        )
        self.reward_config = reward_config or RewardConfig()
        self.game = SnakeGame(seed=seed, board_size=board_size, silent_mode=silent_mode)
        self.steps_without_food = 0
        self.episode_return = 0.0
        self.seen_states_since_food: set[tuple] = set()

    @property
    def step_cost(self) -> float:
        return self.reward_config.total_time_budget / self.max_episode_steps

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, int]:
        state = self.game.reset(seed=seed)
        self.steps_without_food = 0
        self.episode_return = 0.0
        self.seen_states_since_food = {self._state_fingerprint(state)}
        return self.encode_state(state)

    def step(
        self, relative_action: int
    ) -> tuple[tuple[np.ndarray, int], float, bool, dict]:
        if relative_action not in (0, 1, 2):
            raise ValueError("relative_action must be 0 (straight), 1 (right), or 2 (left)")

        direction_index = DIRECTIONS.index(self.game.direction)
        if relative_action == 1:
            direction_index = (direction_index + 1) % 4
        elif relative_action == 2:
            direction_index = (direction_index - 1) % 4
        absolute_action = ABSOLUTE_ACTIONS[DIRECTIONS[direction_index]]

        game_done, info = self.game.step(absolute_action)
        ate_food = bool(info["food_obtained"])
        self.steps_without_food = 0 if ate_food else self.steps_without_food + 1
        fingerprint = self._state_fingerprint(info)
        cycle_detected = not game_done and not ate_food and fingerprint in self.seen_states_since_food
        if ate_food:
            self.seen_states_since_food = {fingerprint}
        else:
            self.seen_states_since_food.add(fingerprint)

        if info["steps"] >= self.max_episode_steps:
            truncation_reason = "step_limit"
        elif self.steps_without_food >= self.max_steps_without_food:
            truncation_reason = "food_timeout"
        elif cycle_detected:
            truncation_reason = "cycle"
        else:
            truncation_reason = None
        truncated = truncation_reason is not None
        done = game_done or truncated

        reward = -self.step_cost
        if ate_food:
            reward += self.reward_config.food_reward
        if done:
            reward -= self.reward_config.terminal_cost
        self.episode_return += reward

        info = dict(info)
        info.update(
            {
                "terminated": game_done,
                "truncated": truncated and not game_done,
                "truncation_reason": truncation_reason if not game_done else None,
                "relative_action": RELATIVE_ACTION_NAMES[relative_action],
                "episode_return": self.episode_return,
            }
        )
        return self.encode_state(info), reward, done, info

    @staticmethod
    def _state_fingerprint(state: dict) -> tuple:
        food = None if state["food_pos"] is None else tuple(map(int, state["food_pos"]))
        return tuple(state["snake"]), food, state["direction"]

    def encode_state(self, state: dict | None = None) -> tuple[np.ndarray, int]:
        """Encode backend state as 3 board channels plus a direction index."""
        state = self.game.get_state() if state is None else state
        board = np.zeros((3, self.board_size, self.board_size), dtype=np.uint8)
        for row, col in state["snake"]:
            board[0, row, col] = 1
        head_row, head_col = state["snake"][0]
        board[1, head_row, head_col] = 1
        if state["food_pos"] is not None:
            food_row, food_col = map(int, state["food_pos"])
            board[2, food_row, food_col] = 1
        return board, DIRECTIONS.index(state["direction"])

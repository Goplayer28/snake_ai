"""Minimal CNN + PPO smoke test for the local Snake environment.

This module intentionally implements only enough PPO to validate the complete
data path: vectorized environment collection, GAE, a clipped PPO update, and a
temporary checkpoint round trip.  It is not a long-running training script.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn import functional as F

from snake_rl import SnakeRLEnv


State = tuple[np.ndarray, int]


@dataclass(frozen=True)
class PPOConfig:
    """Small, fixed configuration used by the smoke test."""

    board_size: int = 12
    num_envs: int = 2
    rollout_steps: int = 32
    max_episode_steps: int = 16
    gamma: float = 0.99
    gae_lambda: float = 0.95
    learning_rate: float = 2.5e-4
    clip_coef: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    update_epochs: int = 2
    minibatch_size: int = 32

    def validate(self) -> None:
        batch_size = self.num_envs * self.rollout_steps
        if self.board_size <= 0 or self.num_envs <= 0 or self.rollout_steps <= 0:
            raise ValueError("board_size, num_envs, and rollout_steps must be positive")
        if self.minibatch_size <= 0 or self.minibatch_size > batch_size:
            raise ValueError("minibatch_size must be in [1, rollout batch size]")
        if self.update_epochs <= 0:
            raise ValueError("update_epochs must be positive")


def choose_device(requested: str) -> torch.device:
    """Resolve auto/cpu/mps/cuda and fail clearly for unavailable devices."""

    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    device = torch.device(requested)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


class ActorCritic(nn.Module):
    """Two-layer CNN with shared features and separate actor/critic heads."""

    def __init__(self, board_size: int = 12, num_actions: int = 3) -> None:
        super().__init__()
        self.board_size = board_size
        self.num_actions = num_actions
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        encoded_size = 32 * board_size * board_size
        self.shared = nn.Sequential(
            nn.Linear(encoded_size + 4, 128),
            nn.ReLU(),
        )
        self.actor = nn.Linear(128, num_actions)
        self.critic = nn.Linear(128, 1)

    def forward(
        self, board: torch.Tensor, direction: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(board.float())
        direction_one_hot = F.one_hot(direction.long(), num_classes=4).float()
        features = self.shared(torch.cat((encoded, direction_one_hot), dim=1))
        return self.actor(features), self.critic(features).squeeze(-1)

    def get_action_and_value(
        self,
        board: torch.Tensor,
        direction: torch.Tensor,
        action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self(board, direction)
        distribution = Categorical(logits=logits)
        if action is None:
            action = distribution.sample()
        return action, distribution.log_prob(action), distribution.entropy(), value

    def get_value(self, board: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
        return self(board, direction)[1]


def simple_reward(info: dict) -> float:
    """Return +1 for food, -1 for a collision, and zero otherwise.

    A board-filling win also obtains food, so it receives +1 without a terminal
    penalty.  A time-limit truncation is not a collision and receives zero.
    """

    if bool(info.get("food_obtained", False)):
        return 1.0
    if bool(info.get("terminated", False)) and not bool(info.get("won", False)):
        return -1.0
    return 0.0


def stack_states(states: Sequence[State]) -> tuple[np.ndarray, np.ndarray]:
    boards = np.stack([state[0] for state in states], axis=0)
    directions = np.asarray([state[1] for state in states], dtype=np.int64)
    return boards, directions


@dataclass
class VectorStep:
    """One synchronous step from all environments."""

    states: list[State]
    transition_next_states: list[State]
    rewards: np.ndarray
    episode_ends: np.ndarray
    terminated: np.ndarray
    infos: list[dict]


class SyncSnakeVectorEnv:
    """Tiny synchronous vector runner with deterministic automatic resets."""

    def __init__(self, config: PPOConfig, seed: int) -> None:
        config.validate()
        self.config = config
        self.seed = seed
        self.envs = [
            SnakeRLEnv(
                board_size=config.board_size,
                seed=seed + index,
                max_episode_steps=config.max_episode_steps,
                silent_mode=True,
            )
            for index in range(config.num_envs)
        ]
        self.episode_counts = np.zeros(config.num_envs, dtype=np.int64)
        self.completed_episodes = 0
        self.states = self.reset()

    def _seed_for(self, env_index: int) -> int:
        return int(
            self.seed
            + env_index
            + self.episode_counts[env_index] * self.config.num_envs
        )

    def reset(self) -> list[State]:
        self.episode_counts.fill(0)
        self.completed_episodes = 0
        self.states = [
            env.reset(seed=self.seed + index) for index, env in enumerate(self.envs)
        ]
        return self.states

    def step(self, actions: Sequence[int]) -> VectorStep:
        if len(actions) != self.config.num_envs:
            raise ValueError("one action is required for each environment")

        next_states: list[State] = []
        transition_next_states: list[State] = []
        rewards = np.zeros(self.config.num_envs, dtype=np.float32)
        episode_ends = np.zeros(self.config.num_envs, dtype=np.bool_)
        terminated = np.zeros(self.config.num_envs, dtype=np.bool_)
        infos: list[dict] = []

        for index, (env, action) in enumerate(zip(self.envs, actions)):
            transition_state, _, done, info = env.step(int(action))
            transition_next_states.append(transition_state)
            rewards[index] = simple_reward(info)
            episode_ends[index] = done
            terminated[index] = bool(info["terminated"])
            infos.append(info)

            if done:
                self.completed_episodes += 1
                self.episode_counts[index] += 1
                next_states.append(env.reset(seed=self._seed_for(index)))
            else:
                next_states.append(transition_state)

        self.states = next_states
        return VectorStep(
            states=next_states,
            transition_next_states=transition_next_states,
            rewards=rewards,
            episode_ends=episode_ends,
            terminated=terminated,
            infos=infos,
        )


def states_to_tensors(
    states: Sequence[State], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    boards, directions = stack_states(states)
    return (
        torch.as_tensor(boards, dtype=torch.uint8, device=device),
        torch.as_tensor(directions, dtype=torch.long, device=device),
    )


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    episode_ends: torch.Tensor,
    terminated: torch.Tensor,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute GAE without propagating across reset boundaries.

    True terminal transitions never bootstrap.  Truncations bootstrap from the
    final observation, but their advantage recursion still stops at the reset.
    All tensors use the shape ``(rollout_steps, num_envs)``.
    """

    expected_shape = rewards.shape
    tensors = (values, next_values, episode_ends, terminated)
    if rewards.ndim != 2 or any(tensor.shape != expected_shape for tensor in tensors):
        raise ValueError("all GAE tensors must have the same two-dimensional shape")

    advantages = torch.zeros_like(rewards)
    last_advantage = torch.zeros(rewards.shape[1], device=rewards.device)
    for step in reversed(range(rewards.shape[0])):
        bootstrap_mask = (~terminated[step].bool()).float()
        continuation_mask = (~episode_ends[step].bool()).float()
        delta = (
            rewards[step]
            + gamma * bootstrap_mask * next_values[step]
            - values[step]
        )
        last_advantage = (
            delta + gamma * gae_lambda * continuation_mask * last_advantage
        )
        advantages[step] = last_advantage
    return advantages, advantages + values


@dataclass
class RolloutBatch:
    boards: torch.Tensor
    directions: torch.Tensor
    actions: torch.Tensor
    log_probs: torch.Tensor
    values: torch.Tensor
    rewards: torch.Tensor
    next_values: torch.Tensor
    episode_ends: torch.Tensor
    terminated: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor

    @property
    def sample_count(self) -> int:
        return int(self.actions.numel())

    def flattened(self) -> dict[str, torch.Tensor]:
        return {
            "boards": self.boards.flatten(0, 1),
            "directions": self.directions.flatten(),
            "actions": self.actions.flatten(),
            "log_probs": self.log_probs.flatten(),
            "values": self.values.flatten(),
            "advantages": self.advantages.flatten(),
            "returns": self.returns.flatten(),
        }


@torch.no_grad()
def collect_rollout(
    envs: SyncSnakeVectorEnv,
    network: ActorCritic,
    config: PPOConfig,
    device: torch.device,
) -> RolloutBatch:
    network.eval()
    boards_buffer: list[torch.Tensor] = []
    directions_buffer: list[torch.Tensor] = []
    actions_buffer: list[torch.Tensor] = []
    log_probs_buffer: list[torch.Tensor] = []
    values_buffer: list[torch.Tensor] = []
    rewards_buffer: list[torch.Tensor] = []
    next_values_buffer: list[torch.Tensor] = []
    episode_ends_buffer: list[torch.Tensor] = []
    terminated_buffer: list[torch.Tensor] = []

    for _ in range(config.rollout_steps):
        boards, directions = states_to_tensors(envs.states, device)
        actions, log_probs, _, values = network.get_action_and_value(boards, directions)
        result = envs.step(actions.cpu().tolist())
        next_boards, next_directions = states_to_tensors(
            result.transition_next_states, device
        )
        next_values = network.get_value(next_boards, next_directions)

        boards_buffer.append(boards)
        directions_buffer.append(directions)
        actions_buffer.append(actions)
        log_probs_buffer.append(log_probs)
        values_buffer.append(values)
        rewards_buffer.append(torch.as_tensor(result.rewards, device=device))
        next_values_buffer.append(next_values)
        episode_ends_buffer.append(
            torch.as_tensor(result.episode_ends, dtype=torch.bool, device=device)
        )
        terminated_buffer.append(
            torch.as_tensor(result.terminated, dtype=torch.bool, device=device)
        )

    rewards = torch.stack(rewards_buffer)
    values = torch.stack(values_buffer)
    next_values = torch.stack(next_values_buffer)
    episode_ends = torch.stack(episode_ends_buffer)
    terminated = torch.stack(terminated_buffer)
    advantages, returns = compute_gae(
        rewards,
        values,
        next_values,
        episode_ends,
        terminated,
        config.gamma,
        config.gae_lambda,
    )
    return RolloutBatch(
        boards=torch.stack(boards_buffer),
        directions=torch.stack(directions_buffer),
        actions=torch.stack(actions_buffer),
        log_probs=torch.stack(log_probs_buffer),
        values=values,
        rewards=rewards,
        next_values=next_values,
        episode_ends=episode_ends,
        terminated=terminated,
        advantages=advantages,
        returns=returns,
    )


def ppo_update(
    network: ActorCritic,
    optimizer: torch.optim.Optimizer,
    rollout: RolloutBatch,
    config: PPOConfig,
) -> dict[str, float]:
    """Run the small clipped PPO update and return averaged diagnostics."""

    network.train()
    batch = rollout.flattened()
    advantages = batch["advantages"]
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    metric_values: dict[str, list[float]] = {
        "actor_loss": [],
        "value_loss": [],
        "entropy": [],
        "total_loss": [],
        "grad_norm": [],
    }

    for _ in range(config.update_epochs):
        indices = torch.randperm(rollout.sample_count, device=advantages.device)
        for start in range(0, rollout.sample_count, config.minibatch_size):
            minibatch = indices[start : start + config.minibatch_size]
            _, new_log_prob, entropy, new_value = network.get_action_and_value(
                batch["boards"][minibatch],
                batch["directions"][minibatch],
                batch["actions"][minibatch],
            )
            log_ratio = new_log_prob - batch["log_probs"][minibatch]
            ratio = log_ratio.exp()
            mb_advantages = advantages[minibatch]
            unclipped_loss = -mb_advantages * ratio
            clipped_loss = -mb_advantages * torch.clamp(
                ratio, 1.0 - config.clip_coef, 1.0 + config.clip_coef
            )
            actor_loss = torch.maximum(unclipped_loss, clipped_loss).mean()
            value_loss = 0.5 * F.mse_loss(new_value, batch["returns"][minibatch])
            entropy_mean = entropy.mean()
            total_loss = (
                actor_loss
                + config.value_coef * value_loss
                - config.entropy_coef * entropy_mean
            )
            if not torch.isfinite(total_loss):
                raise RuntimeError("PPO update produced a non-finite loss")

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(
                network.parameters(), config.max_grad_norm
            )
            if not torch.isfinite(grad_norm):
                raise RuntimeError("PPO update produced a non-finite gradient")
            optimizer.step()

            metric_values["actor_loss"].append(float(actor_loss.detach().cpu()))
            metric_values["value_loss"].append(float(value_loss.detach().cpu()))
            metric_values["entropy"].append(float(entropy_mean.detach().cpu()))
            metric_values["total_loss"].append(float(total_loss.detach().cpu()))
            metric_values["grad_norm"].append(float(grad_norm.detach().cpu()))

    return {name: float(np.mean(values)) for name, values in metric_values.items()}


def save_smoke_checkpoint(path: Path, network: ActorCritic, config: PPOConfig) -> None:
    torch.save(
        {
            "algorithm": "minimal CNN PPO smoke test",
            "config": asdict(config),
            "model_state": network.state_dict(),
        },
        path,
    )


def load_smoke_checkpoint(
    path: Path, device: torch.device
) -> tuple[ActorCritic, dict]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    network = ActorCritic(board_size=int(config["board_size"])).to(device)
    network.load_state_dict(checkpoint["model_state"])
    network.eval()
    return network, checkpoint


def run_smoke_test(device_name: str = "auto", seed: int = 42) -> dict:
    """Exercise collection, one PPO update, and a checkpoint round trip."""

    config = PPOConfig()
    config.validate()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = choose_device(device_name)

    envs = SyncSnakeVectorEnv(config, seed)
    network = ActorCritic(config.board_size).to(device)
    optimizer = torch.optim.Adam(network.parameters(), lr=config.learning_rate)
    before_update = [parameter.detach().clone() for parameter in network.parameters()]

    rollout = collect_rollout(envs, network, config, device)
    metrics = ppo_update(network, optimizer, rollout, config)
    parameters_updated = any(
        not torch.equal(before, after.detach())
        for before, after in zip(before_update, network.parameters())
    )

    sample_boards, sample_directions = states_to_tensors(envs.states, device)
    network.eval()
    with torch.no_grad():
        expected_logits, expected_values = network(sample_boards, sample_directions)

    with tempfile.TemporaryDirectory(prefix="snake-ppo-smoke-") as temp_dir:
        checkpoint_path = Path(temp_dir) / "checkpoint.pt"
        save_smoke_checkpoint(checkpoint_path, network, config)
        loaded_network, checkpoint = load_smoke_checkpoint(checkpoint_path, device)
        with torch.no_grad():
            loaded_logits, loaded_values = loaded_network(
                sample_boards, sample_directions
            )
        save_reload_match = bool(
            checkpoint["algorithm"] == "minimal CNN PPO smoke test"
            and torch.equal(expected_logits, loaded_logits)
            and torch.equal(expected_values, loaded_values)
        )

    finite_metrics = all(math.isfinite(value) for value in metrics.values())
    success = bool(parameters_updated and save_reload_match and finite_metrics)
    return {
        "success": success,
        "device": str(device),
        "seed": seed,
        "num_envs": config.num_envs,
        "steps_per_env": config.rollout_steps,
        "samples": rollout.sample_count,
        "completed_episodes": envs.completed_episodes,
        "actor_loss": metrics["actor_loss"],
        "value_loss": metrics["value_loss"],
        "entropy": metrics["entropy"],
        "total_loss": metrics["total_loss"],
        "grad_norm": metrics["grad_norm"],
        "finite_metrics": finite_metrics,
        "parameters_updated": parameters_updated,
        "save_reload_match": save_reload_match,
    }


def smoke_test_command(args: argparse.Namespace) -> None:
    result = run_smoke_test(args.device, args.seed)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    if not result["success"]:
        raise SystemExit(1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    smoke_parser = subparsers.add_parser(
        "smoke-test", help="run one minimal CNN + PPO validation update"
    )
    smoke_parser.add_argument(
        "--device", default="auto", choices=("auto", "cpu", "mps", "cuda")
    )
    smoke_parser.add_argument("--seed", type=int, default=42)
    smoke_parser.set_defaults(func=smoke_test_command)
    return parser


if __name__ == "__main__":
    cli_args = build_parser().parse_args()
    cli_args.func(cli_args)

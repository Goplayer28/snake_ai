import unittest

import torch

from ppo.snake_ppo import (
    ActorCritic,
    PPOConfig,
    SyncSnakeVectorEnv,
    compute_gae,
    run_smoke_test,
    simple_reward,
)


class ActorCriticTests(unittest.TestCase):
    def test_network_shapes_and_probabilities(self):
        network = ActorCritic(board_size=12)
        boards = torch.zeros((5, 3, 12, 12), dtype=torch.uint8)
        directions = torch.tensor([0, 1, 2, 3, 0])
        logits, values = network(boards, directions)

        self.assertEqual(tuple(logits.shape), (5, 3))
        self.assertEqual(tuple(values.shape), (5,))
        probabilities = logits.softmax(dim=1)
        torch.testing.assert_close(probabilities.sum(dim=1), torch.ones(5))


class SimpleRewardTests(unittest.TestCase):
    def test_reward_mapping(self):
        self.assertEqual(simple_reward({"food_obtained": True}), 1.0)
        self.assertEqual(
            simple_reward(
                {"food_obtained": True, "terminated": True, "won": True}
            ),
            1.0,
        )
        self.assertEqual(
            simple_reward(
                {"food_obtained": False, "terminated": True, "won": False}
            ),
            -1.0,
        )
        self.assertEqual(
            simple_reward(
                {"food_obtained": False, "terminated": False, "truncated": True}
            ),
            0.0,
        )
        self.assertEqual(
            simple_reward({"food_obtained": False, "terminated": False}), 0.0
        )


class GAETests(unittest.TestCase):
    def test_terminal_does_not_bootstrap_or_cross_episode(self):
        rewards = torch.tensor([[1.0], [2.0]])
        values = torch.zeros_like(rewards)
        next_values = torch.tensor([[10.0], [20.0]])
        episode_ends = torch.tensor([[True], [False]])
        terminated = torch.tensor([[True], [False]])

        advantages, returns = compute_gae(
            rewards,
            values,
            next_values,
            episode_ends,
            terminated,
            gamma=0.5,
            gae_lambda=1.0,
        )

        expected = torch.tensor([[1.0], [12.0]])
        torch.testing.assert_close(advantages, expected)
        torch.testing.assert_close(returns, expected)

    def test_truncation_bootstraps_but_stops_recursion(self):
        rewards = torch.tensor([[0.0], [100.0]])
        values = torch.zeros_like(rewards)
        next_values = torch.tensor([[4.0], [0.0]])
        episode_ends = torch.tensor([[True], [False]])
        terminated = torch.tensor([[False], [False]])

        advantages, _ = compute_gae(
            rewards,
            values,
            next_values,
            episode_ends,
            terminated,
            gamma=0.5,
            gae_lambda=1.0,
        )

        self.assertEqual(float(advantages[0, 0]), 2.0)
        self.assertEqual(float(advantages[1, 0]), 100.0)


class VectorEnvironmentTests(unittest.TestCase):
    def test_parallel_step_truncates_and_auto_resets(self):
        config = PPOConfig(
            num_envs=2,
            rollout_steps=1,
            max_episode_steps=1,
            minibatch_size=2,
        )
        envs = SyncSnakeVectorEnv(config, seed=10)
        result = envs.step([0, 0])

        self.assertTrue(result.episode_ends.all())
        self.assertFalse(result.terminated.any())
        self.assertEqual(envs.completed_episodes, 2)
        self.assertEqual(len(result.states), 2)
        for board, direction in result.states:
            self.assertEqual(board.shape, (3, 12, 12))
            self.assertEqual(int(board[0].sum()), 3)
            self.assertIn(direction, range(4))


class PPOPipelineTests(unittest.TestCase):
    def test_cpu_smoke_pipeline(self):
        result = run_smoke_test(device_name="cpu", seed=42)

        self.assertTrue(result["success"])
        self.assertEqual(result["samples"], 64)
        self.assertTrue(result["finite_metrics"])
        self.assertTrue(result["parameters_updated"])
        self.assertTrue(result["save_reload_match"])


if __name__ == "__main__":
    unittest.main()

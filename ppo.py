"""
Proximal Policy Optimization (PPO) Implementation for CartPole Environment

This module implements PPO, an on-policy reinforcement learning algorithm, to solve
the CartPole-v1 environment from OpenAI Gymnasium. PPO is known for its stability
and reliability in training deep RL agents.

The implementation includes:
- A neural network architecture with shared features and separate policy/value heads
- Generalized Advantage Estimation (GAE) for computing advantages
- Clipped surrogate objective for stable policy updates
- Value function loss for better state value estimation
- Policy entropy tracking for monitoring exploration

Key Features:
- Shared neural network architecture for policy and value functions
- Experience collection using trajectory buffer
- PPO-specific objective function with clipping
- Visualization of training progress and policy entropy

Example usage:
    python ppo.py

The script will train the agent and generate training curves once solved.
"""

from typing import List, Tuple, Optional
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

# Hyperparameters
LEARNING_RATE: float = 5e-4  # Learning rate for Adam optimizer
DISCOUNT_FACTOR: float = 0.98  # Discount factor for future rewards
GAE_LAMBDA: float = 0.95  # Lambda parameter for GAE
EPSILON_CLIP: float = 0.1  # PPO clipping parameter
EPOCHS_PER_UPDATE: int = 3  # Number of epochs per policy update
TIMESTEPS_PER_BATCH: int = 20  # Number of timesteps per training batch
PRINT_FREQUENCY: int = 20  # Episodes between printing progress
MAX_EPISODES: int = 10_000  # Maximum number of training episodes


class PPO(nn.Module):
    """
    PPO agent implementation using PyTorch.
    
    This class implements both the policy and value networks using a shared
    feature extractor followed by separate heads. It handles experience
    collection, advantage computation, and policy updates.
    """
    
    def __init__(self) -> None:
        """
        Initialize the PPO agent with neural network architecture and optimizer.
        """
        super(PPO, self).__init__()
        self.trajectory_buffer: List[Tuple] = []

        # Neural network architecture
        self.shared_layer = nn.Linear(4, 256)  # Shared features
        self.policy_head = nn.Linear(256, 2)  # Policy output (2 actions)
        self.value_head = nn.Linear(256, 1)  # Value function
        self.optimizer = optim.Adam(self.parameters(), lr=LEARNING_RATE)

    def get_policy(self, state: torch.Tensor, softmax_dim: int = 0) -> torch.Tensor:
        """
        Compute policy distribution over actions for given state.

        Args:
            state: Current state observation
            softmax_dim: Dimension to apply softmax (default=0)

        Returns:
            Action probabilities as a tensor
        """
        features = F.relu(self.shared_layer(state))
        logits = self.policy_head(features)
        action_probs = F.softmax(logits, dim=softmax_dim)
        return action_probs

    def get_value(self, state: torch.Tensor) -> torch.Tensor:
        """
        Compute value function estimate for given state.

        Args:
            state: Current state observation

        Returns:
            Estimated state value as a tensor
        """
        features = F.relu(self.shared_layer(state))
        state_value = self.value_head(features)
        return state_value

    def store_transition(self, transition: Tuple) -> None:
        """
        Store a transition tuple in the trajectory buffer.

        Args:
            transition: Tuple of (state, action, reward, next_state, action_prob, done)
        """
        self.trajectory_buffer.append(transition)

    def prepare_batch(self) -> Tuple[torch.Tensor, ...]:
        """
        Prepare training batch from stored trajectories.

        Returns:
            Tuple of tensors containing batch data:
            (states, actions, rewards, next_states, dones, action_probs)
        """
        states: List = []
        actions: List[List[int]] = []
        rewards: List[List[float]] = []
        next_states: List = []
        action_probs: List[List[float]] = []
        dones: List[List[float]] = []

        for transition in self.trajectory_buffer:
            state, action, reward, next_state, action_prob, done = transition

            states.append(state)
            actions.append([action])
            rewards.append([reward])
            next_states.append(next_state)
            action_probs.append([action_prob])
            done_mask = 0.0 if done else 1.0
            dones.append([done_mask])

        batch_states = torch.tensor(states, dtype=torch.float)
        batch_actions = torch.tensor(actions)
        batch_rewards = torch.tensor(rewards)
        batch_next_states = torch.tensor(next_states, dtype=torch.float)
        batch_dones = torch.tensor(dones, dtype=torch.float)
        batch_action_probs = torch.tensor(action_probs)

        self.trajectory_buffer = []
        return (
            batch_states,
            batch_actions,
            batch_rewards,
            batch_next_states,
            batch_dones,
            batch_action_probs,
        )

    def train_net(self) -> None:
        """
        Update policy and value networks using PPO algorithm.
        
        Implements the core PPO training loop including:
        - GAE advantage computation
        - Policy ratio clipping
        - Value function updates
        - Combined loss optimization
        """
        (
            states,
            actions,
            rewards,
            next_states,
            done_masks,
            old_action_probs,
        ) = self.prepare_batch()

        for epoch in range(EPOCHS_PER_UPDATE):
            # Compute value targets and advantages
            value_targets = (
                rewards + DISCOUNT_FACTOR * self.get_value(next_states) * done_masks
            )
            value_error = value_targets - self.get_value(states)
            value_error = value_error.detach().numpy()

            # Compute Generalized Advantage Estimation (GAE)
            advantages: List[List[float]] = []
            gae: float = 0.0
            for delta in value_error[::-1]:
                gae = DISCOUNT_FACTOR * GAE_LAMBDA * gae + delta[0]
                advantages.append([gae])
            advantages.reverse()
            advantages = torch.tensor(advantages, dtype=torch.float)

            # Get current policy distribution
            current_policy = self.get_policy(states, softmax_dim=1)
            selected_action_probs = current_policy.gather(1, actions)

            # Compute probability ratio and clipped surrogate objective
            prob_ratio = torch.exp(
                torch.log(selected_action_probs) - torch.log(old_action_probs)
            )
            surrogate1 = prob_ratio * advantages
            surrogate2 = (
                torch.clamp(prob_ratio, 1 - EPSILON_CLIP, 1 + EPSILON_CLIP) * advantages
            )

            # Compute total loss (policy loss + value loss)
            policy_loss = -torch.min(surrogate1, surrogate2)
            value_loss = F.smooth_l1_loss(
                self.get_value(states), value_targets.detach()
            )
            total_loss = policy_loss + value_loss

            # Perform optimization step
            self.optimizer.zero_grad()
            total_loss.mean().backward()
            self.optimizer.step()


def main() -> None:
    """
    Main training loop for the PPO agent on CartPole environment.
    
    Handles:
    - Environment interaction
    - Experience collection
    - Training updates
    - Progress tracking and visualization
    """
    env = gym.make("CartPole-v1")
    agent = PPO()
    total_reward: float = 0.0
    print_interval: int = PRINT_FREQUENCY

    # Initialize tracking variables
    rewards_history: List[float] = []
    policy_entropy_history: List[float] = []
    avg_reward: float = 0.0

    for episode in range(MAX_EPISODES):
        state, _ = env.reset()
        episode_done: bool = False
        episode_reward: float = 0.0

        while not episode_done:
            for timestep in range(TIMESTEPS_PER_BATCH):
                action_probs = agent.get_policy(torch.from_numpy(state).float())
                action_distribution = Categorical(action_probs)
                action = action_distribution.sample().item()
                next_state, reward, episode_done, truncated, info = env.step(action)

                agent.store_transition(
                    (
                        state,
                        action,
                        reward / 100.0,  # Normalize rewards
                        next_state,
                        action_probs[action].item(),
                        episode_done,
                    )
                )
                state = next_state

                episode_reward += reward
                if episode_done:
                    break

            agent.train_net()

            # Track policy entropy
            with torch.no_grad():
                current_probs = agent.get_policy(torch.from_numpy(state).float())
                entropy = -torch.sum(current_probs * torch.log(current_probs + 1e-10))
                policy_entropy_history.append(entropy.item())

        total_reward += episode_reward
        rewards_history.append(episode_reward)

        if episode % print_interval == 0 and episode != 0:
            avg_reward = total_reward / print_interval
            print(f"Episode: {episode}, Average Reward: {avg_reward:.1f}")
            total_reward = 0.0

        if avg_reward > 495:
            print("Solved!")
            # Save model state dict
            torch.save(agent.state_dict(), "ppo_model.pt")

            # Plot average reward curve with min/max shaded region
            plt.figure(figsize=(10, 5))
            rewards_array: NDArray = np.array(rewards_history)
            window_size: int = 10
            min_rewards: List[float] = []
            max_rewards: List[float] = []
            mean_rewards: List[float] = []

            for i in range(len(rewards_array) - window_size + 1):
                window = rewards_array[i : i + window_size]
                min_rewards.append(np.min(window))
                max_rewards.append(np.max(window))
                mean_rewards.append(np.mean(window))

            x = range(window_size - 1, len(rewards_array))
            plt.fill_between(x, min_rewards, max_rewards, alpha=0.2, color="lightblue")
            plt.plot(
                x, mean_rewards, color="darkblue", linewidth=2, label="Mean Reward"
            )
            plt.title("PPO Training Progress")
            plt.xlabel("Episode")
            plt.ylabel("Reward")
            plt.grid(True)
            plt.legend()
            plt.savefig("training_rewards.png")
            plt.close()

            # Plot policy distribution over time
            plt.figure(figsize=(10, 5))
            plt.plot(policy_entropy_history, label="Policy Entropy")
            plt.title("Policy Distribution Over Time")
            plt.xlabel("Episode")
            plt.ylabel("Policy Entropy")
            plt.legend()
            plt.grid(True)
            plt.savefig("policy_entropy.png")
            plt.close()

            break

    env.close()


if __name__ == "__main__":
    main()

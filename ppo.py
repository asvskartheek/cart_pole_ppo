import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import numpy as np

# Hyperparameters
LEARNING_RATE = 5e-4
DISCOUNT_FACTOR = 0.98
GAE_LAMBDA = 0.95
EPSILON_CLIP = 0.1
EPOCHS_PER_UPDATE = 3
TIMESTEPS_PER_BATCH = 20
PRINT_FREQUENCY = 20
MAX_EPISODES = 10_000


class PPO(nn.Module):
    def __init__(self):
        super(PPO, self).__init__()
        self.trajectory_buffer = []

        self.shared_layer = nn.Linear(4, 256)
        self.policy_head = nn.Linear(256, 2)
        self.value_head = nn.Linear(256, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=LEARNING_RATE)

    def get_policy(self, state, softmax_dim=0):
        features = F.relu(self.shared_layer(state))
        logits = self.policy_head(features)
        action_probs = F.softmax(logits, dim=softmax_dim)
        return action_probs

    def get_value(self, state):
        features = F.relu(self.shared_layer(state))
        state_value = self.value_head(features)
        return state_value

    def store_transition(self, transition):
        self.trajectory_buffer.append(transition)

    def prepare_batch(self):
        states, actions, rewards, next_states, action_probs, dones = (
            [],
            [],
            [],
            [],
            [],
            [],
        )
        for transition in self.trajectory_buffer:
            state, action, reward, next_state, action_prob, done = transition

            states.append(state)
            actions.append([action])
            rewards.append([reward])
            next_states.append(next_state)
            action_probs.append([action_prob])
            done_mask = 0 if done else 1
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

    def train_net(self):
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
            advantages = []
            gae = 0.0
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


def main():
    env = gym.make("CartPole-v1")
    agent = PPO()
    total_reward = 0.0
    print_interval = PRINT_FREQUENCY

    # Initialize tracking variables
    rewards_history = []
    policy_entropy_history = []
    avg_reward = 0.0

    for episode in range(MAX_EPISODES):
        state, _ = env.reset()
        episode_done = False
        episode_reward = 0.0

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
                        reward / 100.0,
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
            rewards_array = np.array(rewards_history)
            window_size = 10
            min_rewards = []
            max_rewards = []
            mean_rewards = []

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

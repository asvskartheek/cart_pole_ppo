import pytest
import torch
import gymnasium as gym
from ppo import PPO

def test_ppo_initialization():
    env = gym.make('CartPole-v1')
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    
    agent = PPO(obs_dim, act_dim)
    assert isinstance(agent, PPO)
    assert agent.shared_layer.in_features == obs_dim
    assert agent.policy_head.out_features == act_dim
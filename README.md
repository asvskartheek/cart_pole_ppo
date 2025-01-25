# Proximal Policy Optimization (PPO) Implementation

A clean, type-annotated implementation of Proximal Policy Optimization (PPO) in PyTorch for OpenAI Gymnasium environments. This implementation focuses on clarity, stability, and reproducibility while maintaining strong performance across classic control tasks.

## Features

- 🧠 Neural network architecture with shared features and separate policy/value heads
- 📊 Generalized Advantage Estimation (GAE) for robust advantage computation
- 🎯 Clipped surrogate objective for stable policy updates
- 📈 Comprehensive training visualization including rewards and policy entropy
- 🔧 Highly configurable through command-line arguments
- ✅ Type-annotated codebase for better maintainability

## Supported Environments

- CartPole-v1
- LunarLander-v2
- Acrobot-v1
- MountainCar-v0

## Installation

1. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Basic usage with default parameters:
```bash
python ppo.py -e "CartPole-v1"
```

Customized training:
```bash
python ppo.py -e "LunarLander-v2" --lr 1e-4 --batch-size 32 --gamma 0.99
```

### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `-e`, `--env` | Environment name | CartPole-v1 |
| `--lr` | Learning rate | 5e-4 |
| `--gamma` | Discount factor | 0.98 |
| `--gae-lambda` | GAE lambda parameter | 0.95 |
| `--epsilon` | PPO clipping parameter | 0.1 |
| `--epochs` | Number of epochs per update | 3 |
| `--batch-size` | Timesteps per batch | 20 |
| `--max-episodes` | Maximum training episodes | 10000 |
| `--print-freq` | Progress print frequency | 20 |

## Implementation Details

The implementation includes several key components of modern PPO:

1. **Neural Network Architecture**
   - Shared feature extractor (256 units)
   - Separate policy and value heads
   - ReLU activation functions

2. **Training Process**
   - Experience collection with trajectory buffer
   - GAE advantage computation
   - Policy ratio clipping
   - Combined policy and value function loss
   - Policy entropy tracking

3. **Monitoring and Visualization**
   - Training progress curves
   - Policy entropy analysis
   - Min/max reward ranges
   - Automatic model saving upon solving

## Results

Upon solving an environment, the implementation generates:

1. Trained model weights (`ppo_model_{env_name}.pt`)
2. Training reward curves (`training_rewards_{env_name}.png`)
3. Policy entropy visualization (`policy_entropy_{env_name}.png`)

## Environment-Specific Solving Criteria

| Environment | Solving Threshold |
|-------------|------------------|
| CartPole-v1 | 495 |
| LunarLander-v2 | 200 |
| Acrobot-v1 | -100 |
| MountainCar-v0 | -110 |

## Requirements

Key dependencies:
- PyTorch 2.5.1
- Gymnasium 0.29.1
- NumPy 2.2.2
- Matplotlib 3.10.0

For a complete list, see `requirements.txt`.

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

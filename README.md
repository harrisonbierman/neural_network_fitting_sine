# Neural Network Sin Wave Fitting

A Python implementation of a multi-layer perceptron that learns to fit a sine wave without using backpropagation. The network uses random weight adjustments with an adaptive noise scale to find optimal parameters.

## Network Learning
![Network Learning](assets/model_only.gif)

## Features

- Three-layer neural network architecture
- ReLU activation functions
- Mean Squared Error loss
- Real-time visualization of fitting progress
- Adaptive noise scaling based on loss
- Performance tracking with loss history

## IMPORTANT

This project uses uv which is a dependency manager for python. All you need to do is install uv through pip and run with the command below in the usage section. 

## Requirements

- Python 3.x
- uv

## Installation

```bash
git clone [your-repository-url]
cd [repository-name]
pip install uv 
```

## Usage

Run the script:

```bash
uv run fit_sin.py
```

## The program will:
1. Generate training data points from a sine wave
2. Initialize a neural network with 3 dense layers
3. Display two real-time plots:
   - Sine wave fitting progress
   - Loss history over iterations

## Output of Loss over Iteration
![With Loss](assets/model_with_loss.gif)

## Network Architecture

- Input layer: 1 neuron
- Hidden layer 1: 32 neurons with ReLU activation
- Hidden layer 2: 32 neurons with ReLU activation
- Output layer: 1 neuron

## Hyperparameters

- `hidden_layer_neurons`: 32
- `noise_scale`: Adaptive (0.0001 to 0.01)
- `accept_worse`: Adaptive (1.0001 to 1.001)
- `mse_accuracy_error`: 0.01
- `iterations_per_frame`: 10000

## Training Method

Instead of traditional backpropagation, this implementation uses:
1. Random weight perturbations with adaptive noise scaling
2. Acceptance of changes that improve loss or are within tolerance
3. Parameter reversion when changes worsen performance beyond tolerance

## Performance Metrics

- Accuracy: Percentage of predictions within ±0.01 of target
- Loss: Mean squared error between predictions and targets
- Visual feedback through real-time plotting

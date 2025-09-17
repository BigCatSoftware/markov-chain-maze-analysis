# Markov Chain Maze Analysis

Statistical modeling and steady-state analysis of random walks in maze environments using Markov chains and probabilistic methods.

## Overview

This project implements a comprehensive system for analyzing random walk processes in grid-based environments. Using Markov chain theory, the system computes transition probabilities, evolves probability distributions over time, and determines steady-state convergence behavior for complex maze navigation scenarios.

## Key Features

- **Transition Matrix Computation**: Generates stochastic matrices representing movement probabilities between grid states
- **Multi-Step Evolution**: Computes n-step probability distributions using iterative matrix multiplication
- **Steady-State Analysis**: Implements power iteration algorithm for equilibrium distribution computation
- **Random Walk Simulation**: Performs Monte Carlo simulations of pathfinding to goal states
- **Performance Visualization**: Creates visual representations of successful navigation paths
- **Statistical Validation**: Comprehensive unit testing suite for mathematical correctness

## Mathematical Foundation

The system models maze navigation as a discrete-time Markov chain where:

- **States**: Grid coordinates (r,c) representing agent positions
- **Transition Probabilities**: Uniform distribution over valid adjacent moves
- **Steady-State**: Long-term equilibrium distribution π where π = πP
- **Convergence**: Power iteration method with configurable tolerance levels

## Applications

- System reliability analysis and performance optimization
- Operational research for multi-state systems
- Predictive modeling of random processes
- Mathematical validation of probabilistic algorithms

## Usage

### Basic Analysis
```python
# Initialize maze environment
runner = MarkovMazeRunner(size=10, loop_percent=0)

# Analyze state transition evolution over time
runner.analyze_state_transitions(n_steps=100)

# Simulate pathfinding to goal states
runner.simulate_pathfinding(attempts=20)

# Compute steady-state distribution
runner.compute_steady_state()
```

### Command Line Interface
```bash
# Run analysis with default 10x10 maze, no loops
python MazeRunner.py

# Custom maze size and loop density
python MazeRunner.py 15 50
```

### Parameters
- `size`: Maze dimensions (minimum 2x2)
- `loop_percent`: Maze complexity (0 or 50)

## Dependencies

```python
numpy>=1.21.0
pyamaze>=1.0
matplotlib>=3.5.0
networkx>=2.6
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/BigCatSoftware/markov-chain-maze-analysis.git
cd markov-chain-maze-analysis
```

2. Create virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate  # Windows
```

3. Install dependencies:
```bash
pip install numpy pyamaze matplotlib networkx
```

4. Run the analysis:
```bash
cd src
python MazeRunner.py 10 0
```

## Output

The system generates comprehensive analysis reports including:

- **Transition matrices** at regular time intervals
- **Probability distributions** and most likely agent positions
- **Successful navigation paths** with step counts
- **Steady-state distributions** with convergence metrics
- **Visual maze representations** showing pathfinding results

## Testing

Run the validation suite to verify mathematical correctness:

```bash
cd src
python test_markov.py
```

Tests validate:
- Stochastic matrix properties (rows sum to 1)
- Probability distribution evolution
- Steady-state convergence behavior

## Technical Implementation

- **Markov.py**: Core probabilistic modeling and matrix operations
- **MazeRunner.py**: Analysis orchestration and visualization
- **test_markov.py**: Mathematical validation and unit testing

The implementation uses explicit matrix multiplication for numerical stability and provides configurable parameters for convergence tolerance and iteration limits.

## License

This project is available under the MIT License.

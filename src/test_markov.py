"""
Unit tests for Markov chain implementation.
Validates transition matrix properties and distribution evolution.
"""

import numpy as np
from Markov import Markov


def create_open_grid(rows: int, cols: int) -> dict:
    """Create a fully connected grid for testing purposes."""
    grid = {}
    for r in range(1, rows + 1):
        for c in range(1, cols + 1):
            grid[(r, c)] = {"N": 1, "E": 1, "S": 1, "W": 1}
    return grid


def test_transition_matrix_validity():
    """Test that transition matrix rows sum to 1 (stochastic property)."""
    grid = create_open_grid(3, 3)
    markov_model = Markov(grid, 3, 3, goal=(3, 3))
    P = markov_model.transition_matrix
    row_sums = P.sum(axis=1)
    assert np.allclose(row_sums, 1.0), "Each row of transition matrix must sum to 1"


def test_distribution_evolution():
    """Test that probability distributions evolve correctly and remain valid."""
    grid = create_open_grid(2, 2)
    markov_model = Markov(grid, 2, 2, goal=(2, 2))
    start_state = 0  # Starting at coordinate (1,1)

    # Evolve distribution for 2 steps
    distribution = markov_model.get_distribution_after_n_steps(2, start_state=start_state)

    # Verify distribution properties
    assert np.isclose(distribution.sum(), 1.0), "Distribution must sum to 1"
    assert np.all(distribution >= -1e-12), "All probabilities must be non-negative"


def test_steady_state_convergence():
    """Test that steady-state computation converges to a valid distribution."""
    grid = create_open_grid(3, 3)
    markov_model = Markov(grid, 3, 3, goal=(3, 3))

    steady_state = markov_model.solve_steady_state(tol=1e-8, max_iter=1000)

    # Verify steady-state properties
    assert np.isclose(steady_state.sum(), 1.0), "Steady-state must sum to 1"
    assert np.all(steady_state >= 0), "All steady-state probabilities must be non-negative"


if __name__ == "__main__":
    test_transition_matrix_validity()
    test_distribution_evolution()
    test_steady_state_convergence()
    print("All tests passed - Markov chain implementation verified.")
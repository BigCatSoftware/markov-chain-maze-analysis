"""
Markov Chain Analysis for Maze Navigation

Implements probabilistic modeling of random walks in grid environments
with transition matrix computation and steady-state analysis.
"""

from typing import Dict, List, Tuple, Optional
import random
import numpy as np

Coord = Tuple[int, int]  # (row, col)

class Markov:
    """
    Markov chain model for analyzing random walks in grid environments.

    Builds transition matrices from maze structures and provides utilities
    to evolve probability distributions, compute P^n, and simulate random walks.
    """

    def __init__(self, grid: Dict[Coord, Dict[str, int]], rows: int, cols: int, goal: Coord,
                 max_steps: int = 10_000,) -> None:
        self.grid = grid
        self.rows = rows
        self.cols = cols
        self.goal = goal
        self.max_steps = max_steps

        # State indexing: (1..rows, 1..cols) -> 0..(rows*cols-1)
        self.state_to_coord: Dict[int, Coord] = {}
        self.coord_to_state: Dict[Coord, int] = {}
        s = 0
        for r in range(1, rows + 1):
            for c in range(1, cols + 1):
                self.state_to_coord[s] = (r, c)
                self.coord_to_state[(r, c)] = s
                s += 1
        self.num_states = rows * cols

        self.transition_matrix: np.ndarray = self._create_transition_matrix()
        self.current_matrix: Optional[np.ndarray] = None  # Cache for last P^n computation

    def _get_valid_moves(self, coord: Coord) -> List[Coord]:
        """
        Returns all neighbor coordinates reachable in one step from coord.
        Uses cell dictionary where N,E,S,W -> 1 indicates open passages.
        """
        r, c = coord
        cell = self.grid.get((r, c), {})
        moves: List[Coord] = []
        if cell.get("N", 0) == 1 and r > 1:
            moves.append((r - 1, c))
        if cell.get("E", 0) == 1 and c < self.cols:
            moves.append((r, c + 1))
        if cell.get("S", 0) == 1 and r < self.rows:
            moves.append((r + 1, c))
        if cell.get("W", 0) == 1 and c > 1:
            moves.append((r, c - 1))
        return moves

    def _create_transition_matrix(self) -> np.ndarray:
        """
        Create the (rows*cols) x (rows*cols) transition matrix P.

        For each state i, distribute probability equally among valid moves.
        If no moves are available, assign probability 1.0 to staying in place.
        """
        P = np.zeros((self.num_states, self.num_states), dtype=float)
        for i in range(self.num_states):
            coord = self.state_to_coord[i]
            nbrs = self._get_valid_moves(coord)
            if not nbrs:
                P[i, i] = 1.0
            else:
                p = 1.0 / len(nbrs)
                for nb in nbrs:
                    j = self.coord_to_state[nb]
                    P[i, j] = p
        return P

    def get_transition_matrix_at_step(self, n: int) -> np.ndarray:
        """
        Compute and return P^n using iterative matrix multiplication.

        Args:
            n: Number of time steps

        Returns:
            The n-step transition matrix P^n
        """
        if n == 0:
            Pn = np.eye(self.num_states)
        elif n == 1:
            Pn = self.transition_matrix.copy()
        else:
            # Use iterative matrix multiplication for numerical stability
            Pn = self.transition_matrix.copy()
            for _ in range(n - 1):
                Pn = np.matmul(Pn, self.transition_matrix)

        self.current_matrix = Pn
        return Pn

    def get_distribution_after_n_steps(
        self,
        n: int,
        start_state: Optional[int] = None,
        initial_distribution: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Evolve a probability distribution forward n steps.

        Args:
            n: Number of time steps
            start_state: Single starting state (creates one-hot distribution)
            initial_distribution: Custom initial probability distribution

        Returns:
            Probability distribution after n steps
        """
        if start_state is not None:
            dist = np.zeros(self.num_states, dtype=float)
            dist[start_state] = 1.0
        elif initial_distribution is not None:
            dist = np.array(initial_distribution, dtype=float)
            dist = dist / dist.sum()
        else:
            dist = np.full(self.num_states, 1.0 / self.num_states, dtype=float)

        # Evolve distribution using matrix multiplication
        for step in range(n):
            dist = np.matmul(dist, self.transition_matrix)

        # Cache P^n for analysis
        self.get_transition_matrix_at_step(n)
        return dist

    def get_distribution_after_n_steps_from_random_start(self, n: int) -> Tuple[np.ndarray, Coord]:
        """
        Start from a randomly selected location and evolve for n steps.

        Returns:
            Tuple of (final_distribution, random_starting_coordinate)
        """
        # Pick a random starting state
        start_state = random.randrange(self.num_states)
        start_coord = self.state_to_coord[start_state]

        # Evolve for n steps
        dist = self.get_distribution_after_n_steps(n, start_state=start_state)

        return dist, start_coord

    def simulate_path_to_goal(self, start: Optional[Coord] = None,
                              max_steps: Optional[int] = None,) -> Tuple[List[Coord], int]:
        """
        Simulate a random walk until reaching the goal or hitting step limit.

        Args:
            start: Starting coordinate (random if None)
            max_steps: Maximum steps before termination

        Returns:
            Tuple of (path_coordinates, step_count)
        """
        if max_steps is None:
            max_steps = self.max_steps

        if start is None:
            start_idx = random.randrange(self.num_states)
            pos = self.state_to_coord[start_idx]
        else:
            pos = start

        path: List[Coord] = [pos]
        steps = 0

        while steps < max_steps and pos != self.goal:
            nbrs = self._get_valid_moves(pos)
            if not nbrs:
                # Trapped - stay in place
                steps += 1
                path.append(pos)
                continue
            pos = random.choice(nbrs)
            path.append(pos)
            steps += 1

        return path, steps

    def solve_steady_state(self, tol: float = 1e-10, max_iter: int = 10000) -> np.ndarray:
        """
        Compute steady-state distribution using power iteration algorithm.

        Args:
            tol: Convergence tolerance (L1 norm)
            max_iter: Maximum iterations before termination

        Returns:
            Steady-state probability distribution
        """
        # Initialize with uniform distribution
        v = np.full(self.num_states, 1.0 / self.num_states)

        for iteration in range(max_iter):
            v_next = np.matmul(v, self.transition_matrix)

            # Check convergence using L1 norm
            if np.linalg.norm(v_next - v, ord=1) < tol:
                return v_next
            v = v_next

        return v

    def print_transition_matrix(self, matrix: np.ndarray, step: int, limit: int = 10) -> None:
        """
        Print a formatted view of the transition matrix.

        Args:
            matrix: Matrix to display
            step: Time step for header
            limit: Maximum dimensions to show
        """
        print(f"\nTransition matrix after {step} steps (showing {limit}x{limit}):")
        rmax, cmax = min(limit, matrix.shape[0]), min(limit, matrix.shape[1])
        for i in range(rmax):
            row = " ".join(f"{matrix[i, j]:.4f}" for j in range(cmax))
            print("  " + row)

    def format_distribution(self, dist: np.ndarray, thresh: float = 1e-3) -> str:
        """
        Format probability distribution for display.

        Args:
            dist: Probability distribution
            thresh: Minimum probability to display

        Returns:
            Formatted string of significant probabilities
        """
        lines: List[str] = ["State Probabilities:"]
        for i, p in enumerate(dist):
            if p >= thresh:
                lines.append(f"  {self.state_to_coord[i]}: {p:.6f}")
        return "\n".join(lines)
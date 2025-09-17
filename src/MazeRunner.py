"""
Markov Chain Maze Analysis System

Statistical modeling and visualization of random walk processes in maze environments.
Provides comprehensive analysis of state transitions, convergence behavior, and pathfinding.
"""

import sys
import random
from typing import Dict, List, Tuple, Optional

import numpy as np
import pyamaze as maze

from Markov import Markov

Coord = Tuple[int, int]


class MarkovMazeRunner:
    def __init__(self, size: int = 10, loop_percent: int = 0) -> None:
        self.size = size
        self.loop_percent = loop_percent

        # Build maze
        self.m = maze.maze(rows=size, cols=size)
        self.m.CreateMaze(loopPercent=loop_percent)  # pyamaze API

        # pyamaze stores cells in .maze_map: {(r,c): {'N':0/1,'E':0/1,'S':0/1,'W':0/1}}
        self.grid: Dict[Coord, Dict[str, int]] = self.m.maze_map

        # Random goal position for pathfinding analysis
        self.goal_pos: Coord = (random.randint(1, size), random.randint(1, size))

        # Markov model
        self.markov_model = Markov(self.grid, size, size, goal=self.goal_pos)

        # Analysis results storage
        self.transition_results: List[Dict] = []
        self.initial_start_coord: Optional[Coord] = None
        self.successful_paths: List[List[Coord]] = []
        self.path_step_counts: List[int] = []
        self.first_success_steps: Optional[int] = None
        self.goal_transition_matrix: Optional[np.ndarray] = None
        self.steady_state: Optional[np.ndarray] = None

    # ---------------- State Transition Analysis ----------------

    def analyze_state_transitions(self, n_steps: int = 100) -> None:
        """
        Analyze probability distributions and most likely positions over time.
        Computes state evolution from random starting positions at regular intervals.
        """
        print("\n" + "=" * 60)
        print("STATE TRANSITION ANALYSIS: Tracking position evolution over time")
        print("=" * 60)

        # Get distribution and starting location for each checkpoint
        for step in range(10, n_steps + 1, 10):
            dist, start_coord = self.markov_model.get_distribution_after_n_steps_from_random_start(step)
            Pn = self.markov_model.current_matrix

            # Store start coord from first iteration for reporting
            if step == 10:
                self.initial_start_coord = start_coord

            # Find most likely position
            argmax_state = int(np.argmax(dist))
            argmax_coord = self.markov_model.state_to_coord[argmax_state]
            argmax_prob = float(dist[argmax_state])

            print(f"\nStep {step}: Starting from {start_coord}, most likely at {argmax_coord} (prob={argmax_prob:.6f})")

            # Store results for output generation
            self.transition_results.append(
                {
                    "step": step,
                    "transition_matrix": Pn,
                    "start_coord": start_coord,
                    "most_likely_state": argmax_state,
                    "most_likely_coord": argmax_coord,
                    "probability": argmax_prob,
                }
            )

            # Print transition matrix
            self.markov_model.print_transition_matrix(Pn, step, limit=10)

    # ---------------- Pathfinding Simulation ----------------

    def simulate_pathfinding(self, attempts: int = 10) -> None:
        """
        Simulate random walk pathfinding to goal states.
        Analyzes convergence behavior and visualizes successful navigation paths.
        """
        self.successful_paths.clear()
        self.path_step_counts.clear()

        print("\n" + "=" * 60)
        print("PATHFINDING SIMULATION: Random-walk navigation to goal")
        print("=" * 60)

        successful_attempts = 0

        for attempt in range(attempts):
            path, steps = self.markov_model.simulate_path_to_goal()
            success = path[-1] == self.goal_pos
            status = "SUCCESS" if success else "FAILED"
            print(f"Attempt {attempt+1:>2}: {status} in {steps} steps; final={path[-1]}")

            if success:
                self.successful_paths.append(path)
                self.path_step_counts.append(steps)
                successful_attempts += 1

                # Record the first successful attempt's details for analysis
                if successful_attempts == 1:
                    self.first_success_steps = steps
                    # Compute transition matrix at the step when goal was reached
                    self.goal_transition_matrix = self.markov_model.get_transition_matrix_at_step(steps)
                    print(f"\n*** First success: Agent reached goal in {steps} steps ***")
                    print("Transition matrix when agent reaches goal state:")
                    self.markov_model.print_transition_matrix(self.goal_transition_matrix, steps, limit=10)

        if successful_attempts == 0:
            print("No successful paths found in the attempts.")
            return

        # Visualize up to 3 successful paths with footprints
        colors = [maze.COLOR.red, maze.COLOR.green, maze.COLOR.blue]
        trace_dict = {}

        for i, path in enumerate(self.successful_paths[:3]):
            agent = maze.agent(self.m, footprints=True, color=colors[i % len(colors)])
            trace_dict[agent] = path

        if trace_dict:
            self.m.tracePath(trace_dict, delay=100)

        # Mark goal cell with a static agent
        gx, gy = self.goal_pos
        goal_agent = maze.agent(self.m, x=gx, y=gy, color=maze.COLOR.yellow, footprints=True, filled=True)

        print(f"\nShowing {min(3, len(self.successful_paths))} successful paths in maze visualization.")
        self.m.run()

    # ---------------- Steady State Analysis ----------------

    def compute_steady_state(self, tol: float = 1e-10, max_iter: int = 10000) -> Optional[np.ndarray]:
        """
        Compute steady-state distribution using power iteration algorithm.
        Analyzes long-term convergence behavior of the Markov chain.
        """
        print("\n" + "=" * 60)
        print("STEADY-STATE ANALYSIS: Computing equilibrium distribution")
        print("=" * 60)

        try:
            self.steady_state = self.markov_model.solve_steady_state(tol, max_iter)
            print("Steady-state distribution computed successfully.")

            # Show top probability cells
            top_indices = np.argsort(self.steady_state)[-5:][::-1]  # top 5
            print("Top 5 cells by steady-state probability:")
            for idx in top_indices:
                coord = self.markov_model.state_to_coord[idx]
                prob = self.steady_state[idx]
                print(f"  {coord}: {prob:.6f}")

            return self.steady_state
        except Exception as e:
            print(f"Error computing steady state: {e}")
            return None

    def generate_analysis_report(self) -> str:
        """
        Generate comprehensive analysis report with transition matrices,
        pathfinding results, and steady-state distributions.
        """
        fname = f"analysis_results_size{self.size}_loop{self.loop_percent}.txt"
        with open(fname, "w") as f:
            f.write(f"Markov Chain Maze Analysis Results\n")
            f.write("=" * 50 + "\n")
            f.write(f"Maze configuration: {self.size}x{self.size} grid, {self.loop_percent}% loop density\n")
            f.write(f"Goal position: {self.goal_pos}\n\n")

            # State Transition Analysis
            f.write("STATE TRANSITION ANALYSIS\n")
            f.write("Probability evolution from random starting positions\n")
            f.write("-" * 50 + "\n")

            if self.initial_start_coord:
                f.write(f"Initial starting location: {self.initial_start_coord}\n\n")

            for result in self.transition_results:
                step = result["step"]
                Pn = result["transition_matrix"]
                f.write(f"Time step {step}:\n")
                rmax, cmax = min(10, Pn.shape[0]), min(10, Pn.shape[1])
                for i in range(rmax):
                    row = " ".join(f"{Pn[i, j]:.4f}" for j in range(cmax))
                    f.write("  " + row + "\n")
                f.write(f"Most probable cell: {result['most_likely_coord']} (probability={result['probability']:.6f})\n\n")

            # Pathfinding Analysis
            f.write("PATHFINDING SIMULATION RESULTS\n")
            f.write("-" * 50 + "\n")

            if self.path_step_counts:
                f.write(f"Steps to reach goal (first success): {self.first_success_steps}\n\n")

                # Transition matrix when agent reaches goal
                if self.goal_transition_matrix is not None:
                    f.write("Transition matrix at goal achievement (first 10x10):\n")
                    rmax, cmax = min(10, self.goal_transition_matrix.shape[0]), min(10, self.goal_transition_matrix.shape[1])
                    for i in range(rmax):
                        row = " ".join(f"{self.goal_transition_matrix[i, j]:.4f}" for j in range(cmax))
                        f.write("  " + row + "\n")
                    f.write("\n")

                # Successful navigation paths (up to 3)
                f.write("Successful navigation paths (up to 3 examples):\n")
                for i, (path, steps) in enumerate(
                    zip(self.successful_paths[:3], self.path_step_counts[:3]), start=1
                ):
                    f.write(f"Path {i} ({steps} steps): {path}\n")
            else:
                f.write("No successful navigation paths found.\n")

            # Steady-State Analysis
            if self.steady_state is not None:
                f.write(f"\nSTEADY-STATE DISTRIBUTION\n")
                f.write("Long-term equilibrium probabilities\n")
                f.write("-" * 50 + "\n")
                for i, prob in enumerate(self.steady_state):
                    if prob >= 1e-3:
                        coord = self.markov_model.state_to_coord[i]
                        f.write(f"  {coord}: {prob:.6f}\n")

        return fname


def parse_arguments(argv: List[str]) -> Tuple[int, int]:
    """Parse command line arguments for maze configuration."""
    if len(argv) == 1:
        return 10, 0  # defaults
    if len(argv) != 3:
        print("Usage: python MazeRunner.py [size] [loop_percent]")
        print("  size: int >= 2 (maze dimensions)")
        print("  loop_percent: 0 or 50 (maze complexity)")
        sys.exit(1)
    try:
        size = int(argv[1])
        loop_percent = int(argv[2])
        if size < 2 or loop_percent not in (0, 50):
            raise ValueError
        return size, loop_percent
    except Exception:
        print("Usage: python MazeRunner.py [size] [loop_percent]")
        print("  size: int >= 2 (maze dimensions)")
        print("  loop_percent: 0 or 50 (maze complexity)")
        sys.exit(1)


def main(argv: List[str]) -> None:
    """
    Main execution function for Markov chain maze analysis.
    Performs state transition analysis, pathfinding simulation, and steady-state computation.
    """
    # Set random seed for reproducible results if desired
    # random.seed(42)
    # np.random.seed(42)

    size, loop_percent = parse_arguments(argv)

    print(f"Initializing {size}x{size} maze with {loop_percent}% loop density...")
    print("Goal position will be randomly selected.")

    runner = MarkovMazeRunner(size=size, loop_percent=loop_percent)
    print(f"Goal position: {runner.goal_pos}")

    # Analysis 1: State transition evolution over time
    runner.analyze_state_transitions(n_steps=100)

    # Analysis 2: Random walk pathfinding to goal states
    runner.simulate_pathfinding(attempts=20)

    # Analysis 3: Steady-state distribution computation
    runner.compute_steady_state()

    # Generate comprehensive analysis report
    report_file = runner.generate_analysis_report()
    print(f"\nAnalysis complete. Results saved to: {report_file}")


if __name__ == "__main__":
    main(sys.argv)
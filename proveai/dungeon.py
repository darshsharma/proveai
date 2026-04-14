"""Dungeon grid generation.

Produces an 8x8 grid with obstacles, one key, and one door.
Agents are placed on random empty cells. Grid edges act as boundaries.
"""

from __future__ import annotations

import random
from collections import deque

from .state import AgentState, Cell, GameState, Grid, Message

SIZE = 8


def _empty_grid() -> list[list[Cell]]:
    return [[Cell.EMPTY for _ in range(SIZE)] for _ in range(SIZE)]


def _reachable(grid: list[list[Cell]], start: tuple[int, int]) -> set[tuple[int, int]]:
    """BFS over passable cells (not obstacles) from start."""
    visited: set[tuple[int, int]] = set()
    q: deque[tuple[int, int]] = deque([start])
    while q:
        r, c = q.popleft()
        if (r, c) in visited:
            continue
        visited.add((r, c))
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < SIZE and 0 <= nc < SIZE and grid[nr][nc] != Cell.OBSTACLE and (nr, nc) not in visited:
                q.append((nr, nc))
    return visited


def generate_dungeon(
    obstacle_density: float = 0.20,
    seed: int | None = None,
) -> GameState:
    """Generate a solvable dungeon and return the initial GameState."""
    rng = random.Random(seed)

    for _ in range(200):  # retry until solvable
        grid = _empty_grid()

        # Place obstacles
        obstacle_count = int(SIZE * SIZE * obstacle_density)
        all_coords = [(r, c) for r in range(SIZE) for c in range(SIZE)]
        rng.shuffle(all_coords)
        placed = 0
        for r, c in all_coords:
            if placed >= obstacle_count:
                break
            grid[r][c] = Cell.OBSTACLE
            placed += 1

        # Collect remaining empty cells
        empties = [(r, c) for r in range(SIZE) for c in range(SIZE) if grid[r][c] == Cell.EMPTY]
        if len(empties) < 4:  # need key, door, 2 agents minimum
            continue

        rng.shuffle(empties)
        key_pos = empties.pop()
        door_pos = empties.pop()
        agent0_pos = empties.pop()
        agent1_pos = empties.pop()

        grid[key_pos[0]][key_pos[1]] = Cell.KEY
        grid[door_pos[0]][door_pos[1]] = Cell.DOOR

        # Verify all four special positions are mutually reachable
        reachable = _reachable(grid, agent0_pos)
        if all(p in reachable for p in [agent1_pos, key_pos, door_pos]):
            frozen_grid: Grid = tuple(tuple(row) for row in grid)

            agents = {
                "agent_0": AgentState(agent_id="agent_0", position=agent0_pos),
                "agent_1": AgentState(agent_id="agent_1", position=agent1_pos),
            }

            return GameState(
                grid=frozen_grid,
                agents=agents,
                pending_messages=(),
                turn=0,
                current_agent_id="agent_0",
            )

    raise RuntimeError("Failed to generate a solvable dungeon after 200 attempts")


def render_grid(state: GameState) -> str:
    """Render the grid as a string for display. Agents shown as 0/1."""
    agent_positions = {a.position: a.agent_id for a in state.agents.values()}
    lines = []
    for r in range(SIZE):
        row_chars = []
        for c in range(SIZE):
            pos = (r, c)
            if pos in agent_positions:
                row_chars.append(agent_positions[pos][-1])  # '0' or '1'
            else:
                row_chars.append(state.grid[r][c].value)
        lines.append(" ".join(row_chars))
    return "\n".join(lines)

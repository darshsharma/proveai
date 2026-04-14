# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

ProveAI — a multi-agent dungeon-solving system with a legibility layer. Two LLM agents cooperate on an 8x8 grid to find a key and reach a door. Agents can only see adjacent cells and communicate via message passing. The system produces a full event trace so a human can understand what happened, why, and what should change.

## Running

```bash
python main.py                        # mock agents, JSONL traces only
python main.py --langfuse             # also send traces to Langfuse
python main.py --seed 7 --max-turns 150 --quiet
```

Core: pure Python 3.10+, no external deps. Optional: `pip install langfuse` for remote tracing.

## Architecture

- **`proveai/state.py`** — Frozen dataclasses (`GameState`, `AgentState`, `Cell`, `BehavioralState`). All state is immutable; updates produce new instances via `dataclasses.replace()`. `GameState` holds grid, agents dict, pending messages, turn counter.

- **`proveai/events.py`** — Append-only `EventBus`. Every tool call emits an `IntentPayload` (what the agent expected) and `ResolutionPayload` (what actually happened). When they diverge, `DRIFT_DETECTED` is auto-emitted with a `DriftPayload`.

- **`proveai/tools.py`** — Four agent tools: `move`, `observe`, `pickup`, `send_message`. Each returns a `ToolResult(state, text, success)` — new immutable state + human-readable output. All tool calls flow through `execute_tool()` dispatcher.

- **`proveai/legibility.py`** — Infers behavioral states (`EXPLORING_BLIND`, `NAVIGATING_TO_KEY`, `DELIVERING_KEY`, `STUCK_RECOVERY`, `BACKTRACKING`, etc.) from game state + event history. Provides `format_turn_summary()` and `format_game_summary()` for human-readable traces.

- **`proveai/game_loop.py`** — Central turn-based loop in `run_game()`. Alternates agents, delivers messages, executes one tool per turn, checks win condition (both agents at door + key held), updates behavioral states. Orchestrates all observability spans.

- **`proveai/agent.py`** — `BaseAgent` ABC with `decide()` method returning `ToolCall` (includes `DecisionMetrics` for LLM latency/tokens). `MockAgent` uses heuristics as placeholder.

- **`proveai/observability.py`** — Langfuse integration + structured JSONL records. `TracerFacade` wraps Langfuse with graceful no-op fallback. `StepRecordLogger` writes per-run JSONL files to `traces/`. `build_step_record()` constructs the full diagnostic schema. Langfuse hierarchy: `Trace(game-run) → Span(turn) → Span(decide) + Span(tool) + Span(behavioral-update) + Event(step-record)`.

## Key design constraints

- **Immutable state**: `GameState` is `frozen=True`. The game loop holds one state reference that is replaced each turn — never mutated.
- **One tool per turn**: Each agent calls exactly one tool on their turn.
- **Intent/Resolution duality**: Every tool call emits both payloads to the event bus before the state advances. Drift detection compares them automatically.
- **Message delivery is deferred**: `send_message` queues a message; it's delivered to the recipient at the start of their next turn.
- **Langfuse is optional**: `observability.py` does `try: from langfuse import Langfuse` inside `__init__`. If missing, `TracerFacade.active` is False and all methods are no-ops. JSONL logging always works.
- **Step records are always written**: Every turn produces a `StepRecord` with run_id, turn, agent_id, position, tool intent/result, behavioral state, distance_to_goal, exploration_coverage, partner_distance, etc. Files land in `traces/{run_id}.jsonl`.

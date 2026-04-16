"""Central game loop.

Maintains an immutable GameState that is replaced each turn. Orchestrates:
  - Agent turn order
  - Message delivery
  - Tool execution
  - Win/loss condition checks
  - Legibility updates (behavioral state inference, event emission)
  - Observability (Langfuse tracing + structured JSONL records)
"""

from __future__ import annotations

from .agent import BaseAgent
from .events import Event, EventBus, EventType
from .legibility import format_turn_summary, update_behavioral_state
from .observability import (
    SpanHandle,
    StepRecordLogger,
    TracerFacade,
    build_step_record,
    generate_run_id,
    write_run_summary,
)
from .report import generate_report
from .state import Cell, GameState
from .tools import execute_tool


def check_win_condition(state: GameState) -> tuple[bool, str]:
    """Check if both agents are at the door and one has the key.

    Returns (is_won, reason).
    """
    door_positions: list[tuple[int, int]] = []
    for r in range(len(state.grid)):
        for c in range(len(state.grid[0])):
            if state.grid[r][c] == Cell.DOOR:
                door_positions.append((r, c))

    if not door_positions:
        return False, ""

    door_pos = door_positions[0]

    agents_at_door = [
        aid for aid, a in state.agents.items() if a.position == door_pos
    ]
    any_has_key = any(state.agents[aid].has_key for aid in agents_at_door)

    if len(agents_at_door) == len(state.agents) and any_has_key:
        return True, "Both agents at door, key held — dungeon solved!"

    return False, ""


def _extract_map_metadata(state: GameState, seed: int | None) -> dict:
    """Build rich metadata dict from the initial game state for Langfuse."""
    grid = state.grid
    rows, cols = len(grid), len(grid[0])

    obstacles: list[list[int]] = []
    key_pos: list[int] | None = None
    door_pos: list[int] | None = None
    for r in range(rows):
        for c in range(cols):
            cell = grid[r][c]
            if cell == Cell.OBSTACLE:
                obstacles.append([r, c])
            elif cell == Cell.KEY:
                key_pos = [r, c]
            elif cell == Cell.DOOR:
                door_pos = [r, c]

    # Build a compact text grid for easy reading in the dashboard.
    grid_text = "\n".join(" ".join(cell.value for cell in row) for row in grid)

    return {
        "seed": seed,
        "max_turns": state.max_turns,
        "grid_rows": rows,
        "grid_cols": cols,
        "grid_text": grid_text,
        "agent_0_start": list(state.agents["agent_0"].position),
        "agent_1_start": list(state.agents["agent_1"].position),
        "key_position": key_pos,
        "door_position": door_pos,
        "obstacles": obstacles,
        "obstacle_count": len(obstacles),
        "agent_ids": sorted(state.agents.keys()),
    }


def run_game(
    initial_state: GameState,
    agents: dict[str, BaseAgent],
    bus: EventBus,
    verbose: bool = True,
    tracer: TracerFacade | None = None,
    run_id: str | None = None,
    record_logger: StepRecordLogger | None = None,
    seed: int | None = None,
) -> GameState:
    """Run the game loop until win, loss, or max turns.

    Returns the final GameState.
    """
    # --- Observability setup ---
    if run_id is None:
        run_id = generate_run_id()

    if tracer is None:
        tracer = TracerFacade(enabled=False)

    tracer.start_game_trace(run_id, _extract_map_metadata(initial_state, seed))

    jsonl_path = None
    if record_logger is not None:
        jsonl_path = record_logger.open(run_id)

    state = initial_state

    while not state.game_over and state.turn < state.max_turns:
        agent_id = state.current_agent_id
        agent_ctrl = agents[agent_id]
        agent_state = state.agents[agent_id]

        # --- Open turn span ---
        turn_span = tracer.start_turn_span(state.turn, agent_id, {
            "position": agent_state.position,
            "has_key": agent_state.has_key,
            "behavioral_state": agent_state.behavioral_state.value,
        })

        # --- Deliver pending messages ---
        state, delivered = state.pop_messages_for(agent_id)
        message_texts = [f"[From {m.sender} on turn {m.turn}]: {m.content}" for m in delivered]

        for m in delivered:
            bus.emit(Event(
                EventType.MESSAGE_DELIVERED,
                state.turn,
                agent_id,
                {"sender": m.sender, "content": m.content},
            ))

        # --- Deliver pending tool output from the agent's previous turn ---
        prev_tool_output = agent_state.pending_tool_output
        if prev_tool_output is not None:
            # Consume it — agent only sees each result once.
            state = state.with_agent(agent_id, pending_tool_output=None)
            agent_state = state.agents[agent_id]

        # --- Build observation context (physical facts only) ---
        r, c = agent_state.position
        current_cell = state.grid[r][c]
        at_door = current_cell == Cell.DOOR
        on_key = current_cell == Cell.KEY
        observation = (
            f"You are {agent_id} at ({r},{c}). "
            f"Current cell: '{current_cell.value}' "
            f"({'DOOR' if at_door else 'KEY' if on_key else 'EMPTY'}). "
            f"Has key: {agent_state.has_key}. "
            f"At door: {at_door}. "
            f"Turn: {state.turn}."
        )

        # --- Agent decides a tool call (LLM generation span) ---
        decide_span = tracer.start_decide_span(
            turn_span, agent_id,
            {
                "observation": observation,
                "messages": message_texts,
                "prev_tool_output": prev_tool_output,
            },
            model=getattr(agent_ctrl, "model_name", "mock"),
        )
        tool_call = agent_ctrl.decide(state, observation, message_texts, prev_tool_output)
        tracer.end_decide_span(decide_span, {
            "tool": tool_call.tool_name,
            "args": tool_call.tool_args,
        }, metrics=tool_call.metrics)

        # --- Execute the tool (tool span) ---
        tool_span = tracer.start_tool_span(turn_span, tool_call.tool_name, tool_call.tool_args)
        result = execute_tool(state, agent_id, tool_call.tool_name, tool_call.tool_args, bus)
        state = result.state
        # Store the tool's result so it reaches the agent on its NEXT turn
        # (async call/response semantics — observe output arrives one turn later).
        state = state.with_agent(agent_id, pending_tool_output=result.text)
        tracer.end_tool_span(tool_span, {
            "success": result.success,
            "text": result.text,
        })

        # --- Update behavioral state (legibility span) ---
        old_bs = state.agents[agent_id].behavioral_state.value
        behav_span = tracer.start_behavioral_span(turn_span)
        state = update_behavioral_state(state, agent_id, bus)
        new_bs = state.agents[agent_id].behavioral_state.value
        tracer.end_behavioral_span(behav_span, old_bs, new_bs)

        # --- Log structured step record ---
        record = build_step_record(
            run_id=run_id,
            trace_id=tracer.trace_id,
            turn=state.turn,
            agent_id=agent_id,
            agent_state=state.agents[agent_id],
            game_state=state,
            tool_call=tool_call,
            tool_result=result,
            messages=message_texts,
            bus=bus,
        )
        tracer.log_step_record(turn_span, record)
        if record_logger is not None:
            record_logger.log(record)

        # --- Close turn span ---
        turn_span.end()

        # --- Check win condition ---
        won, reason = check_win_condition(state)
        if won:
            bus.emit(Event(EventType.GAME_OVER, state.turn, agent_id, {"result": "WIN", "reason": reason}))
            state = state.set_win()

        # --- Verbose output ---
        if verbose:
            bs = state.agents[agent_id].behavioral_state.value
            print(format_turn_summary(state.turn, bus))
            print(f"  BEHAV   [{agent_id}] {bs}")
            print()

        # --- Advance turn ---
        state = state.next_turn()

    # --- Game over ---
    game_result = "WIN" if state.win else "LOSS"
    if not state.game_over:
        bus.emit(Event(EventType.GAME_OVER, state.turn, "system", {"result": "LOSS", "reason": "Max turns reached"}))
        state = state.set_loss("Max turns reached")
        if verbose:
            print(f"GAME OVER — max turns ({state.max_turns}) reached without solving the dungeon.")

    tracer.log_game_over({"result": game_result, "total_turns": state.turn, "win": state.win})
    tracer.end_game_trace({
        "result": game_result,
        "total_turns": state.turn,
        "agent_0_end": list(state.agents["agent_0"].position),
        "agent_1_end": list(state.agents["agent_1"].position),
        "agent_0_has_key": state.agents["agent_0"].has_key,
        "agent_1_has_key": state.agents["agent_1"].has_key,
    })
    tracer.score_game(state, bus)
    tracer.flush()

    if record_logger is not None:
        record_logger.close()
        if verbose and jsonl_path:
            print(f"Step records written to: {jsonl_path}")

    # Write the cross-turn summary (always, regardless of logger config).
    summary_path = write_run_summary(run_id, state, bus)
    if verbose:
        print(f"Run summary written to: {summary_path}")

    # Generate the post-game report via LLM.
    agent_model = "mock"
    for a in agents.values():
        if hasattr(a, "model_name") and a.model_name != "mock":
            agent_model = a.model_name
            break
    try:
        report_path = generate_report(
            run_id=run_id,
            seed=seed,
            final_state=state,
            bus=bus,
            model_name=agent_model,
        )
        if verbose:
            print(f"Post-game report written to: {report_path}")
            print()
            print(report_path.read_text())
    except Exception as exc:
        if verbose:
            print(f"Report generation failed: {exc}")

    return state

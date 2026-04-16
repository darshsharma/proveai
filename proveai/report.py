"""Post-game report generator.

Collects all tracked metrics (summary, scores, legibility, event log) into
a structured data bundle, sends it to Qwen via Groq for a natural-language
narrative, then writes the full report to traces/{run_id}.report.txt.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .events import EventBus
    from .state import GameState

GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "qwen/qwen3-32b"

REPORT_SYSTEM_PROMPT = """\
You are a game analyst writing a structured post-game report for a cooperative
multi-agent dungeon run. You will receive a JSON blob with all the metrics.

Write a report in EXACTLY this format (use plain text, box-drawing characters
for tables, tree characters for hierarchy). Do NOT use markdown headers (#),
do NOT use markdown tables (|---|), do NOT use code fences. Use plain text only.

FORMAT:

Trace: {run_id}

Metadata
  seed:           ...
  grid:           {rows} x {cols}
  agent_0_start:  [r, c]
  agent_1_start:  [r, c]
  key_position:   [r, c]
  door_position:  [r, c]
  obstacles:      N cells
  grid:
    (paste the grid_text exactly, indented 4 spaces)

Output (end of game)
  result:          WIN or LOSS
  total_turns:     N
  agent_0_end:     [r, c]    has_key: true/false    (annotate if at door)
  agent_1_end:     [r, c]    has_key: true/false    (annotate if at door)

Scores
  (draw a box table with columns: Score, Value, Type)
  Include ALL scores from the data.

Per-Agent Statistics
  (for each agent, show: total_moves, total_revisits, max_visits_on_single_cell,
   pickup_turn, moves_with_key)

Information Propagation
  Key:  picked up by {agent} on turn {N}, partner learned on turn {M} (delay: {D} turns)
  Door: first seen by {agent} on turn {N}, partner learned on turn {M} (delay: {D} turns)
  (use "never" if null)

Behavioral State Timeline
  (list each transition: Turn N [agent] STATE_A -> STATE_B)

Drift Summary
  Total drifts: N
  (per agent: count and types)

Span Hierarchy
  Trace: game-run
  +-- Span: turn-0-{agent}
  |    +-- Generation: decide-{agent}  (model: {model})
  |    +-- Span: tool-{name}
  |    +-- Span: behavioral-update
  |    +-- Event: step-record
  +-- ...
  +-- Event: game-over
  (show first 2 turns fully, then "..." for the rest, then game-over)

Analysis
  Write 3-5 sentences analyzing:
  - Did the agents cooperate effectively?
  - What was the main bottleneck (exploration, communication, navigation)?
  - How close did they get to winning?
  - What should change to improve the outcome?

The very last line of the report should be:
  Langfuse: https://us.cloud.langfuse.com -> project proveai -> Traces -> search {first 8 chars of run_id}

Do NOT write "End with:" literally. Just end the report with the Langfuse line.
"""


def _collect_report_data(
    run_id: str,
    seed: int | None,
    final_state: "GameState",
    bus: "EventBus",
    model_name: str,
) -> dict[str, Any]:
    """Gather all metrics into a single dict for the LLM prompt."""
    from .events import EventType
    from .state import Cell

    grid = final_state.grid
    rows, cols = len(grid), len(grid[0])

    # Map features
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

    grid_text = "\n".join(" ".join(cell.value for cell in row) for row in grid)

    # Scores
    key_pickups = [
        e for e in bus.events
        if e.event_type == EventType.TOOL_RESOLUTION
        and e.payload.tool_name == "pickup"
        and e.payload.success
    ]
    door_sighted_by: set[str] = set()
    for e in bus.events:
        if (e.event_type == EventType.TOOL_RESOLUTION
                and e.payload.tool_name == "observe"
                and e.payload.success):
            desc = e.payload.description
            if ": D" in desc or desc.startswith("D") or ", D" in desc:
                door_sighted_by.add(e.agent_id)

    door_tuple = tuple(door_pos) if door_pos else None
    agents_at_door_final = sum(
        1 for a in final_state.agents.values()
        if a.position == door_tuple
    )
    total_drifts = sum(
        1 for e in bus.events if e.event_type == EventType.DRIFT_DETECTED
    )
    total_messages = sum(
        1 for e in bus.events if e.event_type == EventType.MESSAGE_SENT
    )

    scores = [
        {"name": "success", "value": 1 if final_state.win else 0, "type": "BOOLEAN"},
        {"name": "key_found", "value": 1 if key_pickups else 0, "type": "BOOLEAN"},
        {"name": "door_found", "value": 1 if door_sighted_by else 0, "type": "BOOLEAN"},
        {"name": "agents_with_key_final", "value": sum(1 for a in final_state.agents.values() if a.has_key), "type": "NUMERIC"},
        {"name": "agents_at_door_final", "value": agents_at_door_final, "type": "NUMERIC"},
        {"name": "door_found_by_n_agents", "value": len(door_sighted_by), "type": "NUMERIC"},
        {"name": "total_turns", "value": final_state.turn, "type": "NUMERIC"},
        {"name": "total_drifts", "value": total_drifts, "type": "NUMERIC"},
        {"name": "total_messages_sent", "value": total_messages, "type": "NUMERIC"},
    ]

    # Per-agent stats
    per_agent: dict[str, Any] = {}
    for aid in sorted(final_state.agents.keys()):
        visits: dict[tuple[int, int], int] = {}
        total_moves = 0
        moves_with_key = 0
        pickup_turn: int | None = None
        for ev in bus.events:
            if ev.agent_id != aid or ev.event_type != EventType.TOOL_RESOLUTION:
                continue
            if not ev.payload.success:
                continue
            if ev.payload.tool_name == "pickup" and pickup_turn is None:
                pickup_turn = ev.turn
            if ev.payload.tool_name == "move":
                total_moves += 1
                pos = ev.payload.actual_position
                if pos is not None:
                    visits[pos] = visits.get(pos, 0) + 1
                if pickup_turn is not None and ev.turn >= pickup_turn:
                    moves_with_key += 1
        total_revisits = sum(v - 1 for v in visits.values() if v > 1)
        max_visits = max(visits.values()) if visits else 0
        per_agent[aid] = {
            "final_position": list(final_state.agents[aid].position),
            "has_key_final": final_state.agents[aid].has_key,
            "at_door_final": door_tuple is not None and final_state.agents[aid].position == door_tuple,
            "total_moves": total_moves,
            "total_revisits": total_revisits,
            "max_visits_on_single_cell": max_visits,
            "pickup_turn": pickup_turn,
            "moves_with_key": moves_with_key,
        }

    # Info propagation — key
    key_pickup_turn: int | None = None
    key_pickup_by: str | None = None
    for ev in bus.events:
        if (ev.event_type == EventType.TOOL_RESOLUTION
                and ev.payload.tool_name == "pickup"
                and ev.payload.success):
            key_pickup_turn = ev.turn
            key_pickup_by = ev.agent_id
            break
    partner_learned_key_turn: int | None = None
    if key_pickup_by is not None:
        for ev in bus.events:
            if (ev.event_type == EventType.MESSAGE_DELIVERED
                    and ev.agent_id != key_pickup_by
                    and ev.turn >= (key_pickup_turn or 0)):
                content = ev.payload.get("content", "").upper()
                if any(k in content for k in ("KEY", "PICKED", "GOT THE KEY", "HAVE THE KEY", "HAS KEY")):
                    partner_learned_key_turn = ev.turn
                    break

    # Info propagation — door
    first_door_sight_turn: int | None = None
    door_sighter_id: str | None = None
    for ev in bus.events:
        if (ev.event_type == EventType.TOOL_RESOLUTION
                and ev.payload.tool_name == "observe"
                and ev.payload.success):
            desc = ev.payload.description
            if ": D" in desc or desc.startswith("D") or ", D" in desc:
                first_door_sight_turn = ev.turn
                door_sighter_id = ev.agent_id
                break
    partner_learned_door_turn: int | None = None
    if door_sighter_id is not None:
        for ev in bus.events:
            if (ev.event_type == EventType.MESSAGE_DELIVERED
                    and ev.agent_id != door_sighter_id
                    and ev.turn >= (first_door_sight_turn or 0)
                    and "DOOR" in ev.payload.get("content", "").upper()):
                partner_learned_door_turn = ev.turn
                break

    # Behavioral state timeline
    transitions = []
    for ev in bus.events:
        if ev.event_type == EventType.STATE_TRANSITION:
            transitions.append({
                "turn": ev.turn,
                "agent_id": ev.agent_id,
                "from": ev.payload["from"],
                "to": ev.payload["to"],
            })

    # Drift per agent
    drift_details: dict[str, dict[str, int]] = {}
    for ev in bus.events:
        if ev.event_type == EventType.DRIFT_DETECTED:
            aid = ev.agent_id
            dtype = ev.payload.drift_type
            drift_details.setdefault(aid, {})
            drift_details[aid][dtype] = drift_details[aid].get(dtype, 0) + 1

    # Messages sent (for narrative context)
    messages_sent = []
    for ev in bus.events:
        if ev.event_type == EventType.MESSAGE_SENT:
            messages_sent.append({
                "turn": ev.turn,
                "sender": ev.agent_id,
                "recipient": ev.payload.get("recipient"),
                "content": ev.payload.get("content"),
            })

    # Initial agent positions — scan for earliest events per agent to get
    # start positions. Alternatively read from event bus turn-0.
    agent_0_start: list[int] | None = None
    agent_1_start: list[int] | None = None
    for ev in bus.events:
        if ev.event_type == EventType.TOOL_INTENT:
            desc = ev.payload.description
            if "from (" in desc or "at (" in desc:
                import re
                m = re.search(r"\((\d+),(\d+)\)", desc)
                if m:
                    pos = [int(m.group(1)), int(m.group(2))]
                    if ev.agent_id == "agent_0" and agent_0_start is None:
                        agent_0_start = pos
                    elif ev.agent_id == "agent_1" and agent_1_start is None:
                        agent_1_start = pos
        if agent_0_start and agent_1_start:
            break

    return {
        "run_id": run_id,
        "seed": seed,
        "grid_rows": rows,
        "grid_cols": cols,
        "grid_text": grid_text,
        "agent_0_start": agent_0_start,
        "agent_1_start": agent_1_start,
        "key_position": key_pos,
        "door_position": door_pos,
        "obstacle_count": len(obstacles),
        "result": "WIN" if final_state.win else "LOSS",
        "total_turns": final_state.turn,
        "max_turns": final_state.max_turns,
        "scores": scores,
        "per_agent": per_agent,
        "key_info_propagation": {
            "key_pickup_turn": key_pickup_turn,
            "key_pickup_by": key_pickup_by,
            "partner_learned_key_turn": partner_learned_key_turn,
            "key_info_delay_turns": (
                partner_learned_key_turn - key_pickup_turn
                if partner_learned_key_turn is not None and key_pickup_turn is not None
                else None
            ),
        },
        "door_info_propagation": {
            "first_door_sight_turn": first_door_sight_turn,
            "door_sighter": door_sighter_id,
            "partner_learned_door_turn": partner_learned_door_turn,
            "door_info_delay_turns": (
                partner_learned_door_turn - first_door_sight_turn
                if partner_learned_door_turn is not None and first_door_sight_turn is not None
                else None
            ),
        },
        "behavioral_transitions": transitions,
        "drift_details": drift_details,
        "messages_sent": messages_sent,
        "model_name": model_name,
    }


def _call_groq(api_key: str, system: str, user: str) -> str:
    """Call Qwen via Groq and return the assistant text."""
    import re as _re

    payload = json.dumps({
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.4,
        "max_tokens": 3000,
    }).encode("utf-8")

    req = urllib.request.Request(
        GROQ_ENDPOINT,
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "proveai/0.1",
            "Accept": "application/json",
        },
        method="POST",
    )

    max_attempts = 5
    for attempt in range(max_attempts):
        try:
            with urllib.request.urlopen(req, timeout=90) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")
            if e.code == 429 and attempt < max_attempts - 1:
                m = _re.search(r"try again in ([\d.]+)s", err_body)
                wait = float(m.group(1)) + 0.5 if m else (2 ** attempt)
                time.sleep(wait)
                continue
            raise RuntimeError(f"Groq HTTP {e.code}: {err_body}") from e
    raise RuntimeError("Groq: exhausted retries")


def generate_report(
    run_id: str,
    seed: int | None,
    final_state: "GameState",
    bus: "EventBus",
    model_name: str = "mock",
    output_dir: str | Path = "traces",
    api_key: str | None = None,
) -> Path:
    """Generate and write the post-game report. Returns path to .report.txt."""
    api_key = api_key or os.environ.get("GROQ_API_KEY", "")

    data = _collect_report_data(run_id, seed, final_state, bus, model_name)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / f"{run_id}.report.txt"

    if not api_key:
        # No API key — write raw JSON data as fallback.
        report_path.write_text(
            f"[No GROQ_API_KEY — raw report data]\n\n"
            + json.dumps(data, indent=2, default=str)
        )
        return report_path

    user_prompt = (
        "Here is the full game data as JSON. Write the report following "
        "the format in your system prompt EXACTLY.\n\n"
        + json.dumps(data, indent=2, default=str)
    )

    report_text = _call_groq(api_key, REPORT_SYSTEM_PROMPT, user_prompt)

    # Qwen3 may emit <think>...</think> reasoning tags; strip them.
    import re
    report_text = re.sub(r"<think>.*?</think>\s*", "", report_text, flags=re.DOTALL)
    # Remove literal "End with:" artifacts from prompt leakage.
    report_text = re.sub(r"\n\s*End with:\s*\n", "\n", report_text)

    report_path.write_text(report_text.strip() + "\n")
    return report_path

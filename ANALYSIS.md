
Used  (Qwen3-32B via Groq) as LLM agent to cooperate on an 8x8 grid dungeon
The agents must find a key, reach the door, and coordinate — all while being stateless, seeing only adjacent cells, and communicating through asynchronous messages. 
Central State Management:
Game state is immutable built on frozen Python dataclasses. A single `GameState` object holds the grid, both agent states, pending messages, and the turn counter.
`AgentState` tracks physical facts only: position, whether the agent holds the key, and a `pending_tool_output` buffer for async tool-result delivery. Agents are stateless — each LLM call receives a fresh prompt containing only current position, current cell contents, partner messages, and the previous tool's result.

Recording every tool call and change in state using a structured `StepRecord` is written to JSONL and Langfuse.

For Intent/Resolution Drift Detection:
Every tool call emits two payloads to an append-only event bus: an Intent Payload (what the agent expected) and a ResolutionPayload (what actually happened). When they diverge — a move blocked by an obstacle, a pickup on an empty cell — a `DRIFT_DETECTED` event fires automatically.


For Legibility Layer: 

Each agent is classified into one of eight behavioral states, inferred from game state and the event log — not from the agent's internal reasoning:

| State | Trigger |
|---|---|
| `EXPLORING_BLIND` | No key/door sighted yet, actively moving |
| `NAVIGATING_TO_KEY` | Agent has observed the key in a prior `observe` call |
| `NAVIGATING_TO_DOOR` | Agent has observed the door |
| `WAITING_AT_DOOR` | Agent is standing on the door cell |
| `GUIDING_PARTNER` | Recently sent a message while holding useful info |
| `STUCK_RECOVERY` | 3+ consecutive failed tool calls (drifts) |
| `BACKTRACKING` | Revisiting the same 2 positions in the last 4 moves |
| `UNKNOWN` | No pattern matched |


Also tracking:

Information Propagation Tracking

The system measures how quickly agents share discoveries:

- Key propagation delay: turns between one agent picking up the key and the partner receiving a message about it.
- Door propagation delay: turns between one agent first observing the door and the partner being informed.

These are computed from the event bus by matching `TOOL_RESOLUTION` (pickup/observe) events against subsequent `MESSAGE_DELIVERED` events containing relevant keywords.


Structured logging schema:

Every turn produces a `StepRecord` written to `traces/{run_id}.jsonl`:

- Identity: `run_id`, `turn_number`, `agent_id`, `trace_id`
- LLM Metrics: `llm_latency_ms`, `prompt_tokens`, `completion_tokens`, `total_tokens`
- Physical State: `agent_location_x/y`, `has_key`, `at_door`, `distance_to_goal`, `partner_distance`
- Tool Call: `intended_tool`, `intended_args`, `resolved_tool_output`, `tool_success`
- Cross-Turn: `position_revisit_count` (how many times this cell was visited before), `moves_with_key_count` (moves since pickup)
- Behavioral: `state` (current behavioral state), `consecutive_drift_count`
- Raw LLM Output: `llm_raw_content`, `llm_raw_tool_calls_json`

Output: 

Each run produces three files in `traces/`:

| File | Contents |
|---|---|
| `{run_id}.jsonl` | Per-turn StepRecords — the raw data for cross-run analysis |
| `{run_id}.summary.json` | Aggregated metrics: total revisits, moves with key, info propagation delays, per-agent statistics |
| `{run_id}.report.txt` | LLM-generated narrative report with scores table, behavioral timeline, span hierarchy, and analysis |

All data is also pushed to Langfuse*(when enabled) as a hierarchical trace: `game-run > turn > decide(generation) + tool(span) + behavioral-update(span) + step-record(event)`, with 9 scores attached (success, key_found, door_found, agents_at_door_final, total_drifts, total_messages_sent, etc.).

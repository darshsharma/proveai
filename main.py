"""Entry point — run a mock dungeon game and print the legibility trace."""

import argparse
from dataclasses import replace

from proveai.agent import MockAgent
from proveai.dungeon import generate_dungeon, render_grid
from proveai.events import EventBus
from proveai.game_loop import run_game
from proveai.legibility import format_game_summary
from proveai.observability import StepRecordLogger, TracerFacade


def main(
    seed: int = 42,
    max_turns: int = 80,
    verbose: bool = True,
    enable_langfuse: bool = False,
) -> None:
    bus = EventBus()
    tracer = TracerFacade(enabled=enable_langfuse)
    record_logger = StepRecordLogger(output_dir="traces")

    state = generate_dungeon(seed=seed)
    if verbose:
        print("=== INITIAL DUNGEON ===")
        print(render_grid(state))
        print()
        for aid, a in state.agents.items():
            print(f"  {aid} starts at {a.position}")
        if tracer.active:
            print("  Langfuse tracing: ENABLED")
        else:
            print("  Langfuse tracing: disabled (JSONL only)")
        print()

    agents = {
        "agent_0": MockAgent("agent_0", seed=seed),
        "agent_1": MockAgent("agent_1", seed=seed + 1),
    }

    if max_turns != 200:
        state = replace(state, max_turns=max_turns)

    final_state = run_game(
        state, agents, bus,
        verbose=verbose,
        tracer=tracer,
        record_logger=record_logger,
    )

    print()
    print(render_grid(final_state))
    print()
    print(format_game_summary(bus))
    print()
    if final_state.win:
        print("RESULT: WIN")
    else:
        print("RESULT: LOSS (max turns)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ProveAI dungeon solver")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-turns", type=int, default=80)
    parser.add_argument("--langfuse", action="store_true", help="Enable Langfuse tracing")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    main(
        seed=args.seed,
        max_turns=args.max_turns,
        verbose=not args.quiet,
        enable_langfuse=args.langfuse,
    )

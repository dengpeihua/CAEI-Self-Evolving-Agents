"""
Filter trajectory data for OPSD-INTERN training (strategy internalization).

Unlike filter_sft_data.py which keeps (strategy_prompt → strategy_response)
pairs from a single step, this script emits ONE SAMPLE PER STEP for the
"internalize strategy" OPSD objective:

    student_prompt   = curr_prompt                         (problem only)
    teacher_prompt   = curr_prompt + reference_strategy    (problem + privileged hint)
    output           = final_response                      (the action the agent took)

Each successful trajectory of N steps becomes N training samples, so we
multiply the dataset by the average trajectory length (~25-50x).

Usage:
    python filter_traj_intern_data.py \
        --traj_file 'logs/alfworld/*/traj_eval_in_distribution_*.json' \
        --output_file sft_data_intern.json
"""

import argparse
import json
import random
from glob import glob

import sys


def _truncate_trajectory_history(curr_prompt: str) -> str:
    """Strip the trailing admissible actions block from a long curr_prompt
    so the student sees the *current* state without prior trajectories.
    For student_prompt we just keep the original curr_prompt verbatim —
    this helper is currently unused but kept for reference.
    """
    return curr_prompt


def _build_teacher_prompt(curr_prompt: str, reference_strategy: str) -> str:
    """Append the reference strategy to the student's prompt so the teacher
    sees the privileged hint when computing its action distribution.

    The format mirrors the existing `action_prompt` convention (strategy
    prefixed with `Below are some strategy items...`) so the teacher's
    forward is consistent with the eval-time action generation template.
    """
    if not reference_strategy:
        return curr_prompt
    suffix = (
        "\n\nBelow is a reference strategy you should follow for this step "
        "(privileged hint — not available to the student at inference):\n"
        "=== Reference Strategy ===\n"
        f"{reference_strategy}\n"
        "=== End Reference Strategy ===\n"
    )
    return curr_prompt + suffix


def _build_student_prompt(curr_prompt: str) -> str:
    """Student sees ONLY the curr_prompt (no retrieved experiences,
    no strategy). This mirrors the inference-time setting where the model
    has to act on its own without an explicit strategy."""
    return curr_prompt


def main():
    parser = argparse.ArgumentParser(
        description="Filter trajectory data for OPSD-INTERN (strategy internalization)"
    )
    parser.add_argument("--traj_file", required=True,
                        help="Glob pattern for trajectory JSON file(s)")
    parser.add_argument("--output_file", default="data/sft_data_intern.json",
                        help="Output JSON for OPSD-INTERN samples")
    parser.add_argument("--require_action_change", action="store_true",
                        help="If set, only keep steps where the strategy actually "
                             "changed the action (final_action != initial_action)")
    parser.add_argument("--max_per_traj", type=int, default=None,
                        help="Cap steps per trajectory (default: keep all)")
    parser.add_argument("--shuffle", action="store_true",
                        help="Shuffle the output samples")
    args = parser.parse_args()

    data = []
    for infile in glob(args.traj_file):
        print(f"  loading: {infile}")
        with open(infile) as f:
            data.extend(json.load(f))
    print(f"loaded {len(data)} trajectories total")

    correct_trajectories = [t for t in data if t.get("won") is True]
    print(f"correct (won=True) trajectories: {len(correct_trajectories)}")

    sft_data = []
    rejected = {"no_curr_prompt": 0, "no_strategy": 0, "no_action": 0, "no_action_change": 0}

    for ti, traj in enumerate(correct_trajectories):
        steps = traj.get("steps", [])
        # Optionally cap
        if args.max_per_traj is not None:
            steps = steps[: args.max_per_traj]
        for si, step in enumerate(steps):
            curr_prompt = step.get("curr_prompt")
            strategy    = step.get("strategy")            # <strategy> bullets
            strategy_response = step.get("strategy_response")
            final_response    = step.get("final_response")
            final_action      = step.get("final_action")
            initial_action    = step.get("initial_action")

            if not curr_prompt:
                rejected["no_curr_prompt"] += 1
                continue
            if not (strategy and strategy.strip()):
                # Strategy can be empty for steps where the model decided no
                # refresh — we still keep them as "natural" student examples.
                pass
            if not (final_response and final_response.strip()):
                rejected["no_action"] += 1
                continue
            if args.require_action_change:
                if (final_action or "") == (initial_action or ""):
                    rejected["no_action_change"] += 1
                    continue

            student_prompt = _build_student_prompt(curr_prompt)
            teacher_prompt = _build_teacher_prompt(curr_prompt, strategy or "")

            sft_data.append({
                "instruction":      student_prompt,        # student input
                "teacher_prompt":   teacher_prompt,        # teacher input (privileged)
                "input":            "",
                "output":           final_response,        # what the agent emitted
                # extra metadata (kept for analysis; trainer ignores them)
                "_meta": {
                    "traj_idx":    ti,
                    "step_idx":    si,
                    "strategy":    strategy,
                    "final_action": final_action,
                },
            })

    if args.shuffle:
        random.shuffle(sft_data)

    with open(args.output_file, "w") as f:
        json.dump(sft_data, f, indent=2)

    print(f"\n=== summary ===")
    print(f"output samples:    {len(sft_data)}")
    print(f"rejected:          {rejected}")
    print(f"saved to:          {args.output_file}")
    if sft_data:
        avg_sp = sum(len(s["instruction"]) for s in sft_data) / len(sft_data)
        avg_tp = sum(len(s["teacher_prompt"]) for s in sft_data) / len(sft_data)
        avg_out = sum(len(s["output"]) for s in sft_data) / len(sft_data)
        print(f"avg student_prompt len:  {avg_sp:.0f} chars")
        print(f"avg teacher_prompt  len: {avg_tp:.0f} chars")
        print(f"avg output len:          {avg_out:.0f} chars")


if __name__ == "__main__":
    main()
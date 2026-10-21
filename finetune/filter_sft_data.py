import json
import random
import argparse
import importlib
from glob import glob


SUBSET_CNT = 1e9 # 10000


def _avg_len_by_qwen_tokens(items, tokenizer, field="output"):
    if not items:
        return 0.0
    return sum(len(tokenizer.encode(x[field])) for x in items) / len(items)


def _build_history_str(steps, current_step_idx, max_steps=8):
    """Build a human-readable history string from steps up to (but not including) current_step_idx."""
    history_items = []
    start = max(0, current_step_idx - max_steps)
    for i in range(start, current_step_idx):
        s = steps[i]
        obs = s.get("curr_prompt", "")[:200]
        act = s.get("final_action", "") or s.get("initial_action", "")
        history_items.append(f"Step {i}: [Obs] {obs}... -> [Action] {act}")
    return "\n".join(history_items) if history_items else "(no history yet)"


def _extract_task_description(prompt: str) -> str:
    """Extract the task goal line from an ALFWorld prompt."""
    if not prompt:
        return ""
    for line in prompt.split("\n"):
        line = line.strip()
        if line.startswith("Your task is to:") or line.startswith("Your task is"):
            return line
        if line.startswith("-= Welcome") or line.startswith("Welcome to"):
            continue
    # Fallback: first 3 non-empty lines
    lines = [l.strip() for l in prompt.split("\n") if l.strip()]
    return "\n".join(lines[:3]) if lines else ""


def main():
    parser = argparse.ArgumentParser(description="Filter SFT data from trajectory file")
    parser.add_argument("--traj_file", required=True, help="Path pattern to the trajectory JSON file")
    parser.add_argument(
        "--mode", required=True,
        choices=[
            "outcome-mft", "action_only", "strategy_and_action", "step-mft",
            "opsd-action", "opsd-reflection",
        ],
        help=(
            "outcome-mft: keep strategies from all successful trajectories. "
            "step-mft: keep only strategies that changed the agent action (process-level filter). "
            "action_only / strategy_and_action: include action generation targets. "
            "opsd-action: OPSD Stage-3a data — student sees task+state, teacher sees task+state+strategy+reflection (action internalization). "
            "opsd-reflection: OPSD Stage-3b data — student sees task+history, teacher sees task+history+reflection (reflection learning)."
        ),
    )
    parser.add_argument("--output_file", default="sft_data.json", help="Output JSON file for SFT data")
    parser.add_argument("--rejected_output_file", default=None, help="Optional JSON file to save rejected strategies in step-mft mode")
    parser.add_argument("--tokenizer_name", default="Qwen/Qwen3-4B-Thinking-2507", help="Qwen tokenizer name for length counting")
    parser.add_argument("--min_reflection_len", type=int, default=40,
                        help="Minimum reflection length (chars) to be considered valid (opsd-reflection mode)")
    args = parser.parse_args()

    transformers_module = importlib.import_module("transformers")
    tokenizer = transformers_module.AutoTokenizer.from_pretrained(args.tokenizer_name)

    # Load trajectory data
    data = []
    for infile in glob(args.traj_file):
        print(f"Loading data from {infile}")
        with open(infile, 'r') as f:
            data.extend(json.load(f))

    sft_data = []
    accepted_step_mft = []
    rejected_success_no_change = []
    rejected_failure = []

    # Filter correct trajectories
    correct_trajectories = [traj for traj in data if traj.get("won") == True]

    for traj in correct_trajectories:
        steps = traj.get("steps", [])
        for step in steps:
            if args.mode in ["outcome-mft", "strategy_and_action"]:
                strategy_prompt = step.get("strategy_prompt")
                strategy_response = step.get("strategy_response")
                if strategy_prompt and strategy_response:
                    sft_data.append({
                        "instruction": strategy_prompt,
                        "input": "",
                        "output": strategy_response
                    })

            if args.mode == "step-mft":
                final_action = step.get("final_action", "")
                initial_action = step.get("initial_action", "")
                strategy_prompt = step.get("strategy_prompt")
                strategy_response = step.get("strategy_response")
                if strategy_prompt and strategy_response:
                    item = {
                        "instruction": strategy_prompt,
                        "input": "",
                        "output": strategy_response
                    }
                    if final_action != initial_action and not (final_action.startswith("search[") and initial_action.startswith("search[")):
                        sft_data.append({
                            "instruction": strategy_prompt,
                            "input": "",
                            "output": strategy_response
                        })
                        accepted_step_mft.append(item)
                    else:
                        rejected_success_no_change.append(item)

            if args.mode in ["action_only", "strategy_and_action"]:
                action_prompt = step.get("action_prompt")
                final_response = step.get("final_response")
                if action_prompt and final_response:
                    sft_data.append({
                        "instruction": action_prompt,
                        "input": "",
                        "output": final_response
                    })

    # ---- OPSD Stage 3a: Action Internalization ----
    # Student prompt: basic task + state (no strategy, no reflection)
    # Teacher prompt: task + state + strategy + reflection (privileged hindsight)
    # Target output: the final action response
    # Filter: only successful trajectories, only steps where strategy changed the action
    if args.mode == "opsd-action":
        for traj in correct_trajectories:
            steps = traj.get("steps", [])
            for step_idx, step in enumerate(steps):
                final_action = step.get("final_action", "")
                initial_action = step.get("initial_action", "")
                action_prompt = step.get("action_prompt")
                final_response = step.get("final_response")
                strategy = step.get("strategy", "")
                reflection = (
                    step.get("reflection_response", "")
                    or step.get("refresh_response", "")
                    or step.get("reflection", "")
                    or ""
                )

                # Only include steps where strategy/reflection actually changed the action
                # (i.e. where the privileged info is useful — action advantage > 0)
                if not action_prompt or not final_response:
                    continue
                if final_action == initial_action and final_action and initial_action:
                    # Action didn't change — strategy was redundant, skip
                    if not (final_action.startswith("search[") and initial_action.startswith("search[")):
                        continue

                # Student sees only the basic action prompt (no strategy / reflection)
                # We construct a minimal prompt with just the task and observation
                curr_prompt = step.get("curr_prompt", "") or ""
                task_desc = _extract_task_description(curr_prompt)

                student_prompt = (
                    f"{task_desc}\n\n"
                    f"Current observation:\n{curr_prompt[:800]}\n\n"
                    f"You are an agent solving this task. Think step by step and then choose one admissible action."
                )

                # Teacher sees the full strategy + reflection (privileged hindsight)
                strategy_section = f"\n\nStrategy from past experience:\n{strategy}" if strategy else ""
                reflection_section = f"\n\nMetacognitive Reflection:\n{reflection}" if reflection else ""
                teacher_prompt = student_prompt + strategy_section + reflection_section

                sft_data.append({
                    "instruction": student_prompt,
                    "teacher_prompt": teacher_prompt,
                    "input": "",
                    "output": final_response,
                    "_meta": {
                        "traj_idx": traj.get("env_num", traj.get("task_id", -1)),
                        "step_idx": step_idx,
                        "initial_action": initial_action,
                        "final_action": final_action,
                        "action_changed": final_action != initial_action,
                        "has_strategy": bool(strategy),
                        "has_reflection": bool(reflection),
                    },
                })

    # ---- OPSD Stage 3b: Reflection Learning ----
    # Student prompt: task + history up to t-1 (learns WHEN and HOW to reflect)
    # Teacher prompt: task + history + high-quality reflection content (privileged)
    # Target output: the reflection content itself
    # Filter: only successful trajectories, only steps where reflection was triggered AND
    #         the reflection actually changed the action (action advantage > 0)
    if args.mode == "opsd-reflection":
        for traj in correct_trajectories:
            steps = traj.get("steps", [])
            for step_idx, step in enumerate(steps):
                reflection_prompt = (
                    step.get("reflection_prompt")
                    or step.get("refresh_prompt")
                    or step.get("strategy_prompt")
                    or ""
                )
                reflection_response = (
                    step.get("reflection_response")
                    or step.get("refresh_response")
                    or step.get("reflection")
                    or ""
                )
                final_action = step.get("final_action", "")
                initial_action = step.get("initial_action", "")
                refresh_decision = step.get("refresh_decision", False)

                # Must have valid reflection content
                if not reflection_response:
                    continue
                if len(reflection_response.strip()) < args.min_reflection_len:
                    continue

                # Action-advantage filter: only keep reflections that changed the action
                # (A(s, f_t) > 0 — reflection successfully corrected a wrong action)
                action_changed = (
                    final_action != initial_action
                    and not (final_action.startswith("search[") and initial_action.startswith("search["))
                )
                if not action_changed:
                    continue

                # Build student prompt: task description + history up to current step
                curr_prompt = step.get("curr_prompt", "") or ""
                task_desc = _extract_task_description(curr_prompt)
                history_str = _build_history_str(steps, step_idx, max_steps=8)

                student_prompt = (
                    f"{task_desc}\n\n"
                    f"Recent interaction history:\n{history_str}\n\n"
                    f"You are an agent and you have been hesitating or getting stuck. "
                    f"Perform a brief metacognitive reflection to diagnose what's going wrong "
                    f"and propose a concrete corrective step."
                )

                # Teacher sees the full reflection prompt (which includes retrieved
                # trajectories and other privileged context) as extra guidance.
                # Fallback: use strategy (retrieved from successful trajectories) as
                # privileged context when the original reflection prompt is unavailable.
                strategy = step.get("strategy", "") or ""
                if reflection_prompt:
                    teacher_prompt = reflection_prompt
                else:
                    strategy_section = (
                        f"\n\nStrategy from past successful trajectories:\n{strategy}"
                        if strategy else ""
                    )
                    teacher_prompt = student_prompt + strategy_section

                sft_data.append({
                    "instruction": student_prompt,
                    "teacher_prompt": teacher_prompt,
                    "input": "",
                    "output": reflection_response,
                    "_meta": {
                        "traj_idx": traj.get("env_num", traj.get("task_id", -1)),
                        "step_idx": step_idx,
                        "initial_action": initial_action,
                        "final_action": final_action,
                        "action_changed": True,
                        "refresh_decision": refresh_decision,
                    },
                })

    if args.mode == "step-mft":
        failed_trajectories = [traj for traj in data if traj.get("won") != True]
        for traj in failed_trajectories:
            steps = traj.get("steps", [])
            for step in steps:
                strategy_prompt = step.get("strategy_prompt")
                strategy_response = step.get("strategy_response")
                if strategy_prompt and strategy_response:
                    rejected_failure.append({
                        "instruction": strategy_prompt,
                        "input": "",
                        "output": strategy_response
                    })

    sft_data = random.sample(sft_data, min(int(SUBSET_CNT), len(sft_data)))
    # Save to output file
    with open(args.output_file, 'w') as f:
        json.dump(sft_data, f, indent=2)

    print(f"Total SFT items curated: {len(sft_data)}")
    print(f"Saved to {args.output_file}")

    if args.mode == "step-mft":
        rejected_all = rejected_success_no_change + rejected_failure
        print("\n[step-mft stats]")
        print(f"Accepted (success + action changed): {len(accepted_step_mft)}")
        print(f"Rejected (success but no action change): {len(rejected_success_no_change)}")
        print(f"Rejected (all failure trajectories): {len(rejected_failure)}")
        print(f"Rejected (all): {len(rejected_all)}")

        accepted_avg_tokens = _avg_len_by_qwen_tokens(accepted_step_mft, tokenizer)
        rejected_avg_tokens = _avg_len_by_qwen_tokens(rejected_all, tokenizer)
        print(f"Accepted avg output length (Qwen tokens): {accepted_avg_tokens:.2f}")
        print(f"Rejected avg output length (Qwen tokens): {rejected_avg_tokens:.2f}")

        if accepted_avg_tokens > rejected_avg_tokens:
            print("Accepted strategies are longer on average (by Qwen tokens).")
        else:
            print("Accepted strategies are NOT longer on average (by Qwen tokens).")

        if args.rejected_output_file:
            with open(args.rejected_output_file, 'w') as f:
                json.dump({
                    "rejected_success_no_change": rejected_success_no_change,
                    "rejected_failure": rejected_failure,
                    "rejected_all": rejected_all
                }, f, indent=2)
            print(f"Saved rejected strategies to {args.rejected_output_file}")

    elif args.mode in ["opsd-action", "opsd-reflection"]:
        with_action_change = sum(1 for d in sft_data if d.get("_meta", {}).get("action_changed"))
        with_strategy = sum(1 for d in sft_data if d.get("_meta", {}).get("has_strategy"))
        with_reflection = sum(1 for d in sft_data if d.get("_meta", {}).get("has_reflection"))
        avg_out = _avg_len_by_qwen_tokens(sft_data, tokenizer, field="output")
        avg_instr = _avg_len_by_qwen_tokens(sft_data, tokenizer, field="instruction")
        avg_teacher = _avg_len_by_qwen_tokens(sft_data, tokenizer, field="teacher_prompt")

        print(f"\n[{args.mode} stats]")
        print(f"  Total samples: {len(sft_data)}")
        print(f"  With action change: {with_action_change}")
        print(f"  With strategy (teacher): {with_strategy}")
        print(f"  With reflection (teacher): {with_reflection}")
        print(f"  Avg instruction tokens (student prompt): {avg_instr:.1f}")
        print(f"  Avg teacher prompt tokens: {avg_teacher:.1f}")
        print(f"  Avg output tokens: {avg_out:.1f}")

if __name__ == "__main__":
    main()

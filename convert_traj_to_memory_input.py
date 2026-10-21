"""
Convert raw trajectory JSON to the input format expected by `skills/generate_memory_alfworld.py`.

Input (`--input_file`): a CAEI trajectory dump, e.g.
    logs/.../traj_eval_in_distribution.json

Output (`--output_file`): JSON list of entries:
    [
        {
            "env_id":   "env_<n>",
            "task":     "put some candle on toilet",   # extracted from curr_prompt
            "type":     "all_success" | "failure",
            "trajectories": [
                [
                    {"step_id": <int>,
                     "action":  <first action taken at this step>,
                     "observation": <observation text>,
                     "reward":  1.0 | 0.0,
                     "done":    <True | False>},
                    ...
                ]
            ]
        },
        ...
    ]

Usage:
    python convert_traj_to_memory_input.py \\
        --input_file  logs/.../traj_eval_in_distribution.json \\
        --output_file skills/processed_trajectories.json
"""

import argparse
import json
import re
from collections import defaultdict


_TASK_RE   = re.compile(r"Your task is to:\s*(.+?)\.")
_OBS_START = re.compile(r"Your current observation is:")


def extract_task(curr_prompt: str) -> str:
    """Return the raw task description (e.g. 'put some candle on toilet')"""
    m = _TASK_RE.search(curr_prompt)
    return m.group(1).strip() if m else ""


def extract_observation(curr_prompt: str) -> str:
    """Return the 'your current observation is: ...' block — the part the
    agent saw before choosing the action."""
    m = _OBS_START.search(curr_prompt)
    if not m:
        return ""
    tail = curr_prompt[m.end():].lstrip()
    # Stop at "Your task is to:" or "Your admissible actions"
    for stop in ("\nYour task is to:", "\nYour admissible actions", "\nNow it's your turn"):
        idx = tail.find(stop)
        if idx >= 0:
            tail = tail[:idx]
    return tail.strip()


def convert_traj_file(input_path: str, output_path: str) -> None:
    with open(input_path) as f:
        trajs = json.load(f)

    # Group trajectories per env_num — same env may have multiple trajectories.
    by_env: dict = defaultdict(list)
    for traj in trajs:
        if not traj.get("steps"):
            continue
        env_num = traj["steps"][0].get("env_num", 0)
        by_env[env_num].append(traj)

    entries = []
    for env_num in sorted(by_env.keys()):
        env_trajs = by_env[env_num]

        # All trajectories in this env share the SAME task (it's the same env),
        # so we pull the task from the first one.
        first_task = extract_task(env_trajs[0]["steps"][0]["curr_prompt"])

        trajectories = []
        all_won = True
        for traj in env_trajs:
            if not traj.get("won"):
                all_won = False
            steps_out = []
            for s in traj["steps"]:
                steps_out.append({
                    "step_id": s.get("step_idx", 0),
                    "action":  s.get("curr_action", ""),
                    "observation": extract_observation(s.get("curr_prompt", "")),
                    "reward":  1.0 if traj.get("won") else 0.0,
                    "done":    bool(traj.get("won")),
                })
            trajectories.append(steps_out)

        entries.append({
            "env_id": f"env_{env_num}",
            "task": first_task,
            "type": "all_success" if all_won else "failure",
            "trajectories": trajectories,
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)

    n_traj = sum(len(e["trajectories"]) for e in entries)
    n_success = sum(1 for t in trajs if t.get("won"))
    print(f"Wrote {len(entries)} env-entries ({n_traj} trajectories) to {output_path}")
    print(f"  successes: {n_success} / {len(trajs)} ({n_success/len(trajs):.0%})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_file", required=True)
    ap.add_argument("--output_file", required=True)
    args = ap.parse_args()
    convert_traj_file(args.input_file, args.output_file)


if __name__ == "__main__":
    main()
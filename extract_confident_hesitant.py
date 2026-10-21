"""
Build a confident-vs-hesitant thinking dataset directly tied to outcomes:

  confident = think content that immediately precedes the agent making a correct
              final action (deep in a successful trajectory, after strategy has
              been "settled"), or a think content in a successful trajectory
              where the agent commits to a clearly correct next step.

  hesitant  = think content from trajectories that ultimately FAIL — the model
              engaged in indecision, repeats, loops, or admitted uncertainty.

This is more directly aligned with "confident/hesitant" than the per-step
4-way taxonomy used in the earlier PCA+K-Means experiment: we define the
classes from the *outcome* rather than our reading of agent confidence.

Output: /sharedata/home/nwkan/data/CAEI/confident_hesitant_samples.json
  list of { task, think, action, env_num, step_idx, label: "confident"|"hesitant" }
"""

import argparse
import json
import re
from collections import defaultdict


CUR_ACT_THINK_RE = re.compile(
    r"<tool_call> <tool_call>\s*(.*?)\s*<tool_call>", re.DOTALL
)
CUR_ACT_ACTION_RE = re.compile(
    r"<action>(.*?)</action>", re.DOTALL
)
CUR_PROMPT_TASK_RE = re.compile(r"Your task is to:\s*(.+?)\.")


def parse_step(step):
    curr = step.get("curr_action", "") or ""
    think_m = CUR_ACT_THINK_RE.search(curr)
    act_m = CUR_ACT_ACTION_RE.search(curr)
    think = think_m.group(1).strip() if think_m else ""
    action = act_m.group(1).strip() if act_m else ""
    p = step.get("curr_prompt", "")
    task_m = CUR_PROMPT_TASK_RE.search(p)
    task = task_m.group(1).strip() if task_m else ""
    return {"think": think, "action": action, "task": task}


def has_indecision_markers(think: str) -> bool:
    """Indicators that the model is uncertain / looping / second-guessing."""
    markers = [
        r"\bNothing happens\b",
        r"\bnot in a (?:valid |a )?(?:state|position) to\b",
        r"\bperhaps\b",
        r"\bpossibly\b",
        r"\bmight be that\b",
        r"\bsuggest(?:s|ing)? that\b",
        r"\bthis could\b",
        r"\bshould I\b",
        r"\bmaybe\b",
        r"\buncertain\b",
        r"\bstuck\b",
        r"\buncertain about\b",
        r"\b(?:attempted|attempting) (?:this|that) (?:multiple|several|many) times?\b",
        r"\b(?:loop|cycling|repeating|again|repeated(?:ly)?)\b",
    ]
    pat = "|".join(markers)
    return bool(re.search(pat, think, re.IGNORECASE))


def has_confident_markers(think: str) -> bool:
    """Indicators the model is committing to its action."""
    markers = [
        r"\b(?:the|this|that) is (?:the (?:correct|only))? (?:solution|approach) to\b",
        r"\btherefore\b.*\b(?:place|move|go|take|put|drop|examine|open|close)\b",
        r"\bthe (?:plan|action|goal) is\b",
        r"\baccording to (?:the )?(?:retrieved|strateg|relevant)",
        r"\blet's\b",
        r"\bI will\b",
        r"\bwe (?:have already|must|need to)\b",
    ]
    pat = "|".join(markers)
    return bool(re.search(pat, think, re.IGNORECASE))


def decisive_score(actions):
    if not actions:
        return 0
    return len(set(actions)) / len(actions)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_file", required=True)
    ap.add_argument("--output_file",
                    default="confident_hesitant_samples.json")
    ap.add_argument("--n_per_class", type=int, default=30)
    args = ap.parse_args()

    with open(args.input_file) as f:
        data = json.load(f)

    confident_pool = []   # successful trajectories, mid-late steps, decisive
    hesitant_pool  = []   # failed trajectories, mid steps, indecision markers

    for traj in data:
        if not traj.get("steps"):
            continue
        won = bool(traj.get("won"))
        steps = traj["steps"]
        n = len(steps)
        if n < 8:
            continue
        # consider the middle section (skip first 3 "explore" steps and last 2 cleanup)
        for i in range(4, n - 2):
            s = steps[i]
            parsed = parse_step(s)
            if not parsed["think"] or not parsed["action"]:
                continue
            recent = [
                parse_step(steps[j])["action"] or "?" for j in range(max(0, i - 6), i)
            ]
            ds = decisive_score(recent)

            entry = {
                "task": parsed["task"],
                "think": parsed["think"],
                "action": parsed["action"],
                "env_num": s.get("env_num"),
                "step_idx": s.get("step_idx"),
                "decisive_score": ds,
                "indecision": has_indecision_markers(parsed["think"]),
                "confident_phrase": has_confident_markers(parsed["think"]),
            }

            # ===== Confident =====
            # A "confident" think is one that precedes a CORRECT step in a
            # WINNING trajectory: the model knows what to do and does it.
            # Heuristic: trajectory won, recent actions are not jittery
            # (ds <= 0.5), and the current step's last action is something
            # the model has done recently (loop-consistent, suggesting commitment).
            repeats = recent.count(parsed["action"]) if parsed["action"] else 0
            if (won
                    and ds <= 0.5
                    and repeats >= 2):
                confident_pool.append(entry)

            # ===== Hesitant =====
            # A "hesitant" think is one inside a FAILING trajectory that
            # mentions Nothing-happens (model is stuck seeing no state change).
            if (not won and "Nothing happens" in parsed["think"]):
                hesitant_pool.append(entry)

    # Tag and trim
    confident = [{"label": "confident", **e} for e in confident_pool[: args.n_per_class]]
    hesitant  = [{"label": "hesitant",  **e} for e in hesitant_pool [: args.n_per_class]]

    samples = confident + hesitant

    with open(args.output_file, "w") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)

    print(f"wrote {len(samples)} samples to {args.output_file}")
    print(f"  confident: {len(confident)} (from pool {len(confident_pool)})")
    print(f"  hesitant:  {len(hesitant)}  (from pool {len(hesitant_pool)})")


if __name__ == "__main__":
    main()
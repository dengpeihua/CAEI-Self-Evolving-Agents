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

# 按轨迹结果与动作重复模式构造第一版自信和犹豫二分类样本。
import argparse
import json
import re
from collections import defaultdict

# 定义思考、动作和任务的解析规则。
CUR_ACT_THINK_RE = re.compile(
    r"<tool_call> <tool_call>\s*(.*?)\s*<tool_call>", re.DOTALL
)
CUR_ACT_ACTION_RE = re.compile(
    r"<action>(.*?)</action>", re.DOTALL
)
CUR_PROMPT_TASK_RE = re.compile(r"Your task is to:\s*(.+?)\.")

# 从当前动作与提示中提取思考正文、动作正文和任务描述。
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

# 检测思考中的不确定或循环表达，作为样本附加信息。
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

# 检测承诺式措辞，作为样本附加信息；这些布尔字段不直接决定最终标签。
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

# 计算近期动作的去重比例，辅助筛选动作模式稳定的步骤。
def decisive_score(actions):
    if not actions:
        return 0
    return len(set(actions)) / len(actions)

# 解析参数并读取轨迹，为两类样本建立候选池。
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_file", required=True)
    ap.add_argument("--output_file",
                    default="data/confident_hesitant_samples.json")
    ap.add_argument("--n_per_class", type=int, default=30)
    args = ap.parse_args()

    with open(args.input_file) as f:
        data = json.load(f)

    confident_pool = []
    hesitant_pool  = []

    # 跳过短轨迹与首尾步骤，解析中段有效思考及其最近六步动作。
    for traj in data:
        if not traj.get("steps"):
            continue
        won = bool(traj.get("won"))
        steps = traj["steps"]
        n = len(steps)
        if n < 8:
            continue
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

            # 自信样本要求轨迹成功、动作去重比例不超过 0.5，且当前动作近期至少重复两次。
            repeats = recent.count(parsed["action"]) if parsed["action"] else 0
            if (won
                    and ds <= 0.5
                    and repeats >= 2):
                confident_pool.append(entry)

            # 犹豫样本要求轨迹失败，且思考包含 Nothing happens；这是启发式代理标签。
            if (not won and "Nothing happens" in parsed["think"]):
                hesitant_pool.append(entry)

    # 按出现顺序截取两类样本、添加标签，保存 JSON 并打印候选数量。
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

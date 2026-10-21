"""
Extract representative thinking content from the v2 ALFWorld trajectory.

Strategy: pick a balanced sample across:
- won=True / won=False trajectories
- "decisive" (action repeats no matter how think) / "hesitant" (lots of
  variant thinking, eventually picks same action as paper strategy)

We hand-tag each sample with simple heuristics:
  tag = "won_decisive"    if won=True and consistent actions
       "won_hesitant"     if won=True but many distinct thinks
       "failed_decisive"  if failed and "wrong action confidently"
       "failed_hesitant"  if failed and "model didn't know what to do"

The exact 4-way labeling is fuzzy and we only use it for the analysis; the
real test is whether PCA+K-Means of the planning tokens recovers a
separation consistent with this taxonomy.
"""

# 从轨迹提取四类思考样本，为隐藏向量的聚类分析准备启发式标签。
import argparse
import json
import re

# 定义思考、动作、任务和无效观测的文本解析规则。
CUR_ACT_THINK_RE = re.compile(
    r"<tool_call> <tool_call>\s*(.*?)\s*<tool_call>", re.DOTALL
)
CUR_ACT_ACTION_RE = re.compile(
    r"<action>(.*?)</action>", re.DOTALL
)
CUR_PROMPT_TASK_RE = re.compile(r"Your task is to:\s*(.+?)\.")
OBS_NOTHING_RE = re.compile(r"Nothing happens")

# 解析单步的思考、动作、任务与观测摘要，并记录提示是否出现 Nothing happens。
def parse_step(step):
    curr_act = step.get("curr_action", "") or ""
    think_m = CUR_ACT_THINK_RE.search(curr_act)
    act_m = CUR_ACT_ACTION_RE.search(curr_act)
    think = think_m.group(1).strip() if think_m else ""
    action = act_m.group(1).strip() if act_m else ""
    prompt = step.get("curr_prompt", "")
    task_m = CUR_PROMPT_TASK_RE.search(prompt)
    task = task_m.group(1).strip() if task_m else ""
    obs = step.get("curr_prompt", "")
    return {
        "think": think,
        "action": action,
        "task": task,
        "obs_excerpt": prompt.split("Your current observation is:")[-1][:1200]
        if "Your current observation is:" in prompt
        else "",
        "nothing_happens": bool(OBS_NOTHING_RE.search(prompt)),
    }

# 用近期动作的去重比例衡量多样性；比例越低，当前规则越倾向于标为果断。
def decisive_score(actions):
    """Lower = more decisive (= same action repeatedly).
    Higher = more hesitant (varied actions, possibly re-trying)."""
    if not actions:
        return 0
    distinct = len(set(actions))
    total = len(actions)
    return distinct / total

# 解析输入输出路径和每类样本数量，读取原始轨迹。
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_file", required=True)
    ap.add_argument("--output_file", default="thinking_samples.json")
    ap.add_argument("--n_per_category", type=int, default=20,
                    help="target sample size per (won|fail) x (decisive|hesitant)")
    args = ap.parse_args()

    with open(args.input_file) as f:
        data = json.load(f)

    # 建立成功或失败与果断或犹豫组合的四个候选池。
    pools = {
        "won_decisive":   [],
        "won_hesitant":    [],
        "failed_decisive": [],
        "failed_hesitant":  [],
    }

    # 从第六步起检查最近五个动作，按去重比例及最终结果将有效思考放入对应候选池。
    for traj in data:
        steps = traj.get("steps", [])
        if not steps:
            continue
        won = bool(traj.get("won"))
        for i in range(5, len(steps)):
            recent_actions = [
                parse_step(steps[j])["action"] or "?" for j in range(max(0, i-5), i)
            ]
            score = decisive_score(recent_actions)
            sample = parse_step(steps[i])
            if not sample["think"] or not sample["action"]:
                continue
            entry = {
                "task": sample["task"],
                "think": sample["think"],
                "action": sample["action"],
                "obs_excerpt": sample["obs_excerpt"],
                "nothing_happens": sample["nothing_happens"],
                "step_idx": steps[i].get("step_idx"),
                "env_num": steps[i].get("env_num"),
                "decisive_score": score,
            }
            decisive = (score <= 0.5)
            hesitant = (not decisive)

            if won and decisive and len(pools["won_decisive"]) < args.n_per_category * 5:
                pools["won_decisive"].append(entry)
            elif won and hesitant and len(pools["won_hesitant"]) < args.n_per_category * 5:
                pools["won_hesitant"].append(entry)
            elif (not won) and decisive and len(pools["failed_decisive"]) < args.n_per_category * 5:
                pools["failed_decisive"].append(entry)
            elif (not won) and hesitant and len(pools["failed_hesitant"]) < args.n_per_category * 5:
                pools["failed_hesitant"].append(entry)

    # 按原始出现顺序截取每类样本并写入标签，再保存 JSON 和类别统计。
    out = []
    for cat, items in pools.items():
        for entry in items[: args.n_per_category]:
            entry["label"] = cat
            out.append(entry)

    with open(args.output_file, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"Total samples: {len(out)}")
    for cat in pools:
        print(f"  {cat}: kept {min(args.n_per_category, len(pools[cat]))}/{len(pools[cat])}")
    print(f"Saved to {args.output_file}")

if __name__ == "__main__":
    main()

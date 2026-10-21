"""
Build a v2 confident-vs-hesitant thinking dataset (larger).

Refined rules:
  CONFIDENT = trajectory won + think has explicit "decision-committing"
              markers ("I will", "the right action is", "the goal is to",...)
              AND think contains explicit inference about why this action
              is the right one. Not just short or repetitive.

  HESITANT  = trajectory failed + think contains markers of the model
              admitting uncertainty ("perhaps", "I'm not sure", "might be
              wrong", "I haven't found", "not certain whether",
              "unclear whether", "may need to", "possibly", "I will try")
              PLUS the trajectory is in a state of indecision (multiple
              different actions were tried recently).

Output: confident_hesitant_samples_v2.json
"""

# 结合结果与思考措辞构造第二版自信和犹豫样本，替代第一版的重复动作筛选。
import argparse
import json
import re
from collections import defaultdict

# 定义思考、动作、任务与无效观测的解析规则。
CUR_ACT_THINK_RE = re.compile(
    r"<tool_call> <tool_call>\s*(.*?)\s*<tool_call>", re.DOTALL
)
CUR_ACT_ACTION_RE = re.compile(
    r"<action>(.*?)</action>", re.DOTALL
)
CUR_PROMPT_TASK_RE = re.compile(r"Your task is to:\s*(.+?)\.")
OBS_NOTHING_RE = re.compile(r"Nothing happens")

# 解析步骤内容并截取观测摘要，为标签筛选和后续特征提取保留上下文。
def parse_step(step):
    curr = step.get("curr_action", "") or ""
    think_m = CUR_ACT_THINK_RE.search(curr)
    act_m = CUR_ACT_ACTION_RE.search(curr)
    think = think_m.group(1).strip() if think_m else ""
    action = act_m.group(1).strip() if act_m else ""
    p = step.get("curr_prompt", "")
    task_m = CUR_PROMPT_TASK_RE.search(p)
    task = task_m.group(1).strip() if task_m else ""
    obs = p.split("Your current observation is:")[-1][:1200] \
        if "Your current observation is:" in p else ""
    return {"think": think, "action": action, "task": task, "obs_excerpt": obs,
            "nothing_happens": bool(OBS_NOTHING_RE.search(p))}

# 定义并统计承诺式表达，用于衡量思考是否已经形成明确行动意图。
COMMITMENT_RE = re.compile(
    r"\b(?:"
    r"the (?:right|correct|optimal|best|suitable|only) action(?: to (?:take|perform))? is\b"
    r"|the (?:plan|approach|strategy) is\b"
    r"|I (?:will|should|need to|am going to|am about to) (?:then )?actually\b"
    r"|this (?:is the (?:next )?)?(?:correct|right|only) (?:step|action|move|approach)\b"
    r"|according to (?:the )?(?:above|strat|retrieved|relevant|recent)|" +
    r"\bwe have (?:already )?(?:completed|achieved|done|finished|acquired|cleaned|heated|cooled)\b"
    r"|(?:therefore|thus|so),?\s*(?:I|we) (?:will|should|need to|now)\b"
    r"|(?:the )?next step(?: would| is to| should be)?\b"
    r")",
    re.IGNORECASE,
)

def looks_committed(think: str) -> int:
    """Return how many commitment markers in the think."""
    return len(COMMITMENT_RE.findall(think))

# 定义并统计不确定表达；正则只提供代理信号，部分肯定措辞也可能被匹配。
UNCERTAINTY_RE = re.compile(
    r"\b(?:"
    r"I(?:'m| am) (?:not )?(?:sure|certain|confident)(?: about| whether)?\b"
    r"|(?:perhaps|possibly|maybe|likely|seem[s]?|might|may)\b"
    r"|(?:could be|may be|is (?:not|un)clear)\b"
    r"|(?:don't|do not) know (?:whether|if|what|where|which)\b"
    r"|(?:unclear|uncertain|unsure|confused) (?:whether|about|if|what|how)\b"
    r"|(?:nothing happens|not (?:working|progress|advancing|succeeding))\b"
    r"|(?:I )?(?:have|has|haven't|hasn't) (?:not yet )?(?:found|located|seen|acquired|achieved|completed)\b"
    r"|(?:unclear|unknown|not visible|not present|missing|absent) (?:in|at|where|from)\b"
    r"|(?:try|trying|attempt|attempting) (?:a|an)?\s*(?:different|another|alternative)\b"
    r")",
    re.IGNORECASE,
)

def looks_uncertain(think: str) -> int:
    """Return how many uncertainty markers in the think."""
    return len(UNCERTAINTY_RE.findall(think))

# 计算动作多样性供记录与分析；本版本最终筛选不使用该比例。
def decisive_score(actions):
    if not actions:
        return 0
    return len(set(actions)) / len(actions)

# 读取轨迹与采样参数，准备自信和犹豫两个候选池。
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_file", required=True)
    ap.add_argument("--output_file",
                    default="data/confident_hesitant_samples_v2.json")
    ap.add_argument("--n_per_class", type=int, default=200)
    args = ap.parse_args()

    with open(args.input_file) as f:
        data = json.load(f)

    confident_pool = []
    hesitant_pool = []

    # 跳过过短轨迹与首尾步骤，解析中段思考并计算词汇命中次数。
    for ti, traj in enumerate(data):
        if not traj.get("steps"):
            continue
        won = bool(traj.get("won"))
        steps = traj["steps"]
        n = len(steps)
        if n < 6:
            continue

        for i in range(3, n - 2):
            s = steps[i]
            parsed = parse_step(s)
            if not parsed["think"] or not parsed["action"]:
                continue

            recent = [parse_step(steps[j])["action"] or "?"
                      for j in range(max(0, i - 6), i)]
            ds = decisive_score(recent)
            commit_n = looks_committed(parsed["think"])
            uncert_n = looks_uncertain(parsed["think"])

            entry = {
                "task": parsed["task"],
                "think": parsed["think"],
                "action": parsed["action"],
                "obs_excerpt": parsed["obs_excerpt"],
                "env_num": s.get("env_num"),
                "step_idx": s.get("step_idx"),
                "traj_idx": ti,
                "decisive_score": ds,
                "commitment_markers": commit_n,
                "uncertainty_markers": uncert_n,
                "nothing_happens": parsed["nothing_happens"],
            }

            # 自信：轨迹成功，承诺标记至少一次、不确定标记至多一次，且提示无 Nothing happens。
            if (won
                    and commit_n >= 1
                    and uncert_n <= 1
                    and not parsed["nothing_happens"]):
                confident_pool.append(entry)

            # 犹豫：轨迹失败，不确定标记至少两次，承诺标记至多一次。
            if (not won
                    and uncert_n >= 2
                    and commit_n <= 1):
                hesitant_pool.append(entry)

    # 按出现顺序截取每类样本，添加标签并保存 JSON 和筛选统计。
    confident = [{"label": "confident", **e}
                 for e in confident_pool[: args.n_per_class]]
    hesitant = [{"label": "hesitant",  **e}
                for e in hesitant_pool[: args.n_per_class]]

    samples = confident + hesitant

    with open(args.output_file, "w") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)

    print(f"wrote {len(samples)} samples to {args.output_file}")
    print(f"  confident: {len(confident)} (from pool {len(confident_pool)})")
    print(f"  hesitant:  {len(hesitant)}  (from pool {len(hesitant_pool)})")
    print(f"\nsampling logic (deeper than v1):")
    print(f"  CONFIDENT requires: won=True, has commitment marker,")
    print(f"                          ≤1 uncertainty marker, no Nothing-happens.")
    print(f"  HESITANT  requires: won=False, ≥2 uncertainty markers,")
    print(f"                          ≤1 commitment marker.")

if __name__ == "__main__":
    main()

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
    python construct/index/utils/convert_traj_to_memory_input.py \\
        --input_file  logs/.../traj_eval_in_distribution.json \\
        --output_file skills/data/processed_trajectories.json
"""

# 将轨迹日志整理为按环境分组的嵌套 JSON，供后续策略记忆生成使用。
import argparse
import json
import re
from collections import defaultdict

# 准备任务描述与当前观测的文本定位规则。
_TASK_RE   = re.compile(r"Your task is to:\s*(.+?)\.")
_OBS_START = re.compile(r"Your current observation is:")

# 从提示中提取任务描述，取任务标记之后、首个句点之前的内容。
def extract_task(curr_prompt: str) -> str:
    """Return the raw task description (e.g. 'put some candle on toilet')"""
    m = _TASK_RE.search(curr_prompt)
    return m.group(1).strip() if m else ""

# 截取当前观测文本，并在任务、可行动作或下一轮交互提示之前停止。
def extract_observation(curr_prompt: str) -> str:
    """Return the 'your current observation is: ...' block — the part the
    agent saw before choosing the action."""
    m = _OBS_START.search(curr_prompt)
    if not m:
        return ""
    tail = curr_prompt[m.end():].lstrip()
    for stop in ("\nYour task is to:", "\nYour admissible actions", "\nNow it's your turn"):
        idx = tail.find(stop)
        if idx >= 0:
            tail = tail[:idx]
    return tail.strip()

# 读取轨迹并按 env_num 分组，允许同一环境包含多条尝试轨迹。
def convert_traj_file(input_path: str, output_path: str) -> None:
    with open(input_path) as f:
        trajs = json.load(f)

    by_env: dict = defaultdict(list)
    for traj in trajs:
        if not traj.get("steps"):
            continue
        env_num = traj["steps"][0].get("env_num", 0)
        by_env[env_num].append(traj)

    # 逐环境构造任务与轨迹记录；只有该环境所有轨迹成功时才标为 all_success。
    entries = []
    for env_num in sorted(by_env.keys()):
        env_trajs = by_env[env_num]

        first_task = extract_task(env_trajs[0]["steps"][0]["curr_prompt"])

        trajectories = []
        all_won = True
        for traj in env_trajs:
            if not traj.get("won"):
                all_won = False

            # 转换步骤字段；动作保留完整 curr_action，reward 与 done 在每一步复制轨迹级结果。
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

    # 保存转换后的 UTF-8 JSON，并打印环境、轨迹及成功数量。
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)

    n_traj = sum(len(e["trajectories"]) for e in entries)
    n_success = sum(1 for t in trajs if t.get("won"))
    print(f"Wrote {len(entries)} env-entries ({n_traj} trajectories) to {output_path}")
    print(f"  successes: {n_success} / {len(trajs)} ({n_success/len(trajs):.0%})")

# 解析输入和输出文件参数，调用格式转换函数。
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_file", required=True)
    ap.add_argument("--output_file", required=True)
    args = ap.parse_args()
    convert_traj_file(args.input_file, args.output_file)

if __name__ == "__main__":
    main()

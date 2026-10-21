"""
Generate skill memories from preprocessed ALFWorld trajectories.

Each memory captures the strategic structure of a trajectory (success or
failure) so it can be retrieved at distillation time. Each entry has:

    - contextual_description: 1-line abstraction of the task + outcome
    - refined_trajectory:     causally-pruned action chain (success only)
    - strategic_guidelines:   planning_pattern + mistakes_to_avoid

We invoke the LLM (default deepseek-chat, configurable) once per section per
trajectory. By default we only generate one memory per env (the first
trajectory in `trajectories`); pass --all_trajectories to generate one
per trajectory.

Usage:
    # API key is read from .env (DEEPSEEK_API_KEY) in the project root.
    python generate_memory_alfworld.py \\
        --input_file processed_trajectories_alfworld_cleaned.json \\
        --output_file generated_memories_alfworld.json \\
        --model deepseek-chat \\
        --api_base https://api.deepseek.com
"""

# 数据流：预处理轨迹 JSON -> 逐轨迹调用模型 -> 结构化记忆 JSON。
# 输入为记录列表，每项包含 env_id、task、type、trajectories；
# trajectories 中每项是一条轨迹的步骤列表。输出供 aggregate_skills.py 汇总，
# 本脚本不训练策略模型，也不直接生成 SkillProvider 所读取的 Markdown 文件。

import argparse
import json
import os
import re
import uuid

from openai import OpenAI


def _load_api_key_from_env_file() -> str:
    """读取项目根目录 .env，优先返回 DeepSeek 密钥，其次为 OpenAI 密钥。

    当前实现要求 .env 文件存在；缺少文件时直接返回空串。
    文件中的键值会写入 os.environ，覆盖同名环境变量。
    """
    import os, pathlib
    env_path = pathlib.Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return ""
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        v = v.strip()
        # Strip matching quotes (single or double) the user may have written.
        if len(v) >= 2 and v[0] == v[-1] and v[0] in ('"', "'"):
            v = v[1:-1]
        os.environ[k.strip()] = v
    return os.environ.get("DEEPSEEK_API_KEY", "") or os.environ.get("OPENAI_API_KEY", "")


# 固定单价仅用于末尾的费用估算，不会按 --model 或服务商自动更新。
PRICE_INPUT = 0.00125 / 1000
PRICE_OUTPUT = 0.0100 / 1000

PROMPTS = {
    "contextual_description": """
You are an expert **RAG Abstraction Engine** for an autonomous agent memory system.
Your objective is to generate a **"contextual_description"** that classifies the task type and summarizes the execution logic.

**Target Output Format:**
"AlfWorld task classified as '[Task Category]' to [Goal String]. [Outcome Description]."

**Rules:**
1. **Task Category (Strict):** exactly one of: `pick_and_place`, `pick_two_obj_and_place`, `look_at_obj_in_light`, `pick_heat_then_place_in_recep`, `pick_cool_then_place_in_recep`, `pick_clean_then_place_in_recep`.
2. **Goal String:** copy the exact 'Goal' string but remove all instance numbers/IDs.
3. **Outcome Description:**
    * IF Success: "Solved by [Sequential Action Summary]"
    * IF Failure: "Unsolved due to [Root Cause]"

Return ONLY the description string.
""",
    "refined_trajectory": """
You are an expert **Trajectory Refinement & Abstraction Engine** using a **"Backward Causal Chaining"** algorithm.
Your goal is to extract the minimal `refined_trajectory` from a raw log AND **generalize** the content into semantic placeholders.

**Phase 1: Refinement (Strictly Backward Logic)**
1. Start from the **Last Successful Step**. Call this `Current_Step`.
2. Recursively: identify the **Required Precondition** for `Current_Step`, scan backwards, find the nearest preceding step that produced it (`Preceding_Step`), prune everything between them, then set `Current_Step = Preceding_Step` and repeat.

**Phase 2: Abstraction**
1. Keep specific object names but **remove instance IDs** (e.g., "mug 1" -> "mug").
2. Replace specific values with `[*_Constraint]` placeholders.
3. Action/observation strings must use the abstracted entities.

**Output:** JSON list `refined_trajectory` (chronological).
* `step_index`: original index
* `action`: GENERALIZED action string
* `critical_observation`: GENERALIZED observation focusing on state changes
* `reasoning`: ONE highly-generalizable sentence on strategic intent. For exploration steps, emphasize **search for unknown** rather than "Navigate".

Output ONLY the JSON object.
""",
    "strategic_guidelines_alfworld": """
You are an expert **Strategic Analyst**.
Your goal is to extract high-level `strategic_guidelines` focusing on the execution skeleton and error avoidance.

### **CASE 1: SUCCESS**
1. **`planning_pattern`:** Logical chain using " -> " separators.
   - **NEVER** use specific names. Replace objects with `[Object_1]`, `[Object_2]`, search areas with `[Location]`, fixed destinations with `[Target_Location]`.
   - **Mandatory Search Precondition:** before interacting with any object, prefix with **"Search [Location] where possibly having [Object_X]"** -> **"[Action_Verb] [Object_X]"**. Use verbs like Acquire / Use / Heat / Cool / Clean / Place where they fit.
   - **Multi-Object:** if same location, combine searches; if different, distinguish.
   - **Navigation:** `"Navigate to [Target_Location]"` ONLY for fixed appliance/placement destinations after acquiring the object.
2. **`mistakes_to_avoid`:** `[]`. (Don't Need)

### **CASE 2: FAILURE**
1. **`planning_pattern`:** `null`. (Don't Need)
2. **`mistakes_to_avoid`:** abstract list of `{trigger_condition, bad_action}` using `[Target_Object]`, `[Container]`, `[Location]` etc. Never specific instance IDs.

Output ONLY the JSON object with keys `planning_pattern` and `mistakes_to_avoid`.
""",
}


def trajectory_to_string(steps):
    """将步骤列表序列化为模型输入，保留动作、奖励、终止标记和观测。

    此处只整理文本，不判断任务成败，也不删去失败或重复动作；
    轨迹精简由后续模型提示词完成。
    """
    out = []
    for s in steps:
        out.append(
            f"{s.get('step_id', 'Step ?')} | Action: {s.get('action', 'None')} | "
            f"Reward: {s.get('reward', 0.0)} | Done: {s.get('done', False)}\n"
            f"Obs: {(s.get('observation') or '').strip()}\n"
        )
    return "\n".join(out)


def extract_json(text):
    """依次尝试完整 JSON、json 代码块、花括号或方括号片段。

    全部失败时返回 None；解析成功仅代表语法有效，不校验字段结构。
    """
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        for pat in (r"```json\s*(.*?)\s*```", r"(\{.*\}|\[.*\])"):
            m = re.search(pat, text, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group(1))
                except Exception:
                    continue
    return None


class MemoryGenerator:
    """按轨迹生成记忆，并累计该实例的接口 token 用量。"""

    def __init__(self, client, model_name):
        self.client = client
        self.model = model_name
        self.input_tokens = 0
        self.output_tokens = 0

    def _run(self, system_prompt, user_content):
        # 每次仅发起一次模型请求，异常时记录错误并返回 None，不在此处重试。
        # create() 会把缺失响应保存为空描述或空字段，调用方需留意数据完整性。
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                temperature=0,
            )
            self.input_tokens += resp.usage.prompt_tokens
            self.output_tokens += resp.usage.completion_tokens
            return resp.choices[0].message.content
        except Exception as e:
            print(f"LLM error: {e}")
            return None

    def create(self, env, goal, outcome, raw_traj_str):
        """构造一条记忆：描述 + 成功轨迹精简 + 策略或错误归纳。

        正常路径下成功轨迹调用模型 3 次，失败轨迹调用 2 次。
        三类提示词分别读取同一份原始轨迹，后一请求不依赖前一请求的结果。
        """
        ctx = (
            f"**Input Data:**\nEnvironment: {env}\nGoal: {goal}\n"
            f"Outcome: {outcome}\nRaw Trajectory:\n{raw_traj_str}"
        )

        description = (
            (self._run(PROMPTS["contextual_description"], ctx) or "").strip().strip('"')
        )

        refined = None
        # 失败轨迹不提炼可模仿的成功动作链，但仍提取描述和应避免的错误。
        if outcome.lower() == "success":
            refined = extract_json(self._run(PROMPTS["refined_trajectory"], ctx))

        strategic = extract_json(
            self._run(PROMPTS["strategic_guidelines_alfworld"], ctx)
        )

        return {
            # 随机 ID 区分记忆记录；它不是轨迹内容哈希，重复运行不会自动去重。
            "memory_id": f"mem_{env.lower()}_{uuid.uuid4().hex[:8]}",
            "contextual_description": description,
            "tags": {"environment": env, "outcome": outcome},
            "content": {
                "task_meta": {"original_goal": goal},
                "refined_trajectory": refined,
                "strategic_guidelines": strategic,
            },
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument(
        "--api_base",
        default="https://api.deepseek.com",
        help="OpenAI-compatible endpoint (use https://api.deepseek.com for DeepSeek).",
    )
    parser.add_argument("--env_name", default="Alfworld")
    parser.add_argument(
        "--all_trajectories",
        action="store_true",
        help="Generate one memory per trajectory (default: only the first per env).",
    )
    args = parser.parse_args()

    api_key = _load_api_key_from_env_file()
    if not api_key:
        raise SystemExit(
            "API key not found. Set DEEPSEEK_API_KEY in .env (project root)."
        )
    client = OpenAI(api_key=api_key, base_url=args.api_base)


    with open(args.input_file, "r", encoding="utf-8") as f:
        entries = json.load(f)

    gen = MemoryGenerator(client, args.model)
    memories = []

    for entry in entries:
        env_id = entry.get("env_id", "Unknown")
        goal = entry.get("task", "")
        trajs = entry.get("trajectories", [])
        # 成败沿用输入记录的 type，而非重新检查每步 reward / done。
        # 同一记录内所有轨迹共用此标签；非 all_success 的值统一标为 Failure。
        outcome = "Success" if entry.get("type", "") == "all_success" else "Failure"

        for idx, steps in enumerate(trajs):
            try:
                mem = gen.create(
                    args.env_name, goal, outcome, trajectory_to_string(steps)
                )
                mem["origin_env_id"] = env_id
                memories.append(mem)
                print(f"[{env_id} traj {idx}] [{outcome}] OK")
            except Exception as e:
                print(f"[{env_id} traj {idx}] FAILED: {e}")
            if not args.all_trajectories:
                # 默认只尝试列表中的第一条；即使该条抛出异常，也不会转试下一条。
                break

    # 全部处理结束后一次性写出，覆盖指定输出文件；没有逐条检查点或断点续跑。
    os.makedirs(
        os.path.dirname(os.path.abspath(args.output_file)) or ".", exist_ok=True
    )
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(memories, f, indent=2, ensure_ascii=False)

    cost = gen.input_tokens * PRICE_INPUT + gen.output_tokens * PRICE_OUTPUT
    print(f"\nSaved {len(memories)} memories to {args.output_file}")
    print(f"Estimated cost (gpt-4o pricing): ${cost:.4f}")


if __name__ == "__main__":
    main()

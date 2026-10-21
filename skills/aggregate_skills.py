"""
Aggregate raw per-trajectory memories into a polished skill bank.

The output schema mirrors `skill_generation/{alfworld,webshop,search}.py` so
the same JSON can feed both the RL runtime (`SkillsOnlyMemory`) and the SFT
distillation pipeline.

Output schema:
    {
      "general_skills": [
        {"skill_id": "gen_001",
         "title": "Systematic Exploration",
         "principle": "Search every plausible ...",
         "when_to_apply": "When starting any task ..."},
        ...
      ],
      "task_specific_skills": {                # alfworld + webshop
        "<category>": [
          {"skill_id": "<prefix>_001",
           "title": "...",
           "principle": "...",
           "when_to_apply": "..."},
          ...
        ],
        ...
      },
      "query_type_skills": {...},              # search only (matches skill_generation/search.py)
      "common_mistakes": [
        {"mistake_id": "err_001",
         "description": "...",
         "why_it_happens": "...",
         "how_to_avoid": "..."},
        ...
      ]
    }

Usage:
    export OPENAI_API_KEY=...
    python aggregate_skills.py \\
        --input_file generated_memories_alfworld.json \\
        --output_file alfworld_skill_bank.json \\
        --env alfworld \\
        --model gpt-4o
"""

# 数据流：轨迹级记忆 JSON -> 成功/失败分流 -> 模型归纳 -> 技能库 JSON。
# 本脚本输出结构化文本数据，不执行环境动作，也不自动写出 general_skills.md
# 等文件。SkillProvider 读取的 Markdown + skill_mapping.json 是另一种存储接口。
# 上面的 skill_generation/、SkillsOnlyMemory 是原说明中的接口名称；
# 当前脚本不导入它们，也不依赖它们执行汇总。

import argparse
import json
import os
import pathlib
import re

from openai import OpenAI


def _load_api_key_from_env_file() -> str:
    """从项目根目录 .env 装载配置，优先使用 DeepSeek 密钥。

    文件不存在时返回空串；文件内键值会覆盖 os.environ 中的同名变量。
    此处只配置客户端，实际使用哪个模型与地址由命令行参数决定。
    """
    env_path = pathlib.Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return ""
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        v = v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in ('"', "'"):
            v = v[1:-1]
        os.environ[k.strip()] = v
    return os.environ.get("DEEPSEEK_API_KEY", "") or os.environ.get("OPENAI_API_KEY", "")

# ---------------------------------------------------------------------------
# Category sets — kept aligned with `skill_generation/{alfworld,webshop,search}.py`
# ---------------------------------------------------------------------------

ALFWORLD_CATEGORIES = [
    "pick_and_place",
    "look_at_obj_in_light",
    "clean",
    "heat",
    "cool",
    "examine",
]

ALFWORLD_CATEGORY_DISPLAY = {
    "pick_and_place": "Pick And Place",
    "look_at_obj_in_light": "Look At Object In Light",
    "clean": "Clean",
    "heat": "Heat",
    "cool": "Cool",
    "examine": "Examine",
}

WEBSHOP_CATEGORIES = [
    "apparel",
    "footwear",
    "home_decor",
    "electronics",
    "accessories",
    "beauty_health",
    "other",
]

SEARCH_CATEGORIES = [
    "direct_retrieval",
    "multi_hop_reasoning",
    "entity_attribute_lookup",
    "comparison",
]


# ---------------------------------------------------------------------------
# Prompts — instruct the LLM to emit the skill_generation/ field schema
# ---------------------------------------------------------------------------

GENERAL_PRINCIPLES_PROMPT = """You are an expert agent-memory summarizer.

You will be given a list of `planning_pattern` strings extracted from successful agent trajectories. Each is a short action chain template like:
    "Search [Location] -> Acquire [Object] -> Navigate to [Target_Location] -> Place [Object]"

Your job: produce 5-7 high-level GENERAL SKILLS that capture the common,
universally-applicable strategies behind these patterns. Each skill should
be polished into clear English prose, ready to drop into a system prompt.

Output format (strict JSON list, no preamble, no markdown fences):
[
  {
    "title": "<Title Case short name, 3-5 words>",
    "principle": "<one or two sentences ending with period; the core actionable insight>",
    "when_to_apply": "<one sentence ending with period; the trigger condition>"
  },
  ...
]

Constraints:
- DO NOT use bracketed placeholders like [Object] or [Location] in the output. Write natural English.
- 5 to 7 entries total.
- Each entry should be self-contained and actionable."""


CATEGORY_SKILLS_PROMPT = """You are an expert agent-memory summarizer.

You will be given a list of `planning_pattern` strings from successful trajectories of one specific task category: **{category_label}**. Each is a short action chain template.

Your job: produce 5-6 CATEGORY-SPECIFIC SKILLS that capture the
distinctive techniques for this task type.

Output format (strict JSON list, no preamble, no markdown fences):
[
  {{
    "title": "<Title Case short name, 3-5 words>",
    "principle": "<one or two sentences ending with period; the actionable insight specific to this task category>",
    "when_to_apply": "<one sentence ending with period; the trigger condition>"
  }},
  ...
]

Constraints:
- DO NOT use bracketed placeholders. Write natural English.
- Skills should be specific to the {category_label} task type, not generic.
- 5 to 6 entries total."""


MISTAKES_PROMPT = """You are an expert agent-memory summarizer.

You will be given a list of `mistakes_to_avoid` items extracted from failed agent trajectories. Each item has a `trigger_condition` and a `bad_action`.

Your job: produce 5 COMMON MISTAKES entries that capture the most common,
generalizable failure modes across these failed trajectories. Dedupe
similar mistakes and polish into clear English.

Output format (strict JSON list, no preamble, no markdown fences):
[
  {
    "description": "<one sentence describing the bad behavior>",
    "why_it_happens": "<one sentence explaining why agents make this mistake>",
    "how_to_avoid": "<one or two sentences describing the concrete correction>"
  },
  ...
]

Constraints:
- DO NOT use bracketed placeholders. Write natural English.
- Exactly 5 entries."""


# ---------------------------------------------------------------------------
# LLM call + JSON parsing
# ---------------------------------------------------------------------------


def call_llm(
    client: OpenAI, model: str, system_prompt: str, user_payload: dict
) -> list:
    """将候选经验作为 JSON 发给模型，再解析模型返回的技能或错误列表。

    先去掉代码围栏，完整 JSON 解析失败后尝试提取方括号片段。
    不校验列表元素字段，也不自动重试；接口或解析异常会向调用方传播。
    """
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ],
        temperature=0,
    )
    text = resp.choices[0].message.content
    text = re.sub(r"^```json\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\[.*\]", text, re.DOTALL)
        if m:
            return json.loads(m.group(0))
        raise ValueError(f"LLM did not return valid JSON: {text[:300]}")


# ---------------------------------------------------------------------------
# Classifiers (per-env). Map a memory record to one of the category keys.
# ---------------------------------------------------------------------------


def classify_alfworld_memory(mem: dict) -> str | None:
    """按规则将记忆归入一个 ALFWorld 类别，无法识别时返回 None。

    优先匹配原始任务目标，再用 contextual_description 兜底。
    判断按代码顺序返回首个结果，因此 clean/heat/cool 优先于普通放置任务；
    这里没有调用分类模型，也没有计算语义相似度。
    """
    goal = (
        mem.get("content", {}).get("task_meta", {}).get("original_goal") or ""
    ).lower()
    if "look at" in goal and "under" in goal:
        return "look_at_obj_in_light"
    if "clean" in goal:
        return "clean"
    if "heat" in goal or "hot" in goal:
        return "heat"
    if "cool" in goal or "cold" in goal:
        return "cool"
    if "examine" in goal or "find" in goal:
        return "examine"
    if "put" in goal or "place" in goal:
        return "pick_and_place"
    desc = (mem.get("contextual_description") or "").lower()
    if "look_at_obj_in_light" in desc:
        return "look_at_obj_in_light"
    if "clean" in desc:
        return "clean"
    if "heat" in desc:
        return "heat"
    if "cool" in desc:
        return "cool"
    if "examine" in desc:
        return "examine"
    if "pick_and_place" in desc or "pick_two_obj_and_place" in desc:
        return "pick_and_place"
    return None


WEBSHOP_KEYWORD_TABLE = [
    (
        "apparel",
        [
            "shirt",
            "dress",
            "t-shirt",
            "polo",
            "pants",
            "jeans",
            "jacket",
            "coat",
            "sweater",
            "blouse",
            "skirt",
            "shorts",
            "underwear",
            "swimsuit",
            "swimwear",
            "hoodie",
            "vest",
            "cardigan",
            "suit",
            "blazer",
            "tee",
            "top",
        ],
    ),
    (
        "footwear",
        [
            "shoe",
            "boot",
            "sandal",
            "sneaker",
            "slipper",
            "loafer",
            "heel",
            "flat",
            "oxford",
            "pump",
            "moccasin",
            "flip-flop",
            "footwear",
        ],
    ),
    (
        "home_decor",
        [
            "pillow",
            "curtain",
            "rug",
            "mat",
            "blanket",
            "bedding",
            "towel",
            "lamp",
            "decor",
            "furniture",
            "cushion",
            "sheet",
            "tablecloth",
            "vase",
        ],
    ),
    (
        "electronics",
        [
            "phone",
            "laptop",
            "tablet",
            "computer",
            "headphone",
            "earphone",
            "earbud",
            "speaker",
            "charger",
            "cable",
            "mouse",
            "keyboard",
            "monitor",
            "camera",
            "smartwatch",
            "battery",
            "electronic",
            "device",
            "gadget",
        ],
    ),
    (
        "accessories",
        [
            "bag",
            "wallet",
            "belt",
            "hat",
            "cap",
            "scarf",
            "glove",
            "jewelry",
            "necklace",
            "bracelet",
            "ring",
            "earring",
            "sunglasses",
            "glasses",
            "watch",
            "purse",
            "backpack",
            "handbag",
            "tie",
            "bow",
        ],
    ),
    (
        "beauty_health",
        [
            "makeup",
            "cosmetic",
            "skincare",
            "lotion",
            "cream",
            "shampoo",
            "conditioner",
            "perfume",
            "cologne",
            "brush",
            "bathing",
            "soap",
            "body wash",
            "nail",
            "lipstick",
            "mascara",
            "foundation",
            "serum",
            "moisturizer",
        ],
    ),
]


def classify_webshop_memory(mem: dict) -> str:
    """按关键词表顺序返回第一个命中的商品类别，未命中则归入 other。

    匹配使用子串包含关系，不是完整单词匹配；同一目标不会同时归入多个类别。
    """
    goal = (
        mem.get("content", {}).get("task_meta", {}).get("original_goal") or ""
    ).lower()
    for category, keywords in WEBSHOP_KEYWORD_TABLE:
        if any(kw in goal for kw in keywords):
            return category
    return "other"


def classify_search_memory(mem: dict) -> str:
    """结合数据集来源和问题关键词，区分多跳、比较、属性查询及直接检索。

    只有列出的多跳数据集会继续检查比较类关键词；popqa 单独归为属性查询，
    其余来源默认直接检索。这里的类别名仍需与下游技能映射保持一致。
    """
    data_source = mem.get("tags", {}).get("data_source", "")
    goal = (
        mem.get("content", {}).get("task_meta", {}).get("original_goal") or ""
    ).lower()
    if data_source in ("hotpotqa", "2wikimultihopqa", "musique", "bamboogle"):
        comparison_kw = [
            "both",
            "are the",
            "which of",
            "same",
            "common",
            "more",
            "less",
            "older",
            "younger",
            "taller",
            "shorter",
        ]
        if any(kw in goal for kw in comparison_kw):
            return "comparison"
        return "multi_hop_reasoning"
    if data_source == "popqa":
        return "entity_attribute_lookup"
    return "direct_retrieval"


# ---------------------------------------------------------------------------
# Skill / mistake ID assignment — done after LLM call so IDs follow a clean
# deterministic scheme regardless of what the LLM emits.
# ---------------------------------------------------------------------------


def _id_prefix(category: str) -> str:
    """Match `skill_generation/`'s convention: first three letters of the category."""
    return re.sub(r"[^a-z]", "", category.lower())[:3] or "cat"


def _assign_skill_ids(skills: list, prefix: str) -> list:
    # 用本地序号替换模型给出的 ID，避免依赖模型的编号格式。
    # 序号取自原列表位置：非字典元素被跳过后，编号可能不连续；此处不去重。
    out = []
    for i, sk in enumerate(skills, 1):
        if not isinstance(sk, dict):
            continue
        sk = {k: v for k, v in sk.items() if k != "skill_id"}
        out.append({"skill_id": f"{prefix}_{i:03d}", **sk})
    return out


def _assign_mistake_ids(mistakes: list) -> list:
    # 错误条目采用同样规则，以 err_ 开头；其他字段保留模型原始输出。
    out = []
    for i, m in enumerate(mistakes, 1):
        if not isinstance(m, dict):
            continue
        m = {k: v for k, v in m.items() if k != "mistake_id"}
        out.append({"mistake_id": f"err_{i:03d}", **m})
    return out


# ---------------------------------------------------------------------------
# Aggregation pipeline
# ---------------------------------------------------------------------------

ENV_CONFIG = {
    "alfworld": {
        "categories": ALFWORLD_CATEGORIES,
        "classifier": classify_alfworld_memory,
        "category_key": "task_specific_skills",
        "label_map": ALFWORLD_CATEGORY_DISPLAY,
    },
    "webshop": {
        "categories": WEBSHOP_CATEGORIES,
        "classifier": classify_webshop_memory,
        "category_key": "task_specific_skills",
        "label_map": None,
    },
    "search": {
        "categories": SEARCH_CATEGORIES,
        "classifier": classify_search_memory,
        "category_key": "query_type_skills",
        "label_map": None,
    },
}


def aggregate(memories: list[dict], env: str, client: OpenAI, model: str) -> dict:
    """将记忆汇总为通用技能、分类技能和常见错误三个部分。

    tags.outcome 必须准确写为 Success 或 Failure，其他值不进入这两组。
    成功记录的 content.strategic_guidelines 应为字典；上游生成失败而留下
    None 的记录需要先清理，当前成功分支不会自动修复这类输入。
    """
    cfg = ENV_CONFIG[env]
    successes = [m for m in memories if m.get("tags", {}).get("outcome") == "Success"]
    failures = [m for m in memories if m.get("tags", {}).get("outcome") == "Failure"]

    # 1. 汇总全部成功记录的 planning_pattern，生成跨任务的通用策略。
    # 即使筛选后没有有效 pattern，这里仍会请求模型，不会自动跳过。
    # ---- General skills ------------------------------------------------
    print(f"Aggregating General Skills from {len(successes)} success memories...")
    all_patterns = [
        m.get("content", {}).get("strategic_guidelines", {}).get("planning_pattern", "")
        for m in successes
    ]
    all_patterns = [p for p in all_patterns if p]
    general = call_llm(
        client, model, GENERAL_PRINCIPLES_PROMPT, {"planning_patterns": all_patterns}
    )
    general = _assign_skill_ids(general, prefix="gen")

    # 2. 将成功记录按任务类别分桶，每个非空桶调用模型一次。
    # 未识别类别的记录不参与分类技能汇总，但其有效 pattern 仍可参与通用汇总。
    # ---- Per-category skills -------------------------------------------
    by_cat: dict[str, list[str]] = {c: [] for c in cfg["categories"]}
    for m in successes:
        cat = cfg["classifier"](m)
        if cat is None or cat not in by_cat:
            continue
        pat = (
            m.get("content", {})
            .get("strategic_guidelines", {})
            .get("planning_pattern", "")
        )
        if pat:
            by_cat[cat].append(pat)

    category_skills: dict[str, list] = {}
    for cat in cfg["categories"]:
        patterns = by_cat[cat]
        if not patterns:
            print(f"  [{cat}] no memories, skipping")
            category_skills[cat] = []
            continue
        if cfg["label_map"]:
            label = cfg["label_map"].get(cat, cat.replace("_", " ").title())
        else:
            label = cat.replace("_", " ").title()
        print(f"  [{cat}] aggregating from {len(patterns)} memories...")
        prompt = CATEGORY_SKILLS_PROMPT.format(category_label=label)
        skills = call_llm(client, model, prompt, {"planning_patterns": patterns})
        category_skills[cat] = _assign_skill_ids(skills, prefix=_id_prefix(cat))

    # 3. 仅从失败记忆收集 mistakes_to_avoid，交给模型归纳错误及改进方法。
    # 没有错误条目时直接返回空列表；错误去重要求写在提示词中，并非本地算法。
    # ---- Common mistakes -----------------------------------------------
    print(f"Aggregating Common Mistakes from {len(failures)} failure memories...")
    all_mistakes: list = []
    for m in failures:
        sg = m.get("content", {}).get("strategic_guidelines", {}) or {}
        for mis in sg.get("mistakes_to_avoid", []) or []:
            all_mistakes.append(mis)
    if all_mistakes:
        mistakes = call_llm(client, model, MISTAKES_PROMPT, {"mistakes": all_mistakes})
        mistakes = _assign_mistake_ids(mistakes)
    else:
        mistakes = []

    # ALFWorld/WebShop 用 task_specific_skills，Search 用 query_type_skills。
    # 分类为空时仍保留该类别键，便于调用方区分“已知类别无样本”和“未知类别”。
    return {
        "general_skills": general,
        cfg["category_key"]: category_skills,
        "common_mistakes": mistakes,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file", required=True, help="generated_memories_*.json from stage 3"
    )
    parser.add_argument("--output_file", required=True)
    parser.add_argument(
        "--env", choices=["alfworld", "webshop", "search"], required=True
    )
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument(
        "--api_base",
        default="https://api.deepseek.com",
        help="Custom API base URL (e.g., https://api.deepseek.com)",
    )
    args = parser.parse_args()

    api_key = _load_api_key_from_env_file()
    if not api_key:
        raise SystemExit(
            "API key not found. Set DEEPSEEK_API_KEY in .env (project root)."
        )

    client = OpenAI(api_key=api_key, base_url=args.api_base)

    with open(args.input_file, "r", encoding="utf-8") as f:
        memories = json.load(f)

    # 当前一次读入全部记忆、串行发起请求，最后统一保存；不提供分批或断点续跑。
    skill_bank = aggregate(memories, args.env, client, args.model)

    os.makedirs(
        os.path.dirname(os.path.abspath(args.output_file)) or ".", exist_ok=True
    )
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(skill_bank, f, indent=2, ensure_ascii=False)

    cat_key = ENV_CONFIG[args.env]["category_key"]
    print(f"\nWrote skill bank to {args.output_file}")
    print(f"  general_skills: {len(skill_bank['general_skills'])} items")
    for cat, items in skill_bank[cat_key].items():
        print(f"  {cat_key}[{cat}]: {len(items)} items")
    print(f"  common_mistakes: {len(skill_bank['common_mistakes'])} items")


if __name__ == "__main__":
    main()

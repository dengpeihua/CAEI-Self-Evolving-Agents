"""
RLSD (Reinforcement Learning with Self-Distillation) utilities.

This module provides skill-based privileged information loading and
token-level advantage computation for the RLSD algorithm.
Skill files and task keyword mappings are loaded from skill_mapping.json.
Dataset-specific routing rules remain in get_privileged_info_from_data_source.
"""

# 本模块提供两类功能：
# 1. SkillProvider：按任务读取经验文本，供调用方拼入观测或教师输入；
# 2. compute_rlsd_token_advantage：根据教师/学生概率差重加权序列优势。
# 它不执行“加热/清洗”等技能动作，也不负责模型生成、环境交互或优化器更新。

import json
import os
from typing import Dict, List, Optional

import torch


def load_skill_mapping(skills_dir: str) -> dict:
    """读取技能文件表、任务到技能映射，以及可选的提示词匹配关键词。

    文件缺失或 JSON 非法时直接抛出异常，不会自动创建默认技能库。
    """
    mapping_path = os.path.join(skills_dir, "skill_mapping.json")
    with open(mapping_path, "r") as f:
        return json.load(f)


def load_skill_content(skills_dir: str, skill_mapping: dict) -> Dict[str, str]:
    """按 skill_files 配置一次性加载 Markdown，缓存为技能名到正文的字典。

    路径相对 skills_dir；只读取文本，不解析 Markdown，也不执行文件内容。
    配置引用的任一文件缺失都会报错，不会跳过该项继续初始化。
    """
    contents = {}
    for skill_name, filename in skill_mapping["skill_files"].items():
        filepath = os.path.join(skills_dir, filename)
        with open(filepath, "r") as f:
            contents[skill_name] = f.read().strip()
    return contents


class SkillProvider:
    """加载并缓存任务经验，按不同输入方式返回可拼入提示词的文本。

    skill_mapping.json 中：
      - skill_files：技能名 -> Markdown 文件名；
      - task_to_skill：任务类型 -> 技能名，多个任务可以共享同一技能；
      - task_keywords：任务类型 -> 关键词列表，可省略。

    路径匹配取首个命中任务；提示词匹配则保留所有命中任务，每类命中任一
    关键词即可。skill_all=True 时忽略任务匹配，返回全部已配置技能。
    """

    def __init__(self, skills_dir: str, skill_all: bool = False):
        self.skills_dir = skills_dir
        self.skill_all = skill_all
        self.skill_mapping = load_skill_mapping(skills_dir)
        self.skill_contents = load_skill_content(skills_dir, self.skill_mapping)
        self.task_to_skill = self.skill_mapping["task_to_skill"]
        # 仅初始化时读取文件；后续修改磁盘内容不会自动刷新当前实例的缓存。
        # 提示词匹配使用 any：一类中任一关键词出现即可命中。
        self.task_keywords: Dict[str, List[str]] = self.skill_mapping.get("task_keywords", {})

        if self.skill_all:
            self._all_skills_text = self._build_all_skills_text()

    def _build_all_skills_text(self) -> str:
        """通用技能放在开头，其余技能按文件映射顺序拼接，不做长度截断。"""
        general = self.skill_contents.get("general_skills", "")
        parts = [general]
        for skill_name, content in self.skill_contents.items():
            if skill_name != "general_skills":
                parts.append(content)
        return "\n\n".join(parts)

    def _get_skill_text(self, task_type: Optional[str]) -> str:
        """拼接通用技能和单个任务技能；未知任务或缺少映射时只返回通用部分。"""
        general = self.skill_contents.get("general_skills", "")
        parts = [general]
        if task_type:
            mapped_name = self.task_to_skill.get(task_type)
            if mapped_name and mapped_name in self.skill_contents:
                parts.append(self.skill_contents[mapped_name])
        return "\n\n".join(parts)

    def get_privileged_info(self, gamefile: str) -> str:
        """在环境文件路径中查找任务类型，按映射顺序取首个子串命中。

        例如路径含 pick_heat_then_place_in_recep 时，可通过映射选取 heat。
        当前匹配区分大小写，未命中时回退到通用技能。
        """
        if self.skill_all:
            return self._all_skills_text
        matched_task = None
        for task_type in self.task_to_skill:
            if task_type in gamefile:
                matched_task = task_type
                break
        return self._get_skill_text(matched_task)

    def get_privileged_info_from_prompt(self, prompt_text: str) -> str:
        """将提示词转小写后，以关键词子串匹配任务，拼接所有命中的任务技能。

        关键词配置应使用小写；这里不做词边界判断或语义检索。
        通用技能只追加一次，但任务技能不去重：如 put two 与 put 同时命中，
        两个任务都映射到 pick_and_place 时，该技能正文会被追加两次。
        """
        if self.skill_all:
            return self._all_skills_text
        text_lower = prompt_text.lower()
        matched_tasks = []
        for task_type, keywords in self.task_keywords.items():
            if keywords and any(kw in text_lower for kw in keywords):
                matched_tasks.append(task_type)

        if not matched_tasks:
            return self._get_skill_text(None)

        general = self.skill_contents.get("general_skills", "")
        parts = [general]
        for task_type in matched_tasks:
            mapped_name = self.task_to_skill.get(task_type)
            if mapped_name and mapped_name in self.skill_contents:
                parts.append(self.skill_contents[mapped_name])
        return "\n\n".join(parts)

    def get_privileged_info_from_data_source(self, data_source: str, prompt_text: str) -> str:
        """先按数据来源路由，再按问题文本兜底，顺序与下方分支一致。

        popqa -> 属性查询；nq/triviaqa -> 直接检索；hotpotqa -> 多跳。
        其余来源先检查 which、or 且无 for 的子串规则，命中时取 compare；
        否则 2wikimultihopqa/musique/bamboogle 取多跳。
        text 来源在上述规则均未选中任务时，再走提示词关键词匹配。
        其他情况返回通用技能。compare 等任务名必须在映射中存在才能选到专用技能。
        """
        if self.skill_all:
            return self._all_skills_text

        task_type = None
        if data_source == "popqa":
            task_type = "entity_attribute_lookup"
        elif data_source in ("nq", "triviaqa"):
            task_type = "direct_retrieval"
        elif data_source == "hotpotqa":
            task_type = "multi_hop_reasoning"
        else:
            text_lower = prompt_text.lower()
            if "which" in text_lower and "or" in text_lower and "for" not in text_lower:
                task_type = "compare"
            elif data_source in ("2wikimultihopqa", "musique", "bamboogle"):
                task_type = "multi_hop_reasoning"

        if task_type is None and data_source == "text":
            return self.get_privileged_info_from_prompt(prompt_text)
        return self._get_skill_text(task_type)


def compute_rlsd_token_advantage(
    seq_advantages: torch.Tensor,
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    rlsd_lambda: float = 0.5,
    rlsd_clip_eps: float = 0.2,
) -> torch.Tensor:
    """
    Compute token-level RLSD advantages.

    计算流程：取每条响应的序列优势 A -> 计算教师与学生的 log 概率差
    delta -> 得到权重 w = exp(sign(A) * delta) -> 对称裁剪权重
    -> 输出 A * ((1 - lambda) + lambda * w)，最后屏蔽无效 token。
    这里裁剪的是优势权重，不是 PPO 新旧策略概率比，也不是 DAPO Clip-Higher。

    Args:
        seq_advantages: (bs, response_length) — standard GRPO sequence-level advantage
            broadcast to all tokens (same value per sequence).
        student_log_probs: (bs, response_length) — log π_θ(y_t | x, y_<t).
        teacher_log_probs: (bs, response_length) — log π_θ(y_t | x, r, y_<t).
        response_mask: (bs, response_length) — mask for valid response tokens.
        rlsd_lambda: mixing coefficient λ. When 0, degrades to standard GRPO.
        rlsd_clip_eps: clipping bound ε_w for token weights.

    Returns:
        token_advantages: (bs, response_length) — token-level advantage Â_t.
    """
    with torch.no_grad():
        # seq_advantages 的有效 token 应共享同一序列优势，因此取首个有效位置。
        # 若一行 mask 全零，argmax 返回 0；该行有限的输出值最终会被 mask 置零。
        first_valid = response_mask.int().argmax(dim=-1)  # (bs,)
        batch_indices = torch.arange(seq_advantages.size(0), device=seq_advantages.device)
        A_seq = seq_advantages[batch_indices, first_valid]  # (bs,)

        # 两套概率必须对应同一批响应 token 和前缀；教师输入额外包含技能信息。
        # 本函数只消费预先算好的概率，不在这里运行教师模型。
        delta_t = teacher_log_probs - student_log_probs  # (bs, response_length)

        sign_A = torch.sign(A_seq).unsqueeze(-1)  # (bs, 1)
        # A>0 时，更受教师支持的 token 得到更强奖励；
        # A<0 时，教师较不支持的 token 得到更强惩罚。裁剪限制权重调整幅度。
        w_t = torch.exp(sign_A * delta_t)  # (bs, response_length)
        w_t = torch.clamp(w_t, 1.0 - rlsd_clip_eps, 1.0 + rlsd_clip_eps)

        A_seq_expanded = A_seq.unsqueeze(-1)  # (bs, 1)
        # lambda=0 保留原序列优势；lambda=1 完全采用加权优势。
        # no_grad 防止梯度穿过优势构造，梯度更新由外部策略损失负责。
        token_advantages = A_seq_expanded * ((1.0 - rlsd_lambda) + rlsd_lambda * w_t)
        token_advantages = token_advantages * response_mask

    return token_advantages

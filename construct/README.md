# construct 中文逐文件说明

本目录覆盖事实记忆索引、认知特征与分类器、辅助诊断，不全部属于项目笔记 2.2.1。

## 文件职责

| 文件 | 作用 | 章节关系 |
|---|---|---|
| `index/build_index.py` | 读取轨迹，筛选成功或失败轨迹，按有效步骤间隔取 curr_prompt 作为检索键；把状态、动作、后续轨迹和结果作为返回值，调用嵌入服务并保存 HNSW。 | 直接对应 2.2.1；向量编码和索引保存还对应 2.2.2 |
| `index/build_index_traj_level.py` | 每条轨迹使用第一个步骤的提示作为键，最后有效提示中恢复的整条轨迹及最终结果作为值；用于轨迹级检索基线。 | 不属于状态拆分；属于轨迹级替代方案 |
| `index/build_index_reasoningbank.py` | 把任务、轨迹和结果交给语言模型归纳最多三个记忆项；并发请求后解析 Markdown，记忆项文本同时作为键和值，再编码建索引。 | 不属于状态拆分；属于策略级替代方案 |
| `index/experimental/build_failure_index.py` | 读取硬编码的评测轨迹，恢复历史后排除成功轨迹，逐步生成失败经验；保存 eval_in_distribution_v2_ 前缀的索引。 | 包含状态拆分，但属于评测轨迹实验分支 |
| `index/experimental/build_failure_only_index.py` | 先统计并跳过成功轨迹，再生成失败步骤条目；创建输出目录，空集合时停止建索引，保留成功和失败数量诊断。 | 包含状态拆分，但属于失败经验实验分支 |
| `index/experimental/build_mixed_index.py` | 同一份硬编码评测日志中保留成功与失败轨迹，逐步生成经验并统计轨迹数量，使用 mixed 前缀区分产物。 | 包含状态拆分，但属于混合经验实验分支 |
| `index/utils/check_data_structure.py` | 检查首条轨迹及特定轨迹字段，打印前二十个 LDA 决策，统计首步和后续步骤、无穷值和普通分数的刷新情况。 | 不属于；只读诊断，不生成检索条目 |
| `index/utils/convert_traj_to_memory_input.py` | 按 env_num 合并同环境的多条轨迹，提取任务和观测，构造 env_id、type、trajectories 等字段，供独立记忆生成脚本使用。 | 不属于；输出嵌套轨迹 JSON，不建立 HNSW |
| `lda/extract_thinking_samples.py` | 解析思考与动作，用最近五步动作的去重比例和最终成功标记构造成功或失败与果断或犹豫的四类标签。 | 不属于；为认知表示分析准备启发式标签 |
| `lda/extract_confident_hesitant.py` | 成功轨迹中选择近期动作去重比例不超过 0.5 且当前动作重复至少两次的步骤；失败轨迹中选择思考包含 Nothing happens 的步骤。 | 不属于；为 LDA 准备 confident 和 hesitant 标签 |
| `lda/extract_confident_hesitant_v2.py` | 成功样本要求承诺标记至少一次、不确定标记不超过一次且提示不含 Nothing happens；失败样本要求不确定标记至少两次且承诺标记不超过一次。 | 不属于；改进认知标签规则 |
| `lda/extract_planning_tokens.py` | 用任务、截断观测、思考和动作重建对话，执行模型前向传播，取指定层最后一个有效输入 token 的隐藏向量，并保存对应标签。 | 不属于；提取 LDA 或聚类输入特征 |
| `lda/cluster_planning_tokens.py` | 先 PCA 降维，再扫描 K-Means 簇数，按轮廓系数选择最佳簇数，计算 ARI、NMI、列联表和 Cramér V，保存投影和聚类结果。 | 不属于；无监督探索，不是必经训练步骤 |
| `lda/lda_planning_classifier.py` | 标准化向量，训练全量 LDA 检查一维投影，再分层交叉验证；保存分数和指标，并检查分数与思考字符长度的相关性。 | 不属于；训练认知方向分类器 |
| `lda/lda_planning_classifier_v2.py` | 每个训练折单独拟合标准化器，对比 LDA、逻辑回归和四隐藏单元 MLP；计算折外指标，打印极端投影样本并与第一版数据比较。 | 不属于；认知分类器训练与比较 |
| `lda/calculate_trajectory_stats.py` | 按数据集和记忆模式选择日志与输入输出字段，用固定 Qwen tokenizer 统计动作数和 token 数；对 adamem-low 区分记忆和动作成本。 | 不属于；运行成本分析 |
| `__init__.py` | 原文件为空；标记该目录为 Python 包，本身不读取轨迹、不建索引、不训练分类器。 | 不属于；包结构文件 |
| `index/__init__.py` | 原文件为空；标记该目录为 Python 包，本身不读取轨迹、不建索引、不训练分类器。 | 不属于；包结构文件 |
| `lda/__init__.py` | 原文件为空；标记该目录为 Python 包，本身不读取轨迹、不建索引、不训练分类器。 | 不属于；包结构文件 |

## 真实依赖流程

- 状态索引：轨迹 → 有效步骤及采样 → key/value → 嵌入服务 → HNSW 与元数据及 values。
- 认知分类：轨迹 → 二分类样本 v1 或 v2 → 显式配置 planning-token 提取器的输入输出 → 同版本分类器。
- 聚类探索：轨迹 → 四分类样本 → 隐藏向量 → PCA 与 K-Means；此分支不是 LDA 必经步骤。
- 转换、日志检查和 token 统计分别独立运行。

## 当前代码边界

- 两个分类器均未导出运行时需要的 lda_model_v2.npz；下方英文 README 的模型清单描述预期产物，并非本目录现存文件。
- v1 先全量标准化再交叉验证；v2 在训练折内标准化，但两者都未按轨迹进行分组验证。
- v2 的动作多样性只记录，不参与筛选；规则标签不等于人工逐步正确性标注。
- experimental 中日志路径硬编码，失败脚本输出名重合；PROJECT_ROOT 层级也不正确，独立执行需注意导入路径。
- 多数 data/ 路径基于运行目录，脚本不统一创建；特征提取默认输出 planning_*，分类器默认读取 lda_*，需显式对齐。
- 中文注释按功能分段，说明每段的主要处理过程、输入输出和必要限制；代码逻辑与字符串保持不变。原始源码备份位于项目内 .construct_annotation_work/original_construct。

## 原英文说明

# Construct — Memory Index Building & LDA Classifier Training

This directory contains scripts for building the long-term memory retrieval index
and training the LDA (Linear Discriminant Analysis) binary classifier used for
the "when to refresh" memory policy.

## Subdirectories

### `index/` — HNSW Memory Index Construction

Build step-level, episode-level, and strategy-level HNSW indexes from successful
training trajectories.

**Core scripts:**

| Script | Purpose |
|---|---|
| `build_index.py` | Step-level index (used by all AdaMEM variants) — supports `--correct_only` |
| `build_index_traj_level.py` | Episode-level index (for synapse baseline) |
| `build_index_reasoningbank.py` | Strategy index (for reasoningbank baseline) |

**Quick start:**
```bash
# Step-level index (used by all AdaMEM variants)
python construct/index/build_index.py \
    --dataset_name alfworld \
    --base_model_name Qwen/Qwen3-4B-Instruct-2507 \
    --correct_only

# Episode-level index (for synapse baseline)
python construct/index/build_index_traj_level.py \
    --dataset_name alfworld \
    --base_model_name Qwen/Qwen3-4B-Instruct-2507

# Strategy index (for reasoningbank baseline)
python construct/index/build_index_reasoningbank.py \
    --dataset_name alfworld \
    --base_model_name Qwen/Qwen3-4B-Instruct-2507
```

**Subdirectories:**
- `utils/` — helper scripts (data structure inspection, trajectory conversion)
- `experimental/` — experimental index variants (failure-only, mixed index, etc.)

### `lda/` — LDA Confidence Classifier

Pipeline for training a linear classifier on planning-token hidden states to
distinguish "confident" vs "hesitant" reasoning. The resulting projection
direction is used as a per-step "should I refresh my strategy?" signal.

**Pipeline (in order):**

1. `extract_thinking_samples.py` — extract representative thinking samples from trajectories (4-way taxonomy)
2. `extract_confident_hesitant_v2.py` — extract confident/hesitant binary labeled samples
3. `extract_planning_tokens.py` — run HF model to extract planning-token hidden states
4. `cluster_planning_tokens.py` — PCA + K-Means clustering of planning tokens (exploratory)
5. `lda_planning_classifier_v2.py` — train LDA / LR / MLP classifiers, cross-validate, save model

**Data & artifacts (in `data/`):**

| File | Description |
|---|---|
| `thinking_samples.json` | Extracted thinking samples with 4-way tags |
| `confident_hesitant_samples_v2.json` | Binary labeled confident/hesitant samples (v2, larger) |
| `lda_vectors_v2.npy` | Planning-token hidden states for LDA training (v2) |
| `lda_labels_v2.json` | Binary labels for LDA training (v2) |
| `lda_model_v2.npz` | **Deployed LDA model** (w, b, scaler) — loaded by agent runtime |
| `lda_proj_v2.npy` | 1D LDA projection of training vectors |
| `lda_result_v2.json` | Cross-validation accuracy / AUC results |

**Sample inspection (in subdirectories):**
- `lda_thinks/` — top-5 confident + top-5 hesitant thinking samples (v1)
- `lda_thinks_v2/` — top-5 confident + top-5 hesitant thinking samples (v2)
- `cluster_thinks/` — K-Means cluster representative samples

**Other utility:**
- `calculate_trajectory_stats.py` — token-count statistics across trajectories

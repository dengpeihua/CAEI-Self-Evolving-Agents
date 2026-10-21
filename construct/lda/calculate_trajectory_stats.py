# 统计不同记忆模式下的动作数量与 token 成本，并拆分记忆和动作生成开销。
import json
import os
import sys
from transformers import AutoTokenizer
from tqdm import tqdm
from glob import glob

# 读取数据集、记忆模式和模型名称，按对应日志命名规则定位输入文件。
if len(sys.argv) != 4:
    print("Usage: python calculate_trajectory_stats.py <dataset_name> <mem_type> <model_name>")
    sys.exit(1)

dataset_name = sys.argv[1]
mem_type = sys.argv[2]
model_name = sys.argv[3]

base_dir = f"logs/{dataset_name}/{model_name.replace('/', '_')}"
if mem_type == "no_memory":
    if dataset_name == "alfworld_old":
        file_list = [
            os.path.join(base_dir, "traj_eval_in_distribution.json"),
        ]
    elif dataset_name == "webshop":
        file_list = [os.path.join(base_dir, "traj_eval.json")]
else:
    if dataset_name == "alfworld_old":
        file_list = [
            os.path.join(base_dir, f"traj_eval_in_distribution_{mem_type}.json"),
        ]
    elif dataset_name == "webshop":
        print(f"{base_dir}/traj_eval_{mem_type}.json")
        print(glob(f"{base_dir}/traj_eval_{mem_type}.json"))
        file_list = [sorted(glob(f"{base_dir}/traj_eval_{mem_type}.json"))[-1]]

# 使用固定 Qwen tokenizer 统一统计口径，再按记忆模式选择输入输出字段。
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Thinking-2507")

if mem_type == "adamem-high":
    input_fields = ["initial_prompt", "strategy_prompt", "action_prompt"]
    output_fields = ["initial_response", "strategy_response", "final_response"]
elif mem_type in ["adamem-max", "adamem-max-without-trajectory-memory"]:
    input_fields = ["strategy_prompt", "action_prompt"]
    output_fields = ["strategy_response", "final_response"]
elif mem_type == "adamem-low":
    input_fields = ["refresh_prompt", "strategy_prompt", "action_prompt"]
    output_fields = ["refresh_response", "strategy_response", "final_response"]
elif mem_type == "adamem-max-without-strategy-memory":
    input_fields = ["action_prompt"]
    output_fields = ["final_response"]
else:
    input_fields = ["curr_prompt"]
    output_fields = ["curr_action"]

print("input_fields", input_fields)
print("output_fields", output_fields)

# 初始化轨迹动作数、输入输出 token 及记忆与动作成本的累计值。
num_actions_list = []
total_input_tokens = 0
total_output_tokens = 0
total_actions = 0

total_memory_tokens = 0
total_action_tokens = 0

# 逐文件读取轨迹，过滤无效动作并为每条轨迹初始化计数。
for filename in file_list:
    filepath = filename
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        continue
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    for traj in tqdm(data, desc=f"Processing {filename}"):
        steps = traj['steps']
        valid_steps = [step for step in steps if step.get('curr_action') != 'None']
        num_actions = len(valid_steps)
        num_actions_list.append(num_actions)
        
        input_tokens_traj = 0
        output_tokens_traj = 0
        memory_tokens_traj = 0
        action_tokens_traj = 0
        
        # 对 adamem-low 推断当前步骤是否刷新策略，以决定相关 token 的成本归属。
        for step_idx, step in enumerate(valid_steps):
            if mem_type == "adamem-low":
                if step["refresh_prompt"] is None:
                    assert step_idx == 0
                    refresh = True
                elif step["action_prompt"] is None:
                    refresh = False
                else:
                    refresh = True

            # 统计有效输入字段的 token；按刷新状态归入记忆成本或动作成本。
            for field in input_fields:
                if field in step and step[field] and step[field] != 'None':
                    token_cnt = len(tokenizer.encode(step[field]))
                    input_tokens_traj += token_cnt
                    if mem_type == "adamem-low":
                        if field == "strategy_prompt":
                            memory_tokens_traj += token_cnt
                        elif field == "action_prompt":
                            action_tokens_traj += token_cnt
                        elif field == "refresh_prompt":
                            if refresh:
                                memory_tokens_traj += token_cnt
                            else:
                                action_tokens_traj += token_cnt
                        else:
                            assert False, f"Unexpected field {field} for mem_type {mem_type}"
                    else:
                        action_tokens_traj += token_cnt

            # 确认对应输入存在后统计输出 token，并按相同规则划分成本。
            for idx, field in enumerate(output_fields):
                if field in step and step[field] and step[input_fields[idx]] and step[field] != 'None':
                    token_cnt = len(tokenizer.encode(step[field]))
                    output_tokens_traj += token_cnt
                    if mem_type == "adamem-low":
                        if field == "strategy_response":
                            memory_tokens_traj += token_cnt
                        elif field == "final_response":
                            action_tokens_traj += token_cnt
                        elif field == "refresh_response":
                            if refresh:
                                memory_tokens_traj += token_cnt
                            else:
                                action_tokens_traj += token_cnt
                    else:
                        action_tokens_traj += token_cnt

        # 将当前轨迹的各项计数累加到全局统计。
        total_input_tokens += input_tokens_traj
        total_output_tokens += output_tokens_traj
        total_actions += num_actions
        total_memory_tokens += memory_tokens_traj
        total_action_tokens += action_tokens_traj

# 计算平均动作数与 token 成本，检查输入输出总量与成本拆分总量一致后打印。
if num_actions_list:
    num_traj = len(num_actions_list)
    avg_num_actions = sum(num_actions_list) / len(num_actions_list)
    if total_actions > 0:
        avg_input_tokens_per_action = total_input_tokens / total_actions
        avg_output_tokens_per_action = total_output_tokens / total_actions
    else:
        avg_input_tokens_per_action = 0
        avg_output_tokens_per_action = 0

    assert total_input_tokens + total_output_tokens == total_memory_tokens + total_action_tokens, "Token count mismatch!"
    
    print(f"Total trajectories processed: {len(num_actions_list)}")
    print(f"Average number of actions per trajectory: {avg_num_actions:.1f}")
    print(f"Average input tokens per action: {avg_input_tokens_per_action:.0f}")
    print(f"Average output tokens per action: {avg_output_tokens_per_action:.0f}")
    print(f"Average total tokens per action: {(avg_input_tokens_per_action + avg_output_tokens_per_action):.0f}")
    print("---- Memory vs Action Tokens ----")
    print(f"Average memory tokens per trajectory: {(total_memory_tokens / num_traj):.1f}")
    print(f"Average action tokens per trajectory: {(total_action_tokens / num_traj):.1f}")
    print(f"Average total tokens per trajectory: {((total_memory_tokens + total_action_tokens) / num_traj):.1f}")
else:
    print("No trajectories found.")

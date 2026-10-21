
import json
from collections import defaultdict

traj_file = "logs/alfworld/Qwen_Qwen3-4B-Instruct-2507/traj_eval_in_distribution_adamem-low_20260714_175615_849396e0.json"

with open(traj_file, "r") as f:
    data = json.load(f)

print(f"Number of trajectories: {len(data)}")
print()

total_tasks = 0
success_tasks = 0
traj_info = []

for env_idx, traj in enumerate(data):
    steps = traj.get("steps", [])
    won = traj.get("won", False)

    total_tasks += 1

    if won:
        success_tasks += 1

    # Try to get task type from first step
    task_type = None
    if len(steps) > 0:
        first_step = steps[0]
        # Check if we can infer task type from prompt
        if "curr_prompt" in first_step:
            prompt = first_step["curr_prompt"]
            if "pick and place" in prompt.lower():
                task_type = "pick_and_place"
            elif "pick two" in prompt.lower() or "pick_heat" in prompt.lower() or "pick_cool" in prompt.lower():
                task_type = "pick_two"
            elif "look" in prompt.lower() and "clean" in prompt.lower():
                task_type = "look_and_clean"
            elif "put" in prompt.lower() and "clean" in prompt.lower():
                task_type = "put_and_clean"
            else:
                task_type = "other"

    traj_info.append({
        "env_idx": env_idx,
        "success": bool(won),
        "task_type": task_type,
        "num_steps": len(steps),
        "trajectory": traj
    })

print(f"Total tasks: {total_tasks}")
print(f"Successful tasks: {success_tasks}")
print(f"Failed tasks: {total_tasks - success_tasks}")
print(f"Success rate: {success_tasks/total_tasks*100:.2f}%")
print()

# Count by task type
task_type_counts = defaultdict(lambda: {"total": 0, "success": 0})
for info in traj_info:
    task_type = info["task_type"] or "unknown"
    task_type_counts[task_type]["total"] += 1
    if info["success"]:
        task_type_counts[task_type]["success"] += 1

print("Task type breakdown:")
for task_type, counts in sorted(task_type_counts.items()):
    sr = counts["success"]/counts["total"]*100 if counts["total"] >0 else 0
    print(f"  {task_type}: {counts['success']}/{counts['total']} ({sr:.1f}%)")
print()

print(f"Average steps per successful task: {sum(info['num_steps'] for info in traj_info if info['success'])/success_tasks if success_tasks else 0:.1f}")
print(f"Average steps per failed task: {sum(info['num_steps'] for info in traj_info if not info['success'])/(total_tasks-success_tasks) if (total_tasks-success_tasks) else 0:.1f}")

# Save the full traj info for building index
with open("logs/alfworld/traj_info_with_success.json", "w") as f:
    # Save without the full trajectory to save space
    out_info = []
    for info in traj_info:
        info_copy = info.copy()
        # Keep only first 2 and last 2 steps for reference
        traj = info_copy.pop("trajectory")
        steps = traj.get("steps", [])
        info_copy["steps_sample"] = steps[:2] + steps[-2:] if len(steps) >4 else steps
        out_info.append(info_copy)
    json.dump(out_info, f, indent=2)

print("\nTrajectory info saved to logs/alfworld/traj_info_with_success.json")
print()

# Save a format suitable for build_index.py
print("Preparing trajectories for index building...")
index_format_trajs = []
for info in traj_info:
    traj = info["trajectory"]
    steps = traj.get("steps", [])
    if len(steps) == 0:
        continue

    # Extract the initial prompt (task description)
    first_step = steps[0]
    task_description = first_step.get("curr_prompt", "")

    index_format_trajs.append({
        "env_idx": info["env_idx"],
        "task_description": task_description,
        "success": info["success"],
        "steps": steps,
        "won": traj.get("won", False)
    })

with open("logs/alfworld/trajectories_for_index.json", "w") as f:
    json.dump(index_format_trajs, f, indent=2)

print(f"Saved {len(index_format_trajs)} trajectories to logs/alfworld/trajectories_for_index.json")

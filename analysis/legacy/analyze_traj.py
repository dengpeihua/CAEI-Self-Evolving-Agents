
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
    if not steps:
        continue

    total_tasks += 1
    # Check if task was successful
    success = False
    final_reward = 0.0
    for step in reversed(steps):
        info = step.get("info", {})
        if "success" in info:
            success = bool(info["success"])
        if "reward" in step:
            final_reward = step["reward"]

    if success:
        success_tasks += 1

    task_type = None
    if steps and "task_type" in steps[0].get("info", {}):
        task_type = steps[0]["info"]["task_type"]
    elif steps and "extra.gamefile" in steps[0].get("info", {}):
        gamefile = steps[0]["info"]["extra.gamefile"]
        if gamefile:
            task_type = gamefile.split("/")[-1].split("-")[0]

    traj_info.append({
        "env_idx": env_idx,
        "success": success,
        "task_type": task_type,
        "num_steps": len(steps),
        "final_reward": final_reward
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

# Save trajectory info for reference
with open("logs/alfworld/traj_analysis.json", "w") as f:
    json.dump(traj_info, f, indent=2)
print("\nTrajectory analysis saved to logs/alfworld/traj_analysis.json")

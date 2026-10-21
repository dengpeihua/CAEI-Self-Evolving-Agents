
import json

traj_file = "logs/alfworld/Qwen_Qwen3-4B-Instruct-2507/traj_eval_in_distribution_adamem-low_20260714_175615_849396e0.json"

with open(traj_file, "r") as f:
    data = json.load(f)

print(f"Type of data: {type(data)}")
print(f"Length of data: {len(data)}")
print()

if len(data) > 0:
    first_traj = data[0]
    print("First trajectory keys:", list(first_traj.keys()))
    print()

    if "steps" in first_traj:
        steps = first_traj["steps"]
        print(f"Number of steps in first traj: {len(steps)}")
        print()
        if len(steps) > 0:
            first_step = steps[0]
            print("First step keys:", list(first_step.keys()))
            print()
            if "info" in first_step:
                print("First step info keys:", list(first_step["info"].keys()))
                print()
                print("First step info:", first_step["info"])
            print()
        # Check last step
        if len(steps) > 1:
            last_step = steps[-1]
            print("Last step keys:", list(last_step.keys()))
            if "info" in last_step:
                print("Last step info keys:", list(last_step["info"].keys()))
                print("\nLast step info:")
                for k, v in last_step["info"].items():
                    print(f"  {k}: {v}")
                print()

print("\n--- Checking success indicator ---")
for i in range(min(5, len(data))):
    traj = data[i]
    steps = traj.get("steps", [])
    success = None
    for step in steps:
        info = step.get("info", {})
        if "success" in info:
            success = info["success"]
        if "reward" in step and step["reward"] > 0:
            print(f"Traj {i}: reward = {step['reward']}")

    print(f"Traj {i}: success = {success}, num_steps = {len(steps)}")


import pickle

values_path = "retrieval_data/alfworld/Qwen_Qwen3-4B-Instruct-2507/eval_in_distribution_all_values.pkl"

with open(values_path, "rb") as f:
    values = pickle.load(f)

print(f"Total items: {len(values)}")
print()

success_count = 0
failure_count = 0

for i, v in enumerate(values[:20]):  # Check first 20 items
    if "Final Outcome:" in v:
        if "success" in v:
            success_count += 1
        elif "failure" in v:
            failure_count += 1

print(f"First 20 items: {success_count} success, {failure_count} failure")
print()

print("Sample value 0:")
print(values[0][:500])
print()
print("Sample value 1:")
print(values[1][:500])

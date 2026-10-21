
import pickle

values_path = "retrieval_data/alfworld/Qwen_Qwen3-4B-Instruct-2507/eval_in_distribution_all_values.pkl"

with open(values_path, "rb") as f:
    values = pickle.load(f)

print(f"Total items: {len(values)}")
print()

success_count = 0
failure_count = 0

for i, v in enumerate(values):
    if "Final Outcome:" in v:
        if "success" in v:
            success_count += 1
        elif "failure" in v:
            failure_count += 1

print(f"All items: {success_count} success, {failure_count} failure")
print(f"Success rate: {success_count/(success_count+failure_count)*100:.1f}%")
print()

# Find a success example
print("Looking for a success example...")
for i, v in enumerate(values):
    if "Final Outcome:" in v and "success" in v:
        print(f"\nSuccess example at index {i}:")
        print(v[:600])
        break

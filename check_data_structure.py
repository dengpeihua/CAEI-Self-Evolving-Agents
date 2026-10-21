#!/usr/bin/env python3
"""Check the actual data structure in the trajectory file."""

import json

file_path = "logs/alfworld/Qwen_Qwen3-4B-Instruct-2507/traj_eval_in_distribution_adamem_lda_20260731_102528_a2bc0ee7.json"

with open(file_path, 'r') as f:
    trajectories = json.load(f)

print(f"Number of trajectories: {len(trajectories)}")

# Check first trajectory structure
print(f"\n{'='*80}")
print(f"First trajectory structure:")
print(f"{'='*80}")
traj0 = trajectories[0]
print(f"Keys: {list(traj0.keys())}")
print(f"Number of steps: {len(traj0.get('steps', []))}")

if traj0.get('steps'):
    print(f"\nStep 0 keys: {list(traj0['steps'][0].keys())}")
    print(f"\nStep 0 data:")
    for k, v in traj0['steps'][0].items():
        if k in ['initial_response', 'strategy_prompt', 'final_response']:
            print(f"  {k}: {repr(str(v)[:200])}...")
        else:
            print(f"  {k}: {v}")

# Look for step 0 of traj 4 (which had LDA=inf but refresh=True)
print(f"\n{'='*80}")
print(f"Trajectory 4, step 0 (LDA=inf case):")
print(f"{'='*80}")
if len(trajectories) > 4 and trajectories[4].get('steps'):
    step = trajectories[4]['steps'][0]
    for k, v in step.items():
        if k in ['initial_response', 'strategy_prompt', 'final_response']:
            print(f"  {k}: {repr(str(v)[:200])}...")
        else:
            print(f"  {k}: {v}")

# Let's manually check refresh decisions vs LDA score
print(f"\n{'='*80}")
print(f"Manual check of first 20 steps:")
print(f"{'='*80}")
print(f"{'Step':<10} {'LDA score':<15} {'LDA decision':<15} {'Refresh':<10} {'Reason'}")
print(f"{'-'*80}")

count = 0
for traj_idx, traj in enumerate(trajectories):
    if count >= 20:
        break
    for step in traj.get('steps', []):
        if count >= 20:
            break
        if 'lda_score' in step:
            lda_score = step.get('lda_score')
            lda_decision = step.get('lda_decision')
            refresh = step.get('refresh_decision')
            reason = step.get('refresh_reason', '')[:50]
            print(f"{traj_idx}-{step.get('step_idx'):<5} {str(lda_score):<15} {str(lda_decision):<15} {str(refresh):<10} {reason}")
            count += 1

# Look specifically at first step vs later steps
print(f"\n{'='*80}")
print(f"Step 0 vs later steps - refresh rate:")
print(f"{'='*80}")
step0_count = 0
step0_refresh = 0
later_count = 0
later_refresh = 0

for traj in trajectories:
    for step in traj.get('steps', []):
        if 'lda_score' not in step:
            continue
        step_idx = step.get('step_idx', -1)
        refresh = step.get('refresh_decision', False)
        if step_idx == 0:
            step0_count += 1
            if refresh:
                step0_refresh += 1
        else:
            later_count += 1
            if refresh:
                later_refresh += 1

print(f"Step 0: {step0_refresh}/{step0_count} ({step0_refresh/step0_count*100:.1f}%) refresh")
print(f"Later steps: {later_refresh}/{later_count} ({later_refresh/later_count*100:.1f}%) refresh")

# Look for "inf" scores
print(f"\n{'='*80}")
print(f"inf score analysis:")
print(f"{'='*80}")
inf_count = 0
inf_refresh = 0
normal_count = 0
normal_refresh = 0

for traj in trajectories:
    for step in traj.get('steps', []):
        if 'lda_score' not in step:
            continue
        lda_score = step.get('lda_score')
        refresh = step.get('refresh_decision', False)
        if lda_score == float('inf') or lda_score == 'inf':
            inf_count += 1
            if refresh:
                inf_refresh += 1
        else:
            normal_count += 1
            if refresh:
                normal_refresh += 1

print(f"inf scores: {inf_count} steps, refresh={inf_refresh} ({inf_refresh/inf_count*100:.1f}% if inf_count>0 else 0)")
print(f"normal scores: {normal_count} steps, refresh={normal_refresh} ({normal_refresh/normal_count*100:.1f}% if normal_count>0 else 0)")

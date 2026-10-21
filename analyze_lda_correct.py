#!/usr/bin/env python3
"""Correct analysis of LDA decisions in trajectory file."""

import json
import re
import numpy as np
from collections import defaultdict, Counter
import argparse


def extract_thinking_content(text):
    """Extract thinking content from model response."""
    # Pattern 1: <tool_call> <tool_call> ... <tool_call>
    pattern1 = r'<tool_call> <tool_call>\s*(.*?)\s*<tool_call>'
    match1 = re.search(pattern1, text, re.DOTALL)
    if match1:
        return match1.group(1).strip()

    # Pattern 2: Look for thinking section
    pattern2 = r'<think>\s*(.*?)\s*</think>'
    match2 = re.search(pattern2, text, re.DOTALL)
    if match2:
        return match2.group(1).strip()

    # Return first 500 chars if no pattern found
    return text[:500]


def analyze_thinking_quality(thinking_text):
    """
    Analyze thinking content more carefully.
    """
    if not thinking_text:
        return {"quality": "empty", "indicators": []}

    indicators = []
    text_lower = thinking_text.lower()

    # Features to look for
    features = {
        "has_step_by_step": "step" in text_lower and ("1" in text_lower or "first" in text_lower),
        "has_goal_analysis": "goal" in text_lower or "task" in text_lower,
        "has_location_reasoning": any(loc in text_lower for loc in ["cabinet", "drawer", "dresser", "shelf"]),
        "has_uncertainty": any(w in text_lower for w in ["maybe", "might", "could", "if", "not sure"]),
        "has_certainty": any(w in text_lower for w in ["definitely", "certainly", "must", "need to", "will"]),
        "has_retrieval_ref": "strategy" in text_lower or "retrieved" in text_lower or "past" in text_lower,
        "length": len(thinking_text),
    }

    # Simple heuristic: longer thinking with more structure = more thoughtful (maybe more hesitant?)
    thoughtfulness_score = 0
    if features["has_step_by_step"]:
        thoughtfulness_score += 1
    if features["has_goal_analysis"]:
        thoughtfulness_score += 1
    if features["has_location_reasoning"]:
        thoughtfulness_score += 1
    if features["length"] > 200:
        thoughtfulness_score += 1

    return {
        "features": features,
        "thoughtfulness_score": thoughtfulness_score,
    }


def main(file_path):
    print(f"Loading trajectory file: {file_path}")
    with open(file_path, 'r') as f:
        trajectories = json.load(f)

    print(f"Loaded {len(trajectories)} trajectories")

    # Collect all steps with LDA decisions (EXCLUDING step 0 which always refreshes)
    all_steps = []
    step0_steps = []
    non_step0_steps = []

    for traj_idx, traj in enumerate(trajectories):
        if not traj.get('steps'):
            continue

        for step in traj['steps']:
            if 'lda_score' not in step:
                continue

            step_data = {
                'traj_idx': traj_idx,
                'step_idx': step.get('step_idx', -1),
                'won': traj.get('won'),
                'lda_score': step.get('lda_score'),
                'lda_decision': step.get('lda_decision'),
                'refresh_decision': step.get('refresh_decision'),
                'refresh_reason': step.get('refresh_reason', ''),
                'initial_response': step.get('initial_response', ''),
                'strategy_prompt': step.get('strategy_prompt'),
                'has_cached_strategy': step.get('step_idx', -1) > 0 and step.get('step_idx', -1) is not None,
            }

            # Extract thinking content
            thinking = extract_thinking_content(step_data['initial_response'])
            step_data['thinking'] = thinking

            # Analyze thinking quality
            step_data['thinking_analysis'] = analyze_thinking_quality(thinking)

            all_steps.append(step_data)

            if step_data['step_idx'] == 0:
                step0_steps.append(step_data)
            else:
                non_step0_steps.append(step_data)

    # Focus on non-step0 steps where there IS a cached strategy to reuse
    print(f"\n{'='*80}")
    print(f"OVERALL STATISTICS")
    print(f"{'='*80}")
    print(f"Total steps with LDA data: {len(all_steps)}")
    print(f"  - Step 0 (always refresh): {len(step0_steps)}")
    print(f"  - Non-step0 (actual decision): {len(non_step0_steps)}")

    non_step0 = non_step0_steps
    print(f"\nNon-step0 decisions:")
    print(f"  - Refresh: {sum(1 for s in non_step0 if s['refresh_decision'])} ({sum(1 for s in non_step0 if s['refresh_decision'])/len(non_step0)*100:.1f}%)")
    print(f"  - Reuse: {sum(1 for s in non_step0 if not s['refresh_decision'])} ({sum(1 for s in non_step0 if not s['refresh_decision'])/len(non_step0)*100:.1f}%)")

    # LDA score distribution (excluding inf for stats)
    valid_scores = [s['lda_score'] for s in non_step0 if s['lda_score'] != float('inf') and s['lda_score'] is not None]
    inf_count = sum(1 for s in non_step0 if s['lda_score'] == float('inf'))

    print(f"\nLDA score statistics (excluding inf):")
    if valid_scores:
        print(f"  Mean: {np.mean(valid_scores):.3f}")
        print(f"  Std:  {np.std(valid_scores):.3f}")
        print(f"  Min:  {np.min(valid_scores):.3f}")
        print(f"  Max:  {np.max(valid_scores):.3f}")
    print(f"  inf count: {inf_count}")

    # Decision accuracy analysis - THIS IS THE KEY
    print(f"\n{'='*80}")
    print(f"DECISION LOGIC ANALYSIS")
    print(f"{'='*80}")

    print(f"\nDecision rule:")
    print(f"  - LDA score < 0  → refresh strategy")
    print(f"  - LDA score >= 0 → reuse strategy")
    print(f"  - inf score → reuse strategy")

    # Check if decisions follow the rule
    correct_decisions = 0
    incorrect_decisions = 0
    incorrect_examples = []

    for s in non_step0:
        lda_score = s['lda_score']
        refresh = s['refresh_decision']

        # What should the decision be?
        should_refresh = False
        if lda_score == float('inf'):
            should_refresh = False
        elif lda_score < 0:
            should_refresh = True
        else:
            should_refresh = False

        if refresh == should_refresh:
            correct_decisions += 1
        else:
            incorrect_decisions += 1
            incorrect_examples.append({
                'step': s,
                'lda_score': lda_score,
                'refresh': refresh,
                'should_refresh': should_refresh,
            })

    print(f"\nDecision accuracy:")
    print(f"  Correct: {correct_decisions} ({correct_decisions/len(non_step0)*100:.2f}%)")
    print(f"  Incorrect: {incorrect_decisions} ({incorrect_decisions/len(non_step0)*100:.2f}%)")

    if incorrect_examples:
        print(f"\nIncorrect decision examples ({len(incorrect_examples)} total):")
        for i, ex in enumerate(incorrect_examples[:5]):
            s = ex['step']
            print(f"\n  {i+1}: Traj {s['traj_idx']}, Step {s['step_idx']}")
            print(f"     LDA score: {ex['lda_score']}")
            print(f"     Actual refresh: {ex['refresh']}")
            print(f"     Should refresh: {ex['should_refresh']}")
            print(f"     Reason: {s['refresh_reason']}")

    # Now look at the thinking content vs LDA score
    print(f"\n{'='*80}")
    print(f"THINKING CONTENT vs LDA SCORE")
    print(f"{'='*80}")

    # Group by LDA score ranges and see thinking characteristics
    print(f"\nThinking analysis by LDA score bin:")
    print(f"{'LDA Bin':<20} {'Count':<8} {'Refresh':<10} {'Avg Thought Len':<20} {'Has Step-by-Step':<15}")
    print(f"{'-'*80}")

    bins = [
        ('<-5', lambda x: x != float('inf') and x < -5),
        ('-5 to 0', lambda x: x != float('inf') and -5 <= x < 0),
        ('0 to 5', lambda x: x != float('inf') and 0 <= x < 5),
        ('>=5', lambda x: x != float('inf') and x >= 5),
        ('inf', lambda x: x == float('inf')),
    ]

    bin_stats = {}
    for bin_name, bin_func in bins:
        bin_steps = [s for s in non_step0 if bin_func(s['lda_score'])]
        if bin_steps:
            avg_len = np.mean([len(s['thinking']) for s in bin_steps])
            has_step = sum(1 for s in bin_steps if s['thinking_analysis']['features']['has_step_by_step'])
            refresh_rate = sum(1 for s in bin_steps if s['refresh_decision']) / len(bin_steps) * 100
            print(f"{bin_name:<20} {len(bin_steps):<8} {refresh_rate:>6.1f}% {avg_len:>15.1f} chars {has_step/len(bin_steps)*100:>12.1f}%")
            bin_stats[bin_name] = {
                'steps': bin_steps,
                'avg_len': avg_len,
                'has_step': has_step / len(bin_steps),
            }

    # Now let's look at some concrete examples - this is what you asked for!
    print(f"\n{'='*80}")
    print(f"DEEP DIVE: LDA=hesitant (score < 0) → refresh")
    print(f"  Is the agent actually hesitant in these cases?")
    print(f"{'='*80}")

    hesitant_steps = [s for s in non_step0 if s['lda_score'] != float('inf') and s['lda_score'] < 0]
    print(f"\nNumber of hesitant (LDA < 0) cases: {len(hesitant_steps)}")
    print(f"Refresh rate for hesitant: {sum(1 for s in hesitant_steps if s['refresh_decision'])/len(hesitant_steps)*100:.1f}%")

    print(f"\nExamples of LDA=hesitant (score < 0):")
    for i, s in enumerate(hesitant_steps[:5]):
        print(f"\n--- Example {i+1}: Traj {s['traj_idx']}, Step {s['step_idx']}, LDA={s['lda_score']:.2f} ---")
        print(f"Refresh decision: {s['refresh_decision']}")
        print(f"Thinking (truncated):")
        thinking = s['thinking']
        # Print in a readable way, break into lines
        if len(thinking) > 800:
            thinking = thinking[:800] + "..."
        print(re.sub(r'(.{80})', r'\1\n', thinking))

    print(f"\n{'='*80}")
    print(f"DEEP DIVE: LDA=confident (score >= 0 or inf) → reuse")
    print(f"  Is the agent actually confident in these cases?")
    print(f"{'='*80}")

    confident_steps = [s for s in non_step0 if s['lda_score'] == float('inf') or (s['lda_score'] != float('inf') and s['lda_score'] >= 0)]
    print(f"\nNumber of confident (LDA >= 0 or inf) cases: {len(confident_steps)}")
    print(f"Reuse rate for confident: {sum(1 for s in confident_steps if not s['refresh_decision'])/len(confident_steps)*100:.1f}%")

    print(f"\nExamples of LDA=confident (score >= 0 or inf):")
    # Mix of inf and positive scores
    inf_examples = [s for s in confident_steps if s['lda_score'] == float('inf')][:3]
    pos_examples = [s for s in confident_steps if s['lda_score'] != float('inf')][:3]

    for i, s in enumerate(inf_examples):
        print(f"\n--- Example {i+1} (inf): Traj {s['traj_idx']}, Step {s['step_idx']} ---")
        print(f"Refresh decision: {s['refresh_decision']}")
        print(f"Thinking (truncated):")
        thinking = s['thinking']
        if len(thinking) > 800:
            thinking = thinking[:800] + "..."
        print(re.sub(r'(.{80})', r'\1\n', thinking))

    for i, s in enumerate(pos_examples):
        print(f"\n--- Example {i+3} (score={s['lda_score']:.2f}): Traj {s['traj_idx']}, Step {s['step_idx']} ---")
        print(f"Refresh decision: {s['refresh_decision']}")
        print(f"Thinking (truncated):")
        thinking = s['thinking']
        if len(thinking) > 800:
            thinking = thinking[:800] + "..."
        print(re.sub(r'(.{80})', r'\1\n', thinking))

    # Success rate analysis
    print(f"\n{'='*80}")
    print(f"SUCCESS RATE ANALYSIS")
    print(f"{'='*80}")

    successful_trajs = [t for t in trajectories if t.get('won')]
    failed_trajs = [t for t in trajectories if not t.get('won')]

    print(f"\nSuccessful trajectories: {len(successful_trajs)}/{len(trajectories)} ({len(successful_trajs)/len(trajectories)*100:.1f}%)")

    # Get LDA scores from successful vs failed
    def get_non_step0_scores(trajs):
        scores = []
        for t in trajs:
            for s in t.get('steps', []):
                if s.get('step_idx', -1) > 0 and 'lda_score' in s:
                    sc = s['lda_score']
                    if sc != float('inf'):
                        scores.append(sc)
        return scores

    success_scores = get_non_step0_scores(successful_trajs)
    failed_scores = get_non_step0_scores(failed_trajs)

    if success_scores:
        print(f"\nSuccessful trajs (non-step0):")
        print(f"  Avg LDA score: {np.mean(success_scores):.3f}")
        print(f"  Fraction hesitant (LDA < 0): {sum(1 for s in success_scores if s < 0)/len(success_scores)*100:.1f}%")

    if failed_scores:
        print(f"\nFailed trajs (non-step0):")
        print(f"  Avg LDA score: {np.mean(failed_scores):.3f}")
        print(f"  Fraction hesitant (LDA < 0): {sum(1 for s in failed_scores if s < 0)/len(failed_scores)*100:.1f}%")

    # Most extreme scores
    print(f"\n{'='*80}")
    print(f"EXTREME SCORE EXAMPLES")
    print(f"{'='*80}")

    valid_non_step0 = [s for s in non_step0 if s['lda_score'] != float('inf')]
    if valid_non_step0:
        most_negative = sorted(valid_non_step0, key=lambda x: x['lda_score'])[:3]
        most_positive = sorted(valid_non_step0, key=lambda x: -x['lda_score'])[:3]

        print(f"\nMost negative LDA scores:")
        for s in most_negative:
            print(f"  {s['lda_score']:.2f}: Traj {s['traj_idx']}, Step {s['step_idx']}, refresh={s['refresh_decision']}")

        print(f"\nMost positive LDA scores:")
        for s in most_positive:
            print(f"  {s['lda_score']:.2f}: Traj {s['traj_idx']}, Step {s['step_idx']}, refresh={s['refresh_decision']}")

    # Finally, let's save some qualitative examples
    output = {
        "hesitant_examples": hesitant_steps[:10],
        "confident_inf_examples": inf_examples[:10],
        "confident_pos_examples": pos_examples[:10],
        "incorrect_examples": incorrect_examples[:10],
    }

    output_file = file_path.replace('.json', '_qualitative_analysis.json')
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nQualitative examples saved to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="Path to trajectory JSON file")
    args = parser.parse_args()
    main(args.file)

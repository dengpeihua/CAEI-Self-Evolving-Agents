#!/usr/bin/env python3
"""Analyze LDA decisions in trajectory file."""

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
    pattern2 = r'<thinking>\s*(.*?)\s*</thinking>'
    match2 = re.search(pattern2, text, re.DOTALL)
    if match2:
        return match2.group(1).strip()

    # Return first 500 chars if no pattern found
    return text[:500]


def analyze_confidence_from_thinking(thinking_text):
    """
    Analyze thinking text to heuristically determine confidence.
    Returns a dict with confidence indicators.
    """
    if not thinking_text:
        return {"confidence": "unknown", "indicators": []}

    indicators = []
    text_lower = thinking_text.lower()

    # Hesitant / uncertain indicators
    hesitant_keywords = [
        "not sure", "uncertain", "maybe", "could be", "might", "possibly",
        "i think", "i guess", "perhaps", "let me see", "wait", "confused",
        "don't know", "unsure", "difficult", "hard to", "struggle",
        "not certain", "doubt", "worried", "concerned"
    ]

    # Confident indicators
    confident_keywords = [
        "definitely", "certainly", "sure", "clearly", "obviously",
        "i know", "i should", "need to", "must", "will", "going to",
        "easy", "simple", "straightforward", "next step"
    ]

    hesitant_count = sum(1 for kw in hesitant_keywords if kw in text_lower)
    confident_count = sum(1 for kw in confident_keywords if kw in text_lower)

    if hesitant_count > confident_count:
        indicators.append(f"hesitant_keywords={hesitant_count}")
        indicators.append(f"confident_keywords={confident_count}")
        return {"confidence": "hesitant", "indicators": indicators}
    elif confident_count > hesitant_count:
        indicators.append(f"hesitant_keywords={hesitant_count}")
        indicators.append(f"confident_keywords={confident_count}")
        return {"confidence": "confident", "indicators": indicators}
    else:
        indicators.append(f"hesitant_keywords={hesitant_count}")
        indicators.append(f"confident_keywords={confident_count}")
        return {"confidence": "neutral", "indicators": indicators}


def main(file_path):
    print(f"Loading trajectory file: {file_path}")
    with open(file_path, 'r') as f:
        trajectories = json.load(f)

    print(f"Loaded {len(trajectories)} trajectories")

    # Collect all steps with LDA decisions
    all_steps = []
    refresh_steps = []
    no_refresh_steps = []

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
            }

            # Extract thinking content
            thinking = extract_thinking_content(step_data['initial_response'])
            step_data['thinking'] = thinking

            # Analyze thinking confidence
            confidence_analysis = analyze_confidence_from_thinking(thinking)
            step_data['thinking_confidence'] = confidence_analysis['confidence']
            step_data['thinking_indicators'] = confidence_analysis['indicators']

            all_steps.append(step_data)

            if step_data['refresh_decision']:
                refresh_steps.append(step_data)
            else:
                no_refresh_steps.append(step_data)

    print(f"\n{'='*80}")
    print(f"OVERALL STATISTICS")
    print(f"{'='*80}")
    print(f"Total steps with LDA data: {len(all_steps)}")
    print(f"  - Refresh steps: {len(refresh_steps)} ({len(refresh_steps)/len(all_steps)*100:.1f}%)")
    print(f"  - No-refresh steps: {len(no_refresh_steps)} ({len(no_refresh_steps)/len(all_steps)*100:.1f}%)")

    # LDA score distribution
    lda_scores = [s['lda_score'] for s in all_steps if s['lda_score'] is not None]
    print(f"\nLDA score statistics:")
    print(f"  Mean: {np.mean(lda_scores):.3f}")
    print(f"  Std:  {np.std(lda_scores):.3f}")
    print(f"  Min:  {np.min(lda_scores):.3f}")
    print(f"  Max:  {np.max(lda_scores):.3f}")
    print(f"  < 0 (hesitant): {sum(1 for s in lda_scores if s < 0)} ({sum(1 for s in lda_scores if s < 0)/len(lda_scores)*100:.1f}%)")
    print(f"  >=0 (confident): {sum(1 for s in lda_scores if s >= 0)} ({sum(1 for s in lda_scores if s >= 0)/len(lda_scores)*100:.1f}%)")

    # Decision accuracy analysis
    print(f"\n{'='*80}")
    print(f"DECISION ACCURACY ANALYSIS")
    print(f"{'='*80}")

    # True Positives: LDA said hesitant, agent refreshed
    tp = sum(1 for s in all_steps
             if s['lda_decision'] == 'hesitant' and s['refresh_decision'])

    # True Negatives: LDA said confident, agent didn't refresh
    tn = sum(1 for s in all_steps
             if s['lda_decision'] == 'confident' and not s['refresh_decision'])

    # False Positives: LDA said confident, but agent refreshed anyway
    fp = sum(1 for s in all_steps
             if s['lda_decision'] == 'confident' and s['refresh_decision'])

    # False Negatives: LDA said hesitant, but agent didn't refresh
    fn = sum(1 for s in all_steps
             if s['lda_decision'] == 'hesitant' and not s['refresh_decision'])

    print(f"\nContingency table:")
    print(f"                    LDA=confident  LDA=hesitant  Total")
    print(f"  No refresh        {tn:12d}  {fn:11d}  {tn+fn:5d}")
    print(f"  Refresh           {fp:12d}  {tp:11d}  {fp+tp:5d}")
    print(f"  Total             {tn+fp:12d}  {fn+tp:11d}  {len(all_steps):5d}")

    accuracy = (tp + tn) / len(all_steps) if all_steps else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\nClassification metrics:")
    print(f"  Accuracy:  {accuracy:.3f}")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall:    {recall:.3f}")
    print(f"  F1:        {f1:.3f}")

    # Thinking content analysis vs LDA decision
    print(f"\n{'='*80}")
    print(f"THINKING CONTENT ANALYSIS vs LDA DECISION")
    print(f"{'='*80}")

    # Build a contingency table of thinking confidence vs LDA decision
    thinking_vs_lda = defaultdict(Counter)
    for s in all_steps:
        thinking_vs_lda[s['lda_decision']][s['thinking_confidence']] += 1

    print(f"\nThinking confidence vs LDA decision:")
    print(f"                    thinking=confident  thinking=neutral  thinking=hesitant  thinking=unknown")
    for lda_dec in ['confident', 'hesitant']:
        counts = thinking_vs_lda[lda_dec]
        print(f"  LDA={lda_dec:10s}  {counts['confident']:18d}  {counts['neutral']:16d}  {counts['hesitant']:17d}  {counts['unknown']:16d}")

    # Let's look at some specific examples
    print(f"\n{'='*80}")
    print(f"EXAMPLE ANALYSIS")
    print(f"{'='*80}")

    # Case 1: LDA says hesitant AND agent refreshed (good decision)
    print(f"\n[CASE 1] LDA=hesitant, Refresh=True (True Positive)")
    tp_examples = [s for s in all_steps if s['lda_decision'] == 'hesitant' and s['refresh_decision']][:3]
    for i, ex in enumerate(tp_examples):
        print(f"\n  Example {i+1}:")
        print(f"    Traj: {ex['traj_idx']}, Step: {ex['step_idx']}, LDA score: {ex['lda_score']:.3f}")
        print(f"    Thinking (truncated): {repr(ex['thinking'][:300])}")
        print(f"    Thinking confidence: {ex['thinking_confidence']}")
        print(f"    Indicators: {ex['thinking_indicators']}")

    # Case 2: LDA says confident AND agent didn't refresh (good decision)
    print(f"\n[CASE 2] LDA=confident, Refresh=False (True Negative)")
    tn_examples = [s for s in all_steps if s['lda_decision'] == 'confident' and not s['refresh_decision']][:3]
    for i, ex in enumerate(tn_examples):
        print(f"\n  Example {i+1}:")
        print(f"    Traj: {ex['traj_idx']}, Step: {ex['step_idx']}, LDA score: {ex['lda_score']:.3f}")
        print(f"    Thinking (truncated): {repr(ex['thinking'][:300])}")
        print(f"    Thinking confidence: {ex['thinking_confidence']}")
        print(f"    Indicators: {ex['thinking_indicators']}")

    # Case 3: LDA says confident BUT agent refreshed (False Positive - questionable)
    print(f"\n[CASE 3] LDA=confident, Refresh=True (False Positive - why refresh?)")
    fp_examples = [s for s in all_steps if s['lda_decision'] == 'confident' and s['refresh_decision']][:3]
    for i, ex in enumerate(fp_examples):
        print(f"\n  Example {i+1}:")
        print(f"    Traj: {ex['traj_idx']}, Step: {ex['step_idx']}, LDA score: {ex['lda_score']:.3f}")
        print(f"    Refresh reason: {ex['refresh_reason']}")
        print(f"    Thinking (truncated): {repr(ex['thinking'][:300])}")
        print(f"    Thinking confidence: {ex['thinking_confidence']}")
        print(f"    Indicators: {ex['thinking_indicators']}")

    # Case 4: LDA says hesitant BUT agent didn't refresh (False Negative - missed opportunity?)
    print(f"\n[CASE 4] LDA=hesitant, Refresh=False (False Negative - why no refresh?)")
    fn_examples = [s for s in all_steps if s['lda_decision'] == 'hesitant' and not s['refresh_decision']][:3]
    for i, ex in enumerate(fn_examples):
        print(f"\n  Example {i+1}:")
        print(f"    Traj: {ex['traj_idx']}, Step: {ex['step_idx']}, LDA score: {ex['lda_score']:.3f}")
        print(f"    Thinking (truncated): {repr(ex['thinking'][:300])}")
        print(f"    Thinking confidence: {ex['thinking_confidence']}")
        print(f"    Indicators: {ex['thinking_indicators']}")

    # Success rate analysis
    print(f"\n{'='*80}")
    print(f"SUCCESS RATE ANALYSIS")
    print(f"{'='*80}")

    successful_steps = [s for s in all_steps if s['won']]
    failed_steps = [s for s in all_steps if not s['won']]

    print(f"\nSuccessful trajectories: {len([t for t in trajectories if t.get('won')])}/{len(trajectories)}")
    print(f"Steps in successful trajs: {len(successful_steps)}")
    print(f"Steps in failed trajs: {len(failed_steps)}")

    if successful_steps:
        success_lda_mean = np.mean([s['lda_score'] for s in successful_steps if s['lda_score'] is not None])
        print(f"\n  Success trajs:")
        print(f"    Avg LDA score: {success_lda_mean:.3f}")
        print(f"    Refresh rate: {sum(1 for s in successful_steps if s['refresh_decision'])/len(successful_steps)*100:.1f}%")

    if failed_steps:
        failed_lda_mean = np.mean([s['lda_score'] for s in failed_steps if s['lda_score'] is not None])
        print(f"\n  Failed trajs:")
        print(f"    Avg LDA score: {failed_lda_mean:.3f}")
        print(f"    Refresh rate: {sum(1 for s in failed_steps if s['refresh_decision'])/len(failed_steps)*100:.1f}%")

    # LDA score distribution by bins
    print(f"\n{'='*80}")
    print(f"LDA SCORE DISTRIBUTION")
    print(f"{'='*80}")
    bins = [(-np.inf, -2.0), (-2.0, -1.0), (-1.0, -0.5), (-0.5, 0.0),
            (0.0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, np.inf)]
    for low, high in bins:
        count = sum(1 for s in lda_scores if low <= s < high)
        refresh_count = sum(1 for s in all_steps if s['lda_score'] is not None and
                           low <= s['lda_score'] < high and s['refresh_decision'])
        if count > 0:
            print(f"  [{low:.1f}, {high:.1f}): {count:5d} steps, refresh={refresh_count:4d} ({refresh_count/count*100:5.1f}%)")

    # Save detailed examples for deeper inspection
    output = {
        "tp_examples": tp_examples[:10],
        "tn_examples": tn_examples[:10],
        "fp_examples": fp_examples[:10],
        "fn_examples": fn_examples[:10],
    }

    output_file = file_path.replace('.json', '_analysis_examples.json')
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nDetailed examples saved to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="Path to trajectory JSON file")
    args = parser.parse_args()
    main(args.file)

"""
Train Linear Discriminant Analysis (LDA) on planning tokens.

Goal: probe whether the model encodes "the agent's confidence / intent to
refresh" in its last-layer hidden state. We:
  1. Load lda_vectors.npy (120, 2560) and lda_labels.json (120)
  2. Standardize the vectors
  3. Run 5-fold cross-validated LDA
  4. Compute accuracy, AUC, confusion matrix
  5. Project to 1D and visualize the decision boundary
  6. Test whether the LDA projection correlates with the existing
     "thinking length" feature (sanity check)

This shows whether a single linear direction in planning-token space separates
"confident committed reasoning" from "hesitant stuck reasoning".

If the cross-validated AUC is significantly above 0.5 (chance),
LDA has found a meaningful direction in the planning-token space — we can
then USE that direction as a "should refresh?" signal for an Adamem-style
memory policy (replacing the broken argmax-prob approach).
"""

import argparse
import json
import numpy as np
from collections import Counter
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    confusion_matrix,
    balanced_accuracy_score,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vectors", default="lda_vectors.npy")
    ap.add_argument("--labels", default="lda_labels.json")
    ap.add_argument("--samples", default="confident_hesitant_samples.json")
    ap.add_argument("--n_splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    V = np.load(args.vectors).astype(np.float64)
    with open(args.labels) as f:
        labels = json.load(f)
    with open(args.samples) as f:
        samples = json.load(f)
    print(f"V: {V.shape}, label distribution: {Counter(labels)}")

    label_map = {"confident": 1, "hesitant": 0}
    y = np.array([label_map[l] for l in labels])

    # 1. Standardize
    scaler = StandardScaler()
    Z = scaler.fit_transform(V)
    print(f"Standardized: mean={Z.mean(axis=0).mean():.3f}, std={Z.std(axis=0).mean():.3f}")

    # 2. Fit LDA on full data to inspect its projection direction
    lda_full = LinearDiscriminantAnalysis()
    lda_full.fit(Z, y)
    proj_full = lda_full.decision_function(Z)   # 1D projection
    # descriptive stats per class
    proj_pos = proj_full[y == 1]
    proj_neg = proj_full[y == 0]
    print(f"\nFull-data LDA projection (1D):")
    print(f"  confident (y=1): mean={proj_pos.mean():.3f} std={proj_pos.std():.3f}")
    print(f"  hesitant  (y=0): mean={proj_neg.mean():.3f} std={proj_neg.std():.3f}")
    print(f"  separation (Cohen's d): {(proj_pos.mean()-proj_neg.mean())/np.sqrt((proj_pos.var()+proj_neg.var())/2):.3f}")

    # 3. 5-fold cross-validated LDA
    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    preds_cv = np.zeros_like(y)
    scores_cv = np.zeros_like(y, dtype=np.float64)
    for fold, (train_idx, test_idx) in enumerate(skf.split(Z, y)):
        lda = LinearDiscriminantAnalysis()
        lda.fit(Z[train_idx], y[train_idx])
        preds_cv[test_idx] = lda.predict(Z[test_idx])
        scores_cv[test_idx] = lda.decision_function(Z[test_idx])
        print(f"  fold {fold}: train_acc={accuracy_score(y[train_idx], lda.predict(Z[train_idx])):.3f}, test_acc={accuracy_score(y[test_idx], preds_cv[test_idx]):.3f}")

    # 4. Aggregate metrics
    acc = accuracy_score(y, preds_cv)
    balanced = balanced_accuracy_score(y, preds_cv)
    auc = roc_auc_score(y, scores_cv)
    cm = confusion_matrix(y, preds_cv)
    print(f"\n=== Cross-validated results ({args.n_splits}-fold) ===")
    print(f"  accuracy    : {acc:.3f}")
    print(f"  balanced acc: {balanced:.3f}  (= (TPR+TNR)/2, chance=0.5)")
    print(f"  AUC         : {auc:.3f}  (chance=0.5)")
    print(f"  confusion matrix [rows=true, cols=pred]:")
    print(f"    {cm}")

    # 5. Save the decision scores for visualization
    np.save("lda_proj.npy", proj_full)
    np.save("lda_scores_cv.npy", scores_cv)
    with open("lda_result.json", "w") as f:
        json.dump({
            "accuracy": float(acc),
            "balanced_accuracy": float(balanced),
            "auc": float(auc),
            "confusion_matrix": cm.tolist(),
            "cohens_d": float((proj_pos.mean() - proj_neg.mean()) /
                              np.sqrt((proj_pos.var() + proj_neg.var()) / 2)),
            "projection_mean_confident": float(proj_pos.mean()),
            "projection_mean_hesitant": float(proj_neg.mean()),
            "projection_std_confident": float(proj_pos.std()),
            "projection_std_hesitant": float(proj_neg.std()),
        }, f, indent=2)

    # 6. Sanity check: does LDA correlate with think length?
    think_lens = np.array([len(s["think"]) for s in samples])
    print(f"\n=== Sanity check: LDA projection vs think length ===")
    from numpy import corrcoef
    r = float(corrcoef(scores_cv, think_lens)[0, 1])
    print(f"  Pearson r(proj_cv, think_len) = {r:.3f}")
    if abs(r) > 0.6:
        print("  -- LDA is mostly recovering think-length (not a NEW signal)")
    elif abs(r) < 0.3:
        print("  -- LDA is recovering something BEYOND think-length")

    # 7. Interpret — exact decision boundary info
    w = lda_full.coef_.ravel()
    b = lda_full.intercept_[0]
    print(f"\n=== Decision boundary (full-fit LDA) ===")
    print(f"  decision_function: dot(w, scaled_vector) + {b:.3f}")
    print(f"  positive => confident, negative => hesitant")
    print(f"  w norm: {np.linalg.norm(w):.3f}")
    print(f"  hint: positive bias means the class balance lean a certain way")


if __name__ == "__main__":
    main()
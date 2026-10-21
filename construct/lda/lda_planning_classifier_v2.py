"""
Train LDA on the v2 confident/hesitant dataset (353 samples),
compare to v1 (120 samples), and also compare to MLP for fairness.

Outputs to lda_result_v2.json + dumps top/bottom 10 samples for inspection.
"""

# 对第二版认知特征比较 LDA、逻辑回归和小型 MLP，并与第一版数据对照。
import argparse
import json
import numpy as np
from collections import Counter

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
)

# 分层交叉验证中，仅用训练折拟合标准化器和分类器，汇总折外概率与评估指标。
def eval_classifier(name, X, y, k_folds, clf_factory):
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=0)
    train_accs, test_accs = [], []
    test_scores = np.zeros_like(y, dtype=np.float64)
    for tr, te in skf.split(X, y):
        sc = StandardScaler().fit(X[tr])
        Xtr = sc.transform(X[tr])
        Xte = sc.transform(X[te])
        clf = clf_factory()
        clf.fit(Xtr, y[tr])
        s = clf.predict_proba(Xte)[:, 1] if hasattr(clf, "predict_proba") \
            else clf.decision_function(Xte)
        p = (s >= 0.5).astype(int)
        test_scores[te] = s
        train_accs.append(accuracy_score(y[tr], clf.predict(Xtr)))
        test_accs.append(accuracy_score(y[te], p))
    overall_acc = accuracy_score(y, (test_scores >= 0.5).astype(int))
    overall_balanced = balanced_accuracy_score(y, (test_scores >= 0.5).astype(int))
    overall_auc = roc_auc_score(y, test_scores)
    return {
        "name": name,
        "train_acc_mean": float(np.mean(train_accs)),
        "train_acc_std": float(np.std(train_accs)),
        "test_acc_mean": float(np.mean(test_accs)),
        "test_acc_std": float(np.std(test_accs)),
        "overall_acc": float(overall_acc),
        "overall_balanced": float(overall_balanced),
        "overall_auc": float(overall_auc),
        "test_scores": test_scores,
    }, (test_scores >= 0.5).astype(int)

# 解析两版数据路径和折数，读取第二版特征、标签与样本。
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vectors", default="data/lda_vectors_v2.npy")
    ap.add_argument("--labels", default="data/lda_labels_v2.json")
    ap.add_argument("--samples", default="data/confident_hesitant_samples_v2.json")
    ap.add_argument("--v1_vectors", default="data/lda_vectors.npy")
    ap.add_argument("--v1_labels", default="data/lda_labels.json")
    ap.add_argument("--v1_samples", default="data/confident_hesitant_samples.json")
    ap.add_argument("--k_folds", type=int, default=5)
    args = ap.parse_args()

    V = np.load(args.vectors).astype(np.float64)
    with open(args.labels) as f:
        labels = json.load(f)
    with open(args.samples) as f:
        samples = json.load(f)

    print(f"=== v2 dataset ({args.vectors}) ===")
    print(f"V: {V.shape}, label dist: {Counter(labels)}")

    y = np.array([1 if l == "confident" else 0 for l in labels])
    print(f"y distribution (y=1: confident, y=0: hesitant): "
          f"{Counter(y.tolist())}")

    # 使用相同交叉验证流程评估三种分类器，打印训练与测试指标。
    print("\n--- Training 3 classifiers on v2 (353 samples) ---")
    r_lda, p_lda = eval_classifier(
        "LDA", V, y, args.k_folds,
        lambda: LinearDiscriminantAnalysis(),
    )
    r_lr, p_lr = eval_classifier(
        "LogisticRegression(C=1)", V, y, args.k_folds,
        lambda: LogisticRegression(C=1.0, max_iter=1000),
    )
    r_mlp, p_mlp = eval_classifier(
        "MLP(4)", V, y, args.k_folds,
        lambda: MLPClassifier(hidden_layer_sizes=(4,), max_iter=300, random_state=0),
    )
    print(f"\n{'Method':<24s} {'Train':>16s} {'Test':>16s} "
          f"{'Overall acc':>14s} {'AUC':>8s}")
    print("-" * 80)
    for r in [r_lda, r_lr, r_mlp]:
        train_str = f"{r['train_acc_mean']:.3f}+-{r['train_acc_std']:.3f}"
        test_str  = f"{r['test_acc_mean']:.3f}+-{r['test_acc_std']:.3f}"
        print(f"{r['name']:<24s} {train_str:>16s} {test_str:>16s} "
              f"{r['overall_acc']:>14.3f} {r['overall_auc']:>8.3f}")

    # 另用全部第二版数据拟合 LDA，计算投影并打印分数最高、最低各五个样本供检查。
    lda_full = LinearDiscriminantAnalysis().fit(
        StandardScaler().fit_transform(V), y
    )
    from sklearn.preprocessing import StandardScaler as SS
    scaler = SS().fit(V)
    Z_full = scaler.transform(V)
    proj_full = lda_full.decision_function(Z_full)

    order = np.argsort(proj_full)
    top5_conf = order[-5:][::-1]
    top5_hes  = order[:5]
    print(f"\nTop 5 most CONFIDENT (proj highest, true=confident):")
    for idx in top5_conf:
        print(f"  proj={proj_full[idx]:+.2f}, label={labels[idx]}, task={samples[idx]['task'][:50]}")
    print(f"\nTop 5 most HESITANT (proj lowest, true=hesitant):")
    for idx in top5_hes:
        print(f"  proj={proj_full[idx]:+.2f}, label={labels[idx]}, task={samples[idx]['task'][:50]}")

    # 读取第一版特征与标签做同口径 LDA 验证，对比两版 AUC。
    print(f"\n=== v1 baseline (120 samples) ===")
    V1 = np.load(args.v1_vectors).astype(np.float64)
    with open(args.v1_labels) as f:
        labels_v1 = json.load(f)
    y1 = np.array([1 if l == "confident" else 0 for l in labels_v1])
    r1_lda, _ = eval_classifier(
        "LDA (v1 120 samples)", V1, y1, args.k_folds,
        lambda: LinearDiscriminantAnalysis(),
    )
    print(f"  LDA test acc (v1): {r1_lda['test_acc_mean']:.3f}, "
          f"AUC={r1_lda['overall_auc']:.3f}")
    print(f"  LDA test acc (v2): {r_lda['test_acc_mean']:.3f}, "
          f"AUC={r_lda['overall_auc']:.3f}")
    delta = r_lda['overall_auc'] - r1_lda['overall_auc']
    print(f"  Δ AUC: {delta:+.4f}")

    # 保存第二版投影、折外分数和比较指标；没有导出部署用 lda_model_v2.npz。
    np.save("data/lda_proj_v2.npy", proj_full)
    np.save("data/lda_scores_v2_cv.npy", r_lda["test_scores"])

    result = {
        "v1": {"n_samples": int(V1.shape[0]), "lda_test_acc": r1_lda["test_acc_mean"],
               "lda_auc": r1_lda["overall_auc"]},
        "v2": {
            "n_samples": int(V.shape[0]),
            "lda_test_acc": r_lda["test_acc_mean"],
            "lda_auc": r_lda["overall_auc"],
            "lr_test_acc": r_lr["test_acc_mean"],
            "lr_auc": r_lr["overall_auc"],
            "mlp_test_acc": r_mlp["test_acc_mean"],
            "mlp_auc": r_mlp["overall_auc"],
        },
        "auc_delta_v2_over_v1": float(delta),
    }
    with open("data/lda_result_v2.json", "w") as f:
        json.dump(result, f, indent=2)

if __name__ == "__main__":
    main()

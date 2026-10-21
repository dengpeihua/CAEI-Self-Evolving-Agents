"""
Cluster + analyse planning tokens per the paper.

Steps:
  1. Load planning_vectors.npy (80, 2560) + planning_labels.json
  2. PCA to 50 components (paper's first "noise-suppression" step)
  3. K-Means cluster into k=2..6 clusters, evaluate with
     silhouette score (paper uses a similar quality metric)
  4. Use the BEST k to label each planning token
  5. Build a contingency table: predicted_cluster vs ground_truth_label
  6. Run a chi-square-like independence test (Cramer V)
  7. Visualize with PCA-2D scatter
  8. Print final analysis: does clustering recover the 4 categories?
     does it cleanly separate won/failed? decisive/hesitant?

The key research question we want to answer:
  -- does the planning token (the model's last-layer CoT encoding) actually
     cluster into semantically meaningful reasoning patterns
     (e.g., confident reasoning vs hesitant reasoning)?

A "no" answer = the planning token is essentially noise / doesn't encode
reasoning mode info.
A "yes" answer = clustering yields interpretable groups matching the labels.
"""
import argparse
import json
from collections import Counter, defaultdict

import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    accuracy_score,
    confusion_matrix,
)


def best_k_by_silhouette(X, ks=range(2, 8)):
    """Return (best_k, scores) selected by silhouette."""
    scores = {}
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X)
        scores[k] = silhouette_score(X, km.labels_)
    best_k = max(scores, key=scores.get)
    return best_k, scores


def contingency(label_a, label_b):
    a_unique = sorted(set(label_a))
    b_unique = sorted(set(label_b))
    matrix = np.zeros((len(a_unique), len(b_unique)), dtype=int)
    aidx = {a: i for i, a in enumerate(a_unique)}
    bidx = {b: i for i, b in enumerate(b_unique)}
    for a, b in zip(label_a, label_b):
        matrix[aidx[a], bidx[b]] += 1
    return a_unique, b_unique, matrix


def cramers_v(matrix):
    """Cramér's V in [0, 1]: 0 = no association, 1 = perfect."""
    chi2 = 0.0
    n = matrix.sum()
    row_sums = matrix.sum(axis=1)
    col_sums = matrix.sum(axis=0)
    if n == 0:
        return 0.0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            expected = row_sums[i] * col_sums[j] / n
            if expected > 0:
                chi2 += (matrix[i, j] - expected) ** 2 / expected
    k = min(matrix.shape)
    if k <= 1:
        return 0.0
    return float(np.sqrt(chi2 / (n * (k - 1))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vectors", default="planning_vectors.npy")
    ap.add_argument("--labels", default="planning_labels.json")
    ap.add_argument("--n_components", type=int, default=50)
    ap.add_argument("--max_k", type=int, default=7)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    V = np.load(args.vectors)
    with open(args.labels) as f:
        labels = json.load(f)
    V = V.astype(np.float64)
    print(f"V: {V.shape}, labels: {Counter(labels)}")

    # ---- Step 1: PCA ----------
    pca = PCA(n_components=args.n_components, random_state=args.seed)
    Z = pca.fit_transform(V)
    print(f"PCA explained variance ratio (top 5): "
          f"{pca.explained_variance_ratio_[:5].round(3).tolist()}, "
          f"sum={pca.explained_variance_ratio_.sum():.3f}")

    # ---- Step 2: K-Means over a range of k ----------
    best_k, scores = best_k_by_silhouette(Z, ks=range(2, args.max_k + 1))
    print(f"\nSilhouette scores (k=2..{args.max_k}): "
          f"{ {k: round(s, 3) for k, s in scores.items()} }")
    print(f"  best k by silhouette: k={best_k}, score={scores[best_k]:.3f}")

    km = KMeans(n_clusters=best_k, n_init=10, random_state=args.seed).fit(Z)
    pred = km.labels_.tolist()
    pred_strs = [f"c{i}" for i in pred]

    # ---- Step 3: Cluster quality vs ground-truth labels ----------
    ari = adjusted_rand_score(labels, pred_strs)
    nmi = normalized_mutual_info_score(labels, pred_strs)
    print(f"\nCluster quality vs ground-truth 4-class label:")
    print(f"  Adjusted Rand Index (ARI): {ari:.4f}")
    print(f"  Normalized Mutual Info (NMI): {nmi:.4f}")

    a_unique, b_unique, matrix = contingency(labels, pred_strs)
    print(f"\nContingency table (rows=ground-truth, cols=cluster):")
    header = ["label\\cluster"] + [f"{c:>12s}" for c in b_unique]
    print("  " + "  ".join(f"{x:<14s}" for x in header))
    for i, a in enumerate(a_unique):
        row = [f"{a:<14s}"]
        for j in range(len(b_unique)):
            row.append(f"{int(matrix[i, j]):>12d}")
        print("  " + "  ".join(row))

    # ---- Step 4: Binary reduction (won vs failed, decisive vs hesitant) ----------
    won_vs_fail = ["won" if l.startswith("won") else "fail" for l in labels]
    dec_vs_hes = ["decisive" if l.endswith("decisive") else "hesitant"
                   for l in labels]

    # For each binary axis, do the same best_k clustering and check if
    # the cluster assignment correlates.
    print("\n=== Binary axis: won vs fail ===")
    best_k2, _ = best_k_by_silhouette(Z, ks=range(2, 4))
    km2 = KMeans(n_clusters=best_k2, n_init=10, random_state=args.seed).fit(Z)
    pred2 = km2.labels_.tolist()
    ari2 = adjusted_rand_score(won_vs_fail, [f"c{i}" for i in pred2])
    nmi2 = normalized_mutual_info_score(won_vs_fail, [f"c{i}" for i in pred2])
    print(f"  best k={best_k2}, ARI={ari2:.4f}, NMI={nmi2:.4f}")
    au, bu, mat = contingency(won_vs_fail, [f"c{i}" for i in pred2])
    for i, a in enumerate(au):
        row = [f"{a:<14s}"]
        for j in range(len(bu)):
            row.append(f"{int(mat[i, j]):>12d}")
        print("  " + "  ".join(row))
    cv2 = cramers_v(mat)
    print(f"  Cramér's V (won vs fail axis): {cv2:.4f}")

    print("\n=== Binary axis: decisive vs hesitant ===")
    best_k3, _ = best_k_by_silhouette(Z, ks=range(2, 4))
    km3 = KMeans(n_clusters=best_k3, n_init=10, random_state=args.seed).fit(Z)
    pred3 = km3.labels_.tolist()
    ari3 = adjusted_rand_score(dec_vs_hes, [f"c{i}" for i in pred3])
    nmi3 = normalized_mutual_info_score(dec_vs_hes, [f"c{i}" for i in pred3])
    print(f"  best k={best_k3}, ARI={ari3:.4f}, NMI={nmi3:.4f}")
    au, bu, mat = contingency(dec_vs_hes, [f"c{i}" for i in pred3])
    for i, a in enumerate(au):
        row = [f"{a:<14s}"]
        for j in range(len(bu)):
            row.append(f"{int(mat[i, j]):>12d}")
        print("  " + "  ".join(row))
    cv3 = cramers_v(mat)
    print(f"  Cramér's V (decisive vs hesitant axis): {cv3:.4f}")

    # ---- Step 5: Final interpretation ----------
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    print(f"""
4-class clustering (won/decisive × 2):
  - ARI={ari:.3f}, NMI={nmi:.3f}
  -- Higher = cluster boundaries align with categories.
  -- 0 = random clustering. 1 = perfect.

Binary won/fail axis:
  - Cramér's V={cv2:.3f} (0=independent, 1=perfect)

Binary decisive/hesitant axis:
  - Cramér's V={cv3:.3f}

Silhouette by k={best_k}: {scores[best_k]:.3f}
  -- Typical "good cluster" silhouette is > 0.4.
  -- > 0.5 is "well separated", < 0.25 means clusters overlap heavily.

These numbers tell us whether the planning token encoding carries the
semantic distinctions we hypothesized.
""")

    # ---- Step 6: save cluster assignments for plotting -------
    np.save("planning_pca.npy", Z[:, :2])
    with open("planning_clusters.json", "w") as f:
        json.dump({
            "best_k": int(best_k),
            "pred": [int(p) for p in pred],
            "labels": labels,
            "pca_variance_top10":
                [float(x) for x in pca.explained_variance_ratio_[:10]],
            "silhouette_by_k": {str(k): float(s) for k, s in scores.items()},
            "ari_4class": float(ari),
            "nmi_4class": float(nmi),
            "cv_won_fail": float(cv2),
            "cv_dec_hes": float(cv3),
        }, f, indent=2)


if __name__ == "__main__":
    main()
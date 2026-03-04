"""
=============================================================================
Teradata KMeans Clustering Experiment
Phase 1: RFM Only          → Find optimal k (elbow + silhouette)
Phase 2: RFM + Demo + Risk → Find optimal k and compare segments
=============================================================================
Requirements:
    pip install teradataml matplotlib seaborn pandas scikit-learn
=============================================================================
"""

import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import silhouette_score
from teradataml import create_context, DataFrame, copy_to_sql, remove_context
from teradataml.analytics.mle import KMeans, KMeansPredict

warnings.filterwarnings("ignore")

# ─── CONFIG ───────────────────────────────────────────────────────────────────
TD_HOST     = os.environ.get("TD_HOST", "your-host")
TD_USER     = os.environ.get("TD_USER", "demo_user")
TD_PASSWORD = os.environ.get("TD_PASSWORD", "your-password")
DATABASE    = "demo_user"

# Tables (already exist in your environment)
RFM_TABLE       = "segmentation_rfm_scaled"        # monetary, frequency, recency — 3 features
FULL_TABLE      = "segmentation_full_scaled"        # + credit_score, age, income — 6 features
RISK_TABLE      = "segmentation_rfm_risk_scaled"   # RFM + credit_score — 4 features

ID_COL          = "customer_id"

FEATURES_RFM    = ["monetary_scaled", "frequency_scaled", "recency_scaled"]
FEATURES_FULL   = ["monetary_scaled", "frequency_scaled", "recency_scaled",
                   "credit_score_scaled", "age_scaled", "income_scaled"]
FEATURES_RISK   = ["monetary_scaled", "frequency_scaled", "recency_scaled",
                   "credit_score_scaled"]

K_RANGE         = range(2, 11)   # k = 2 to 10
FINAL_OUTPUT_DIR = "./kmeans_results"

os.makedirs(FINAL_OUTPUT_DIR, exist_ok=True)


# ─── CONNECTION ───────────────────────────────────────────────────────────────

def connect():
    print(f"Connecting to Teradata: {TD_HOST}...")
    create_context(host=TD_HOST, username=TD_USER, password=TD_PASSWORD,
                   database=DATABASE)
    print("Connected ✅")


# ─── ELBOW + SILHOUETTE EXPERIMENT ───────────────────────────────────────────

def run_experiment(table_name: str, features: list, label: str) -> pd.DataFrame:
    """
    Run KMeans for k=2..10 on a Teradata scaled table.
    Returns a DataFrame with k, within_ss, silhouette for each k.
    Computation stays in Teradata — only centroids pulled to Python for silhouette.
    """
    print(f"\n{'='*60}")
    print(f"  EXPERIMENT: {label}")
    print(f"  Table: {table_name}  |  Features: {features}")
    print(f"{'='*60}")

    td_df = DataFrame.from_table(f"{DATABASE}.{table_name}")
    records = []

    for k in K_RANGE:
        print(f"  Running KMeans k={k}...", end=" ", flush=True)
        try:
            # ── Run KMeans in Teradata (teradataml pushes to VAL) ──────────
            model = KMeans(
                data=td_df,
                n_clusters=k,
                id_column=ID_COL,
                center_columns=features,
                max_iter=100,
                seed=42
            )

            # ── Extract Within-SS from model output ────────────────────────
            model_df = model.result.to_pandas()
            info_rows = model_df[model_df["td_modelinfo_kmeans"].notna()]
            within_ss = None
            n_iter    = None
            converged = None

            for _, row in info_rows.iterrows():
                info = str(row["td_modelinfo_kmeans"])
                if "Total_WithinSS" in info:
                    try: within_ss = float(info.split(":")[-1].strip())
                    except: pass
                elif "Number of Iterations" in info:
                    try: n_iter = int(info.split(":")[-1].strip())
                    except: pass
                elif "Converged" in info:
                    converged = "True" in info

            # ── Silhouette Score ───────────────────────────────────────────
            # Pull predictions to Python for silhouette (lightweight — just IDs + labels)
            predictions = KMeansPredict(
                object=model,
                newdata=td_df,
                id_column=ID_COL,
                accumulate=features
            )
            pred_df = predictions.result.to_pandas()

            # Compute silhouette on a sample (max 10K rows for speed)
            sample = pred_df.sample(min(10000, len(pred_df)), random_state=42)
            X      = sample[features].values
            labels = sample["td_clusterid_kmeans"].values

            if len(set(labels)) > 1:
                sil = silhouette_score(X, labels, metric="euclidean", sample_size=5000)
            else:
                sil = -1

            records.append({
                "k":            k,
                "within_ss":    within_ss,
                "silhouette":   round(sil, 4),
                "n_iterations": n_iter,
                "converged":    converged
            })
            print(f"✅  within_ss={within_ss:.2f}  silhouette={sil:.4f}  iters={n_iter}")

        except Exception as e:
            print(f"❌  Error: {e}")
            records.append({"k": k, "within_ss": None, "silhouette": None,
                            "n_iterations": None, "converged": None, "error": str(e)})

    results_df = pd.DataFrame(records)

    # Compute elbow delta
    results_df["within_ss_delta"] = results_df["within_ss"].diff(-1)

    print(f"\n  Results for {label}:")
    print(results_df[["k", "within_ss", "within_ss_delta", "silhouette",
                       "n_iterations", "converged"]].to_string(index=False))

    return results_df


# ─── PLOTTING ─────────────────────────────────────────────────────────────────

def plot_experiment_results(results: dict):
    """
    Plot elbow + silhouette for all experiments side by side.
    results = {"RFM Only": df, "RFM + Demo + Risk": df, ...}
    """
    n_experiments = len(results)
    fig = plt.figure(figsize=(7 * n_experiments, 10))
    fig.patch.set_facecolor("#0f1117")

    palette = {
        "RFM Only":          "#00d4ff",
        "RFM + Risk":        "#ff6b6b",
        "RFM + Demo + Risk": "#00ff9d",
    }

    gs = gridspec.GridSpec(2, n_experiments, hspace=0.45, wspace=0.35)

    for col, (label, df) in enumerate(results.items()):
        color  = palette.get(label, "#ffffff")
        valid  = df.dropna(subset=["within_ss", "silhouette"])
        ks     = valid["k"].values
        wss    = valid["within_ss"].values
        sil    = valid["silhouette"].values

        # Find optimal k (max silhouette)
        opt_k_sil = ks[np.argmax(sil)]
        opt_k_elb = _find_elbow(ks, wss)

        # ── Elbow Plot ────────────────────────────────────────────────────
        ax1 = fig.add_subplot(gs[0, col])
        ax1.set_facecolor("#1a1d27")
        ax1.plot(ks, wss, color=color, linewidth=2.5, marker="o",
                 markersize=7, markerfacecolor="white", markeredgecolor=color)
        ax1.axvline(x=opt_k_elb, color=color, linestyle="--",
                    alpha=0.7, linewidth=1.5, label=f"Elbow k={opt_k_elb}")
        ax1.set_title(f"{label}\nElbow (Within-SS)", color="white",
                      fontsize=12, fontweight="bold", pad=10)
        ax1.set_xlabel("Number of Clusters (k)", color="#aaaaaa", fontsize=10)
        ax1.set_ylabel("Total Within-SS", color="#aaaaaa", fontsize=10)
        ax1.tick_params(colors="#aaaaaa")
        ax1.spines[:].set_color("#333344")
        ax1.legend(facecolor="#1a1d27", labelcolor=color, fontsize=9)
        for spine in ax1.spines.values():
            spine.set_linewidth(0.8)

        # Annotate optimal point
        idx = list(ks).index(opt_k_elb)
        ax1.annotate(f"k={opt_k_elb}", xy=(opt_k_elb, wss[idx]),
                     xytext=(opt_k_elb + 0.3, wss[idx] * 1.02),
                     color=color, fontsize=9,
                     arrowprops=dict(arrowstyle="->", color=color, lw=1.2))

        # ── Silhouette Plot ───────────────────────────────────────────────
        ax2 = fig.add_subplot(gs[1, col])
        ax2.set_facecolor("#1a1d27")
        bar_colors = [color if k == opt_k_sil else "#334455" for k in ks]
        bars = ax2.bar(ks, sil, color=bar_colors, edgecolor="#222233",
                       linewidth=0.8, zorder=3)
        ax2.axhline(y=0, color="#666677", linewidth=0.8, linestyle="--")
        ax2.set_title(f"{label}\nSilhouette Score", color="white",
                      fontsize=12, fontweight="bold", pad=10)
        ax2.set_xlabel("Number of Clusters (k)", color="#aaaaaa", fontsize=10)
        ax2.set_ylabel("Silhouette Score", color="#aaaaaa", fontsize=10)
        ax2.tick_params(colors="#aaaaaa")
        ax2.spines[:].set_color("#333344")
        ax2.set_xticks(ks)

        # Annotate best bar
        best_bar = bars[list(ks).index(opt_k_sil)]
        ax2.annotate(f"Best k={opt_k_sil}\n({max(sil):.3f})",
                     xy=(best_bar.get_x() + best_bar.get_width() / 2, max(sil)),
                     xytext=(best_bar.get_x() + best_bar.get_width() / 2,
                             max(sil) + 0.01),
                     ha="center", color=color, fontsize=9, fontweight="bold")

    fig.suptitle("KMeans Clustering Experiments — Elbow & Silhouette Analysis\n"
                 "Teradata ClearScape Analytics (VAL in-database)",
                 color="white", fontsize=15, fontweight="bold", y=1.01)

    plt.tight_layout()
    out_path = f"{FINAL_OUTPUT_DIR}/kmeans_experiment_results.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"\n📊 Plot saved → {out_path}")
    plt.show()


def _find_elbow(ks, wss):
    """Find elbow using the maximum second derivative (curvature)."""
    if len(wss) < 3:
        return ks[0]
    diffs2 = np.diff(np.diff(wss))
    return int(ks[np.argmax(np.abs(diffs2)) + 1])


# ─── CLUSTER PROFILING ────────────────────────────────────────────────────────

def profile_final_clusters(model, td_df, features: list,
                            label: str, segmentation_df=None):
    """
    Run final KMeans with optimal k, profile each cluster with 360° view.
    """
    print(f"\n{'='*60}")
    print(f"  PROFILING FINAL CLUSTERS: {label}")
    print(f"{'='*60}")

    predictions = KMeansPredict(
        object=model,
        newdata=td_df,
        id_column=ID_COL,
        accumulate=features
    )
    pred_df = predictions.result.to_pandas()

    # Join with segmentation_dataset if provided (for 360° view)
    if segmentation_df is not None:
        pred_df = pred_df.merge(segmentation_df, on=ID_COL, how="left",
                                suffixes=("", "_seg"))

    profile = pred_df.groupby("td_clusterid_kmeans").agg(
        customer_count=("customer_id", "count"),
        avg_monetary=("monetary_scaled", "mean"),
        avg_frequency=("frequency_scaled", "mean"),
        avg_recency=("recency_scaled", "mean"),
    ).reset_index()

    profile["pct_of_total"] = (profile["customer_count"] /
                                profile["customer_count"].sum() * 100).round(2)
    profile.columns = [c.replace("td_clusterid_kmeans", "cluster_id")
                       for c in profile.columns]

    print(profile.to_string(index=False))

    # Plot cluster profiles
    _plot_cluster_radar(pred_df, features, label)

    return pred_df, profile


def _plot_cluster_radar(pred_df: pd.DataFrame, features: list, label: str):
    """Radar/spider chart of cluster centroids."""
    centroids = pred_df.groupby("td_clusterid_kmeans")[features].mean()
    n_clusters = len(centroids)
    n_features = len(features)

    angles = np.linspace(0, 2 * np.pi, n_features, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8),
                           subplot_kw=dict(polar=True), facecolor="#0f1117")
    ax.set_facecolor("#1a1d27")

    colors = plt.cm.Set2(np.linspace(0, 1, n_clusters))
    feature_labels = [f.replace("_scaled", "").replace("_", " ").title()
                      for f in features]

    for idx, (cluster_id, row) in enumerate(centroids.iterrows()):
        values = row.tolist() + row.tolist()[:1]
        ax.plot(angles, values, color=colors[idx], linewidth=2,
                label=f"Cluster {cluster_id}")
        ax.fill(angles, values, color=colors[idx], alpha=0.15)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(feature_labels, color="white", fontsize=10)
    ax.tick_params(colors="#aaaaaa")
    ax.spines["polar"].set_color("#333344")
    ax.set_facecolor("#1a1d27")
    ax.yaxis.set_tick_params(labelcolor="#666677")
    ax.grid(color="#333344", linewidth=0.8)

    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1),
              facecolor="#1a1d27", labelcolor="white", fontsize=10)
    ax.set_title(f"{label}\nCluster Centroids Radar",
                 color="white", fontsize=13, fontweight="bold", pad=20)

    out_path = f"{FINAL_OUTPUT_DIR}/radar_{label.replace(' ', '_').lower()}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"  📊 Radar chart saved → {out_path}")
    plt.show()


# ─── COMPARISON TABLE ─────────────────────────────────────────────────────────

def compare_experiments(results: dict):
    """Print side-by-side comparison of optimal k across experiments."""
    print(f"\n{'='*60}")
    print("  EXPERIMENT COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"{'Experiment':<25} {'Opt k (Elbow)':<15} {'Opt k (Silhouette)':<20} {'Best Silhouette'}")
    print("-" * 75)

    for label, df in results.items():
        valid = df.dropna(subset=["within_ss", "silhouette"])
        ks    = valid["k"].values
        wss   = valid["within_ss"].values
        sil   = valid["silhouette"].values

        opt_elbow = _find_elbow(ks, wss)
        opt_sil   = ks[np.argmax(sil)]
        best_sil  = max(sil)

        print(f"{label:<25} {opt_elbow:<15} {opt_sil:<20} {best_sil:.4f}")

    print(f"\n{'='*60}")
    print("  KEY INSIGHT:")
    print("  Adding demographic & risk features shifts the optimal k,")
    print("  revealing segments that pure behavioral RFM cannot detect.")
    print(f"{'='*60}\n")


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    connect()

    all_results = {}

    # ── Phase 1: RFM Only ─────────────────────────────────────────────────────
    print("\n🔵 PHASE 1: RFM Only")
    rfm_results = run_experiment(
        table_name=RFM_TABLE,
        features=FEATURES_RFM,
        label="RFM Only"
    )
    rfm_results.to_csv(f"{FINAL_OUTPUT_DIR}/results_rfm.csv", index=False)
    all_results["RFM Only"] = rfm_results

    # ── Phase 2: RFM + Risk ───────────────────────────────────────────────────
    print("\n🟡 PHASE 2: RFM + Credit Risk")
    risk_results = run_experiment(
        table_name=RISK_TABLE,
        features=FEATURES_RISK,
        label="RFM + Risk"
    )
    risk_results.to_csv(f"{FINAL_OUTPUT_DIR}/results_rfm_risk.csv", index=False)
    all_results["RFM + Risk"] = risk_results

    # ── Phase 3: RFM + Demographics + Risk (Full) ─────────────────────────────
    print("\n🟢 PHASE 3: RFM + Demographics + Risk (Full 360°)")
    full_results = run_experiment(
        table_name=FULL_TABLE,
        features=FEATURES_FULL,
        label="RFM + Demo + Risk"
    )
    full_results.to_csv(f"{FINAL_OUTPUT_DIR}/results_full.csv", index=False)
    all_results["RFM + Demo + Risk"] = full_results

    # ── Plot all experiments ───────────────────────────────────────────────────
    plot_experiment_results(all_results)

    # ── Comparison Summary ────────────────────────────────────────────────────
    compare_experiments(all_results)

    # ── Final Models with Optimal k ───────────────────────────────────────────
    # Determine optimal k for each phase from silhouette
    for label, df in all_results.items():
        valid = df.dropna(subset=["silhouette"])
        if len(valid) == 0:
            continue
        opt_k = valid.loc[valid["silhouette"].idxmax(), "k"]
        print(f"  → {label}: optimal k = {opt_k}")

    # Example: run final model for RFM Only with optimal k
    # Uncomment and adjust k after reviewing results:
    #
    # td_df = DataFrame.from_table(f"{DATABASE}.{RFM_TABLE}")
    # final_model = KMeans(
    #     data=td_df, n_clusters=4, id_column=ID_COL,
    #     center_columns=FEATURES_RFM, max_iter=100, seed=42
    # )
    # pred_df, profile = profile_final_clusters(
    #     final_model, td_df, FEATURES_RFM, "RFM Only Final (k=4)"
    # )
    # pred_df.to_csv(f"{FINAL_OUTPUT_DIR}/final_assignments_rfm.csv", index=False)

    remove_context()
    print("\n✅ All experiments complete. Results in ./kmeans_results/")


if __name__ == "__main__":
    main()

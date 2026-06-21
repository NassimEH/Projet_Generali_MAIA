"""Génère figures + métriques sections 5, 6 et pipeline soumission."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from scipy.stats import uniform
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

import visualization
from andrea_models import subsample
from ml_pipeline import gini_scorer, make_submission


def gini_coefficient(y_true, y_pred):
    """Gini (formule type challenge / Kaggle) — tri par proba décroissante."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n = len(y_true)
    if n == 0:
        return 0.0
    order = np.lexsort((np.arange(n), -y_pred))
    y_sorted = y_true[order]
    n_pos = y_sorted.sum()
    if n_pos == 0:
        return 0.0
    gini_sum = (y_sorted * np.arange(1, n + 1)).sum()
    return (2 * gini_sum / (n_pos * n)) - (n + 1) / n


def normalized_gini(y_true, y_pred):
    """Gini normalisé = Gini(modèle) / Gini(prédiction parfaite)."""
    denom = gini_coefficient(y_true, y_true)
    if denom == 0:
        return 0.0
    return gini_coefficient(y_true, y_pred) / denom


def main():
    data_dir = Path(__file__).resolve().parent
    fig_dir = data_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    X_raw = pd.read_csv(data_dir / "X_train.csv", index_col=0)
    y_raw = pd.read_csv(data_dir / "y_train_saegPGl.csv", index_col=0)
    X_test_file = pd.read_csv(data_dir / "X_test.csv", index_col=0)
    df = X_raw.merge(y_raw, on="Identifiant").dropna()
    if df["EXPO"].dtype == "object":
        df["EXPO"] = df["EXPO"].str.replace(",", ".", regex=False).astype(float)

    cols = ["superficief", "EXPO"]
    feat_kmeans = ["superficief", "ft_2_categ", "ft_21_categ", "ft_22_categ", "EXPO"]
    metrics = {}

    # ========== Section 5 — MLP ==========
    df_tr, df_te = train_test_split(df, test_size=0.2, random_state=1)
    X_tr = df_tr[cols]
    X_te = df_te[cols]
    y_tr = df_tr["target"]
    y_te = df_te["target"]
    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr)
    X_te_s = sc.transform(X_te)
    X_bal, y_bal = SMOTE(random_state=42).fit_resample(X_tr_s, y_tr)

    mlp = MLPClassifier(
        activation="logistic",
        learning_rate_init=0.001,
        early_stopping=True,
        hidden_layer_sizes=(40, 40),
        max_iter=300,
        random_state=42,
    )
    mlp.fit(X_bal, y_bal)
    y_mlp = mlp.predict(X_te_s)
    metrics["MLP accuracy (test, SMOTE train)"] = round(accuracy_score(y_te, y_mlp), 4)
    metrics["MLP Gini normalisé (test)"] = round(
        normalized_gini(y_te.values, mlp.predict_proba(X_te_s)[:, 1]), 4
    )

    visualization.plot_conf_mat(
        y_te, y_mlp, np.array(["pas sinistre", "sinistre"]), width=6, height=5
    )
    plt.title("5 — MLP (réseau de neurones)")
    plt.savefig(fig_dir / "05_mlp_confusion.png", dpi=120, bbox_inches="tight")
    plt.close()

    mlp_base = MLPClassifier(random_state=42, max_iter=200, early_stopping=True)
    param_dist = {
        "hidden_layer_sizes": [(40, 40), (50, 30), (30,)],
        "activation": ["logistic", "relu"],
        "alpha": uniform(0.0001, 0.05),
        "learning_rate_init": uniform(0.0005, 0.005),
    }
    search = RandomizedSearchCV(
        mlp_base,
        param_dist,
        n_iter=8,
        scoring=gini_scorer,
        cv=3,
        random_state=45,
        n_jobs=-1,
    )
    search.fit(X_bal, y_bal)
    metrics["MLP RandomizedSearchCV (meilleur CV)"] = round(search.best_score_, 4)
    metrics["MLP RandomizedSearchCV (test Gini)"] = round(
        normalized_gini(y_te.values, search.predict_proba(X_te_s)[:, 1]), 4
    )
    metrics["MLP RandomizedSearchCV (test accuracy)"] = round(
        accuracy_score(y_te, search.predict(X_te_s)), 4
    )

    fig, ax = plt.subplots(figsize=(6, 4))
    test_gini = normalized_gini(y_te.values, search.predict_proba(X_te_s)[:, 1])
    ax.bar(
        ["CV Gini (meilleur)", "Test Gini"],
        [search.best_score_, test_gini],
        color=["#4c72b0", "#c44e52"],
    )
    ax.set_ylim(0, 1)
    ax.set_ylabel("Gini normalisé")
    ax.set_title("5 — MLP après recherche d'hyperparamètres (score = Gini)")
    for i, v in enumerate([search.best_score_, test_gini]):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center")
    plt.tight_layout()
    plt.savefig(fig_dir / "05_mlp_tuning_scores.png", dpi=120, bbox_inches="tight")
    plt.close()

    # ========== Section 6 — K-means ==========
    X_scaled = StandardScaler().fit_transform(df[feat_kmeans].values)
    K = 3
    n_samples = min(8000, len(X_scaled))
    X_sub = subsample(X_scaled, n_samples, random_state=42)
    repeats = list(range(10, 201, 20))
    inertia_values = []
    for r in repeats:
        km = KMeans(K, n_init=r, random_state=42)
        km.fit(X_sub)
        inertia_values.append(km.inertia_)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(repeats, inertia_values, marker="o")
    ax.set_xlabel("n_init (initialisations)")
    ax.set_ylabel("Inertia")
    ax.set_title("6 — K-means : stabilisation de l'inertie")
    plt.tight_layout()
    plt.savefig(fig_dir / "06_kmeans_inertia.png", dpi=120, bbox_inches="tight")
    plt.close()

    n_init = 80
    clusters = KMeans(K, n_init=n_init, random_state=42).fit_predict(X_sub)
    metrics["K-means silhouette moyen (K=3)"] = round(
        silhouette_score(X_sub, clusters), 4
    )

    plt.figure(figsize=(10, 6))
    visualization.silhouette_diagram(X_sub, clusters, K)
    plt.title("6 — Diagramme de silhouette (K=3)")
    plt.savefig(fig_dir / "06_kmeans_silhouette.png", dpi=120, bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(clusters, bins=K, rwidth=0.8, color="#55a868", edgecolor="black")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Effectif")
    ax.set_title("6 — Répartition des bâtiments en 3 clusters")
    plt.tight_layout()
    plt.savefig(fig_dir / "06_kmeans_clusters_hist.png", dpi=120, bbox_inches="tight")
    plt.close()

    # ========== Section 7 — Soumission challenge (pipeline df_cleaned complet) ==========
    # Ancienne version (2 features + dropna sur test) remplacée par make_submission()
    # pour couvrir les 3 412 lignes de X_test — voir ml_pipeline.py
    submission, sub_metrics = make_submission(data_dir, random_state=5, tune=True)
    submission.to_csv(data_dir / "submission_proba.csv", index=False)
    metrics["Lignes soumission"] = sub_metrics["n_test_submission"]
    metrics["Gini normalisé (validation hold-out 20%)"] = sub_metrics[
        "gini_validation_holdout"
    ]
    metrics["Features encodées (soumission)"] = sub_metrics["n_features"]
    metrics["Gini CV (meilleur, soumission)"] = sub_metrics.get("gini_cv_best")

    proba_test = submission["target"].values
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(proba_test, bins=30, color="#8172b3", edgecolor="white")
    ax.set_xlabel("Probabilité prédite de sinistre")
    ax.set_ylabel("Nombre de bâtiments (test)")
    ax.set_title(
        f"Soumission — {len(submission)} bâtiments, "
        f"{sub_metrics['n_features']} features encodées"
    )
    plt.tight_layout()
    plt.savefig(fig_dir / "07_submission_proba_hist.png", dpi=120, bbox_inches="tight")
    plt.close()

    print("=== Métriques sections 5-6 + soumission ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")
    print("submission_proba.csv écrit.")


if __name__ == "__main__":
    main()

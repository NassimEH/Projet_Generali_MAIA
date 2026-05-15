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
        scoring="accuracy",
        cv=3,
        random_state=45,
        n_jobs=-1,
    )
    search.fit(X_bal, y_bal)
    metrics["MLP RandomizedSearchCV (meilleur CV)"] = round(search.best_score_, 4)
    metrics["MLP RandomizedSearchCV (test)"] = round(search.score(X_te_s, y_te), 4)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(
        ["CV (meilleur)", "Test"],
        [search.best_score_, search.score(X_te_s, y_te)],
        color=["#4c72b0", "#c44e52"],
    )
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.set_title("5 — MLP après recherche d'hyperparamètres")
    for i, v in enumerate([search.best_score_, search.score(X_te_s, y_te)]):
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

    # ========== Soumission challenge ==========
    X_test_df = X_test_file.dropna().copy()
    if X_test_df["EXPO"].dtype == "object":
        X_test_df["EXPO"] = X_test_df["EXPO"].str.replace(",", ".", regex=False).astype(
            float
        )
    ids_test = X_test_df["Identifiant"].copy()
    X_test_m = X_test_df[cols]

    sc_sub = StandardScaler()
    X_full_s = sc_sub.fit_transform(df[cols])
    X_test_s = sc_sub.transform(X_test_m)
    X_full_b, y_full_b = SMOTE(random_state=42).fit_resample(X_full_s, df["target"])

    rf_sub = RandomForestClassifier(
        n_estimators=100, max_leaf_nodes=32, random_state=5, n_jobs=-1
    )
    rf_sub.fit(X_full_b, y_full_b)
    proba_test = rf_sub.predict_proba(X_test_s)[:, 1]

    sub = pd.DataFrame({"Identifiant": ids_test, "target": proba_test})
    sub.to_csv(data_dir / "submission_proba.csv", index=False)
    metrics["Lignes soumission"] = len(sub)
    metrics["Gini normalisé (validation hold-out 20%)"] = round(
        normalized_gini(
            df_te["target"].values,
            rf_sub.predict_proba(sc_sub.transform(df_te[cols]))[:, 1],
        ),
        4,
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(proba_test, bins=30, color="#8172b3", edgecolor="white")
    ax.set_xlabel("Probabilité prédite de sinistre")
    ax.set_ylabel("Nombre de bâtiments (test)")
    ax.set_title("Soumission — distribution des probabilités sur X_test")
    plt.tight_layout()
    plt.savefig(fig_dir / "07_submission_proba_hist.png", dpi=120, bbox_inches="tight")
    plt.close()

    print("=== Métriques sections 5-6 + soumission ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")
    print("submission_proba.csv écrit.")


if __name__ == "__main__":
    main()

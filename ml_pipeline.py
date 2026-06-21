"""Prétraitement commun (train / test) et métriques Gini — challenge Generali."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer


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


def _gini_scorer_func(y_true, y_score):
    """Wrapper sklearn : accepte proba binaire (1 colonne ou matrice n×2)."""
    y_score = np.asarray(y_score)
    if y_score.ndim == 2:
        y_score = y_score[:, 1]
    return normalized_gini(y_true, y_score)


gini_scorer = make_scorer(_gini_scorer_func, response_method="predict_proba")


def load_raw_data(data_dir: Path | str):
    """Charge X_train, y_train, X_test depuis le dossier projet."""
    data_dir = Path(data_dir)
    X_train = pd.read_csv(data_dir / "X_train.csv", index_col=0)
    y_train = pd.read_csv(data_dir / "y_train_saegPGl.csv", index_col=0)
    X_test = pd.read_csv(data_dir / "X_test.csv", index_col=0)
    df_train = X_train.merge(y_train, on="Identifiant").dropna()
    return df_train, X_test


def _replace_commas(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == object:
            out[col] = out[col].astype(str).str.replace(",", ".", regex=False)
    return out


def build_feature_matrices(
    df_train: pd.DataFrame,
    X_test_raw: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, list[str]]:
    """
    Reproduit la logique ``df_cleaned`` du notebook (§3) sur le train,
    applique le même encodage au test **sans dropna** (imputation train).

    Returns
    -------
    X_train, y_train, X_test, test_ids, feature_names
    """
    train = _replace_commas(df_train.copy())
    test = _replace_commas(X_test_raw.copy())

    y_train = train["target"].astype(int)
    test_ids = test["Identifiant"].copy()

    non_numeric_cols: list[str] = []
    for col in train.columns:
        if col in ("Identifiant", "target"):
            continue
        try:
            train[col] = train[col].astype(float)
        except (ValueError, TypeError):
            non_numeric_cols.append(col)

    for col in train.columns:
        if col in ("Identifiant", "target"):
            continue
        if col in non_numeric_cols:
            fill = (
                train[col].mode(dropna=True).iloc[0]
                if not train[col].mode(dropna=True).empty
                else "__NA__"
            )
        else:
            fill = train[col].median()
        train[col] = train[col].fillna(fill)
        if col in test.columns:
            if col in non_numeric_cols:
                test[col] = test[col].fillna(fill)
            else:
                test[col] = pd.to_numeric(test[col], errors="coerce").fillna(fill)

    for col in non_numeric_cols:
        train[col] = train[col].astype(str)
        if col in test.columns:
            test[col] = test[col].astype(str)

    train_feat = train.drop(columns=["target"])
    train_enc = pd.get_dummies(train_feat, columns=non_numeric_cols, drop_first=False)
    test_enc = pd.get_dummies(test, columns=non_numeric_cols, drop_first=False)

    feature_names = [c for c in train_enc.columns if c != "Identifiant"]
    X_train = train_enc[feature_names].astype(float)
    X_test = test_enc.reindex(columns=feature_names, fill_value=0).astype(float)

    return X_train, y_train, X_test, test_ids, feature_names


def make_submission(
    data_dir: Path | str | None = None,
    random_state: int = 5,
    tune: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """
    Entraîne une forêt aléatoire (SMOTE + features encodées) et produit la soumission.

    Returns
    -------
    submission_df, metrics_dict
    """
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
    from scipy.stats import randint

    data_dir = Path(data_dir or Path(__file__).resolve().parent)
    df_train, X_test_raw = load_raw_data(data_dir)
    X_train, y_train, X_test, test_ids, feature_names = build_feature_matrices(
        df_train, X_test_raw
    )

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=1, stratify=y_train
    )

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
    rf_grid = {
        "rf__n_estimators": randint(150, 400),
        "rf__max_leaf_nodes": randint(32, 256),
        "rf__max_depth": randint(12, 40),
        "rf__min_samples_leaf": randint(1, 6),
        "rf__min_samples_split": randint(2, 16),
        "rf__max_features": ["sqrt", "log2", 0.3, 0.5],
        "rf__criterion": ["gini", "entropy"],
    }
    rf_grid_plain = {k.replace("rf__", ""): v for k, v in rf_grid.items()}

    candidates = [
        (
            "smote",
            ImbPipeline([
                ("smote", SMOTE(random_state=42)),
                ("rf", RandomForestClassifier(random_state=random_state, n_jobs=-1)),
            ]),
            rf_grid,
        ),
        (
            "class_weight",
            RandomForestClassifier(
                random_state=random_state, n_jobs=-1, class_weight="balanced_subsample"
            ),
            rf_grid_plain,
        ),
        (
            "smote_class_weight",
            ImbPipeline([
                ("smote", SMOTE(random_state=42)),
                (
                    "rf",
                    RandomForestClassifier(
                        random_state=random_state,
                        n_jobs=-1,
                        class_weight="balanced_subsample",
                    ),
                ),
            ]),
            rf_grid,
        ),
    ]

    if tune:
        best_name, best_search = None, None
        for name, est, grid in candidates:
            search = RandomizedSearchCV(
                est,
                grid,
                n_iter=8,
                scoring=gini_scorer,
                cv=cv,
                random_state=random_state,
                n_jobs=-1,
            )
            search.fit(X_tr, y_tr)
            if best_search is None or search.best_score_ > best_search.best_score_:
                best_name, best_search = name, search
        model = best_search.best_estimator_
        best_cv_gini = best_search.best_score_
        strategy = best_name
    else:
        model = ImbPipeline([
            ("smote", SMOTE(random_state=42)),
            (
                "rf",
                RandomForestClassifier(
                    random_state=random_state,
                    n_jobs=-1,
                    class_weight="balanced_subsample",
                    n_estimators=200,
                    max_leaf_nodes=64,
                ),
            ),
        ])
        model.fit(X_tr, y_tr)
        best_cv_gini = None
        strategy = "default"

    val_proba = model.predict_proba(X_val)[:, 1]
    val_gini = normalized_gini(y_val.values, val_proba)

    model.fit(X_train, y_train)
    proba_test = model.predict_proba(X_test)[:, 1]

    submission = pd.DataFrame({"Identifiant": test_ids.values, "target": proba_test})
    metrics = {
        "n_train": len(X_train),
        "n_features": len(feature_names),
        "n_test_submission": len(submission),
        "n_test_expected": len(X_test_raw),
        "gini_validation_holdout": round(val_gini, 4),
        "gini_cv_best": round(best_cv_gini, 4) if best_cv_gini is not None else None,
        "strategy": strategy,
    }
    return submission, metrics

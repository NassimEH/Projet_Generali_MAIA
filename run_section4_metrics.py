"""Métriques clés section 4 pour le README (échantillon rapide, pas le RandomizedSearchCV complet)."""
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from andrea_models import AndreaLinearRegression
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import visualization

DATA = Path(__file__).resolve().parent
FIG = DATA / "figures"
FIG.mkdir(exist_ok=True)

X = pd.read_csv(DATA / "X_train.csv", index_col=0)
y = pd.read_csv(DATA / "y_train_saegPGl.csv", index_col=0)
df = X.merge(y, on="Identifiant").dropna()
if df["EXPO"].dtype == "object":
    df["EXPO"] = df["EXPO"].str.replace(",", ".", regex=False).astype(float)

df_train, df_test = train_test_split(df, test_size=0.3, random_state=19)
cols = ["superficief", "EXPO"]
metrics = {}

# 4.1 Régression linéaire (superficief)
X_tr = df_train[["superficief"]].values.reshape(-1, 1)
y_tr = df_train["target"].values
X_te = df_test[["superficief"]].values.reshape(-1, 1)
y_te = df_test["target"].values
m = AndreaLinearRegression()
m.fit(X_tr, y_tr, column_names=["superficief"])
y_p = m.predict(X_te)
rmse_lin = math.sqrt(mean_squared_error(y_te, y_p))
metrics["RMSE régression linéaire (superficief)"] = round(rmse_lin, 4)

interval = np.linspace(X_tr.min(), X_tr.max(), 300).reshape(-1, 1)
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(X_tr, y_tr, color="blue", s=5, alpha=0.4, label="train")
ax.plot(interval, m.predict(interval), color="red", label="régression")
ax.set_xlabel("superficief")
ax.set_ylabel("target")
ax.set_title("4.1 — Régression linéaire (superficief → sinistre)")
ax.legend()
plt.tight_layout()
plt.savefig(FIG / "04_regression_superficief.png", dpi=120, bbox_inches="tight")
plt.close()

# 4.6 Logistique + SMOTE
df_tr, df_te = train_test_split(df, test_size=0.2, random_state=1)
X_tr = df_tr[cols]
X_te = df_te[cols]
y_tr = df_tr["target"]
y_te = df_te["target"]
scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr)
X_te_s = scaler.transform(X_te)
log = LogisticRegression(C=1, max_iter=1000)
log.fit(X_tr_s, y_tr)
acc_before = accuracy_score(y_te, log.predict(X_te_s))

sm = SMOTE(random_state=42)
X_bal, y_bal = sm.fit_resample(X_tr_s, y_tr)
log.fit(X_bal, y_bal)
y_pred = log.predict(X_te_s)
acc_after = accuracy_score(y_te, y_pred)
metrics["Accuracy logistique (sans SMOTE)"] = round(acc_before, 4)
metrics["Accuracy logistique (avec SMOTE)"] = round(acc_after, 4)

visualization.plot_conf_mat(
    y_te, y_pred, np.array(["pas sinistre", "sinistre"]), width=6, height=5
)
plt.title("4.6 — Matrice de confusion (logistique + SMOTE)")
plt.savefig(FIG / "04_logistic_confusion.png", dpi=120, bbox_inches="tight")
plt.close()

# 4.7 Forêt aléatoire (simple)
X_tr, X_te, y_tr, y_te = train_test_split(df[cols], df["target"], test_size=0.2, random_state=4)
X_b, y_b = SMOTE(random_state=42).fit_resample(X_tr, y_tr)
rf = RandomForestClassifier(n_estimators=100, max_leaf_nodes=16, random_state=5, n_jobs=-1)
rf.fit(X_b, y_b)
y_rf = rf.predict(X_te)
metrics["Accuracy forêt aléatoire (SMOTE, baseline)"] = round(accuracy_score(y_te, y_rf), 4)

visualization.plot_conf_mat(
    y_te, y_rf, np.array(["pas sinistre", "sinistre"]), width=6, height=5
)
plt.title("4.7 — Forêt aléatoire (baseline)")
plt.savefig(FIG / "04_rf_confusion.png", dpi=120, bbox_inches="tight")
plt.close()

plt.figure(figsize=(7, 4))
visualization.plot_feature_importances(rf.feature_importances_, cols)
plt.title("Importances — superficief vs EXPO")
plt.savefig(FIG / "04_rf_importances.png", dpi=120, bbox_inches="tight")
plt.close()

print("=== Métriques section 4 ===")
for k, v in metrics.items():
    print(f"{k}: {v}")

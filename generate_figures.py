"""Regénère les figures du README (dossier figures/)."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import visualization

DATA_DIR = Path(__file__).resolve().parent
FIG_DIR = DATA_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

X_train = pd.read_csv(DATA_DIR / "X_train.csv", index_col=0)
y_train = pd.read_csv(DATA_DIR / "y_train_saegPGl.csv", index_col=0)
df = X_train.merge(y_train, on="Identifiant")
df_raw = df.copy()

# Section 2
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
df_raw["target"].value_counts().sort_index().plot(
    kind="bar", ax=axes[0], color=["#4c72b0", "#c44e52"]
)
axes[0].set_title("Effectifs par classe (avant dropna)")
axes[0].set_xticklabels(["0 - pas de sinistre", "1 - sinistre"], rotation=0)
df_raw["target"].value_counts(normalize=True).sort_index().plot(
    kind="bar", ax=axes[1], color=["#4c72b0", "#c44e52"]
)
axes[1].set_title("Proportions (~23% de sinistres)")
axes[1].set_ylabel("Proportion")
plt.tight_layout()
plt.savefig(FIG_DIR / "02_target_desequilibre.png", dpi=120, bbox_inches="tight")
plt.close()

miss = df_raw.isna().sum().sort_values(ascending=False)
miss = miss[miss > 0]
fig, ax = plt.subplots(figsize=(8, 4))
miss.plot(kind="barh", ax=ax, color="#55a868")
ax.set_title("Valeurs manquantes par colonne (avant dropna)")
ax.set_xlabel("Nombre de cases vides")
plt.tight_layout()
plt.savefig(FIG_DIR / "02_valeurs_manquantes.png", dpi=120, bbox_inches="tight")
plt.close()

# Section 3
df = df.dropna()
if df["EXPO"].dtype == "object":
    df["EXPO"] = df["EXPO"].str.replace(",", ".", regex=False).astype(float)

fig, ax = plt.subplots(figsize=(7, 5))
df.boxplot(column=["superficief"], ax=ax)
ax.set_title("Superficie (m²) — outliers vers le haut")
ax.set_ylabel("superficief")
plt.tight_layout()
plt.savefig(FIG_DIR / "03_boxplot_superficief.png", dpi=120, bbox_inches="tight")
plt.close()

fig, ax = plt.subplots(figsize=(7, 5))
df.boxplot(column=["EXPO"], ax=ax)
ax.set_title("EXPO (fraction année assurée) — majorité à 1")
ax.set_ylabel("EXPO")
plt.tight_layout()
plt.savefig(FIG_DIR / "03_boxplot_expo.png", dpi=120, bbox_inches="tight")
plt.close()

_cols = ["superficief", "ft_2_categ", "ft_21_categ", "ft_22_categ", "EXPO", "target"]
visualization.plot_corr(df[_cols], width=10, height=8, print_value=True, thresh=0.05)
plt.savefig(FIG_DIR / "03_heatmap_correlations.png", dpi=120, bbox_inches="tight")
plt.close()

fig, ax = plt.subplots(figsize=(8, 5))
df.boxplot(column=["superficief"], by="target", ax=ax)
plt.suptitle("")
ax.set_title("Superficie selon sinistre (target)")
ax.set_xlabel("target (0=non, 1=oui)")
plt.tight_layout()
plt.savefig(FIG_DIR / "03_superficie_par_target.png", dpi=120, bbox_inches="tight")
plt.close()

df[["ft_2_categ", "ft_22_categ", "superficief", "EXPO"]].hist(figsize=(12, 10))
plt.suptitle("Distributions (4 variables clés)", y=1.02)
plt.tight_layout()
plt.savefig(FIG_DIR / "03_histogrammes.png", dpi=120, bbox_inches="tight")
plt.close()

print("Figures enregistrées dans", FIG_DIR)
for p in sorted(FIG_DIR.glob("*.png")):
    print(" ", p.name)

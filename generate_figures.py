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
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
df_raw["target"].value_counts().sort_index().plot(
    kind="bar", ax=axes[0], color=["#4c72b0", "#c44e52"]
)
axes[0].set_title("Effectifs — target (avant dropna)")
axes[0].set_xlabel("target")
axes[0].set_ylabel("Nombre de bâtiments")
axes[0].set_xticklabels(["0 — pas de sinistre", "1 — sinistre"], rotation=0)
axes[0].text(
    0.5,
    -0.28,
    "≈ 7 900 sans sinistre vs ≈ 2 300 avec sinistre.\nLa classe « sinistre » est minoritaire.",
    transform=axes[0].transAxes,
    ha="center",
    va="top",
    fontsize=9,
    color="#333333",
)

df_raw["target"].value_counts(normalize=True).sort_index().plot(
    kind="bar", ax=axes[1], color=["#4c72b0", "#c44e52"]
)
axes[1].set_title("Proportions — déséquilibre des classes")
axes[1].set_xlabel("target")
axes[1].set_ylabel("Proportion")
axes[1].set_xticklabels(["0", "1"], rotation=0)
axes[1].text(
    0.5,
    -0.28,
    "≈ 77 % sans sinistre, ≈ 23 % avec sinistre.\nToujours répondre « 0 » ≈ 77 % de réussite, sans trier les risques.",
    transform=axes[1].transAxes,
    ha="center",
    va="top",
    fontsize=9,
    color="#333333",
)

plt.tight_layout()
plt.subplots_adjust(bottom=0.22)
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

years_ft22 = df["ft_22_categ"].astype(int)
fig, ax = plt.subplots(figsize=(10, 4))
ax.hist(years_ft22, bins=30, color="#55a868", edgecolor="white")
ax.set_title("ft_22_categ — distribution des années")
ax.set_xlabel("Année (hypothèse : construction)")
ax.set_ylabel("Nombre de bâtiments")
ax.text(
    0.5,
    -0.2,
    "Pic vers 1960–1980 ; quelques valeurs très anciennes ou récentes.",
    transform=ax.transAxes,
    ha="center",
    fontsize=9,
    color="#333333",
)
plt.tight_layout()
plt.subplots_adjust(bottom=0.2)
plt.savefig(FIG_DIR / "03_hist_ft_22_categ.png", dpi=120, bbox_inches="tight")
plt.close()

counts_ft2 = df["ft_2_categ"].value_counts().sort_index()
fig, ax = plt.subplots(figsize=(8, 4))
counts_ft2.plot(kind="bar", ax=ax, color="#4c72b0", edgecolor="black")
ax.set_title("ft_2_categ — année d'observation du contrat")
ax.set_xlabel("Année")
ax.set_ylabel("Nombre de bâtiments")
ax.set_xticklabels([int(x) for x in counts_ft2.index], rotation=0)
for i, v in enumerate(counts_ft2.values):
    ax.text(i, v + max(counts_ft2) * 0.01, f"{int(v)}", ha="center", fontsize=9)
ax.text(
    0.5,
    -0.22,
    "Peu de dispersion : tous les contrats sont observés entre 2012 et 2016.",
    transform=ax.transAxes,
    ha="center",
    fontsize=9,
    color="#333333",
)
plt.tight_layout()
plt.subplots_adjust(bottom=0.18)
plt.savefig(FIG_DIR / "03_hist_ft_2_categ.png", dpi=120, bbox_inches="tight")
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

_key = ["ft_2_categ", "ft_22_categ", "superficief", "EXPO"]
fig, axes = plt.subplots(2, 2, figsize=(16, 11))
for ax, col in zip(axes.ravel(), _key):
    s = df[col].dropna()
    nuniq = int(s.nunique())
    b = nuniq if nuniq <= 25 else 30
    ax.hist(s, bins=b, color="#4c72b0", edgecolor="white")
    ax.set_title(col, fontsize=12, fontweight="bold", pad=12)
    ax.tick_params(axis="both", labelsize=10)
    ax.tick_params(axis="x", rotation=25)
fig.suptitle("Distributions (4 variables clés)", fontsize=14, y=1.01)
plt.tight_layout()
plt.subplots_adjust(top=0.92, hspace=0.45, wspace=0.3)
plt.savefig(FIG_DIR / "03_histogrammes.png", dpi=120, bbox_inches="tight")
plt.close()

print("Figures enregistrées dans", FIG_DIR)
for p in sorted(FIG_DIR.glob("*.png")):
    print(" ", p.name)

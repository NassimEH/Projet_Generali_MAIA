# -*- coding: utf-8 -*-
"""Insert markdown interpretations below plot cells in Projet.ipynb."""
import json
from pathlib import Path

NOTEBOOK = Path(__file__).parent / "Projet.ipynb"

LEGENDE_FT = """**Légende — colonnes au nom peu clair** (Generali anonymisé ; indicatif) : `ft_2_categ` = année d'observation (2012–2016) ; `ft_21_categ` = code catégoriel (hyp. type/usage) ; `ft_22_categ` = hyp. année de construction ; `EXPO` = fraction d'année assurée (1 = année pleine) ; `superficief` = superficie (m²) ; `target` = sinistre (0/1)."""

CONF_MAT_MD = """**Interprétation — matrice de confusion**

| | Prédit 0 | Prédit 1 |
|---|----------|----------|
| **Réel 0** (pas de sinistre) | Vrais négatifs — bien classés | Faux positifs — alarmes inutiles |
| **Réel 1** (sinistre) | Faux négatifs — sinistres manqués | Vrais positifs — sinistres détectés |

- **`target` = 0** : pas de sinistre bâtiment ; **`target` = 1** : au moins un sinistre.
- Déséquilibre ~77 % / 23 % : une **accuracy** seule peut tromper — regarder faux négatifs (sinistres ratés).
"""

INSERTIONS = [
    # (after_cell_index, markdown) — applied in reverse order
    (153, """**Interprétation — répartition des clusters (K-means)**

- Chaque barre = nombre de bâtiments assignés à un **groupe** (cluster) après regroupement sur quelques variables numériques.
- Sert surtout à l'**exploration** (profils de parc), pas directement à la note Gini du challenge.
- Des clusters **très déséquilibrés** = un groupe dominant + petits groupes niche.
"""),
    (102, """**Interprétation — régressions linéaire, quadratique et cubique**

- Sur **log(superficie)** : comparaison de complexités (droite, courbe, courbe plus flexible).
- Si la **RMSE** test ne baisse presque pas du linéaire au cubique → pas besoin de modèle très complexe sur cette seule variable.
- Risque du polynôme élevé : **sur-apprentissage** (mieux sur le train, pas sur le test) — d'où la section 4.5 (Ridge).
"""),
    (100, """**Interprétation — régression linéaire vs quadratique sur log(superficie)**

- **Axe X :** `superficief_log` = log(superficie + 1) pour réduire l'effet des très grands bâtiments.
- **Points bleus :** vrais `target` (0 ou 1) ; **rouge** = droite ; **vert** = courbe (degré 2).
- Si la courbe verte n'améliore pas beaucoup la RMSE → relation déjà **presque linéaire** après le log.
- Même limite qu'en 4.1 : prédire du 0/1 avec une régression continue.
"""),
    (85, """**Interprétation — régression linéaire sur log(superficie)** (§ 4.2)

- **Axe X :** `superficief_log` (superficie transformée) ; **Y :** `target` (0/1).
- **Droite rouge :** tendance moyenne après compression des extrêmes — souvent **plus stable** que sur la superficie brute.
- Compare la **RMSE** au graphique 4.1 : si elle baisse peu, le log aide marginalement.
"""),
    (79, """**Interprétation — régression linéaire sur `EXPO` seul** (§ 4.1)

- **Points** en bandes y = 0 et y = 1 ; **gros nuage à EXPO = 1** (années pleines).
- **Droite rouge** légèrement montante : plus de temps assuré → score un peu plus haut (corr. ~0,10 avec `target`).
- **`EXPO` = 1** : contrat sur l'année entière ; valeurs &lt; 1 = couverture partielle.
- Variable **seule insuffisante** (RMSE ~0,40) ; utile **combinée** à d'autres features.
"""),
    (74, """**Interprétation — régression linéaire sur `superficief` seul** (§ 4.1)

- **Points bleus :** chaque bâtiment (superficie en X, sinistre 0/1 en Y).
- **Droite rouge :** plus la surface augmente, plus le score prédit monte (lien ~0,30 avec `target`).
- Le modèle prédit des **nombres hors [0,1]** → normal en régression linéaire sur une cible binaire ; la **logistique** (4.6) corrige cela.
- Nuage très dispersé → la superficie **aide un peu** mais **ne suffit pas**.
"""),
    (70, f"""**Interprétation — matrice de corrélation (section 4)**

- Chaque case = corrélation de **Pearson** entre −1 et +1 (rouge = positif, bleu = négatif).
- **`superficief` × `target` ≈ 0,30** : signal le plus fort ici.
- **`EXPO` × `target` ≈ 0,10** ; **`ft_2`**, **`ft_22`** : liens très faibles en linéaire.
- **`superficief` × `ft_22` ≈ 0,20** : grands bâtiments un peu plus « récents » — faible redondance.

{LEGENDE_FT}
"""),
    (59, f"""**Interprétation — nuage de points superficie × année d'observation**

- Un point = un bâtiment ; **X** = superficie (m²), **Y** = `ft_2_categ` (années **2012–2016** seulement).
- Bandes horizontales = années discrètes ; **pas de tendance claire** « plus récent → plus grand ».
- Zone vide hors 2012–2016 = pas de données à ces dates.

{LEGENDE_FT}
"""),
    (56, f"""**Interprétation — matrice de scatter plots (4 variables)**

- **Diagonale :** histogramme de chaque variable.
- **Hors diagonale :** nuage (X = colonne, Y = ligne) — chercher une forme (montée, bandes verticales, nuage rond).
- **`EXPO` :** pic à 1 ; **`superficief` :** très asymétrique ; **`ft_2` :** 5 années ; **`ft_22` :** type « année » (construction).
- Peu de relations **linéaires fortes** entre ces 4 variables → le sinistre (`target`, absent ici) se lit mieux sur la heatmap et les modèles.

{LEGENDE_FT}
"""),
    (54, """**Interprétation — superficie selon sinistre ou non**

- **Deux boxplots :** `target = 0` (pas de sinistre) vs `target = 1` (sinistre).
- **Ligne verte au centre de la boîte = médiane** (50 % des bâtiments en dessous, 50 % au-dessus) — ce ne sont **pas** des points individuels.
- **Boîte bleue :** milieu de 50 % des bâtiments (Q1–Q3) ; **points noirs** = outliers (très grandes superficies).
- Médiane **plus haute** pour `target = 1` → sinistres un peu plus liés aux **grands** bâtiments ; **fort chevauchement** → la taille seule ne suffit pas.

**Colonnes :** `superficief` = superficie (m²) ; `target` = sinistre (0/1).
"""),
    (52, """**Interprétation — heatmap des corrélations (§ 3.2)**

- Triangle = corrélations **linéaires** (Pearson) ; seuil **0,05** : les très petites valeurs s'affichent comme 0.
- **`superficief` ↔ `target` (~0,30)** : variable la plus liée au sinistre sur ce graphique.
- **`ft_21` ↔ `target` (~0,11)** : piste secondaire ; **`ft_2`**, **`ft_22`** souvent masqués si &lt; 0,05.
- Peu de corrélations fortes **entre** features → peu de redondance à supprimer ici.
"""),
    (49, f"""**Interprétation — histogrammes (3 classes seulement)**

- Même 4 variables qu'au-dessus, mais **3 intervalles** par variable → vue **très grossière**.
- Utile pour voir grossièrement où se concentre la masse (ex. petits bâtiments, EXPO = 1).

{LEGENDE_FT}
"""),
    (48, f"""**Interprétation — histogrammes des 4 variables clés**

| Variable | Lecture rapide |
|----------|----------------|
| `superficief` | Pic à gauche (petits/moyens bâtiments), longue traîne à droite |
| `EXPO` | Majorité à **1** (année pleine) |
| `ft_2_categ` | Répartition 2012–2016 |
| `ft_22_categ` | Masse vers années 1960–1980 (hyp. construction) |

{LEGENDE_FT}
"""),
    (47, f"""**Interprétation — vue d'ensemble (toutes variables numériques)**

- Un panneau par colonne numérique (hors `Identifiant`) : forme de la distribution.
- Compare avec les histogrammes ciblés ci-dessous : ici vue **globale**, parfois illisible si trop de modalités.

{LEGENDE_FT}
"""),
]

# Enhance existing markdown cells
UPDATES = {
    36: """**Interprétation — boxplot `superficief` (superficie en m²)**

- **Boîte bleue :** 50 % centraux des bâtiments (entre Q1 et Q3).
- **Ligne verte :** **médiane** (valeur « typique ») — pas la moyenne, pas un point de données.
- **Moustaches :** étendue « habituelle » ; **cercles noirs** = **outliers** (très grandes superficies, au-dessus du 75e percentile).
- Distribution **très asymétrique** : beaucoup de petits/moyens bâtiments, quelques géants qui tirent la moyenne vers le haut.
""",
    39: """**Interprétation — boxplot `EXPO`**

- **`EXPO`** = fraction de l'année où le bâtiment était **assuré** (1 = année entière, 0,5 ≈ six mois).
- Boîte **collée en haut à 1** : ~**76 %** des contrats sont des années pleines.
- Points en dessous de 1 = couvertures **partielles** (minorité).
- Boxplot peu informatif quand Q1 ≈ Q3 ≈ 1 — l'histogramme ou les effectifs sont plus parlants.
""",
    26: """**Interprétation — déséquilibre de `target`**

- **`target` = 0** : pas de sinistre bâtiment sur la période ; **`target` = 1** : au moins un sinistre.
- Environ **77 %** de 0 et **23 %** de 1 → classe minoritaire = sinistres.
- **Pourquoi c'est important :** un modèle qui prédit toujours « 0 » peut avoir ~77 % d'accuracy sans être utile ; d'où **SMOTE** et métrique **Gini** plus tard.
""",
}

PLOT_HIST_GRID_CELL = '''def plot_hist_grid(data, columns, bins=30, figsize=(18, 12), ncols=3):
    """Histogrammes en grille lisible (grande figure, espacement, titres)."""
    n = len(columns)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.atleast_1d(axes).ravel()
    for ax, col in zip(axes, columns):
        s = data[col].dropna()
        nuniq = int(s.nunique())
        if nuniq <= 25:
            ax.hist(s, bins=nuniq, color="#4c72b0", edgecolor="white")
        else:
            ax.hist(s, bins=bins, color="#4c72b0", edgecolor="white")
        ax.set_title(col, fontsize=12, fontweight="bold", pad=12)
        ax.tick_params(axis="both", labelsize=10)
        ax.tick_params(axis="x", rotation=25)
    for ax in axes[n:]:
        ax.set_visible(False)
    fig.suptitle("Distributions des variables numériques", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.subplots_adjust(top=0.93, hspace=0.55, wspace=0.35)
    plt.show()
    return fig

# Vue d'ensemble (sans Identifiant)
_num_cols = [c for c in df.select_dtypes(include="number").columns if c != "Identifiant"]
plot_hist_grid(df, _num_cols, bins=30, figsize=(22, 16), ncols=3)
'''

CELL_48 = '''_key_vars = ["ft_2_categ", "ft_22_categ", "superficief", "EXPO"]
plot_hist_grid(df, _key_vars, bins=30, figsize=(16, 11), ncols=2)
'''

CELL_49 = (
    "'''chaque histogramme aura 3 intervalles (ou \"buckets\") pour regrouper les valeurs. "
    "Cela simplifie la vue en faisant peu de classes.'''\n"
    "plot_hist_grid(df, _key_vars, bins=3, figsize=(16, 11), ncols=2)\n"
)


def md_cell(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [text if text.endswith("\n") else text + "\n"],
    }


def main():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cells = nb["cells"]

    for idx, text in UPDATES.items():
        if idx < len(cells) and cells[idx]["cell_type"] == "markdown":
            cells[idx]["source"] = [text + "\n"]

    # Fix histogram cells
    if 47 < len(cells):
        cells[47]["source"] = [PLOT_HIST_GRID_CELL]
    if 48 < len(cells):
        cells[48]["source"] = [CELL_48]
    if 49 < len(cells):
        cells[49]["source"] = [CELL_49]

    for after_idx, text in sorted(INSERTIONS, key=lambda x: -x[0]):
        pos = after_idx + 1
        # Skip if next cell already starts with same interpretation title
        if pos < len(cells) and cells[pos]["cell_type"] == "markdown":
            nxt = "".join(cells[pos]["source"])
            if "**Interprétation" in nxt and after_idx not in (52,):
                continue
        cells.insert(pos, md_cell(text))

    # Matrices de confusion restantes (logistique, RF, MLP…)
    conf_extra = 0
    i = 0
    while i < len(cells):
        c = cells[i]
        src = "".join(c.get("source", []))
        if (
            c["cell_type"] == "code"
            and "plot_conf_mat(" in src
            and "import" not in src.split("plot_conf_mat")[0][-80:]
        ):
            if i + 1 >= len(cells) or "**Interprétation — matrice de confusion**" not in "".join(
                cells[i + 1]["source"]
            ):
                cells.insert(i + 1, md_cell(CONF_MAT_MD))
                conf_extra += 1
                i += 2
                continue
        i += 1

    NOTEBOOK.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Done. Inserted/updated interpretations. Extra conf matrices: {conf_extra}")


if __name__ == "__main__":
    main()

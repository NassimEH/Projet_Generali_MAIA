import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, silhouette_samples, silhouette_score,\
				average_precision_score, precision_score, recall_score, \
				precision_recall_curve, roc_curve, roc_auc_score
from sklearn.utils.multiclass import unique_labels
import matplotlib.cm as cm
from matplotlib.ticker import FormatStrFormatter

def plot_corr(df, width, height, print_value, thresh=0, triangular="full"):
	    """Matrice de corrélations (Pearson) avec conversion numérique automatique.

	    Parameters
	    ------------
	    triangular : {'full', 'lower', 'upper'}
	        'full' affiche toute la matrice (lisible pour superficief).
	        'lower' / 'upper' masquent la moitié dupliquée (comportement cours).
	    """
	    data = df.copy()
	    for col in data.columns:
		    if data[col].dtype == object:
			    data[col] = pd.to_numeric(
				    data[col].astype(str).str.replace(",", ".", regex=False),
				    errors="coerce",
			    )

	    cormat = data.corr(numeric_only=True)

	    mask = None
	    if triangular == "lower":
		    mask = np.triu(np.ones_like(cormat, dtype=bool), k=1)
	    elif triangular == "upper":
		    mask = np.tril(np.ones_like(cormat, dtype=bool), k=-1)

	    plot_data = cormat.copy()
	    if thresh > 0:
		    plot_data = plot_data.where(plot_data.abs() >= thresh)

	    f, ax = plt.subplots(figsize=(width, height))
	    sns.heatmap(
		    plot_data,
		    vmin=-1,
		    vmax=1,
		    cmap="coolwarm",
		    annot=cormat if print_value else False,
		    fmt=".2f",
		    mask=mask,
		    square=True,
		    linewidths=0.5,
		    linecolor="white",
		    cbar_kws={"label": "Corrélation de Pearson"},
		    ax=ax,
	    )
	    ax.set_title("Corrélations entre variables numériques", pad=12)
	    plt.tight_layout()
	    return f, ax



def rotate_labels(sm):
		"""
		Rotate the labels of a scatter matrix

		Parameters
		-----------------
		sm: scatter matrix
		"""
		# source: https://stackoverflow.com/a/32568134/2110769
		[s.xaxis.label.set_rotation(45) for s in sm.reshape(-1)]
		[s.yaxis.label.set_rotation(0) for s in sm.reshape(-1)]
 


def plot_conf_mat(y_true, y_pred, class_names, normalize=True, title=None, 
    cmap=plt.cm.viridis, text=True, width=8, height=8 ):
 		"""
 		This function prints and plots the confusion matrix.
 		In case of errors, you may need to do 
 				class_names = np.array(class_names)
 		before calling this function.


 		Parameters:
 		--------------------------
		target: The array of the true categories. It contains as many values 
				as the number of samples. Each value is an integer number 
				corresponding to a certain category. This array represents 
				the true category of each sample.
		
		predicted:	It has the same format, but it does not represent the true 
					category, rather it represents the result of a model.

		class_names:	Array of strings, where the first. The k-th element
						is the name of the k-th class

		normalize: 	(default=True) If False, it just prints the number of values in 
						each cell. Otherwise it prints the frequencies, i.e.
						the sum over each row is 1

		title: 	(default=None) Title of the figure

		cmap: 	(default=plt.cm.viridis) Color map

		text:	(default=True) If True it prints numerical values on each cell. Otherwise
				it just shows the colors


		width: 	(default=8) Of the figure

		height:	(default=8) Of the figure
 		"""
 		if not isinstance(class_names, (np.ndarray) ):
 		  raise TypeError('class_names must be an np.array. It is instead ', 
                     type(class_names), '. Try to convert to arrays before: executing',
                     'class_names = np.array(class_names)')
    
    
 		if not title:
 		  if normalize:
 		    title = 'Normalized confusion matrix'
 		  else:
 		    title = 'Confusion matrix, without normalization'

 		# Compute confusion matrix
 		cm = confusion_matrix(y_true, y_pred)

 		# Only use the labels that appear in the data
 		labels_present = unique_labels(y_true, y_pred)
 		classes = class_names[labels_present]
 		if normalize:
 			cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
 			print("Normalized confusion matrix")
 		else:
 			print('Confusion matrix, without normalization')

 		print(cm)

 		fig, ax = plt.subplots(figsize=(width,height))
 		im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
 		ax.figure.colorbar(im, ax=ax)
 		# We want to show all ticks...
 		ax.set(xticks=np.arange(cm.shape[1]),
 			yticks=np.arange(cm.shape[0]),
 			# ... and label them with the respective list entries
 			xticklabels=classes, yticklabels=classes,
 			title=title,
 			ylabel='True label',
 			xlabel='Predicted label')

 		# Rotate the tick labels and set their alignment.
 		plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
 			rotation_mode="anchor")

 		# Loop over data dimensions and create text annotations.
 		if text == True:
	 		fmt = '.2f' if normalize else 'd'
	 		thresh = cm.max() / 2.
	 		for i in range(cm.shape[0]):
	 			for j in range(cm.shape[1]):
	 				ax.text(j, i, format(cm[i, j], fmt),
	 					ha="center", va="center",
	 					color="white" if cm[i, j] < thresh else "black")
 		fig.tight_layout()
 		return ax


def _logistic_grid_for_feature(X_df, feature_name, feature_names, fixed_features=None, n_points=300):
	"""Grille 1D sur ``feature_name`` ; les autres colonnes restent fixes (médiane par défaut)."""
	if fixed_features is None:
		fixed_features = {
			c: float(X_df[c].median())
			for c in feature_names
			if c != feature_name
		}
	x_grid = np.linspace(float(X_df[feature_name].min()), float(X_df[feature_name].max()), n_points)
	grid = pd.DataFrame({feature_name: x_grid})
	for col in feature_names:
		if col != feature_name:
			grid[col] = fixed_features[col]
	return x_grid, grid


def plot_logistic_probability_curve(
	model,
	X_df,
	y,
	feature_name,
	feature_names,
	scaler=None,
	fixed_features=None,
	ax=None,
	title=None,
	color="C0",
	label=None,
	show_scatter=True,
	scatter_alpha=0.12,
	scatter_size=6,
	jitter=0.025,
	n_points=300,
):
	"""
	Courbe de probabilité P(sinistre=1) d'une régression logistique en fonction d'une variable.

	Les autres variables sont maintenues fixes (médiane du train par défaut).
	"""
	if feature_name not in feature_names:
		raise ValueError(f"{feature_name!r} doit figurer dans feature_names")

	x_grid, grid = _logistic_grid_for_feature(
		X_df, feature_name, feature_names, fixed_features, n_points
	)
	X_in = scaler.transform(grid) if scaler is not None else grid.values
	proba = model.predict_proba(X_in)[:, 1]

	created_fig = ax is None
	if ax is None:
		fig, ax = plt.subplots(figsize=(8, 5))
	else:
		fig = ax.figure

	if show_scatter:
		rng = np.random.default_rng(42)
		y_plot = np.asarray(y) + rng.uniform(-jitter, jitter, size=len(y))
		ax.scatter(
			X_df[feature_name],
			y_plot,
			alpha=scatter_alpha,
			s=scatter_size,
			c="#888888",
			linewidths=0,
			zorder=1,
		)

	ax.plot(x_grid, proba, color=color, linewidth=2.2, label=label, zorder=3)
	ax.axhline(0.5, color="#aaaaaa", linestyle="--", linewidth=0.9, zorder=2)
	ax.set_ylim(-0.05, 1.05)
	ax.set_xlabel(feature_name)
	ax.set_ylabel("P(sinistre = 1)")
	if title:
		ax.set_title(title)
	if label:
		ax.legend(loc="lower right", framealpha=0.9)
	ax.grid(True, alpha=0.25)
	if created_fig:
		fig.tight_layout()
	return fig, ax


def plot_logistic_curves_compare(
	models,
	labels,
	X_df,
	y,
	feature_name,
	feature_names,
	scaler=None,
	fixed_features=None,
	colors=None,
	figsize=(14, 5),
	suptitle=None,
	savepath=None,
):
	"""
	Compare plusieurs modèles logistiques (ex. sans / avec SMOTE) sur deux panneaux côte à côte.
	"""
	if len(models) != len(labels):
		raise ValueError("models et labels doivent avoir la même longueur")
	if colors is None:
		colors = ["#c44e52", "#4c72b0", "#55a868", "#8172b2"][: len(models)]

	fig, axes = plt.subplots(1, len(models), figsize=figsize, sharey=True)
	if len(models) == 1:
		axes = [axes]

	for ax, model, label, color in zip(axes, models, labels, colors):
		plot_logistic_probability_curve(
			model,
			X_df,
			y,
			feature_name,
			feature_names,
			scaler=scaler,
			fixed_features=fixed_features,
			ax=ax,
			title=label,
			color=color,
			label="Courbe logistique",
			show_scatter=True,
		)

	if fixed_features:
		fixed_txt = ", ".join(f"{k}={v:.3g}" for k, v in fixed_features.items())
	else:
		medians = {c: float(X_df[c].median()) for c in feature_names if c != feature_name}
		fixed_txt = ", ".join(f"{k}={v:.3g}" for k, v in medians.items())

	fig.text(
		0.5,
		-0.02,
		f"Autres variables fixées à leur médiane : {fixed_txt}",
		ha="center",
		va="top",
		fontsize=9,
		color="#444444",
	)
	if suptitle:
		fig.suptitle(suptitle, fontsize=13, y=1.02)
	fig.tight_layout()
	if savepath:
		fig.savefig(savepath, dpi=120, bbox_inches="tight")
	return fig, axes


def plot_feature_importances(importances, feature_names):
  """
  Plots the feature importance with bars. 

  To use with Random Forest Classifiers or Regressors.

  Parameters:
  --------------
  importances: the list of values of feature importances. You can get 
  				it as model.feature_importances (if model is the name of
  				your Random Forest model)
  feature_names: the list of feature names

  Credits to spies006: https://stackoverflow.com/a/44102451/2110769
  """
  indices = np.argsort(importances)
  fig, ax = plt.subplots(figsize=(7, max(4, len(indices) * 0.25)))
  ax.set_title('Feature Importances')
  ax.barh(range(len(indices)), importances[indices], color='b', align='center')
  ax.set_yticks(range(len(indices)))
  ax.set_yticklabels([feature_names[i] for i in indices])
  ax.set_xlabel('Relative Importance')
  fig.tight_layout()
  plt.show()




def silhouette_diagram(X_, cluster_labels, n_clusters, sample_size=None,
	random_state = None,
	title="Silhouette diagram"):
  """
  This code is based on scikit learn documentation:
  https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html#sphx-glr-auto-examples-cluster-plot-kmeans-silhouette-analysis-py


  Returns 
  ----------------
  sample_silhouette_values: the silhouette value of each sample
  """

  # The silhouette_score gives the average value for all the samples.
  # This gives a perspective into the density and separation of the formed
  # clusters
  silhouette_avg = silhouette_score(X_, cluster_labels)

  fig, ax1 = plt.subplots()
  fig.set_size_inches(9, 7)
  ax1.set_xlim([-1, 1])

  # The (n_clusters+1)*10 is for inserting blank space between silhouette
  # plots of individual clusters, to demarcate them clearly.
  ax1.set_ylim([0, len(X_) + (n_clusters + 1) * 10])

  # Compute the silhouette scores for each sample
  sample_silhouette_values = silhouette_samples(X_, cluster_labels)
  y_lower = 10
  for i in range(n_clusters):
  	# Aggregate the silhouette scores for samples belonging to
  	# cluster i, and sort them
  	ith_cluster_silhouette_values = \
  		sample_silhouette_values[cluster_labels == i]
  	ith_cluster_silhouette_values.sort()

  	size_cluster_i = ith_cluster_silhouette_values.shape[0]
  	y_upper = y_lower + size_cluster_i

  	color = cm.nipy_spectral(float(i) / n_clusters)
  	ax1.fill_betweenx(np.arange(y_lower, y_upper),
  		0, ith_cluster_silhouette_values,
  		facecolor=color, edgecolor=color, alpha=0.7)

  	# Label the silhouette plots with their cluster numbers at the middle
  	ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

  	# Compute the new y_lower for next plot
  	y_lower = y_upper + 10  # 10 for the 0 samples

  	ax1.set_title(title)
  	ax1.set_xlabel("The silhouette coefficient values")
  	ax1.set_ylabel("Cluster label")

  	# The vertical line for average silhouette score of all the values
  	ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

  return sample_silhouette_values




###########################################
###### ANOMALY DETECTION ##################
###########################################


def false_positive_rate(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fpr = fp/(fp+tn)
    return fpr

def plot_precision_recall_curve(y_train, anomaly_scores, ax, 
                                threshold_selected=None):
  """
  Inspired by 
    http://abhay.harpale.net/blog/machine-learning/threshold-tuning-using-roc/
  """
  
  precision, recall, thresholds = precision_recall_curve(y_train, anomaly_scores)
  pr_auc_score = average_precision_score(y_train,anomaly_scores)

  ax.plot(recall, precision, 
           label='Pr-Re curve (AUC = %0.2f)' % (pr_auc_score))
  ax.set_ylabel('Precision')
  ax.set_xlabel('Recall')

  ax_tw = ax.twinx()  # instantiate a second axes that shares the same x-axis
  ax_tw.plot(recall[:-1], thresholds,linestyle='dashed', color='grey', 
              label='Threshold')
  ax_tw.tick_params(axis='y')
  ax_tw.set_ylabel('Threshold')
  ax.grid(True)
  ax.legend(loc="lower center")
  ax_tw.legend(loc="lower left")
  ax.set_xticks(np.arange(0, 1+0.09, 0.1))
  ax.set_yticks(np.arange(0, 1+0.09, 0.1))

  if threshold_selected != None:
    y_pred = (anomaly_scores >= threshold_selected)
    recall_sc = recall_score(y_train, y_pred)
    print("Precision=", precision_score(y_train, y_pred))
    print("Recall=", recall_sc)
    ax.axvline(recall_sc, 0, 1, color='r', 
               linestyle='-.', label="Threshold selected")
    


def plot_roc_curve(y_train, anomaly_scores, ax, threshold_selected=None):
  """
  Inspired by 
    http://abhay.harpale.net/blog/machine-learning/threshold-tuning-using-roc/
  """
  
  fpr, tpr, thresholds = roc_curve(y_train, anomaly_scores)
  roc_auc_sc = roc_auc_score(y_train, anomaly_scores)
  # The thresholds obtained here are the same as the ones obtained with the
  # precision-recall curve
  
  ax.plot(fpr, tpr, label='ROC (AUC = %0.2f)' % (roc_auc_sc))
  ax.set_ylabel('True Positive Rate (Recall) : TP/(TP+FN) )')
  ax.set_xlabel('False Positive Rate: FP/(FP+TN)')

  # Also plot the thresholds
  ax_tw = ax.twinx()  # instantiate a second axes that shares the same x-axis
  ax_tw.plot(fpr, thresholds,linestyle='dashed', color='grey', 
              label='Threshold')
  ax_tw.tick_params(axis='y')
  ax_tw.set_ylabel('Threshold')
  ax.grid(True)
  ax.legend(loc="center right")
  ax_tw.legend(loc="lower left")
  ax.set_xticks(np.arange(0, 1+0.09, 0.1))
  ax.set_yticks(np.arange(0, 1+0.09, 0.1))


  if threshold_selected != None:
    y_pred = (anomaly_scores >= threshold_selected)
    tpr = recall_score(y_train, y_pred)
    fpr = false_positive_rate(y_train, y_pred)

    print("False Positive Rate = ", fpr)
    print("True Positive Rate = ", tpr)
    ax.axvline(fpr, 0, 1, color='r', 
               linestyle='-.', label="Threshold selected")




def plot_precision_recall_vs_thresholds(y_train, anomaly_scores, ax, 
                                        threshold_selected=None):
  """

    From Fig.3.4 of Geron, Hands-On Machine Learning with Scikit-Learn, 
      Keras, and TensorFlow, O'Reilly 2019
  """
  precision, recall, thresholds = precision_recall_curve(y_train, anomaly_scores)
  ax.plot(thresholds, precision[:-1], 'b--', label='Precision')
  ax.plot(thresholds, recall[:-1], 'g-', label='Recall')
  ax.set_xlabel('Threshold')
  ax.legend(loc='upper center')
  ax.grid(True)
  ax.set_yticks(np.arange(0, 1+0.09, 0.1))
  
  if threshold_selected != None:
    ax.axvline(threshold_selected, 0, 1, color='r', 
      linestyle='-.', label="Threshold selected")




def plot_tpr_fpr_vs_thresholds(y_train, anomaly_scores, ax, 
                               threshold_selected=None):
  """
  
    Inspired by Fig.3.4 of Geron, Hands-On Machine Learning with Scikit-Learn, 
      Keras, and TensorFlow, O'Reilly 2019
  """
  fpr, tpr, thresholds = roc_curve(y_train, anomaly_scores)
  ax.plot(thresholds, fpr, 'b--', label='False Positive rate')
  ax.plot(thresholds, tpr, 'g-', label='True Positive rate')
  ax.set_xlabel('Threshold')
  ax.legend(loc='upper center')
  ax.grid(True)
  #ax.set_xticks(np.arange(min(thresholds),max(thresholds), 
  #                       (max(thresholds) -min(thresholds) )/10  ))
  #ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
  ax.set_yticks(np.arange(0, 1+0.09, 0.1))
  
  if threshold_selected != None:
    ax.axvline(threshold_selected, 0, 1, color='r', 
               linestyle='-.', label="Threshold selected")



def evaluate_anomaly_detector(y_train, anomaly_scores, threshold_selected=None):
  """
    Evaluates the results from anomaly detection
  """
  fig, axs = plt.subplots(2,2, figsize=(16, 16))
  fig.tight_layout(pad=8)
  plt.rcParams["font.size"] = "12"

  plot_precision_recall_curve(y_train, anomaly_scores, axs[0,0], threshold_selected)
  plot_roc_curve(y_train, anomaly_scores, axs[0,1], threshold_selected)  
  plot_precision_recall_vs_thresholds(y_train, anomaly_scores, axs[1,0], 
                                      threshold_selected)
  plot_tpr_fpr_vs_thresholds(y_train, anomaly_scores, axs[1,1], 
                             threshold_selected)


def plot_expo_on_ax(ax, expo, title="EXPO"):
    """Résumé compact EXPO sur un axe (grilles d'histogrammes)."""
    s = pd.to_numeric(expo, errors="coerce").dropna()
    full = int((s == 1).sum())
    partial = int((s < 1).sum())
    labels = ["Pleine (1)", "Partielle (<1)"]
    counts = [full, partial]
    colors = ["#4c72b0", "#c44e52"]
    bars = ax.barh(labels, counts, color=colors, edgecolor="white", height=0.55)
    ax.set_xlabel("Effectif")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    xmax = max(counts) if counts else 1
    for bar, c in zip(bars, counts):
        pct = 100 * c / len(s) if len(s) else 0
        ax.text(
            bar.get_width() + xmax * 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{c} ({pct:.0f} %)",
            va="center",
            fontsize=9,
        )
    ax.set_xlim(0, xmax * 1.25)


def plot_expo_distribution(df, figsize=(12, 5), savepath=None):
    """
    Visualisation lisible de EXPO (fraction d'année assurée).

    Le boxplot est peu informatif quand ~76 % des valeurs valent 1 :
    deux panneaux (global + tranches des couvertures partielles).
    """
    expo = pd.to_numeric(df["EXPO"], errors="coerce").dropna()
    n = len(expo)
    full = int((expo == 1).sum())
    partial = int((expo < 1).sum())

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    cats = ["Année pleine\n(EXPO = 1)", "Couverture partielle\n(EXPO < 1)"]
    counts = [full, partial]
    colors = ["#4c72b0", "#c44e52"]
    bars = axes[0].bar(cats, counts, color=colors, edgecolor="white", width=0.55)
    axes[0].set_ylabel("Nombre de bâtiments")
    axes[0].set_title("Répartition globale")
    ymax = max(counts) if counts else 1
    for bar, c in zip(bars, counts):
        pct = 100 * c / n if n else 0
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + ymax * 0.02,
            f"{c}\n({pct:.1f} %)",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    partial_expo = expo[expo < 1]
    bin_edges = [0, 0.01, 0.25, 0.5, 0.75, 0.95, 1.0]
    bin_labels = ["~0", "0–0,25", "0,25–0,5", "0,5–0,75", "0,75–0,95", "0,95–1"]
    buckets = pd.cut(
        partial_expo, bins=bin_edges, labels=bin_labels, include_lowest=True
    )
    bucket_counts = buckets.value_counts().sort_index()
    x = np.arange(len(bucket_counts))
    bars2 = axes[1].bar(
        x, bucket_counts.values, color="#55a868", edgecolor="white"
    )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(bucket_counts.index, rotation=30, ha="right")
    axes[1].set_ylabel("Nombre de bâtiments")
    axes[1].set_title(f"Détail des couvertures partielles (n = {partial})")
    for bar, v in zip(bars2, bucket_counts.values):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + partial * 0.01,
            str(int(v)),
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.suptitle(
        "EXPO — fraction de l'année où le bâtiment était assuré",
        fontsize=13,
        y=1.02,
    )
    plt.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=120, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_rf_classification_dashboard(
    y_true,
    y_pred,
    y_proba=None,
    feature_importances=None,
    feature_names=None,
    class_labels=None,
    gini_score=None,
    cv_gini=None,
    top_n_features=15,
    suptitle=None,
    savepath=None,
    figsize=(15, 10),
):
    """
    Tableau de bord lisible pour une forêt aléatoire (ou classifieur binaire).

    Panneaux : matrice de confusion (effectifs + % par ligne), rappel/précision,
    courbe précision-rappel (si ``y_proba``), top importances de variables.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if class_labels is None:
        class_labels = np.array(["Pas de sinistre", "Sinistre"])
    else:
        class_labels = np.asarray(class_labels)

    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    acc = (y_true == y_pred).mean()
    prec_per_class, rec_per_class = [], []
    for i in range(len(class_labels)):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        prec_per_class.append(tp / (tp + fp) if (tp + fp) > 0 else 0.0)
        rec_per_class.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.35, wspace=0.3)

    # --- Matrice de confusion (effectifs + % ligne) ---
    ax_cm = fig.add_subplot(gs[0, 0])
    im = ax_cm.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax_cm.set_xticks(range(len(class_labels)))
    ax_cm.set_yticks(range(len(class_labels)))
    ax_cm.set_xticklabels(class_labels, rotation=25, ha="right")
    ax_cm.set_yticklabels(class_labels)
    ax_cm.set_ylabel("Vraie classe")
    ax_cm.set_xlabel("Classe prédite")
    ax_cm.set_title("Matrice de confusion\n(effectif et % par ligne)")
    fig.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04, label="Part de la ligne")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            pct = 100 * cm_norm[i, j]
            ax_cm.text(
                j,
                i,
                f"{cm[i, j]:d}\n({pct:.0f} %)",
                ha="center",
                va="center",
                color="white" if cm_norm[i, j] > 0.5 else "#222222",
                fontsize=11,
                fontweight="bold",
            )

    # --- Rappel / précision par classe ---
    ax_met = fig.add_subplot(gs[0, 1])
    x = np.arange(len(class_labels))
    width = 0.35
    ax_met.bar(x - width / 2, rec_per_class, width, label="Rappel", color="#4c72b0")
    ax_met.bar(x + width / 2, prec_per_class, width, label="Précision", color="#c44e52")
    ax_met.set_xticks(x)
    ax_met.set_xticklabels(class_labels, rotation=15, ha="right")
    ax_met.set_ylim(0, 1.05)
    ax_met.set_ylabel("Score")
    ax_met.set_title("Rappel et précision par classe")
    ax_met.legend(loc="lower right")
    ax_met.axhline(0.5, color="#888888", linestyle="--", linewidth=0.8, alpha=0.7)
    subtitle_parts = [f"Accuracy globale : {acc:.1%}"]
    if gini_score is not None:
        subtitle_parts.append(f"Gini normalisé (test) : {gini_score:.3f}")
    if cv_gini is not None:
        subtitle_parts.append(f"Meilleur Gini CV : {cv_gini:.3f}")
    ax_met.text(
        0.5,
        -0.22,
        " · ".join(subtitle_parts),
        transform=ax_met.transAxes,
        ha="center",
        fontsize=10,
        color="#333333",
    )

    # --- Courbe précision-rappel ---
    ax_pr = fig.add_subplot(gs[1, 0])
    if y_proba is not None:
        y_proba = np.asarray(y_proba)
        prec_curve, rec_curve, _ = precision_recall_curve(y_true, y_proba)
        ap = average_precision_score(y_true, y_proba)
        rec_at_05 = recall_score(y_true, y_pred, zero_division=0)
        prec_at_05 = precision_score(y_true, y_pred, zero_division=0)
        ax_pr.plot(rec_curve, prec_curve, color="#55a868", linewidth=2, label=f"AP = {ap:.3f}")
        ax_pr.scatter([rec_at_05], [prec_at_05], color="#c44e52", s=80, zorder=5, label="Seuil 0,5")
        ax_pr.set_xlabel("Rappel")
        ax_pr.set_ylabel("Précision")
        ax_pr.set_title("Courbe précision–rappel")
        ax_pr.set_xlim(0, 1.02)
        ax_pr.set_ylim(0, 1.05)
        ax_pr.legend(loc="upper right")
        ax_pr.grid(True, alpha=0.3)
    else:
        ax_pr.text(0.5, 0.5, "Probabilités non fournies", ha="center", va="center")
        ax_pr.set_axis_off()

    # --- Top importances ---
    ax_imp = fig.add_subplot(gs[1, 1])
    if feature_importances is not None and feature_names is not None:
        imp = np.asarray(feature_importances)
        names = list(feature_names)
        order = np.argsort(imp)[-top_n_features:]
        top_imp = imp[order]
        top_names = [names[i] for i in order]
        ax_imp.barh(range(len(order)), top_imp, color="#8172b2")
        ax_imp.set_yticks(range(len(order)))
        ax_imp.set_yticklabels(top_names, fontsize=8)
        ax_imp.set_xlabel("Importance relative")
        ax_imp.set_title(f"Top {len(order)} variables (forêt aléatoire)")
    else:
        ax_imp.text(0.5, 0.5, "Importances non disponibles", ha="center", va="center")
        ax_imp.set_axis_off()

    if suptitle:
        fig.suptitle(suptitle, fontsize=14, y=1.01)
    fig.subplots_adjust(top=0.92, hspace=0.4, wspace=0.3)
    if savepath:
        fig.savefig(savepath, dpi=120, bbox_inches="tight")
    return fig


def _describe_kmeans_cluster(sub_df, numeric_cols, categorical_cols):
    """Résumé textuel d'un cluster pour légendes et présentation."""
    parts = []
    if "superficief" in sub_df.columns:
        parts.append(f"superficie moy. {sub_df['superficief'].mean():.0f} m²")
    if "EXPO" in sub_df.columns:
        expo_m = sub_df["EXPO"].mean()
        if expo_m < 0.6:
            parts.append(f"faible exposition (EXPO ~ {expo_m:.2f})")
        else:
            parts.append(f"exposition pleine (EXPO ~ {expo_m:.2f})")
    for col in categorical_cols or []:
        if col in sub_df.columns:
            mode_val = sub_df[col].mode(dropna=True)
            if not mode_val.empty:
                parts.append(f"{col} dominante = {mode_val.iloc[0]}")
    return " · ".join(parts)


def build_kmeans_cluster_labels(profile_df, clusters, numeric_cols=None, categorical_cols=None):
    """Génère un libellé court par cluster à partir des profils moyens."""
    numeric_cols = numeric_cols or []
    categorical_cols = categorical_cols or []
    n_clusters = int(np.max(clusters)) + 1
    labels = []
    for c in range(n_clusters):
        sub = profile_df.iloc[np.asarray(clusters) == c]
        desc = _describe_kmeans_cluster(sub, numeric_cols, categorical_cols)
        if "EXPO" in sub.columns and sub["EXPO"].mean() < 0.6:
            short = f"Cluster {c} — faible exposition"
        elif categorical_cols and "ft_21_categ" in sub.columns:
            mode_ft21 = sub["ft_21_categ"].mode(dropna=True).iloc[0]
            if mode_ft21 in (3, 4):
                short = f"Cluster {c} — cat. ft_21 élevée (3–4)"
            elif c == 0:
                short = f"Cluster {c} — parc standard (majoritaire)"
            else:
                short = f"Cluster {c} — profil intermédiaire"
        else:
            short = f"Cluster {c}"
        labels.append(f"{short}\n{desc}")
    return labels


def summarize_kmeans_profiles(profile_df, clusters, numeric_cols, categorical_cols=None, y_target=None):
    """Tableau récapitulatif : effectifs, moyennes numériques, modes catégorielles."""
    categorical_cols = categorical_cols or []
    rows = []
    n_clusters = int(np.max(clusters)) + 1
    for c in range(n_clusters):
        mask = np.asarray(clusters) == c
        sub = profile_df.iloc[mask]
        row = {"cluster": c, "effectif": int(mask.sum()), "part_%": round(100 * mask.mean(), 1)}
        for col in numeric_cols:
            if col in sub.columns:
                row[f"{col}_moy"] = round(float(sub[col].mean()), 2)
        for col in categorical_cols:
            if col in sub.columns and not sub[col].mode(dropna=True).empty:
                row[f"{col}_mode"] = sub[col].mode(dropna=True).iloc[0]
        if y_target is not None:
            yt = np.asarray(y_target)[mask]
            row["taux_sinistre_%"] = round(100 * yt.mean(), 1)
        rows.append(row)
    return pd.DataFrame(rows)


def plot_kmeans_cluster_dashboard(
    profile_df,
    clusters,
    numeric_cols,
    categorical_cols=None,
    y_target=None,
    cluster_labels=None,
    suptitle="K-means — profils de clusters",
    savepath=None,
    figsize=(14, 10),
):
    """
    Tableau de bord K-means : effectifs, variables numériques, modes catégorielles,
    résumé textuel par cluster.
    """
    categorical_cols = categorical_cols or []
    clusters = np.asarray(clusters)
    n_clusters = int(clusters.max()) + 1
    colors = ["#4c72b0", "#55a868", "#c44e52", "#8172b2", "#ccb974"]

    if cluster_labels is None:
        cluster_labels = build_kmeans_cluster_labels(
            profile_df, clusters, numeric_cols, categorical_cols
        )

    counts = np.bincount(clusters, minlength=n_clusters)
    total = counts.sum()

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.1], hspace=0.38, wspace=0.32)

    # --- Effectifs par cluster ---
    ax_sz = fig.add_subplot(gs[0, 0])
    x = np.arange(n_clusters)
    bars = ax_sz.bar(
        x, counts, color=[colors[i % len(colors)] for i in range(n_clusters)], edgecolor="white"
    )
    ax_sz.set_xticks(x)
    short_xtick = [f"Cluster {i}" for i in range(n_clusters)]
    ax_sz.set_xticklabels(short_xtick)
    ax_sz.set_ylabel("Nombre de bâtiments")
    ax_sz.set_title("Répartition des clusters\n(effectif et % du parc)")
    ax_sz.grid(axis="y", alpha=0.3)
    for i, (bar, n) in enumerate(zip(bars, counts)):
        pct = 100 * n / total
        ax_sz.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + total * 0.01,
            f"{n}\n({pct:.0f} %)",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
        )

    # --- Moyennes numériques ---
    ax_num = fig.add_subplot(gs[0, 1])
    num_present = [c for c in numeric_cols if c in profile_df.columns]
    if num_present:
        width = 0.35
        x = np.arange(n_clusters)
        for j, col in enumerate(num_present[:2]):
            means = [
                profile_df.iloc[clusters == i][col].mean() for i in range(n_clusters)
            ]
            offset = (j - 0.5) * width
            ax_num.bar(x + offset, means, width, label=col, color=colors[j], alpha=0.85)
        ax_num.set_xticks(x)
        ax_num.set_xticklabels([f"Cluster {i}" for i in range(n_clusters)])
        ax_num.set_title("Variables numériques moyennes\npar cluster (échelle originale)")
        ax_num.legend(loc="upper right")
        ax_num.grid(axis="y", alpha=0.3)
        if len(num_present) > 2:
            ax_num.text(
                0.02, 0.02,
                "Autres vars : " + ", ".join(num_present[2:]),
                transform=ax_num.transAxes, fontsize=8, color="#555",
            )
    else:
        ax_num.set_axis_off()

    # --- Modes catégorielles (heatmap simplifiée) ---
    ax_cat = fig.add_subplot(gs[1, 0])
    if categorical_cols:
        cat_data = []
        cat_labels = []
        for col in categorical_cols:
            if col not in profile_df.columns:
                continue
            modes = []
            for i in range(n_clusters):
                sub = profile_df.iloc[clusters == i]
                m = sub[col].mode(dropna=True)
                modes.append(float(m.iloc[0]) if not m.empty else np.nan)
            cat_data.append(modes)
            cat_labels.append(col.replace("_categ", ""))
        if cat_data:
            mat = np.array(cat_data)
            im = ax_cat.imshow(mat, aspect="auto", cmap="YlOrRd")
            ax_cat.set_xticks(range(n_clusters))
            ax_cat.set_xticklabels([f"C{i}" for i in range(n_clusters)])
            ax_cat.set_yticks(range(len(cat_labels)))
            ax_cat.set_yticklabels(cat_labels, fontsize=9)
            ax_cat.set_title("Valeur catégorielle dominante\n(mode par cluster)")
            for r in range(mat.shape[0]):
                for c in range(mat.shape[1]):
                    val = mat[r, c]
                    txt = "—" if np.isnan(val) else f"{val:.0f}"
                    ax_cat.text(c, r, txt, ha="center", va="center", fontsize=9, color="#222")
            fig.colorbar(im, ax=ax_cat, fraction=0.046, pad=0.04)
        else:
            ax_cat.set_axis_off()
    else:
        ax_cat.set_axis_off()

    # --- Résumé textuel par cluster ---
    ax_txt = fig.add_subplot(gs[1, 1])
    ax_txt.axis("off")
    ax_txt.set_title("À quoi correspond chaque cluster ?", loc="left", fontsize=11, pad=12)
    lines = []
    for i in range(n_clusters):
        block = cluster_labels[i] if i < len(cluster_labels) else f"Cluster {i}"
        if y_target is not None:
            mask = clusters == i
            rate = 100 * np.asarray(y_target)[mask].mean()
            block += f"\n  → taux sinistre observé : {rate:.1f} % (exploration, non supervisé)"
        lines.append(block)
    ax_txt.text(
        0.02, 0.98, "\n\n".join(lines),
        transform=ax_txt.transAxes, va="top", ha="left", fontsize=9.5,
        family="sans-serif", linespacing=1.45,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f7f7f7", edgecolor="#cccccc"),
    )

    if suptitle:
        fig.suptitle(suptitle, fontsize=14, y=1.01)
    fig.subplots_adjust(top=0.92)
    if savepath:
        fig.savefig(savepath, dpi=120, bbox_inches="tight")
    return fig


def plot_kmeans_inertia_ninit(repeats, inertia_values, chosen_n_init=None, savepath=None, figsize=(9, 5)):
    """Courbe d'inertie vs nombre d'initialisations K-means (choix de n_init)."""
    repeats = list(repeats)
    inertia_values = list(inertia_values)
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(repeats, inertia_values, color="#4c72b0", linewidth=2, marker="o", markersize=4)
    if chosen_n_init is not None:
        ax.axvline(chosen_n_init, color="#c44e52", linestyle="--", linewidth=1.5,
                   label=f"n_init retenu = {chosen_n_init}")
        ax.legend(loc="upper right")
    ax.set_xlabel("Nombre d'initialisations aléatoires (n_init)")
    ax.set_ylabel("Inertie (somme des distances² au centroïde)")
    ax.set_title(
        "Stabilité de K-means\n"
        "L'inertie se stabilise quand n_init est suffisant (convergence)"
    )
    ax.grid(True, alpha=0.3)
    ymin, ymax = min(inertia_values), max(inertia_values)
    margin = max((ymax - ymin) * 0.15, 1e-6)
    ax.set_ylim(ymin - margin, ymax + margin)
    note = (
        "Plus n_init est élevé, plus K-means relance l'algorithme\n"
        "pour éviter un minimum local. Plateau = réglage fiable."
    )
    ax.text(0.5, -0.22, note, transform=ax.transAxes, ha="center", fontsize=9, color="#444")
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=120, bbox_inches="tight")
    return fig


def plot_kmeans_anomaly_histogram(anomaly_score, savepath=None, figsize=(8, 4.5)):
    """Histogramme du score d'anomalie (distance normalisée au centroïde)."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(anomaly_score, bins=30, color="#c44e52", edgecolor="white", alpha=0.85)
    ax.set_xlabel("Score d'anomalie (0 = proche du centroïde, 1 = très éloigné)")
    ax.set_ylabel("Nombre de bâtiments")
    ax.set_title("Profils atypiques — distance au centroïde du cluster")
    med = float(np.median(anomaly_score))
    ax.axvline(med, color="#4c72b0", linestyle="--", linewidth=1.5, label=f"médiane = {med:.2f}")
    ax.legend(loc="upper right")
    ax.text(
        0.98, 0.75,
        "Les bâtiments à droite sont plus atypiques\n"
        "par rapport au centre de leur cluster.",
        transform=ax.transAxes, ha="right", fontsize=9, color="#444",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#f7f7f7", edgecolor="#ddd"),
    )
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=120, bbox_inches="tight")
    return fig


def show_figure(fig):
    """Affiche une figure matplotlib une seule fois (Jupyter / VS Code / script)."""
    try:
        from IPython import get_ipython
        if get_ipython() is not None:
            from IPython.display import display
            display(fig)
            return fig
    except ImportError:
        pass
    plt.show()
    return fig


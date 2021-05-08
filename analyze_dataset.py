from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
import seaborn as sns

def read_file(input_file, labels):
	"""Reads a tab separated value file."""
	df = pd.read_csv(input_file, 
						encoding="utf-8", 
						sep='\t',
						names=["text", "label", "id"])
	# print(df)
	label_dict = {str(i): v for i, v in enumerate(labels)}
	df["label"] = df["label"].apply(lambda x: ",".join(label_dict[l] for l in x.split(",")))
	df = pd.concat([df, df['label'].str.get_dummies(sep=',')], axis=1)
	df.drop(["label"], axis=1, inplace=True)
	return df


def CheckAgreement(ex, min_agreement, all_emotions, max_agreement=100):
	"""Return the labels that at least min_agreement raters agree on."""
	sum_ratings = ex[all_emotions].sum(axis=0)
	agreement = ((sum_ratings >= min_agreement) & (sum_ratings <= max_agreement))
	return ",".join(sum_ratings.index[agreement].tolist())


def CountLabels(labels):
	if (not isinstance(labels, float)) and labels:
		return len(labels.split(","))
	return 0


def main():
	parser = argparse.ArgumentParser()

	## Required parameters
	parser.add_argument("--data_dir", default="data/original", type=str,
											help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
	parser.add_argument("--label_file", default="labels.txt", type=str)
	parser.add_argument("--train_file", default="train.tsv", type=str)
	parser.add_argument("--sentiment_dict", default="sentiment_dict.json", type=str)
	parser.add_argument("--output_dir", default="dataset_analysis", type=str,
											help="Directory for saving plots and analyses.")

	args = parser.parse_args()
	print(args)

	if not os.path.exists(args.output_dir):
		os.makedirs(args.output_dir)

	print("++++++++++++++++++Loading data++++++++++++++++++++++")
	with open(os.path.join(args.data_dir, args.label_file), "r") as f:
		all_emotions = f.read().splitlines()
	print("%d emotion categories" % len(all_emotions))

	data = read_file(os.path.join(args.data_dir, args.train_file), all_emotions)
	print(data.head(8))
	print("++++++++++++++++++Data loaded++++++++++++++++++++++")
	print("{} training examples".format(len(data)))

	print("++++++++++++Distribution of number of labels per example++++++++++++")
	print(data[all_emotions].sum(axis=1).value_counts() / len(data))
	print("%.5f with more than 3 labels" %
				((data[all_emotions].sum(axis=1) > 3).sum() /
				 len(data)))  # more than 3 labels

	print("Label distributions:")
	print((data[all_emotions].sum(axis=0).sort_values(ascending=False) /
				 len(data) * 100).round(2))

	print("Plotting label correlations...")
	ratings = data.groupby("id")[all_emotions].mean()

	# Compute the correlation matrix
	corr = ratings.corr()

	# Generate a mask for the upper triangle
	mask = np.zeros_like(corr, dtype=np.bool)
	mask[np.triu_indices_from(mask)] = True

	# Set up the matplotlib figure
	fig, _ = plt.subplots(figsize=(11, 9))

	# Generate a custom diverging colormap
	cmap = sns.diverging_palette(220, 10, as_cmap=True)

	# Draw the heatmap with the mask and correct aspect ratio
	sns.heatmap(
			corr,
			mask=mask,
			cmap=cmap,
			vmax=.3,
			center=0,
			square=True,
			linewidths=.5,
			cbar_kws={"shrink": .5})
	fig.savefig(
			args.output_dir + "/correlations.pdf",
			dpi=500,
			format="pdf",
			bbox_inches="tight")

	print("Plotting hierarchical relations...")
	z = linkage(
			pdist(ratings.T, metric="correlation"),
			method="ward",
			optimal_ordering=True)
	fig = plt.figure(figsize=(11, 4), dpi=400)
	plt.xlabel("")
	plt.ylabel("")
	dendrogram(
			z,
			labels=ratings.columns,
			leaf_rotation=90.,  # rotates the x axis labels
			leaf_font_size=12,  # font size for the x axis labels
			color_threshold=1.05,
	)
	fig.savefig(
			args.output_dir + "/hierarchical_clustering.pdf",
			dpi=600,
			format="pdf",
			bbox_inches="tight")

	sent_color_map = {
			"positive": "#BEECAF",
			"negative": "#94bff5",
			"ambiguous": "#FFFC9E"
	}
	with open(os.path.join(args.data_dir, args.sentiment_dict)) as f:
		sent_dict = json.loads(f.read())
	sent_colors = {}
	for e in all_emotions:
		if e in sent_dict["positive"]:
			sent_colors[e] = sent_color_map["positive"]
		elif e in sent_dict["negative"]:
			sent_colors[e] = sent_color_map["negative"]
		else:
			sent_colors[e] = sent_color_map["ambiguous"]

	# Generate a mask for the upper triangle
	mask = np.zeros_like(corr, dtype=np.bool)
	mask[np.diag_indices(mask.shape[0])] = True

	# Generate a custom diverging colormap
	cmap = sns.diverging_palette(220, 10, as_cmap=True)

	row_colors = pd.Series(
			corr.columns, index=corr.columns, name="sentiment").map(sent_colors)

	# Draw the heatmap with the mask and correct aspect ratio
	g = sns.clustermap(
			corr,
			mask=mask,
			cmap=cmap,
			vmax=.3,
			vmin=-0.3,
			center=0,
			row_linkage=z,
			col_linkage=z,
			col_colors=row_colors,
			linewidths=.1,
			cbar_kws={
					"ticks": [-.3, -.15, 0, .15, .3],
					"use_gridspec": False,
					"orientation": "horizontal",
			},
			figsize=(10, 10))

	g.ax_row_dendrogram.set_visible(False)
	g.cax.set_position([.34, -0.05, .5, .03])

	for label in sent_color_map:
		g.ax_col_dendrogram.bar(
				0, 0, color=sent_color_map[label], label=label, linewidth=0)

	g.ax_col_dendrogram.legend(
			title="Sentiment", loc="center", bbox_to_anchor=(1.1, .5))

	g.savefig(args.output_dir + "/hierarchical_corr.pdf", dpi=600, format="pdf")

if __name__ == "__main__":
	main()






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

def read_file(input_file, labels, label_dict, with_id=True):
	"""Reads a tab separated value file."""
	names=["text", "label"]
	if with_id: names.append("id")

	df = pd.read_csv(input_file, 
						encoding="utf-8", 
						sep='\t',
						names=names)
	# print(df)
	if with_id:
		df["label"] = df["label"].apply(lambda x: ",".join(label_dict[l] for l in x.split(",")))
	else:
		df["label"] = df["label"].map(label_dict)
	df = pd.concat([df, df['label'].str.get_dummies(sep=',')], axis=1)

	if with_id:
		# do not drop the label column for analysis on aug-ed dataset
		df.drop(["label"], axis=1, inplace=True)
	return df


def write_label_distributions(output_file, data, labels):
	df = data[labels].sum(axis=0).sort_values(ascending=False)/len(data) * 100
	df.to_json(output_file)


def analyze_aug(args, labels):
	label_dict = {i: v for i, v in enumerate(labels)}
	df = read_file(os.path.join(args.data_dir, args.aug_file), 
					labels,
					label_dict,
					with_id=False)
	print(df.head(8))

	aug_labels = df.label.unique()

	print("++++++++++++++++++Data loaded++++++++++++++++++++++")
	print("{} augmented training examples".format(len(df)))
	print("Labels: {}".format(aug_labels))

	print("Label distributions:")
	print((df[aug_labels].sum(axis=0).sort_values(ascending=False) /
				 len(df) * 100).round(2))

def main():
	parser = argparse.ArgumentParser()

	## Required parameters
	parser.add_argument("--data_dir", default="data/original", type=str,
											help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
	parser.add_argument("--label_file", default="labels.txt", type=str)
	parser.add_argument("--aug", action='store_true',
						help="Whether or not to run analysis on the augmented part of the dataset") # default to False
	parser.add_argument("--aug_file", default="train_aug_9_sanitized.tsv", type=str,
						help="Only used when `aug` flag is set to True")
	parser.add_argument("--train_file", default="train.tsv", type=str)
	parser.add_argument("--label_distributions", default="label_distributions.json", type=str)
	parser.add_argument("--sentiment_dict", default="sentiment_dict.json", type=str)
	parser.add_argument("--output_dir", default="dataset_analysis", type=str,
											help="Directory for saving plots and analyses.")

	args = parser.parse_args()
	print(args)

	if not os.path.exists(args.output_dir):
		os.makedirs(args.output_dir)

	print("++++++++++++++++++Loading Labels++++++++++++++++++++++")
	with open(os.path.join(args.data_dir, args.label_file), "r") as f:
		all_emotions = f.read().splitlines()
	print("%d emotion categories" % len(all_emotions))

	if args.aug:
		analyze_aug(args, all_emotions)
		return

	label_dict = {str(i): v for i, v in enumerate(all_emotions)}
	data = read_file(os.path.join(args.data_dir, args.train_file), 
					all_emotions,
					label_dict)
	print(data.head(8))
	print("++++++++++++++++++Data loaded++++++++++++++++++++++")
	print("{} training examples".format(len(data)))

	print("++++++++++++Distribution of number of labels per example++++++++++++")
	# pd.set_option('display.max_rows', 500)
	# pd.set_option('display.max_columns', 500)
	# pd.set_option('display.width', 1000)

	print(data[all_emotions].sum(axis=1).value_counts() / len(data))
	# m_desire = (data[all_emotions].sum(axis=1) > 1) & (data['desire'] == 1)
	# m_remorse = (data[all_emotions].sum(axis=1) > 1) & (data['remorse'] == 1)
	# print(data[m_desire].shape)
	# print(data[m_remorse].shape)
	# print((data[all_emotions].sum(axis=1) >= 2).sum())

	print("%.5f with more than 3 labels" %
				((data[all_emotions].sum(axis=1) > 3).sum() /
				 len(data)))  # more than 3 labels

	print("Label distributions:")
	print((data[all_emotions].sum(axis=0).sort_values(ascending=False) /
				 len(data) * 100).round(2))
	write_label_distributions(os.path.join(args.data_dir, args.label_distributions), 
								data,
								all_emotions)

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






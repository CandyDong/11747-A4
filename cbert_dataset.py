'''
	form dataset for cbert
	only consider underrepresented data (< 3%)
	label distributions:
		neutral           32.76
		admiration         9.51
		approval           6.77
		gratitude          6.13
		annoyance          5.69
		amusement          5.36
		curiosity          5.05
		love               4.81
		disapproval        4.66
		optimism           3.64
		anger              3.61
		joy                3.34
		confusion          3.15
		sadness            3.05
	*	disappointment     2.92
	*	realization        2.56
	*	caring             2.50
	*	surprise           2.44
	*	excitement         1.96
	*	disgust            1.83
	*	desire             1.48
	*	fear               1.37
	*	remorse            1.26
	*	embarrassment      0.70
	*	nervousness        0.38
	*	relief             0.35
	*	pride              0.26
	*	grief              0.18
'''
import json
import os
import argparse
import pandas as pd

def read_file(input_file, labels):
	"""Reads a tab separated value file."""
	df = pd.read_csv(input_file, 
						encoding="utf-8", 
						sep='\t', 
						names=["text", "label", "id"])
	m = df["label"].isin(labels)
	df = df[m]
	return df

def save_file(output_file, df):
	df.to_csv(output_file, sep = '\t', index=False, header=False)


def main():
	parser = argparse.ArgumentParser()

	## Required parameters
	parser.add_argument("--data_dir", default="data/original", type=str,
						help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
	parser.add_argument("--label_file", default="labels.txt", type=str)
	parser.add_argument("--train_file", default="train.tsv", type=str)
	parser.add_argument("--label_distributions", default="label_distributions.json", type=str)
	parser.add_argument("--threshold", default=3, type=float,
						help="Only collect data with label distribution smaller than the threshold")
	parser.add_argument("--suffix", default="cbert", type=str)
	parser.add_argument("--output_dir", default="data/original", type=str,
						help="The output dir for sub-dataset.")
	args = parser.parse_args()
	print(args)

	with open(os.path.join(args.data_dir, args.label_distributions)) as json_file:
		distributions = json.load(json_file)

	with open(os.path.join(args.data_dir, args.label_file), "r") as f:
		all_emotions = f.read().splitlines()

	labels = []
	for i, e in enumerate(all_emotions):
		if distributions[e] < args.threshold: labels.append(str(i))

	print("++++++++++++++++++++++++++++labels++++++++++++++++++++++++++++")
	for label in labels:
		print(all_emotions[int(label)])

	df = read_file(os.path.join(args.data_dir, args.train_file), labels)
	print("{} training examples".format(len(df)))
	save_file(os.path.join(args.data_dir, "train_{}.tsv".format(args.suffix)), df)

	df = pd.concat([df,  
					df["label"].apply(lambda x: ",".join(all_emotions[int(i)] for i in x.split(","))).reset_index(name="label_name")], 
					axis=1)
	print(df.groupby(['label_name']).size().sort_values(ascending=False))


if __name__ == "__main__":
	main()



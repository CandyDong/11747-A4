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
def read_file(input_file, labels):
	"""Reads a tab separated value file."""
	df = pd.read_csv(input_file, 
						encoding="utf-8", 
						sep='\t',
						names=["text", "label", "id"])
	label_dict = {str(i): v for i, v in enumerate(labels)}
	df["label"] = df["label"].apply(lambda x: ",".join(label_dict[l] for l in x.split(",")))
	df = pd.concat([df, df['label'].str.get_dummies(sep=',')], axis=1)
	df.drop(["label"], axis=1, inplace=True)
	return df


def main():
	parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default="data/original", type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--label_file", default="labels.txt", type=str)
    parser.add_argument("--threshold", default=0.03, type=float,
    					help="Only collect data with label distribution smaller than the threshold")
    parser.add_argument("--suffix", default="_cbert", type=str)
    parser.add_argument("--output_dir", default="data/original", type=str,
                        help="The output dir for sub-dataset.")
    args = parser.parse_args()
    print(args)




if __name__ == "__main__":
    main()



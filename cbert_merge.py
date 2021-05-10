# merge the augmented data and the original training data
# remove sentences if resolves to the same token sequence

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import numpy as np
import pandas as pd
import csv
import shutil

from transformers import BertTokenizer

def rev_wordpiece(str):
    """wordpiece function used in cbert"""
    
    #print(str)
    if len(str) > 1:
        for i in range(len(str)-1, 0, -1):
            if str[i] == '[PAD]':
                str.remove(str[i])
            elif len(str[i]) > 1 and str[i][0]=='#' and str[i][1]=='#':
                str[i-1] += str[i][2:]
                str.remove(str[i])
    return " ".join(str)

def read_file(input_file, with_id=False):
	"""Reads a tab separated value file."""
	names = ["text", "label"]
	if with_id: names.append("id")

	df = pd.read_csv(input_file, 
						encoding="utf-8", 
						sep='\t', 
						names=names)
	return df

def main():
	parser = argparse.ArgumentParser()

	## Required parameters
	parser.add_argument("--data_dir", default="data/original", type=str,
						help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
	parser.add_argument("--bert_model", default="bert-base-cased", type=str)
	parser.add_argument("--aug_data_file", default="train_aug_9.tsv", type=str)
	parser.add_argument("--aug_sanitized_file", default="train_aug_9_sanitized.tsv", type=str,
						help="Where to save the sanitized aug file (corrected tokens + duplicate removed)")
	parser.add_argument("--cbert_data_file", default="train_original_9.tsv", type=str)
	parser.add_argument("--train_data_file", default="train.tsv", type=str)
	parser.add_argument("--merged_data_file", default="train_merge.tsv", type=str)
	args = parser.parse_args()
	print(args)

	# tokenizer = BertTokenizer.from_pretrained(args.bert_model)
	# sent = tokenizer._tokenize("’ s a shame [ NAME ] doesn ’ t get more duis with the amount he needed to")

	# for token, length, real_token in zip(["[N##AM##E]", "[R##EL##IG##ION]"], 
	# 											[5, 6],
	# 											["[NAME]", "[RELIGION]"]):
	# 	print("sent: {}".format(sent))
	# 	indices = []
	# 	for i in range(len(sent)):
	# 		if "".join(sent[i:i+length]) == token: indices.append(i)

	# 	for j in indices:
	# 		print("before: {}".format(sent))
	# 		del sent[j:j+length]
	# 		sent.insert(j, real_token)
	# 		print("after: {}".format(sent))
	# return

	if os.path.exists(os.path.join(args.data_dir, args.merged_data_file)):
		os.remove(os.path.join(args.data_dir, args.merged_data_file))
	# copy the original training data over
	shutil.copy(os.path.join(args.data_dir, args.train_data_file),
				os.path.join(args.data_dir, args.merged_data_file))
	merged_data_file = open(os.path.join(args.data_dir, args.merged_data_file), 'a')
	merged_tsv_writer = csv.writer(merged_data_file, delimiter='\t')

	if os.path.exists(os.path.join(args.data_dir, args.aug_sanitized_file)):
		os.remove(os.path.join(args.data_dir, args.aug_sanitized_file))
	sanitized_data_file = open(os.path.join(args.data_dir, args.aug_sanitized_file), 'w')
	sanitized_tsv_writer = csv.writer(sanitized_data_file, delimiter='\t')

	tokenizer = BertTokenizer.from_pretrained(args.bert_model)

	cbert_df = read_file(os.path.join(args.data_dir, args.cbert_data_file), with_id=True)
	aug_df = read_file(os.path.join(args.data_dir, args.aug_data_file))

	print("Number of original cbert traing examples: {}".format(len(cbert_df)))
	print("Number of augmented cbert traing examples: {}".format(len(aug_df)))

	count = 0
	for index, row in aug_df.iterrows():
		# this is a placeholder for the original comment id so that 
		# the data loader in run_goemotions.py can work properly
		sent_id = cbert_df.at[index, "id"]
		label = cbert_df.at[index, "label"]

		cbert_sent = cbert_df.at[index, "text"]
		aug_sent = row["text"]
		# print("cbert_sent: \n{}".format(cbert_sent))
		# print("aug_sent: \n{}".format(aug_sent))

		cbert_sent_tokenized = tokenizer._tokenize(cbert_sent)
		# print("cbert_sent_tokenized: {}".format(cbert_sent_tokenized))

		aug_sent_tokenized = tokenizer._tokenize(aug_sent)
		# print("aug_sent_tokenized: {}".format(aug_sent_tokenized))

		# solve the problem of [NAME], [RELIGION] being tokenized to seperate tokens
		for token, length, real_token in zip(["[N##AM##E]", "[R##EL##IG##ION]"], 
												[5, 6],
												["[NAME]", "[RELIGION]"]):
			indices = [i for i in range(len(aug_sent_tokenized)) 
						if "".join(aug_sent_tokenized[i:i+length]) == token]
			for j in indices:
				# print("before: {}".format(aug_sent_tokenized))
				del aug_sent_tokenized[j:j+length]
				aug_sent_tokenized.insert(j, real_token)
				# print("after: {}".format(aug_sent_tokenized))

		if cbert_sent_tokenized == aug_sent_tokenized:
			continue

		aug_sent = rev_wordpiece(aug_sent_tokenized)
		# print("aug_sent: \n{}".format(aug_sent))
		
		merged_tsv_writer.writerow([aug_sent, label, sent_id])
		sanitized_tsv_writer.writerow([aug_sent, label])
		count += 1

	print("Augmented training dataset by {} training examples".format(count))
	
if __name__ == "__main__":
	main()
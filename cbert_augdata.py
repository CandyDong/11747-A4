from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import os
import shutil
import logging
import argparse
import random
from tqdm import tqdm, trange
import json

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler

from transformers import BertTokenizer, BertModel, BertForMaskedLM, AdamW

from cbert_utils import *
# import train_text_classifier

#PYTORCH_PRETRAINED_BERT_CACHE = ".pytorch_pretrained_bert"

"""initialize logger"""
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

"""cuda or cpu"""
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def convert_ids_to_str(ids, tokenizer):
    """converts token_ids into str."""
    tokens = []
    for token_id in ids:
        token = tokenizer._convert_id_to_token(token_id)
        tokens.append(token)
    outputs = rev_wordpiece(tokens)
    return outputs

def main():
    parser = argparse.ArgumentParser()

    ## required parameters
    parser.add_argument("--data_dir", default="data/original", type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--output_dir", default="data/original", type=str,
                        help="The output dir for augmented dataset")
    parser.add_argument("--label_file", default="labels.txt", type=str)
    parser.add_argument("--save_model_dir", default="cbert_model", type=str,
                        help="The cache dir for saved model.")
    parser.add_argument("--bert_model", default="bert-base-cased", type=str,
                        help="The path of pretrained bert model.")
    # parser.add_argument("--max_seq_length", default=64, type=int,
    #                     help="The maximum total input sequence length after WordPiece tokenization. \n"
    #                          "Sequences longer than this will be truncated, and sequences shorter \n"
    #                          "than this will be padded.")
    parser.add_argument("--max_seq_length", default=50, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequence longer than this will be truncated, and sequences shorter \n"
                             "than this wille be padded.")
    parser.add_argument("--do_lower_case", default=False, action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size", default=32, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=10.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument('--seed', default=42, type=int, 
                        help="random seed for initialization")
    parser.add_argument('--sample_num', default=1, type=int,
                        help="sample number")
    parser.add_argument('--sample_ratio', default=7, type=int,
                        help="sample ratio")
    parser.add_argument('--gpu', default=0, type=int,
                        help="gpu id")
    parser.add_argument('--temp', default=1.0, type=float,
                        help="temperature")
    

    args = parser.parse_args()
    print(args)
    
    """prepare processors"""
    processor = AugProcessor(args)
    label_list = processor.get_labels()
    num_labels = len(label_list)

    label_texts = processor.get_labels_text()

    ## prepare for model
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    def load_model(model_name):
        weights_path = os.path.join(args.save_model_dir, model_name)
        model = torch.load(weights_path)
        return model

    ## prepare for training
    train_examples = processor.get_train_examples()
    train_features, num_train_steps, train_dataloader = \
        construct_train_dataloader(train_examples, label_list, args.max_seq_length, 
        args.train_batch_size, args.num_train_epochs, tokenizer, device)

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_examples))
    logger.info("  Batch size = %d", args.train_batch_size)
    logger.info("  Num steps = %d", num_train_steps)

    MASK_id = convert_tokens_to_ids(['[MASK]'], tokenizer)[0]
    # best_test_acc = train_text_classifier.train("aug_data_{}_{}_{}_{}".format(args.sample_num, args.sample_ratio, args.gpu, args.temp))
    # print("before augment best acc:{}".format(best_test_acc))

    for e in trange(int(args.num_train_epochs), desc="Epoch"):
        save_train_path = os.path.join(args.output_dir, "train_aug_{}.tsv".format(e))
        save_mask_path = os.path.join(args.output_dir, "train_mask_{}.tsv".format(e))
        save_original_path = os.path.join(args.output_dir, "train_original_{}.tsv".format(e))

        if os.path.exists(save_train_path): os.remove(save_train_path)
        if os.path.exists(save_mask_path): os.remove(save_mask_path)
        if os.path.exists(save_original_path): os.remove(save_original_path)

        torch.cuda.empty_cache()
        cbert_name = "{}_{}".format(args.bert_model, e)
        model = load_model(cbert_name)
        model.cuda()

        save_train_file = open(save_train_path, 'a')
        save_mask_file = open(save_mask_path, 'a')
        save_original_file = open(save_original_path, 'a')

        train_tsv_writer = csv.writer(save_train_file, delimiter='\t')
        mask_tsv_writer = csv.writer(save_mask_file, delimiter='\t')
        original_tsv_writer = csv.writer(save_original_file, delimiter='\t')

        for _, batch in enumerate(train_dataloader):
            model.eval()
            batch = tuple(t.cuda() for t in batch)
            init_ids, _, input_mask, segment_ids, _ = batch
            input_lens = [sum(mask).item() for mask in input_mask]

            for init_id, mask, seg_id in zip(init_ids, input_mask, segment_ids):
                original_str = convert_ids_to_str(init_id[mask==1].cpu().numpy(), tokenizer)
                original_label = seg_id[0].item()
                original_tsv_writer.writerow([original_str, original_label, label_texts[original_label]])

            masked_idx = np.squeeze([np.random.randint(0, l, max(l//args.sample_ratio, 1)) for l in input_lens])
            for ids, idx in zip(init_ids, masked_idx):
                ids[idx] = MASK_id
            predictions = model(init_ids, input_mask, segment_ids)
            predictions = torch.nn.functional.softmax(predictions[0]/args.temp, dim=2)
            for ids, idx, preds, seg in zip(init_ids, masked_idx, predictions, segment_ids):
                preds = torch.multinomial(preds, args.sample_num, replacement=True)[idx]
                if len(preds.size()) == 2:
                    preds = torch.transpose(preds, 0, 1)
                for pred in preds:
                    # logger.info("  input sentence = %s", convert_ids_to_str(ids.cpu().numpy(), tokenizer))
                    mask_str = convert_ids_to_str(ids.cpu().numpy(), tokenizer)
                    mask_tsv_writer.writerow([mask_str, seg[0].item(), label_texts[seg[0].item()]])

                    ids[idx] = pred
                    new_str = convert_ids_to_str(ids.cpu().numpy(), tokenizer)
                    train_tsv_writer.writerow([new_str, seg[0].item()])

                    # logger.info("  new sentence = %s", new_str)
            torch.cuda.empty_cache()
        predictions = predictions.detach().cpu()
        model.cpu()
        torch.cuda.empty_cache()

        # best_test_acc = train_text_classifier.train("aug_data_{}_{}_{}_{}".format(args.sample_num, args.sample_ratio, args.gpu, args.temp))
        # print("epoch {} augment best acc:{}".format(e, best_test_acc))

if __name__ == "__main__":
    main()
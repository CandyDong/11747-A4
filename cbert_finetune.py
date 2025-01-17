from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
import logging
import argparse
import random
import json
from tqdm import tqdm, trange

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from transformers import BertTokenizer, BertModel, BertForMaskedLM, AdamW

from cbert_utils import *

"""initialize logger"""
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

"""cuda or cpu"""
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default="data/original", type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--label_file", default="labels.txt", type=str)
    parser.add_argument("--save_model_dir", default="cbert_model", type=str,
                        help="The cache dir for saved model.")
    parser.add_argument("--bert_model", default="bert-base-cased", type=str,
                        help="The path of pretrained bert model.")
    # parser.add_argument("--max_seq_length", default=64, type=int,
    #                     help="The maximum total input sequence length after WordPiece tokenization. \n"
    #                          "Sequence longer than this will be truncated, and sequences shorter \n"
    #                          "than this wille be padded.")
    parser.add_argument("--max_seq_length", default=50, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequence longer than this will be truncated, and sequences shorter \n"
                             "than this wille be padded.")
    parser.add_argument("--train_batch_size", default=32, type=int,
                        help="Total batch size for training.") 
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=10.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for."
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--save_every_epoch", default=True, action='store_true')

    args = parser.parse_args()
    print(args)
    
    """prepare processors"""
    processor = AugProcessor(args)

    label_list = processor.get_labels()
    num_labels = len(label_list)

    """prepare model"""
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ## leveraging lastest bert module in Transformers to load pre-trained model tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.bert_model)
    
    ## leveraging lastest bert module in Transformers to load pre-trained model (weights)
    model = BertForMaskedLM.from_pretrained(args.bert_model)

    model.bert.embeddings.token_type_embeddings = torch.nn.Embedding(num_labels, 768)
    model.bert.embeddings.token_type_embeddings.weight.data.normal_(mean=0.0, std=0.02)
    logger.info("Token type embedding layer: {}".format(model.bert.embeddings.token_type_embeddings))

    train_examples = processor.get_train_examples()

    logger.info("  Num train examples = %d", len(train_examples))
    train_features, num_train_steps, train_dataloader = \
        construct_train_dataloader(train_examples, label_list, args.max_seq_length, 
        args.train_batch_size, args.num_train_epochs, tokenizer, device)

    ## if you have a GPU, put everything on cuda
    model.cuda()

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_features))
    logger.info("  Batch size = %d", args.train_batch_size)
    logger.info("  Num steps = %d", num_train_steps)

    ## in Transformers, optimizer and schedules are splitted and instantiated like this:
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grounded_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
    ]
    optimizer = AdamW(optimizer_grounded_parameters, lr=args.learning_rate, correct_bias=False)
    model.train()

    if os.path.exists(args.save_model_dir):
        shutil.rmtree(args.save_model_dir)
    os.makedirs(args.save_model_dir)

    for e in trange(int(args.num_train_epochs), desc="Epoch"):
        avg_loss = 0.

        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.cuda() for t in batch)

            _, input_ids, input_mask, segment_ids, masked_ids = batch
            """train generator at each batch"""
            optimizer.zero_grad() 
            outputs = model(input_ids, 
                            input_mask, 
                            segment_ids,
                            masked_lm_labels=masked_ids)
            loss = outputs[0]
            loss.backward()
            avg_loss += loss.item()
            optimizer.step()
            if (step + 1) % 50 == 0:
                print("avg_loss: {}".format(avg_loss / 50))
                avg_loss = 0
        if args.save_every_epoch:
            save_model_name = "{}_{}".format(args.bert_model, e)
            save_model_path = os.path.join(args.save_model_dir, save_model_name)
            torch.save(model, save_model_path)
        else:
            if (e + 1) % 10 == 0:
                save_model_name = "BertForMaskedLM_epoch_" + str(e + 1)
                save_model_path = os.path.join(args.save_model_dir, save_model_name)
                torch.save(model, save_model_path)

if __name__ == "__main__":
    main()
# Conditional BERT contextual augmentation and BERT with Self-Supervised Attention on GoEmotions 

Pytorch Implementation of [GoEmotions](https://github.com/google-research/google-research/tree/master/goemotions), [Conditional BERT contextual augmentation](https://arxiv.org/abs/1812.06705) and [BERT with Self-Supervised Attention](https://arxiv.org/pdf/2004.03808v3.pdf) with [Huggingface Transformers](https://github.com/huggingface/transformers)


## About GoEmotions

Dataset labeled **58000 Reddit comments** with **28 emotions**

- admiration, amusement, anger, annoyance, approval, caring, confusion, curiosity, desire, disappointment, disapproval, disgust, embarrassment, excitement, fear, gratitude, grief, joy, love, nervousness, optimism, pride, realization, relief, remorse, sadness, surprise + neutral

- dataset with three different taxonomies placed in `data`


## Requirements
1. Clone this repo
2. Python 3.6
3. Install Pytorch==1.4.0
4. Install the rest of the requirements: `pip install -r requirements.txt`

## To perform dataset analysis on GoEmotions
```bash
python analyze_dataset.py [--aug]
```
Hyperparameters can be changed from the json files in `config` directory. By default, the python script runs dataset analysis on `data/original/train.tsv`, with labels defined in `data/original/labels.txt`. If run with the `aug` flag on, dataset analysis will be performed on the augmented training dataset stored at `data/original/train_augmented_*.tsv` by default, without reading the labels file (augmented training dataset is generated using CBERT with a label distribution threshold of user's choice).

## To Run Vanilla BERT with GoEmotions

```bash
python run_goemotions.py --taxonomy original
```

## To Run Conditional BERT for generating new examples

First, finetune the conditional BERT model with the original training dataset.

```bash
python cbert_finetune.py 
```

Second, use the model saved in the previous step to generate new examples. The original examples, masked version and predicted version are stored in separate files in `data/original` by default.

```bash
python cbert_augdata.py
```

Third, remove duplicates, sanitize and merge the newly generated into the original training corpus.

```bash
python cbert_merge.py
```

## To Run BERT with Self-Supervised Attention on GoEmotions

```bash
cd ssa_BERT
python run_ssa.py
``` 

To run BERT with Self-Supervised Attention on the augmented GoEmotions dataset, simply change the default value of `train_data_file` defined in `ssa_BERT/run_ssa.py`.


## Reference

- [GoEmotions](https://github.com/google-research/google-research/tree/master/goemotions)
- [GoEmotions Github](https://github.com/google-research/google-research/tree/master/goemotions)
- [Conditional BERT Contextual Augmentation](https://github.com/1024er/cbert_aug)
- [BERT with Self-Supervised Attention](https://github.com/koukoulala/ssa_BERT)
- [Huggingface Transformers](https://github.com/huggingface/transformers)

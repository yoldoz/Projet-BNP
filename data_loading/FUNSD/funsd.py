from PIL import Image, ImageDraw, ImageFont
import json
from torch.nn import CrossEntropyLoss
from transformers import LayoutLMTokenizer
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import logging
import os
import random
import torch
from data_loading.read_txt_utils import convert_examples_to_features, read_examples_from_file

# if there are questions about path, change line 102 and 113.

logger = logging.getLogger(__name__)

# In fact, the codes of FunsdDataset is also included in the library layoutlm.data.funsd, we can use these functions directly by importing this library.
# Here we define the FunsdDataset by ourself and we create the torch type dataset for funsd which can be directly called in main.py to train.


class FunsdDataset(Dataset):
    def __init__(self, args, tokenizer, labels, pad_token_label_id, mode):
        if args.local_rank not in [-1, 0] and mode == "train":
            torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

        # Load data features from cache or dataset file
        cached_features_file = os.path.join(
            args.data_dir,
            "cached_{}_{}_{}".format(
                mode,
                list(filter(None, args.model_name_or_path.split("/"))).pop(),
                str(args.max_seq_length),
            ),
        )
        if os.path.exists(cached_features_file) and not args.overwrite_cache:
            logger.info("Loading features from cached file %s", cached_features_file)
            features = torch.load(cached_features_file)
        else:
            logger.info("Creating features from dataset file at %s", args.data_dir)
            examples = read_examples_from_file(args.data_dir, mode)
            features = convert_examples_to_features(

                examples,
                labels,
                args.max_seq_length,
                tokenizer,
                logger=logger,
                cls_token=tokenizer.cls_token,
                cls_token_segment_id=0,
                sep_token=tokenizer.sep_token,
                pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                pad_token_segment_id=0,
                pad_token_label_id=pad_token_label_id,
            )
            if args.local_rank in [-1, 0]:
                logger.info("Saving features into cached file %s", cached_features_file)
                torch.save(features, cached_features_file)

        if args.local_rank == 0 and mode == "train":
            torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

        self.features = features
        # Convert to Tensors and build dataset
        self.all_input_ids = torch.tensor(
            [f.input_ids for f in features], dtype=torch.long
        )
        self.all_input_mask = torch.tensor(
            [f.input_mask for f in features], dtype=torch.long
        )
        self.all_segment_ids = torch.tensor(
            [f.segment_ids for f in features], dtype=torch.long
        )
        self.all_label_ids = torch.tensor(
            [f.label_ids for f in features], dtype=torch.long
        )
        self.all_bboxes = torch.tensor([f.boxes for f in features], dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return (
            self.all_input_ids[index],
            self.all_input_mask[index],
            self.all_segment_ids[index],
            self.all_label_ids[index],
            self.all_bboxes[index],
        )


# dataset uses the BIOES annotation scheme,
# a given token is at the beginning (B), inside (I), outside (O), at the end (E) or start (S) of a given entity.
# Entities include ANSWER, QUESTION, HEADER and OTHER
def get_labels(path):
    with open(path, "r") as f:
        labels = f.read().splitlines()
    if "O" not in labels:
        labels = ["O"] + labels
    return labels

labels = get_labels("./data_loading/FUNSD/labels.txt")
num_labels = len(labels)
label_map = {i: label for i, label in enumerate(labels)}

# Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
pad_token_label_id = CrossEntropyLoss().ignore_index
# print(labels)

# Create a PyTorch dataset and corresponding dataloader
args = {'local_rank': -1,
        'overwrite_cache': True,
        'data_dir': './data_loading/FUNSD',
        'model_name_or_path':'microsoft/layoutlm-base-uncased',
        'max_seq_length': 512,
        'model_type': 'layoutlm',}

# class to turn the keys of a dict into attributes (thanks Stackoverflow)
class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

args = AttrDict(args)

tokenizer = LayoutLMTokenizer.from_pretrained("microsoft/layoutlm-base-uncased")

# the LayoutLM authors already defined a specific FunsdDataset, so we are going to use this here
train_dataset = FunsdDataset(args, tokenizer, labels, pad_token_label_id, mode="train")
train_sampler = RandomSampler(train_dataset)
train_dataloader = DataLoader(train_dataset,
                              sampler=train_sampler,
                              batch_size=2)

eval_dataset = FunsdDataset(args, tokenizer, labels, pad_token_label_id, mode="test")
eval_sampler = SequentialSampler(eval_dataset)
eval_dataloader = DataLoader(eval_dataset,
                             sampler=eval_sampler,
                            batch_size=2)

print(len(train_dataloader))
print(len(eval_dataloader))
print(eval_dataloader)
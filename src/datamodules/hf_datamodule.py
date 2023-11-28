from argparse import ArgumentError
from typing import Optional, Tuple

import os
import torch
import datasets
from pathlib import Path
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers import DataCollatorForTokenClassification


def get_entity_id(data_name):
    if data_name in ["NCBI-disease", "BC5CDR-disease", "mirna-di", "ncbi_disease", "scai_disease", "variome-di"]:
        entity_id = 1
    elif data_name in ["BC5CDR-chem",  "cdr-ch", "chemdner", "scai_chemicals", "chebi-ch", "BC4CHEMD","drAbreu/bc4chemd_ner"]:
        entity_id = 2
    elif data_name in ["BC2GM", "JNLPBA-protein", "bc2gm","bc2gm_corpus","mirna-gp", "cell_finder-gp", "chebi-gp", "loctext-gp", "deca", "fsu", "gpro", "jnlpba-gp", "bio_infer-gp", "variome-gp", "osiris-gp",  "iepa"]:
        entity_id = 3
    elif data_name in ["s800", "linnaeus", "loctext-sp", "mirna-sp", "chebi-sp", "cell_finder-sp", "variome-sp"]:
        entity_id = 4
    elif data_name in ["JNLPBA-cl", "cell_finder-cl", "jnlpba-cl", "gellus", "cll"]:
        entity_id = 5
    elif data_name in ["JNLPBA-dna", "jnlpba-dna"]:
        entity_id = 6
    elif data_name in ["JNLPBA-rna","jnlpba-rna"]:
        entity_id = 7
    elif data_name in ["JNLPBA-ct","jnlpba-ct"]:
        entity_id = 8
    else:
        entity_id = 0
    return entity_id


class HFDataModule(LightningDataModule):
    """
    LightningDataModule for HF Datasets.
    Requires a pre-processed (tokenized, cleaned...) dataset provided within the `data` folder.
    Might require adjustments if your dataset doesn't follow the structure of SNLI or MNLI.

    A DataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    def __init__(
        self,
        dataset_name: str,
        hf_token:str,
        tokenizer_name: str,
        batch_size: int = 64,
        max_length: int = 128,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.dataset = None
        self.tokenizer = None
        self.collator_fn = None

        self.eval_key = "validation"
        self.test_key = "test"

        if "mnli" in dataset_name:
            self.eval_key += "_matched"
            self.test_key += "_matched"

        self.keep_columns=[
            "idx",
            "input_ids",
            "attention_mask",
            "token_type_ids",
            "labels",
            "bias",
            "teacher_probs",
        ]

    @property
    def num_classes(self) -> int:
        return 3

    def prepare_data(self):
        """
        We should not assign anything here, so this function simply ensures
        that the pre-processed data is available.
        """
        # self.dataset_path = Path(self.hparams.data_dir) / self.hparams.dataset_name

        # if not os.path.exists(self.dataset_path):
        #     raise ValueError("The provided folder does not exist.")

        AutoTokenizer.from_pretrained(self.hparams.tokenizer_name,
                                      use_fast=True,
                                      token=self.hparams.hf_token)  # TODO: Load according to model-name

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.
        This method is called by lightning twice for `trainer.fit()` and `trainer.test()`, so be careful if you do a random split!
        The `stage` can be used to differentiate whether it's called before trainer.fit()` or `trainer.test()`."""

        if not self.tokenizer:
            # TODO: Load according to model-name
            self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.tokenizer_name,
                                                           use_fast=True,
                                                           token=self.hparams.hf_token)

        if not self.collator_fn:
            self.collator_fn = DataCollatorForTokenClassification(tokenizer=self.tokenizer,
                                                                  padding='max_length',
                                                                  max_length=self.hparams.max_length)

        if not self.dataset:
            self.dataset = self.load_datasets()

    def load_datasets(self):
      # Function that will return tokenized dataset with entity id added.
      data_names=self.hparams.dataset_name.split(',')
      final_datasets=[]
      for name in data_names:
        dataset=datasets.load_dataset(name)
        tokenized_dataset=dataset.map(self.tokenize_and_align_labels,
                                  batched=True,
                                  remove_columns=dataset["train"].column_names
                                  )
        #adding entity
        entity_id=get_entity_id(name)

        # Add the new column to each split (train, validation, test)
        new_column_values=[2,2]
        for split in tokenized_dataset.keys():
          entity_values = [[entity_id]*self.hparams.max_length for _ in range(len(tokenized_dataset[split]))]
          tokenized_dataset[split] = tokenized_dataset[split].add_column('entity_id', entity_values)

        final_datasets.append(tokenized_dataset)

      combined_dataset_dict = datasets.DatasetDict({split: datasets.concatenate_datasets([dataset_dict[split] for dataset_dict in final_datasets])
      for split in final_datasets[0].keys()
      })

      return combined_dataset_dict


    def align_labels_with_tokens(self,labels, word_ids):
        new_labels = []
        current_word = None
        for word_id in word_ids:
            if word_id != current_word:
                # Start of a new word!
                current_word = word_id
                label = -100 if word_id is None else labels[word_id]
                new_labels.append(label)
            elif word_id is None:
                # Special token
                new_labels.append(-100)
            else:
                # Same word as previous token
                label = labels[word_id]
                # If the label is B-XXX we change it to I-XXX
                if label % 2 == 1:
                    label += 1
                new_labels.append(label)

        return new_labels

    def tokenize_and_align_labels(self,examples):
        tokenized_inputs = self.tokenizer(
            examples["tokens"], truncation=True, is_split_into_words=True
        )
        all_labels = examples["ner_tags"]
        new_labels = []

        for i, labels in enumerate(all_labels):
            word_ids = tokenized_inputs.word_ids(i)
            new_labels.append(self.align_labels_with_tokens(labels, word_ids))

        tokenized_inputs["labels"] = new_labels

        return tokenized_inputs

    def train_dataloader(self):
        return DataLoader(
            dataset=self.dataset["train"],
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=self.collator_fn,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.dataset[self.eval_key],
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=self.collator_fn,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.dataset[self.test_key],
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=self.collator_fn,
            shuffle=False,
        )


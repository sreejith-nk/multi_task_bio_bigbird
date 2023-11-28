import inspect
from typing import Any, Dict, List, Tuple

import torch
from pytorch_lightning import LightningModule
from torch import nn
from torchmetrics import MaxMetric
from torchmetrics.classification.accuracy import Accuracy
from transformers import AdamW, AutoModel, get_linear_schedule_with_warmup
import torch.nn.functional as F

from src.utils import utils

log = utils.get_logger(__name__)

def match_indexes(entity_id):
    dise_idx = []
    chem_idx = []
    gene_idx = []
    spec_idx = []
    cellline_idx = []
    dna_idx = []
    rna_idx = []
    celltype_idx = []

    for i, example in enumerate(entity_id):
        if example[0] == 1:
            dise_idx.append(i)
        elif example[0] == 2:
            chem_idx.append(i)
        elif example[0] == 3:
            gene_idx.append(i)
        elif example[0] == 4:
            spec_idx.append(i)
        elif example[0] == 5:
            cellline_idx.append(i)
        elif example[0] == 6:
            dna_idx.append(i)
        elif example[0] == 7:
            rna_idx.append(i)
        elif example[0] == 8:
            celltype_idx.append(i)
        else:
            # Handle other cases if needed
            pass
    return dise_idx, chem_idx, gene_idx, spec_idx, cellline_idx, dna_idx, rna_idx, celltype_idx


class Bigbird_multitask(LightningModule):
    """
    Transformer Model for Sequence Classification.

    A LightningModule organizes your PyTorch code into 5 sections:
        - Computations (init).
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Optimizers (configure_optimizers)

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
        self,
        huggingface_model: str,
        num_labels: int,
        hf_token: str,
        learning_rate: float = 2e-5,
        adam_epsilon: float = 1e-8,
        warmup_steps: int = 0,
        weight_decay: float = 0.0,
        batch_size: int = 64,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # Load model and add classification head
        self.model = AutoModel.from_pretrained(self.hparams.huggingface_model,
                                               token=self.hparams.hf_token)

        self.dise_classifier = nn.Linear(self.model.config.hidden_size, self.hparams.num_labels) # disease
        self.chem_classifier = nn.Linear(self.model.config.hidden_size, self.hparams.num_labels) # chemical
        self.gene_classifier = nn.Linear(self.model.config.hidden_size, self.hparams.num_labels) # gene/protein
        self.spec_classifier = nn.Linear(self.model.config.hidden_size, self.hparams.num_labels) # species
        self.cellline_classifier = nn.Linear(self.model.config.hidden_size, self.hparams.num_labels) # cell line
        self.dna_classifier = nn.Linear(self.model.config.hidden_size, self.hparams.num_labels) # dna
        self.rna_classifier = nn.Linear(self.model.config.hidden_size, self.hparams.num_labels) # rna
        # self.protein_classifier = nn.Linear(self.model.config.hidden_size, self.hparams.num_labels) # protein
        self.celltype_classifier = nn.Linear(self.model.config.hidden_size, self.hparams.num_labels) # cell type

        # Init classifier weights according to initialization rules of model
        classifiers=[self.dise_classifier,
            self.chem_classifier,
            self.gene_classifier,
            self.spec_classifier,
            self.cellline_classifier,
            self.dna_classifier,
            self.rna_classifier,
            # self.protein_classifier,
            self.celltype_classifier]

        for classifier in classifiers:
            self.model._init_weights(classifier)

        # Apply dropout rate of model
        dropout_prob = self.model.config.hidden_dropout_prob
        log.info(f"Dropout probability of classifier set to {dropout_prob}.")
        self.dropout = nn.Dropout(dropout_prob)

        # loss function (assuming single-label multi-class classification)
        self.loss_fn = torch.nn.CrossEntropyLoss()  # TODO: Make this customizable

        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        self.train_acc = Accuracy(task='multiclass',
                                     num_classes=self.hparams.num_labels)
        self.val_acc = Accuracy(task='multiclass',
                                     num_classes=self.hparams.num_labels)
        self.test_acc = Accuracy(task='multiclass',
                                     num_classes=self.hparams.num_labels)

        # for logging best so far validation accuracy
        self.val_acc_best = MaxMetric()

        # Collect the forward signature
        params = inspect.signature(self.model.forward).parameters.values()
        params = [param.name for param in params if param.kind == param.POSITIONAL_OR_KEYWORD]
        self.forward_signature = params

    def forward(self, batch: Dict[str, torch.tensor]):
        filtered_batch = {key: batch[key] for key in batch.keys() if key in self.forward_signature}
        outputs = self.model(**filtered_batch, return_dict=True)
        outputs = outputs.last_hidden_state

        dise_logits = self.dise_classifier(outputs) # disease logit value
        chem_logits = self.chem_classifier(outputs) # chemical logit value
        gene_logits = self.gene_classifier(outputs) # gene/protein logit value
        spec_logits = self.spec_classifier(outputs) # species logit value
        cellline_logits = self.cellline_classifier(outputs) # cell line logit value
        dna_logits = self.dna_classifier(outputs) # dna logit value
        rna_logits = self.rna_classifier(outputs) # rna logit value
        # protein_logits = self.protein_classifier(outputs) # protein logit value
        celltype_logits = self.celltype_classifier(outputs) # cell type logit value

        # pooler = outputs.pooler_output
        # pooler = self.dropout(pooler)
        # logits = self.classifier(pooler)
        return dise_logits, chem_logits, gene_logits, spec_logits, cellline_logits, dna_logits, rna_logits, celltype_logits

    def step(self, batch: Dict[str, torch.tensor]):
        logits = self(batch)
        indexes = match_indexes(batch['entity_id'])

        overall_loss=0.0
        overall_preds=torch.zeros([batch['entity_id'].shape[0],batch['entity_id'].shape[1]])
        print(overall_preds.shape)


        for i,task_idx in enumerate(indexes):
            if len(task_idx) > 0:
                labels = batch.get('labels')[task_idx].view(-1)
                preds = logits[i][task_idx].view(-1, self.hparams.num_labels)  # Replace 'logits' with the appropriate variable

                overall_preds[task_idx]=torch.argmax(logits[i][task_idx], dim=2).to(torch.float32)

                # Calculate loss for the current category using a specified loss function
                task_loss = self.loss_fn(preds, labels)
                overall_loss += task_loss



        return overall_loss,overall_preds.view(-1)


        # logits = logits.view(-1, self.hparams.num_labels)
        # labels = batch["labels"].view(-1)
        # loss = self.loss_fn(logits, labels)
        # preds = torch.argmax(logits, dim=1)
        # return loss, preds

    def training_step(self, batch: Dict[str, torch.tensor], batch_idx: int):
        loss, preds = self.step(batch)
        # log train metrics
        acc = self.train_acc(preds, batch["labels"])
        self.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=False)
        self.log("train/acc", acc, on_step=True, on_epoch=False, prog_bar=True)
        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()`` below
        # remember to always return loss from `training_step()` or else backpropagation will fail!
        return {"loss": loss, "preds": preds, "targets": batch["labels"]}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        pass

    def validation_step(self, batch: Dict[str, torch.tensor], batch_idx: int):
        loss, preds = self.step(batch)

        # log val metrics
        acc = self.val_acc(preds, batch["labels"])
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=False)

        return {"loss": loss, "preds": preds, "targets": batch["labels"]}

    def validation_epoch_end(self, outputs: List[Any]):
        acc = self.val_acc.compute()  # get val accuracy from current epoch
        self.val_acc_best.update(acc)
        self.log("val/acc_best", self.val_acc_best.compute(), on_epoch=True, prog_bar=True)

    def test_step(self, batch: Dict[str, torch.tensor], batch_idx: int):
        loss, preds = self.step(batch)

        # log test metrics
        acc = self.test_acc(preds, batch["labels"])
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/acc", acc, on_step=False, on_epoch=True)

        return {"loss": loss, "preds": preds, "targets": batch["labels"]}

    def test_epoch_end(self, outputs: List[Any]):
        pass

    def on_epoch_end(self):
        # reset metrics at the end of every epoch!
        self.train_acc.reset()
        self.test_acc.reset()
        self.val_acc.reset()

    @property
    def total_training_steps(self) -> int:
        """Total training steps inferred from datamodule and devices."""
        if isinstance(self.trainer.limit_train_batches, int) and self.trainer.limit_train_batches != 0:
            dataset_size = self.trainer.limit_train_batches
        elif isinstance(self.trainer.limit_train_batches, float):
            # limit_train_batches is a percentage of batches
            dataset_size = len(self.trainer.datamodule.train_dataloader())
            dataset_size = int(dataset_size * self.trainer.limit_train_batches)
        else:
            dataset_size = len(self.trainer.datamodule.train_dataloader())

        num_devices = max(1, self.trainer.num_gpus, self.trainer.num_processes)
        if self.trainer.tpu_cores:
            num_devices = max(num_devices, self.trainer.tpu_cores)

        effective_batch_size = self.trainer.accumulate_grad_batches * num_devices
        max_estimated_steps = (dataset_size // effective_batch_size) * self.trainer.max_epochs

        if self.trainer.max_steps and 0 < self.trainer.max_steps < max_estimated_steps:
            return self.trainer.max_steps
        return max_estimated_steps

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        print(f"{self.hparams.learning_rate =}")
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.total_training_steps,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]
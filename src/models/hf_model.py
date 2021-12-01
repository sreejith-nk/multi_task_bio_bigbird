from typing import Any, Dict, List, Tuple

import torch
import inspect
from pytorch_lightning import LightningModule
from torch import nn
from torchmetrics import MaxMetric
from torchmetrics.classification.accuracy import Accuracy
from transformers import AutoModel, AdamW, get_linear_schedule_with_warmup


class SequenceClassificationTransformer(LightningModule):
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
        num_train_samples: int, # Need this until lightning updates issue #10760
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
        self.model = AutoModel.from_pretrained(huggingface_model)
        self.classifier = nn.Linear(self.model.config.hidden_size, num_labels)
        self.dropout = nn.Dropout(self.model.config.hidden_dropout_prob)

        # loss function (assuming single-label multi-class classification)
        self.loss_fn = torch.nn.CrossEntropyLoss()  # TODO: Make this customizable

        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()

        # for logging best so far validation accuracy
        self.val_acc_best = MaxMetric()

        # Collect the forward signature
        params = inspect.signature(self.model.forward).parameters.values()
        params = [param.name for param in params if param.kind == param.POSITIONAL_OR_KEYWORD]
        self.forward_signature = params

    def forward(self, batch: Dict[str, torch.tensor]):
        filtered_batch = {key : batch[key] for key in batch.keys() if key in self.forward_signature}
        outputs = self.model(**filtered_batch, return_dict=True)
        pooler = outputs.pooler_output
        pooler = self.dropout(pooler)
        logits = self.classifier(pooler)
        return logits

    def step(self, batch: Dict[str, torch.tensor]):
        logits = self.forward(batch)
        logits = logits.view(-1, self.hparams.num_labels)
        labels = batch["labels"].view(-1)
        loss = self.loss_fn(logits, labels)
        preds = torch.argmax(logits, dim=1)
        return loss, preds

    def training_step(self, batch: Dict[str, torch.tensor], batch_idx: int):
        loss, preds = self.step(batch)
        # log train metrics
        acc = self.train_acc(preds, batch["labels"])
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=False)
        self.log("train/acc", acc, on_step=True, on_epoch=True, prog_bar=True)
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
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

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
    def total_train_steps(self):
        """Steps (including batches and multi-devices) until training is finished.
        Needed for the scheduler. Lightning 1.5 removed self.train_dataloader so the usual way could
        not be used anymore. Until they figure out a new way, we pass `num_train_samples` as param to model.
        Watch issue [#10760](https://github.com/PyTorchLightning/pytorch-lightning/issues/10760 """
        tb_size = self.hparams.batch_size * max(1, self.trainer.gpus)
        ab_size = self.trainer.accumulate_grad_batches * float(self.trainer.max_epochs)
        return (self.hparams.num_train_samples // tb_size) // ab_size

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
        print(f'{self.hparams.learning_rate =}')
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.total_train_steps,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]
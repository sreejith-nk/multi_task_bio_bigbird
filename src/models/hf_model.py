import inspect
from typing import Any, Dict, List, Tuple

import torch
from pytorch_lightning import LightningModule
from torch import nn
from torchmetrics import MaxMetric
from torchmetrics import Accuracy,Precision,Recall,F1Score
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
    protein_idx = []

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
        elif example[0] == 9:
            protein_idx.append(i)
        else:
            # Handle other cases if needed
            pass
    return dise_idx, chem_idx, gene_idx, spec_idx, cellline_idx, dna_idx, rna_idx, celltype_idx, protein_idx


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
                                               use_auth_token=self.hparams.hf_token)

        self.dise_seq = nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size) # disease
        self.chem_seq = nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size)
        self.gene_seq = nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size)
        self.spec_seq = nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size)
        self.cellline_seq = nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size)
        self.dna_seq = nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size)
        self.rna_seq = nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size)
        self.celltype_seq = nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size)
        self.protein_seq = nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size)

        self.dise_classifier = nn.Linear(self.model.config.hidden_size, self.hparams.num_labels) # disease
        self.chem_classifier = nn.Linear(self.model.config.hidden_size, self.hparams.num_labels) # chemical
        self.gene_classifier = nn.Linear(self.model.config.hidden_size, self.hparams.num_labels) # gene/protein
        self.spec_classifier = nn.Linear(self.model.config.hidden_size, self.hparams.num_labels) # species
        self.cellline_classifier = nn.Linear(self.model.config.hidden_size, self.hparams.num_labels) # cell line
        self.dna_classifier = nn.Linear(self.model.config.hidden_size, self.hparams.num_labels) # dna
        self.rna_classifier = nn.Linear(self.model.config.hidden_size, self.hparams.num_labels) # rna
        self.celltype_classifier = nn.Linear(self.model.config.hidden_size, self.hparams.num_labels) # cell type
        self.protein_classifier = nn.Linear(self.model.config.hidden_size, self.hparams.num_labels) # protein

        # Init classifier weights according to initialization rules of model
        seqs=[self.dise_seq,
            self.chem_seq,
            self.gene_seq,
            self.spec_seq,
            self.cellline_seq,
            self.dna_seq,
            self.rna_seq,
            self.celltype_seq,
            self.protein_seq]

        classifiers=[self.dise_classifier,
            self.chem_classifier,
            self.gene_classifier,
            self.spec_classifier,
            self.cellline_classifier,
            self.dna_classifier,
            self.rna_classifier,
            self.celltype_classifier,
            self.protein_classifier,]

        for seq in seqs:
            self.model._init_weights(seq)

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
                                     num_classes=self.hparams.num_labels+1,ignore_index=-100)
        self.val_acc = Accuracy(task='multiclass',
                                     num_classes=self.hparams.num_labels+1,ignore_index=-100)
        self.test_acc = Accuracy(task='multiclass',
                                     num_classes=2*9+1+1,ignore_index=-100)
        self.test_ovrl_f1 = F1Score(task='multiclass',average='micro',
                                        num_classes=2*9+1+1,ignore_index=-100)
        self.test_ovrl_prec = Precision(task='multiclass',average
        ='micro',
                                        num_classes=2*9+1+1,ignore_index=-100)
        self.test_ovrl_rec = Recall(task='multiclass',average='micro',
                                        num_classes=2*9+1+1,ignore_index=-100)

        self.test_prec = Precision(task='multiclass',average=None,
                                        num_classes=2*9+1+1,ignore_index=-100)
        self.test_rec = Recall(task='multiclass',average=None,
                                        num_classes=2*9+1+1,ignore_index=-100)
        self.test_f1 = F1Score(task='multiclass',average=None,
                                        num_classes=2*9+1+1,ignore_index=-100)   
                                                     
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

        dise_outputs = F.relu(self.dise_seq(outputs)) # disease
        chem_outputs = F.relu(self.chem_seq(outputs))
        gene_outputs = F.relu(self.gene_seq(outputs))
        spec_outputs = F.relu(self.spec_seq(outputs))
        cellline_outputs = F.relu(self.cellline_seq(outputs))
        dna_outputs = F.relu(self.dna_seq(outputs))
        rna_outputs = F.relu(self.rna_seq(outputs))
        celltype_outputs = F.relu(self.celltype_seq(outputs))
        protein_outputs = F.relu(self.protein_seq(outputs))

        dise_logits = self.dise_classifier(dise_outputs) # disease logit value
        chem_logits = self.chem_classifier(chem_outputs) # chemical logit value
        gene_logits = self.gene_classifier(gene_outputs) # gene/protein logit value
        spec_logits = self.spec_classifier(spec_outputs) # species logit value
        cellline_logits = self.cellline_classifier(cellline_outputs) # cell line logit value
        dna_logits = self.dna_classifier(dna_outputs) # dna logit value
        rna_logits = self.rna_classifier(rna_outputs) # rna logit value
        celltype_logits = self.celltype_classifier(celltype_outputs) # cell type logit value
        protein_logits = self.protein_classifier(protein_outputs) # protein logit value

        # pooler = outputs.pooler_output
        # pooler = self.dropout(pooler)
        # logits = self.classifier(pooler)
        return dise_logits, chem_logits, gene_logits, spec_logits, cellline_logits, dna_logits, rna_logits, celltype_logits, protein_logits

    def step(self, batch: Dict[str, torch.tensor]):
        logits = self(batch)
        indexes = match_indexes(batch['entity_id'])

        overall_loss=0.0
        overall_preds=torch.zeros([batch['entity_id'].shape[0],batch['entity_id'].shape[1]]).to(logits[0].device)

        for i,task_idx in enumerate(indexes):
            if len(task_idx) > 0:
                labels = batch.get('labels')[task_idx].view(-1)
                preds = logits[i][task_idx].view(-1, self.hparams.num_labels)  # Replace 'logits' with the appropriate variable

                overall_preds[task_idx]=torch.argmax(logits[i][task_idx], dim=2).to(torch.float32).to(logits[i][task_idx].device)

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
        acc = self.train_acc(preds, batch["labels"].view(-1))
        self.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=False)
        self.log("train/acc", acc, on_step=True, on_epoch=False, prog_bar=True)
        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()`` below
        # remember to always return loss from `training_step()` or else backpropagation will fail!
        return {"loss": loss, "preds": preds, "targets": batch["labels"]}

    def on_train_epoch_end(self):
        # `outputs` is a list of dicts returned from `training_step()`
        self.train_acc.reset()

    def validation_step(self, batch: Dict[str, torch.tensor], batch_idx: int):

        loss, preds = self.step(batch)

        # log val metrics
        acc = self.val_acc(preds, batch["labels"].view(-1))
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=False)

        return {"loss": loss, "preds": preds, "targets": batch["labels"]}

    def on_validation_epoch_end(self):
        acc = self.val_acc.compute()  # get val accuracy from current epoch
        self.val_acc_best.update(acc)
        self.log("val/acc_best", self.val_acc_best.compute(), on_epoch=True, prog_bar=True)
        self.val_acc.reset()
    
    def test_labelify(self, pred_label, pred_task):
        test_labels=torch.zeros(pred_label.shape).to(pred_label.device)
        for i,(label,task) in enumerate(zip(pred_label,pred_task)): 
            if label.item()==0 or label.item()==-100:
                test_labels[i]=label.item()
            else:
                test_labels[i]=2*task+label
        return test_labels
    def test_step(self, batch: Dict[str, torch.tensor], batch_idx: int):
        logits = self.forward(batch)
        
        labels=batch['labels'].view(-1)
        tasks=batch['entity_id'].view(-1)-1
        if batch['entity_id'][0][0]!=0:
            test_labels=self.test_labelify(labels,tasks)
        else:
            test_labels=labels
        new_logits = []
        for logit in logits:
            argmax_predictions = torch.argmax(logit, dim=-1)
            max_probabilities = torch.max(logit, dim=-1).values
            new_tensor = torch.stack([argmax_predictions, max_probabilities], dim=-1)
            new_logits.append(new_tensor)
        
        new_logits=torch.stack(new_logits,dim=2)
        new_logits=new_logits.view(-1,new_logits.shape[2],2)
        
        test_pred_label=torch.zeros(test_labels.shape).to(test_labels.device)
        test_pred_task=torch.zeros(test_labels.shape).to(test_labels.device)

        for i,pred in enumerate(new_logits):
            confidences=pred[:,1]
            predictions=pred[:,0]
            if all(predictions[[batch['entity_id'][0][0].item()-1]]==0):
                test_pred_label[i]=0
                test_pred_task[i]=-1
                continue
            else:
                indices = torch.nonzero(predictions!= 0)
                indices_mask=(indices==0) | (indices==1) | (indices==2) | (indices==3) 
                indices=indices[indices_mask]
                predicted_task = indices[torch.argmax(confidences[indices])]

                test_pred_label[i]=predictions[predicted_task]
                test_pred_task[i]=predicted_task

        test_pred=self.test_labelify(test_pred_label,test_pred_task)
        
        self.test_prec.update(test_pred, test_labels)
        self.test_rec.update(test_pred, test_labels)
        self.test_f1.update(test_pred, test_labels)
        
        self.test_acc.update(test_pred, test_labels)
        self.test_ovrl_prec.update(test_pred, test_labels)
        self.test_ovrl_rec.update(test_pred, test_labels)
        self.test_ovrl_f1.update(test_pred, test_labels)

        loss, preds = self.step(batch)

        return {"loss": loss, "preds": preds, "targets": batch["labels"]}

    def on_test_epoch_end(self):
        acc=self.test_acc.compute()
        prec=self.test_prec.compute()
        rec=self.test_rec.compute()
        f1=self.test_f1.compute()

        ovrl_prec=self.test_ovrl_prec.compute()
        ovrl_rec=self.test_ovrl_rec.compute()
        ovrl_f1=self.test_ovrl_f1.compute()

        self.log("test/acc", acc, on_epoch=True, prog_bar=True)
        self.log("test/ovrl_prec", ovrl_prec, on_epoch=True, prog_bar=True)
        self.log("test/ovrl_rec", ovrl_rec, on_epoch=True, prog_bar=True)
        self.log("test/ovrl_f1", ovrl_f1, on_epoch=True, prog_bar=True)

        for i, (p, r, f) in enumerate(zip(prec, rec, f1)):
            self.log(f'precision_class_{i}', p, on_step=False, on_epoch=True)
            self.log(f'recall_class_{i}', r, on_step=False, on_epoch=True)
            self.log(f'f1_class_{i}', f, on_step=False, on_epoch=True)
        
        
        self.test_ovrl_f1.reset()
        self.test_ovrl_prec.reset()
        self.test_ovrl_rec.reset()
        self.test_acc.reset()
        self.test_prec.reset()
        self.test_rec.reset()
        self.test_f1.reset()

    """@property"""
    def total_training_steps(self) -> int:
        print(dir(self.trainer))
        """Total training steps inferred from datamodule and devices."""
        if isinstance(self.trainer.limit_train_batches, int) and self.trainer.limit_train_batches != 0:
            dataset_size = self.trainer.limit_train_batches
        elif isinstance(self.trainer.limit_train_batches, float):
            # limit_train_batches is a percentage of batches
            dataset_size = len(self.trainer.datamodule.train_dataloader())
            dataset_size = int(dataset_size * self.trainer.limit_train_batches)
        else:
            dataset_size = len(self.trainer.datamodule.train_dataloader())
       
        num_devices = max(1, self.trainer.num_devices)

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
            num_training_steps=self.total_training_steps(),
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]

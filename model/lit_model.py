import torch.nn as nn
import pytorch_lightning as lt
from torchmetrics import Accuracy, F1Score, MeanMetric, MaxMetric
import timm
import torch


class LitModel(lt.LightningModule):
    def __init__(self,
                 lr=0.001,
                 num_classes=2):
        super().__init__()
        
        self.save_hyperparameters(logger=False)
        self.net = timm.create_model("convnextv2_tiny", pretrained=True, num_classes=num_classes)
        
        # modify train the FCs only
        for param in self.net.parameters():
            param.requires_grad = False
        for param in self.net.head.parameters():
            param.requires_grad = True
        self.net.head.drop = nn.Dropout(0.5)
        self.lr = lr
        
        # loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # metric objects for calculating and averaging accuracy across batches
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)
        
        self.train_f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")
        self.val_f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")
        self.test_f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")
        
        # for average loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()
        
        self.val_acc_best = MaxMetric()
        
    def forward(self, x):
        return self.net(x)
    
    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acc_best doesn't store accuracy from these checks
        self.val_acc_best.reset()

    def step(self, batch):
        image, label = batch
        # mixed_x, y_a, y_b, lam = self.mixup_data(image, label)#apply mixup
        logits = self.forward(image)
        # loss = self.mixup_criterion(logits, y_a, y_b, lam)
        loss = self.criterion(logits, label)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, label

    def training_step(self, batch, batch_idx: int):
        loss, preds, targets = self.step(batch)
        # update and log metrics
        self.train_loss(loss)
        self.train_f1(preds, targets)
        self.train_acc(preds, targets)
        self.log("train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/f1", self.train_f1, on_step=False, on_epoch=True, prog_bar=True)
        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or backpropagation will fail!
        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_step(self, batch, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.val_f1(preds, targets)
        self.val_acc(preds, targets)
        self.log("val/loss", self.val_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/f1", self.val_f1, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss, "preds": preds, "targets": targets}

    def on_validation_epoch_end(self):
        acc = self.val_acc.compute()  # get current val acc
        self.val_acc_best(acc)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/acc_best", self.val_acc_best.compute(), prog_bar=True)

    def test_step(self, batch, batch_idx: int):
        loss, preds, targets = self.step(batch)
        # plot_top_losses(batch[0], batch[1], batch_idx, loss, targets, preds)
        # save_images_with_confidence(self.img_confs, confidence_scores, targets, file_names)
        # update and log metrics
        # loss, preds, targets = self.step(batch)
        self.test_loss(loss)
        self.test_f1(preds, targets)
        self.test_acc(preds, targets)
        self.log("test/loss", self.test_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/f1", self.test_f1, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss, "preds": preds, "targets": targets}
        
    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-5,
                                      amsgrad=False)
        # optimizer = torch.optim.SGD(self.parameters(), lr=0.1, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200*20)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
                "interval": "step",
                "frequency": 1,
            },
        }
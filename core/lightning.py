
import torch
import torch.nn as nn

import pytorch_lightning as pl

import torchmetrics



class DLM_module(pl.LightningModule):
    def __init__(self, model):
        super(DLM_module, self).__init__()
        self.model=model()
        self.loss = nn.MSELoss()
        self.accuracy = torchmetrics.Accuracy()
        self.auroc = torchmetrics.AUROC()

    def training_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self.model.forward(X)
        ts_loss = self.loss(torch.squeeze(y_hat), y)
        # ts_accuracy = self.accuracy(y_hat, y)
        # ts_auroc = self.auroc(y_hat, y)
        self.log("train_loss", ts_loss, on_epoch=True, logger=True)
        # self.log("train_accuracy", ts_accuracy, on_epoch=True, logger=True)
        # self.log("train_auroc", ts_auroc, on_epoch=True, logger=True)
        return ts_loss


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1.0e-4)
        return optimizer

    def validation_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self.model.forward(X)
        vs_loss = self.loss(torch.squeeze(y_hat), y)
        # vs_accuracy = self.accuracy(y_hat, y)
        # vs_auroc = self.auroc(y_hat, y)
        self.log("validation_loss", vs_loss, on_epoch=True, logger=True)
        # self.log("validation_accuracy", vs_accuracy, on_epoch=True, logger=True)
        # self.log("validation_auroc", vs_auroc, on_epoch=True, logger=True)
        return vs_loss

    def test_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self.model.forward(X)
        ts_loss = self.loss(torch.squeeze(y_hat), y)
        # ts_accuracy = self.accuracy(y_hat, y)
        # ts_auroc = self.auroc(y_hat, y)
        self.log("test_loss", ts_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # self.log("test_accuracy", ts_accuracy, on_epoch=True, logger=True)
        # self.log("test_auroc", ts_auroc, on_epoch=True, logger=True)
        return ts_loss




import argparse
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
import utils.metrics
import utils.losses
from torch.optim.lr_scheduler import ReduceLROnPlateau


class SupervisedForecastTask(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        loss="mse",
        feat_max_val: float = 1.0,
        **kwargs
    ):
        super(SupervisedForecastTask, self).__init__()
        self.save_hyperparameters()
        self.model = model
        self._loss = loss
        self.feat_max_val = feat_max_val

    def forward(self, x):
        batch_size, _, num_nodes = x.size()
        hidden = self.model(x)
        hidden = hidden.reshape((-1, hidden.size(2)))
        predictions = hidden
        predictions = predictions.reshape((batch_size, num_nodes, -1))
        return predictions

    def shared_step(self, batch, batch_idx):
        x, y = batch
        num_nodes = x.size(2)
        predictions = self(x)
        predictions = predictions.transpose(1, 2).reshape((-1, num_nodes))
        y = y.reshape((-1, y.size(2)))
        return predictions, y

    def loss(self, inputs, targets):
        if self._loss == "mse":
            return F.mse_loss(inputs, targets)
        if self._loss == "mse_with_regularizer":
            return utils.losses.mse_with_regularizer_loss(inputs, targets, self)
        raise NameError("Loss not supported:", self._loss)

    def training_step(self, batch, batch_idx):
        predictions, y = self.shared_step(batch, batch_idx)
        loss = self.loss(predictions, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        predictions, y = self.shared_step(batch, batch_idx)
        predictions = predictions * self.feat_max_val
        predictions = torch.round(torch.abs(predictions))
        y = y * self.feat_max_val
        loss = self.loss(predictions, y)
        accuracy = utils.metrics.accuracy(predictions, y)
        r2 = utils.metrics.r2(predictions, y)
        explained_variance = utils.metrics.explained_variance(predictions, y)
        mape = utils.metrics.MAPE(y, predictions)
        mae = utils.metrics.MAE(y, predictions)
        rmse = utils.metrics.RMSE(y, predictions)
        masked_mape = utils.metrics.MASKED_MAPE(y, predictions)
        metrics = {
            "val_loss": loss,
            "RMSE": rmse,
            "MAE": mae,
            "accuracy": accuracy,
            "R2": r2,
            "ExplainedVar": explained_variance,
            "MAPE": mape,
            "MASKED_MAPE": masked_mape
        }
        self.log_dict(metrics)
        return predictions.reshape(batch[1].size()), y.reshape(batch[1].size())

    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(optimizer, factor=0.5, min_lr=1e-5),
                "monitor": "train_loss",
                "interval": "epoch",
                "frequency": 500
            },
        }




    @staticmethod
    def add_task_specific_arguments(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--learning_rate", "--lr", type=float, default=1e-3)
        parser.add_argument("--weight_decay", "--wd", type=float, default=1.5e-3)
        parser.add_argument("--loss", type=str, default="mse")
        return parser

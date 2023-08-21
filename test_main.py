import argparse
import traceback
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info
import models
import tasks
import utils.callbacks
import utils.data
import torch
import pandas as pd
import utils.metrics
import numpy as np
import matplotlib.pyplot as plt
import time
# import utils.email
import utils.logging


DATA_PATHS = {
    "my": {"feat": "my_data/data.csv", "adj": "my_data/adj_matrix_0.9.csv"},
}


def get_model(args, dm):
    model = None
    if args.model_name == "AGCRNN":
        model = models.AGCRNN(adj=dm.adj, hidden_dim=args.hidden_dim)
    return model


def get_task(args, model, dm):
    task = getattr(tasks, args.settings.capitalize() + "ForecastTask")(
        model=model, feat_max_val=dm.feat_max_val, **vars(args)
    )
    return task


def get_callbacks(args):
    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor="train_loss")
    early_stop = pl.callbacks.EarlyStopping(
        monitor='train_loss',
        min_delta=0.0,
        patience=64,
        verbose=False,
        mode='min'
    )
    plot_validation_predictions_callback = utils.callbacks.PlotValidationPredictionsCallback(monitor="train_loss")
    callbacks = [
        checkpoint_callback,
        plot_validation_predictions_callback,
        early_stop,
    ]
    return callbacks


def main_supervised(args):
    dm = utils.data.SpatioTemporalCSVDataModule(
        feat_path=DATA_PATHS[args.data]["feat"], adj_path=DATA_PATHS[args.data]["adj"], **vars(args)
    )
    data = pd.read_csv('my_data/other data/data_test.csv')
    data1 = np.array(data, dtype=np.float32)
    max_val = np.max(data1)
    data1 = data1 / max_val
    test = list()
    label = list()
    for i in range(data1.shape[0] - 7 - 1  ):
        test.append(np.array(data1[i : i + 7]))
        label.append(np.array(data1[i + 7 : i + 7 + 1 ]))
    test = np.array(test)

    label = np.array(label)

    test_data = torch.FloatTensor(test)

    label = torch.FloatTensor(label)
    test_data = torch.unsqueeze(test_data,0)


    model = get_model(args, dm)
    task = get_task(args, model, dm)

    trainer = pl.Trainer(resume_from_checkpoint='lightning_logs/version_73/checkpoints/epoch=2999-step=108000.ckpt',gpus=1,logger=False)

    results = trainer.predict(model = task,ckpt_path='lightning_logs/version_73/checkpoints/epoch=2999-step=108000.ckpt',dataloaders =test_data)
    for i in results:

        predictions = i.transpose(1, 2).reshape((-1, 151))

        predictions = predictions * max_val
        print('pre',predictions)
        predictions_l = torch.round(torch.abs(predictions))


        label = label.reshape((-1, 151))
        label = label * max_val
        print(label)
        label_l = torch.round(torch.abs(label))

        accuracy = utils.metrics.accuracy(predictions, label)
        r2 = utils.metrics.r2(predictions, label)
        explained_variance = utils.metrics.explained_variance(predictions, label)
        mae = utils.metrics.MAE(label, predictions)
        rmse = utils.metrics.RMSE(label, predictions_l)
        print('accuracy:', accuracy)
        print('r2:', r2)
        print('var:', explained_variance)
        print('mae:', mae)
        print('rmse:', rmse)

        plt.rcParams["font.family"] = "Times New Roman"
        fig = plt.figure(figsize=(7, 2), dpi=200)
        plt.plot(
            label[:, 92],
            color="dimgray",
            linestyle="-",
            label="Ground truth",
        )
        plt.plot(
            predictions[:, 92],
            color="deepskyblue",
            linestyle="-",
            label="Predictions",
        )
        plt.legend(loc="best", fontsize=10)
        plt.xlabel("Time/ 15 min")
        plt.ylabel("PSD(dB/Hz)")
        plt.show()
    return results


def main(args):
    rank_zero_info(vars(args))
    results = globals()["main_" + args.settings](args)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)

    parser.add_argument(
        "--data", type=str, help="The name of the dataset", choices=("my"), default="my"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="The name of the model for spatiotemporal prediction",
        choices=("AGCRNN"),
        default="AGCRNN",
    )
    parser.add_argument(
        "--settings",
        type=str,
        help="The type of settings, e.g. supervised learning",
        choices=("supervised"),
        default="supervised",
    )
    parser.add_argument("--log_path", type=str, default='./log', help="Path to the output console log file")
    # parser.add_argument("--send_email", "--email", action="store_true", help="Send email when finished")

    temp_args, _ = parser.parse_known_args()

    parser = getattr(utils.data, temp_args.settings.capitalize() + "DataModule").add_data_specific_arguments(parser)
    parser = getattr(models, temp_args.model_name).add_model_specific_arguments(parser)
    parser = getattr(tasks, temp_args.settings.capitalize() + "ForecastTask").add_task_specific_arguments(parser)

    args = parser.parse_args()
    utils.logging.format_logger(pl._logger)
    if args.log_path is not None:
        utils.logging.output_logger_to_file(pl._logger, args.log_path)

    try:
        results = main(args)
        print(results)
    except:
        traceback.print_exc()

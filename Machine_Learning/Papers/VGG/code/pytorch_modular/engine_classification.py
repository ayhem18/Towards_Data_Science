import os
from _collections_abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Union

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .pytorch_utilities import get_default_device

HOME = os.getcwd()


def accuracy(y_pred: torch.tensor, y: torch.tensor) -> float:
    return (y_pred == y).type(torch.float32).mean().item()


def train_per_epoch(model: nn.Module,
                    dataloader: DataLoader,
                    loss_fn: nn.Module,
                    optimizer: torch.optim.Optimizer,
                    output_layer: nn.Module,
                    metrics: Union[Sequence[callable], callable] = None,
                    device: str = None) -> tuple:
    # first set the default value of the device parameter
    if device is None:
        device = get_default_device()

    # the default metric for classification is accuracy
    if metrics is None:
        metrics = accuracy

    if not isinstance(metrics, Sequence):
        metrics = [metrics]

    # put the mode in train mode
    model.train()

    # set up the train loss of a model
    train_loss = 0
    train_metrics = [0 for _ in metrics]

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(torch.long).to(device)  # convert to Long Type

        # make sure to un-squeeze 'y' if it is only 1 dimensions
        if len(y.shape) == 1:
            y = torch.unsqueeze(y, dim=-1)

        optimizer.zero_grad()
        # get the forward pass first
        y_pred = model(X)

        # calculate the loss
        batch_loss = loss_fn(y_pred, y.float())
        batch_loss.backward()
        # add the batch loss to the general training loss
        optimizer.step()

        train_loss += batch_loss.item()
        y_pred_class = output_layer(y_pred)

        # calculate the different metrics needed:
        metrics_results = [m(y_pred_class, y) for m in metrics]

        # add the batch metrics to the train metrics in general
        for index, mr in enumerate(metrics_results):
            train_metrics[index] += mr

    # adjust metrics to get the average loss and average metrics
    train_loss = train_loss / len(dataloader)
    train_metrics = tuple([m / len(dataloader) for m in train_metrics])

    return (train_loss,) + train_metrics


def test_per_epoch(model: nn.Module,
                   dataloader: DataLoader,
                   loss_fn: nn.Module,
                   output_layer: nn.Module,
                   metrics: Union[Sequence[callable], callable] = None,
                   device: str = None) -> tuple:
    # set the device
    if device is None:
        device = get_default_device()

    # the default metric for classification is accuracy
    if metrics is None:
        metrics = accuracy

    if not isinstance(metrics, Sequence):
        metrics = [metrics]

    # put the model to the evaluation mode
    model.eval()

    # Setup test loss and test accuracy values
    test_loss = 0
    metric_values = [0 for _ in metrics]

    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for _, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            y_pred = model(X)

            # 2. Calculate and accumulate loss
            loss = loss_fn(y_pred, y)
            test_loss += loss.item()

            labels = output_layer(y_pred)
            # calculate the different metrics needed:
            metrics_results = [m(labels, y) for m in metrics]

            # add the batch metrics to the train metrics in general
            for index, mr in enumerate(metrics_results):
                metric_values[index] += mr

    # adjust metrics to get the average loss and average metrics
    loss = test_loss / len(dataloader)
    metric_values = tuple([m / len(dataloader) for m in metric_values])

    return (loss,) + metric_values


def create_summary_writer(experiment_name: str, model_name: str, parent_dir: Union[str, Path] = None,
                          extra_details: Union[list[str], str] = None) -> SummaryWriter:
    # Get timestamp of current date (all experiments on certain day live in same folder)
    current_time = datetime.now()
    current_hour = current_time.hour
    current_minute = current_time.minute
    timestamp = datetime.now().strftime("%Y-%m-%d")  # returns current date in YYYY-MM-DD format
    timestamp += f"-{current_hour}-{current_minute}"  # now it is much more detailed: better tracking

    if parent_dir is None:
        parent_dir = HOME

    log_dir = os.path.join(parent_dir, experiment_name, model_name)

    if extra_details is not None:
        # convert a single string to a list of only one element
        if isinstance(extra_details, str):
            extra_details = [extra_details]

        for detail in extra_details:
            log_dir = os.path.join(log_dir, str(detail))

    # create the directory if needed
    os.makedirs(log_dir, exist_ok=True)

    print(f"[INFO] Created SummaryWriter, saving to: {log_dir}...")
    return SummaryWriter(log_dir=log_dir)

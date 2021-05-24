from logging import info
from typing import Iterator, Tuple
from argparse import ArgumentParser
from config import defaults
import torch
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer, LightningModule

def train(model: LightningModule, datasets: Tuple[Iterator, Iterator], save: bool = True):
    parser = Trainer.add_argparse_args(ArgumentParser())
    args = parser.parse_args()

    (train, valid) = datasets
    trainer = Trainer.from_argparse_args(args)

    info(f'Training with {len(train)} batches')
    info(f'Validate with {len(valid)} batches')

    train.create_batches()
    valid.create_batches()

    trainer.fit(model, train_dataloader=train, val_dataloaders=valid)
    
    if save:
        torch.save(model, f"models/{modelname()}.mdl")

    return model


def modelname(name: str = "transformer"):
    return f'{name}-{defaults[name]["emb_size"]}x{defaults[name]["encoder"]["num_layers"]}x{defaults[name]["encoder"]["heads"]}'

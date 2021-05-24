from logging import info
from typing import Tuple
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer, LightningModule

def train(model: LightningModule, datasets, args):
    (train, valid) = datasets
    trainer = Trainer.from_argparse_args(args)

    info(f'Training with {len(train)} batches')
    info(f'Validate with {len(valid)} batches')

    train.create_batches()
    valid.create_batches()

    # for batch in train:
    #     print(batch.src)
    #     print("-------------------------------------------")
    #     batch.tgt

    # return
    trainer.fit(model, train_dataloader=train)
    return model

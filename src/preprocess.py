from logging import info
from torch.utils.data import DataLoader

from config import defaults
from src.pipelines.tokenization import tokenizer_from
from src.pipelines.dataset import TextDataset, FastIterator

def build_dataset(src_path: str, tgt_path: str, ftype: str = "text"):
    info('Building dataset')
    SOURCE = defaults["src_lang"]
    TARGET = defaults["tgt_lang"]

    src_vocab, tgt_vocab = tokenizer_from(src_lang=SOURCE, tgt_lang=TARGET)
    data_fields = [("src", src_vocab), ("tgt", tgt_vocab)]

    if ftype == "text":
        train, valid = TextDataset(src_path, tgt_path, fields=data_fields, columns=[SOURCE, TARGET])
    
    src_vocab.build_vocab(train, valid)
    tgt_vocab.build_vocab(train, valid)
    
    train_data = FastIterator(train, batch_size=defaults["train_batchlen"], train=True)
    valid_data = FastIterator(valid, batch_size=defaults["val_batchlen"])

    return (src_vocab, tgt_vocab), (train_data, valid_data)

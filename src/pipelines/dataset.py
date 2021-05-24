from logging import info
from typing import List, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchtext.legacy.data.dataset import TabularDataset
from torchtext.legacy.data.iterator import Iterator, batch
from torchtext.legacy.data.field import Field

from config import defaults

def TextDataset(src_path: str, tgt_path: str, fields: List[Tuple[str, Field]], columns: List, equalize: bool = False):
    info('Reading dataset files')
    source = open(src_path, encoding='utf-8').read().split('\n')
    target = open(tgt_path, encoding='utf-8').read().split('\n')

    raw = {
        columns[0]: [line for line in source], 
        columns[1]: [line for line in target]
    }

    df = pd.DataFrame(raw, columns=columns)
    
    df['src_len'] = df[columns[0]].str.count(' ')
    df['tgt_len'] = df[columns[1]].str.count(' ')

    if equalize:
        # remove very long sentences and sentences where translations are 
        # not of roughly equal length
        df = df.query('tgt_len < 80 & src_len < 80')
        df = df.query('tgt_len < src_len * 1.5 & tgt_len * 1.5 > src_len')

    train, valid = train_test_split(df, test_size=defaults["holdout"])

    train.to_csv("data/train.csv", index=False)
    valid.to_csv("data/valid.csv", index=False)

    datasets = TabularDataset.splits(
        path='data', 
        format='csv', 
        train='train.csv', validation='valid.csv', 
        fields=fields)

    return datasets


# code from http://nlp.seas.harvard.edu/2018/04/03/attention.html 
# read text after for description of what it does
global max_src_in_batch, max_tgt_in_batch

class FastIterator(Iterator):
    sort_key = lambda x: (len(x.src), len(x.tgt))

    def batch_size_fn(new, count, sofar):
        "Keep augmenting batch and calculate total number of tokens + padding."
        global max_src_in_batch, max_tgt_in_batch

        if count == 1:
            max_src_in_batch = 0
            max_tgt_in_batch = 0

        max_src_in_batch = max(max_src_in_batch,  len(new.src))
        max_tgt_in_batch = max(max_tgt_in_batch,  len(new.tgt) + 2)

        src_elements = count * max_src_in_batch
        tgt_elements = count * max_tgt_in_batch

        return max(src_elements, tgt_elements)
    
    def create_batches(self):
        if self.train:
            def pool(d, random_shuffler):
                for p in batch(d, self.batch_size * 100):
                    p_batch = batch(sorted(p, key=self.sort_key), self.batch_size, self.batch_size_fn)

                    for b in random_shuffler(list(p_batch)):
                        yield b

            self.batches = pool(self.data(), self.random_shuffler)
            
        else:
            self.batches = []
            for b in batch(self.data(), self.batch_size, self.batch_size_fn):
                self.batches.append(sorted(b, key=self.sort_key))

# While Torchtext is brilliant, it’s sort_key based batching leaves a little to be desired. 
# Often the sentences aren’t of the same length at all, and you end up feeding a lot of 
# padding into your network.

# Additionally, if your RAM can process say 1500 tokens each iteration, and your batch_size 
# is 20, then only when you have batches of length 75 will you be utilising all the memory. 

# An efficient batching mechanism would change the batch size depending on the sequence 
# length to make sure around 1500 tokens were being processed each iteration.

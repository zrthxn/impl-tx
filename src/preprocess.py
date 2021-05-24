from logging import info
from torchtext.legacy.data.iterator import Iterator, batch
from torchtext.legacy.data import TabularDataset
from torch.utils.data import DataLoader

from config import defaults
from src.pipelines.tokenization import tokenizer_from
from src.pipelines import dataset

def build_dataset(src_path: str, tgt_path: str, ftype: str = "text"):
    info('Building dataset')
    SOURCE = defaults["src_lang"]
    TARGET = defaults["tgt_lang"]

    src_vocab, tgt_vocab = tokenizer_from(src_lang=SOURCE, tgt_lang=TARGET)
    data_fields = [("src", src_vocab), ("tgt", tgt_vocab)]

    if ftype == "text":
        train, valid = dataset.from_text(src_path, tgt_path, fields=data_fields, columns=[SOURCE, TARGET])
    
    src_vocab.build_vocab(train, valid)
    tgt_vocab.build_vocab(train, valid)
    
    train_data = FastIterator(train, batch_size=200, 
        train=True,
        sort_key=lambda x: (len(x.src), len(x.tgt)))
    
    valid_data = FastIterator(valid, batch_size=100, 
        sort_key=lambda x: (len(x.src), len(x.tgt)))

    return (src_vocab, tgt_vocab), (train_data, valid_data)


# code from http://nlp.seas.harvard.edu/2018/04/03/attention.html 
# read text after for description of what it does
global max_src_in_batch, max_tgt_in_batch

class FastIterator(Iterator):
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
# padding into your network (as you can see with all the 1s in the last figure).

# Additionally, if your RAM can process say 1500 tokens each iteration, and your batch_size 
# is 20, then only when you have batches of length 75 will you be utilising all the memory. 

# An efficient batching mechanism would change the batch size depending on the sequence 
# length to make sure around 1500 tokens were being processed each iteration.
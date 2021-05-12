from config import defaults
from src.pipelines import tokenization, dataset
from torchtext.legacy.data.dataset import TabularDataset
from torchtext.legacy.data.iterator import Iterator, batch
from logging import info

def build_dataset(src_path, tgt_path):
    SOURCE = defaults["src_lang"]
    TARGET = defaults["tgt_lang"]

    info('Reading dataset files')
    source = open(src_path, encoding='utf-8').read().split('\n')
    target = open(tgt_path, encoding='utf-8').read().split('\n')

    src_tok, tgt_tok = tokenization.tokenizer_from(src_lang=SOURCE, tgt_lang=TARGET)

    raw_data = {
        SOURCE: [line for line in source], 
        TARGET: [line for line in target]
    }

    info('Building dataset')
    data_fields = [("src", src_tok), ("tgt", tgt_tok)]

    tr, vl = dataset.from_raw_csv(raw_data, columns=[SOURCE, TARGET])
    train, valid = TabularDataset.splits(path='data', train=tr, validation=vl, format='csv', fields=data_fields)
    
    src_tok.build_vocab(train, valid)
    tgt_tok.build_vocab(train, valid)

    train_iter = MyIterator(train, batch_size=20, sort_key=lambda x: len(x.tgt), shuffle=True)
    valid_iter = MyIterator(valid, batch_size=20, sort_key=lambda x: len(x.tgt), shuffle=True)

    return train_iter, valid_iter


def embedding():
    pass


# code from http://nlp.seas.harvard.edu/2018/04/03/attention.html 
# read text after for description of what it does
global max_src_in_batch, max_tgt_in_batch

def batch_size_fn(new, count, sofar):
    "Keep augmenting batch and calculate total number of tokens + padding."
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch,  len(new.English))
    max_tgt_in_batch = max(max_tgt_in_batch,  len(new.French) + 2)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)

class MyIterator(Iterator):
    def create_batches(self):
        if self.train:
            def pool(d, random_shuffler):
                for p in batch(d, self.batch_size * 100):
                    p_batch = batch(
                        sorted(p, key=self.sort_key),
                        self.batch_size, self.batch_size_fn)
                    for b in random_shuffler(list(p_batch)):
                        yield b
            self.batches = pool(self.data(), self.random_shuffler)
            
        else:
            self.batches = []
            for b in batch(self.data(), self.batch_size, self.batch_size_fn):
                self.batches.append(sorted(b, key=self.sort_key))

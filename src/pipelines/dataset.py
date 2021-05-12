from typing import List
from config import defaults
from sklearn.model_selection import train_test_split
import pandas as pd

def from_raw_csv(raw: dict, columns: List, equalize: bool = False):
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

    return "train.csv", "valid.csv"

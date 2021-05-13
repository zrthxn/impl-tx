from logging import info, warning
import os
import spacy
from torchtext.legacy.data.field import Field
from spacy.language import Language
from typing import Tuple


def _load_lang(lang):
    info(f'Loading SpaCy pipeline for {lang}')
    try:
        spacy.load(lang)
    except (OSError):
        warning(f'Pipeline for {lang} not found')
        choice = input('Download pipeline? (Y/n)')
        if choice == 'y' or choice == 'Y': 
            os.system(f'python -m spacy download {lang}')
    finally:
        return spacy.load(lang)
    

def tokenizer_from(src_lang: str, tgt_lang: str) -> Tuple[Field]:
    src_lang = _load_lang(src_lang)
    tgt_lang = _load_lang(tgt_lang)

    info('Building tokenizer')
    src_tok = build_tokenizer(src_lang)
    tgt_tok = build_tokenizer(tgt_lang)

    src = Field(tokenize=src_tok)
    tgt = Field(tokenize=tgt_tok)

    return src, tgt

def build_tokenizer(lang: Language):
    return lambda sentence: [tok.text for tok in lang.tokenizer(sentence)]

from typing import List


defaults = {
    # Source and target languages, used by SpaCy
    "src_lang": "en_core_web_trf",
    "tgt_lang": "de_dep_news_trf",
    "holdout": 0.2,

    # Default config options for spacy.Tokenizer
    "tokenizer": {
        "symbols": 40000,
        "args": {
            "mode": "aggressive",
            "joiner_annotate": True,
            "preserve_placeholders": True,
            "case_markup": True,
            "soft_case_regions": True,
            "preserve_segmented_tokens": True
        }
    },

    "vocabulary": {
        "data_type": "text",
        "share_vocab": False,
        "vocab_size_multiple": 1,
        "src_vocab_size": 30000,
        "tgt_vocab_size": 30000,
        "src_words_min_frequency": 1,
        "tgt_words_min_frequency": 1
    },

    # Default config options for Trainer
    "dropout": 0.1,
    "training": {
        "train_steps": 100000,
        "valid_steps": 4000,
        "save_checkpoint_steps": 4000
    },

    # Default config options for models.transformer
    "transformer": {
        "emb_size": 512,
        "learning_rate": 2,
        "seq_length": 10,
        "encoder" : {
            "num_layers": 6,
            "heads": 8,
            "expansion_size": 4096, 
            "dropout": 0.1, 
            "attention_dropout": 0.1,
            "max_relative_positions": 0
        },
        "decoder" : {
            "num_layers": 6,
            "heads": 8,
            "expansion_size": 4096, 
            "dropout": 0.1,
            "attention_dropout": 0.1,
            "max_relative_positions": 0,
            "self_attn_type": "average",
            "copy_attn": True,
            "aan_useffn": False,
            "full_context_alignment": False,
            "alignment_layer": 1,
            "alignment_heads": 0
        }
    }
}


def build_configuration(argv: List[str]):
    global defaults
    config = dict()

    for arg in argv:
        if arg.find("=") == -1:
            continue
        
        name, value = arg.split("=")
        keys = name.split("-")
        
        config = tree_traverse(defaults, keys, value)

    defaults = config


def tree_traverse(tree: dict, keys: List[str], value):
    key = keys.pop(0)

    if key in tree.keys():
        tree[key] = tree_traverse(tree[key], keys, value) if type(tree[key]) == dict else type(tree[key])(value)
        return tree

from torch import nn
from src.models.transformer import Transformer

class BERT(nn.Module):
    def __init__(self) -> None:
        super().__init__()

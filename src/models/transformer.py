import torch
from math import sqrt
from torch import nn, Tensor
from torch import functional as F
from config import defaults

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size: int, heads: int = 5):
        super(MultiHeadAttention, self).__init__()
        
        self.head_dim = emb_size // heads
        assert (self.head_dim * heads == emb_size), "Number of Heads should divide Embedding size evenly"

        self.emb_size = emb_size
        self.heads = heads
        
        self.keys_layer = nn.Linear(in_features=self.head_dim, out_features=self.head_dim)
        self.values_layer = nn.Linear(in_features=self.head_dim, out_features=self.head_dim)
        self.queries_layer = nn.Linear(in_features=self.head_dim, out_features=self.head_dim)

        self.out = nn.Linear(in_features=self.emb_size, out_features=self.emb_size)

    def forward(self, values: Tensor, keys: Tensor, queries: Tensor, mask):
        # Split the K, V, Q tensors into self.heads pieces
        keys = keys.reshape(-1, keys.shape[1], self.heads, self.head_dim)
        values = values.reshape(-1, values.shape[1], self.heads, self.head_dim)
        queries = queries.reshape(-1, queries.shape[1], self.heads, self.head_dim)

        keys = self.keys_layer(keys)
        values = self.values_layer(values)
        queries = self.queries_layer(queries)

        # Scaled Dot-Product Attention
        dot = torch.einsum("nqhd,nkhd->nqhd", queries, keys) / sqrt(self.emb_size)
        # keys:    (n examples, q len, heads, dim)
        # queries: (n examples, k len, heads, dim)
        # dot:     (n examples, q len, heads, dim)

        # Apply mask
        if mask is not None:
            dot.masked_fill(mask == 0, -1e20)
        
        attention = torch.softmax(dot, dim=3) * values
        attention = attention.reshape(-1, queries.shape[1], self.emb_size)
        # Concat by reshape

        out = self.out(attention)
        return out


class EncoderBlock(nn.Module):
    def __init__(self, emb_size: int, heads: int = 5, dropout: float = defaults["dropout"], expansion: int = 4):
        super(EncoderBlock, self).__init__()

        self.attention = MultiHeadAttention(emb_size, heads)
        self.feed_fwd = nn.Sequential(
            nn.Linear(in_features=emb_size, out_features=int(expansion*emb_size)), nn.ReLU,
            nn.Linear(in_features=int(expansion*emb_size), out_features=emb_size)
        )
        self.norm_a = nn.LayerNorm(emb_size)
        self.norm_b = nn.LayerNorm(emb_size)

        self.dropout = nn.Dropout(dropout)

    def forward(self, embedding, mask):
        attention = self.attention(embedding, embedding, embedding, mask)

        norm1 = self.norm_a(attention + embedding)
        norm1 = self.dropout(norm1)

        dense = self.feed_fwd(norm1)
        out = self.dropout(self.norm_b(dense + norm1))
        return out

class Encoder(nn.Module):
    def __init__(self, emb_size: int, heads: int = 5, layers: int = 6):
        super(Encoder, self).__init__()
        # Word embeddings stuff
        
        self.blocks = nn.ModuleList([
            EncoderBlock(emb_size, heads) for _ in range(layers)
        ])

    def forward(self, inputs, mask):
        # Word embeddings stuff

        for block in self.blocks:
            out = block(inputs, mask)

        return out


class DecoderBlock(nn.Module):
    def __init__(self, emb_size: int, heads: int = 5, dropout: float = defaults["dropout"], expansion: int = 4):
        super(DecoderBlock, self).__init__()

        self.attention = MultiHeadAttention(emb_size, heads)
        self.masked_attention = MultiHeadAttention(emb_size, heads)
        self.feed_fwd = nn.Sequential(
            nn.Linear(in_features=emb_size, out_features=int(expansion*emb_size)), nn.ReLU,
            nn.Linear(in_features=int(expansion*emb_size), out_features=emb_size)
        )

        self.outputs_norm = nn.LayerNorm(emb_size)
        self.inputs_norm_a = nn.LayerNorm(emb_size)
        self.inputs_norm_b = nn.LayerNorm(emb_size)

        self.dropout = nn.Dropout(dropout)

    def forward(self, in_embedding, out_embedding, in_mask, out_mask):
        out_attention = self.masked_attention(out_embedding, out_embedding, out_embedding, out_mask)
        query = self.outputs_norm(out_attention + out_embedding)
        query = self.dropout(query)

        in_attention = self.attention(in_embedding, in_embedding, query, in_mask)
        in_norm = self.inputs_norm_a(in_attention + query)
        in_norm = self.dropout(in_norm)

        dense = self.feed_fwd(in_norm)
        out = self.dropout(self.inputs_norm_b(dense + in_norm))
        out = self.dropout(out)
        return out

class Decoder(nn.Module):
    def __init__(self, emb_size: int, heads: int = 5, layers: int = 6):
        super(Decoder, self).__init__()
        # Word embeddings stuff
        
        self.blocks = nn.ModuleList([
            DecoderBlock(emb_size, heads) for _ in range(layers)
        ])


    def forward(self, inputs, outputs, mask):
        # Word embeddings stuff

        for block in self.blocks:
            out = block(inputs, outputs, mask)

        return out


class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()

    def forward(self, x):
        pass

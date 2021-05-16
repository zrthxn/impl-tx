import torch
from math import cos, sin, sqrt
from torch import nn, autograd, Tensor
from torchtext.vocab import Vocab
from config import defaults

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size: int, heads: int):
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


class PositionalEncoder(nn.Module):
    def __init__(self, emb_size: int, seq_length: int):
        super().__init__()
        self.emb_size = emb_size
        
        # create constant 'pe' matrix with values dependant on pos and i
        pe = torch.zeros(seq_length, emb_size)
        for pos in range(seq_length):
            for i in range(0, emb_size, 2):
                pe[pos, i] = sin(pos / (10000 ** ((2 * i)/emb_size)))
                pe[pos, i + 1] = cos(pos / (10000 ** ((2 * (i + 1))/emb_size)))
                
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # make embeddings relatively larger
        x = x * sqrt(self.emb_size)

        #add constant to embedding
        seq_len = x.size(1)
        
        x = x + autograd.Variable(self.pe[:,:seq_len], requires_grad=False)
        return x


class EncoderBlock(nn.Module):
    def __init__(self, emb_size: int, heads: int, 
            dropout: float = defaults["transformer"]["encoder"]["dropout"], 
            attention_dropout: float = defaults["transformer"]["encoder"]["attention_dropout"], 
            expansion_size: int = defaults["transformer"]["encoder"]["expansion_size"]):
        super(EncoderBlock, self).__init__()

        self.attention = MultiHeadAttention(emb_size, heads)
        self.feed_fwd = nn.Sequential(
            nn.Linear(in_features=emb_size, out_features=expansion_size), 
            nn.ReLU(),
            nn.Linear(in_features=expansion_size, out_features=emb_size)
        )
        self.norm_a = nn.LayerNorm(emb_size)
        self.norm_b = nn.LayerNorm(emb_size)

        self.attention_dropout = nn.Dropout(attention_dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, embedding, mask):
        attention = self.attention(embedding, embedding, embedding, mask)

        norm1 = self.norm_a(attention + embedding)
        norm1 = self.attention_dropout(norm1)

        dense = self.feed_fwd(norm1)
        out = self.norm_b(dense + norm1)
        out = self.dropout(out)
        return out

class Encoder(nn.Module):
    def __init__(self, vocab_size: int, emb_size: int, seq_length: int,
            heads: int = defaults["transformer"]["encoder"]["heads"], 
            layers: int = defaults["transformer"]["encoder"]["num_layers"]):
        super(Encoder, self).__init__()
        
        self.inp_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=emb_size)
        self.pos_encoding = PositionalEncoder(emb_size=emb_size, seq_length=seq_length)
        
        self.layers = nn.ModuleList([
            EncoderBlock(emb_size, heads) for _ in range(layers)
        ])

        self.dropout = nn.Dropout(defaults["transformer"]["encoder"]["dropout"])

    def forward(self, inputs, mask):
        inputs = self.inp_embedding(inputs)
        inputs = self.pos_encoding(inputs)
        inputs = self.dropout(inputs)

        for layer in self.layers:
            inputs = layer(inputs, mask)

        return inputs


class DecoderBlock(nn.Module):
    def __init__(self, emb_size: int, heads: int, 
            dropout: float = defaults["transformer"]["decoder"]["dropout"], 
            attention_dropout: float = defaults["transformer"]["decoder"]["attention_dropout"], 
            expansion_size: int = defaults["transformer"]["decoder"]["expansion_size"]):
        super(DecoderBlock, self).__init__()

        self.attention = MultiHeadAttention(emb_size, heads)
        self.masked_attention = MultiHeadAttention(emb_size, heads)
        self.feed_fwd = nn.Sequential(
            nn.Linear(in_features=emb_size, out_features=expansion_size), 
            nn.ReLU(),
            nn.Linear(in_features=expansion_size, out_features=emb_size)
        )

        self.outputs_norm = nn.LayerNorm(emb_size)
        self.inputs_norm_a = nn.LayerNorm(emb_size)
        self.inputs_norm_b = nn.LayerNorm(emb_size)

        self.attention_dropout = nn.Dropout(attention_dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, in_embedding, out_embedding, in_mask, out_mask):
        out_attention = self.masked_attention(out_embedding, out_embedding, out_embedding, out_mask)
        query = self.outputs_norm(out_attention + out_embedding)
        query = self.attention_dropout(query)

        in_attention = self.attention(in_embedding, in_embedding, query, in_mask)
        in_norm = self.inputs_norm_a(in_attention + query)
        in_norm = self.dropout(in_norm)

        dense = self.feed_fwd(in_norm)
        out = self.inputs_norm_b(dense + in_norm)
        out = self.dropout(out)
        return out

class Decoder(nn.Module):
    def __init__(self, vocab_size: int, emb_size: int, seq_length: int,
            heads: int = defaults["transformer"]["decoder"]["heads"],  
            layers: int = defaults["transformer"]["decoder"]["num_layers"]):
        super(Decoder, self).__init__()
        
        self.out_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=emb_size)
        self.pos_encoding = PositionalEncoder(emb_size=emb_size, seq_length=seq_length)
        
        self.layers = nn.ModuleList([
            DecoderBlock(emb_size, heads) for _ in range(layers)
        ])

        self.scale = nn.Linear(in_features=emb_size, out_features=seq_length)
        self.dropout = nn.Dropout(defaults["transformer"]["decoder"]["dropout"])

    def forward(self, inputs, outputs, in_mask, out_mask):
        outputs = self.out_embedding(outputs)
        outputs = self.pos_encoding(outputs)
        outputs = self.dropout(outputs)

        for layer in self.layers:
            outputs = layer(inputs, outputs, in_mask, out_mask)
        
        outputs = self.scale(outputs)
        return outputs


class Transformer(nn.Module):
    def __init__(self, 
            src_vocab: Vocab, 
            tgt_vocab: Vocab, 
            seq_length: int,
            emb_size: int = defaults["transformer"]["emb_size"]):
        super(Transformer, self).__init__()

        src_vocab_size = src_vocab.vocab.__len__()
        tgt_vocab_size = tgt_vocab.vocab.__len__()

        self.encoder = Encoder(vocab_size=src_vocab_size, emb_size=emb_size, seq_length=seq_length)
        self.decoder = Decoder(vocab_size=tgt_vocab_size, emb_size=emb_size, seq_length=seq_length)

        self.src_pad = src_vocab.vocab.stoi["<pad>"]
        self.tgt_pad = tgt_vocab.vocab.stoi["<pad>"]

    def _src_mask(self, src):
        mask = (src != self.src_pad).unsqeeze(1)
        return mask

    def _tgt_mask(self, tgt):
        batch, embed = tgt.shape
        mask = torch.tril(torch.ones((embed, embed))).expand(batch, 1, embed, embed)
        return mask

    def forward(self, src, tgt):
        src_mask = self._src_mask(src)
        tgt_mask = self._tgt_mask(tgt)

        enc = self.encoder(src, src_mask)
        dec = self.decoder(enc, tgt, src_mask, tgt_mask)

        return dec

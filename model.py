import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embeddings = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embeddings(x) * self.d_model ** 0.5


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # Create a tensor of shape (seq_len, d_model)
        pe = torch.empty(seq_len, d_model)
        # Create a tensor of shape (seq_len, 1)
        position = torch.arange(seq_len, dtype=torch.float).unsqueeze(1)
        # Create a tensor of shape (1, d_model/2)
        div_term = torch.exp(torch.arange(0., d_model, 2.) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Create a tensor of shape [1, x.shape[1], d_model]
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)


class LayerNorm(nn.Module):
    """ Layer Normalization

    just use torch.nn.LayerNorm
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.net(x)


class Head(nn.Module):
    def __init__(self, d_model: int, d_k: int, dropout: float):
        super().__init__()
        self.w_q = nn.Linear(d_model, d_k, bias=False)
        self.w_k = nn.Linear(d_model, d_k, bias=False)
        self.w_v = nn.Linear(d_model, d_k, bias=False)
        self.dropout = nn.Dropout(dropout)

    def attention(self, query, key, value, mask, dropout: nn.Dropout):
        d_k = query.size(-1)
        attention_scores = (query @ key.transpose(-2, -1)) * d_k ** -0.5
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask, float('-inf'))
        attention_scores = F.softmax(attention_scores, dim=-1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        return attention_scores @ value

    def forward(self, query, key, value, mask=None):
        q = self.w_q(query)
        k = self.w_k(key)
        v = self.w_v(value)
        return self.attention(q, k, v, mask, self.dropout)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, heads: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.heads = heads
        assert d_model % heads == 0
        d_k = d_model // heads
        self.heads = nn.ModuleList([Head(d_model, d_k, dropout) for _ in range(self.heads)])

    def forward(self, q, k, v, mask=None):
        return torch.cat([head(q, k, v, mask) for head in self.heads], dim=-1)


class AddAndNorm(nn.Module):
    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderBlock(nn.Module):
    def __init__(self, self_attention: MultiHeadAttention, feed_forward: FeedForward, dropout: float):
        super().__init__()
        self.self_attention = self_attention
        self.feed_forward = feed_forward
        d_model = self_attention.d_model
        self.ln1 = AddAndNorm(d_model, dropout)
        self.ln2 = AddAndNorm(d_model, dropout)

    def forward(self, x, src_mask):
        x = self.ln1(x, lambda x: self.self_attention(x, x, x, src_mask))
        x = self.ln2(x, lambda x: self.feed_forward(x))
        return x


class Encoder(nn.Module):
    def __init__(self, encoders: list[EncoderBlock], d_model: int):
        super().__init__()
        self.encoders = nn.ModuleList(encoders)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask):
        for encoder in self.encoders:
            x = encoder(x, mask)
        return self.norm(x)


class DecoderBlock(nn.Module):
    def __init__(self, self_attention_block, cross_attention_block, feed_forward_block, dropout):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        d_model = self_attention_block.d_model
        self.ln1 = AddAndNorm(d_model, dropout)
        self.ln2 = AddAndNorm(d_model, dropout)
        self.ln3 = AddAndNorm(d_model, dropout)

    def forward(self, x, encoder_output, mask_src, mask_dst):
        x = self.ln1(x, lambda xx: self.self_attention_block(xx, xx, xx, mask_dst))
        x = self.ln2(x, lambda xx: self.cross_attention_block(xx, encoder_output, encoder_output, mask_src))
        x = self.ln3(x, lambda xx: self.feed_forward_block(xx))
        return x


class Decoder(nn.Module):
    def __init__(self, decoders: list[DecoderBlock], d_model: int):
        super().__init__()
        self.decoders = nn.ModuleList(decoders)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, encoder_output, mask_src, mask_dst):
        for decoder in self.decoders:
            x = decoder(x, encoder_output, mask_src, mask_dst)
        return self.norm(x)


class Transformer(nn.Module):
    def __init__(self, encoder, decoder, embed_src, embed_dst, pos_src, pos_dst, proj):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.embed_src = embed_src
        self.embed_dst = embed_dst
        self.pos_src = pos_src
        self.pos_dst = pos_dst
        self.proj = proj

    def encode(self, x, mask_src):
        x = self.embed_src(x)
        x = self.pos_src(x)
        return self.encoder(x, mask_src)

    def decode(self, x, encoder_output, mask_src, mask_dst):
        x = self.embed_dst(x)
        x = self.pos_dst(x)
        return self.decoder(x, encoder_output, mask_src, mask_dst)

    def project(self, x):
        x = self.proj(x)
        return F.softmax(x, dim=-1)


def build_transformer(vocab_size_src, vocab_size_dst, seq_len, d_model: int = 512, N: int = 6, heads: int = 8,
                      dropout: float = 0.1,
                      d_ff: int = 2048):
    embed_src = InputEmbeddings(d_model, vocab_size_src)
    embed_dst = InputEmbeddings(d_model, vocab_size_dst)
    pos_src = PositionalEncoding(d_model, seq_len, dropout)
    pos_dst = PositionalEncoding(d_model, seq_len, dropout)
    encoders = [
        EncoderBlock(
            MultiHeadAttention(d_model, heads, dropout),
            FeedForward(d_model, d_ff, dropout),
            dropout)
        for _ in range(N)
    ]
    encoder = Encoder(encoders, d_model)
    decoders = [DecoderBlock(
        MultiHeadAttention(d_model, heads, dropout),
        MultiHeadAttention(d_model, heads, dropout),
        FeedForward(d_model, d_ff, dropout),
        dropout
    ) for _ in range(N)]
    decoder = Decoder(decoders, d_model)
    proj = nn.Linear(d_model, vocab_size_dst)
    transformer = Transformer(encoder, decoder, embed_src, embed_dst, pos_src, pos_dst, proj)

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer

# Implementation of Transformer from scratch 

import math
import torch
import torch.nn as nn


class MultiheadAttention(nn.Module):
    # scaled, multi-head attention with scaling on Q, K and V matrices
    def __init__(self, num_heads, d_k, d_v, d_m):
        super(MultiheadAttention, self).__init__()
        self.proj_K = nn.ModuleList([nn.Linear(d_m, d_k) for _ in range(num_heads)])
        self.proj_Q = nn.ModuleList([nn.Linear(d_m, d_k) for _ in range(num_heads)])
        self.proj_V = nn.ModuleList([nn.Linear(d_m, d_v) for _ in range(num_heads)])
        self.proj_O = nn.Linear(d_v*num_heads, d_m)
        self.softmax = nn.Softmax(dim=1)

    # Q for queries, K for keys and V for values
    def forward(self, Q, K, V):
        # Q, K, V: B x N x d_m
        # a list to hold weighted_V on the fly. BETTER WAY?
        weighted_Vs = []
        for head_idx in range(len(self.proj_K)):
            # B x N x d_k
            Q_proj = self.proj_Q[head_idx](Q)
            # B x N x d_k
            K_proj = self.proj_K[head_idx](K)
            # B x N x d_v
            V_proj = self.proj_V[head_idx](V)
            # B x N x N
            scaled_weight = torch.bmm(Q_proj, K_proj.permute(0, 2, 1)) / math.sqrt(Q_proj.size(2))
            scaled_weight = self.softmax(scaled_weight)
            # B x N x d_v
            # each row of weighted_V is a weighted sum of V_proj where weight is determined by K_proj and Q_proj
            weighted_V = torch.bmm(scaled_weight, V_proj)
            weighted_Vs.append(weighted_V)
        # B x N x d_v*num_heads
        cat_V = torch.cat(weighted_Vs, dim=2)
        # B x N x d_m
        out = self.proj_O(cat_V)

        return out


class FeedforwardNetwork(nn.Module):
    # Positionwise feedforward neural network 
    # performs more projection in addition to the output projection W_O in MultiheadAttention
    def __init__(self, d_m, d_hidden, dropout=0.0):
        super(FeedforwardNetwork, self).__init__()
        self.fc1 = nn.Linear(d_m, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_m)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, x):
        # x: B x N x d_m
        # B x N x d_h
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        # B x N x d_m
        out = self.fc2(out)

        return out

class PositionEncoding(nn.Module):
    # encode position information of input words by overlaying sinusoidal signal
    def __init__(self, d_m, max_len=1000, dropout=0.0):
        super(PositionEncoding, self).__init__()
        # all even numbers from 0 to 1-d_m
        feature_idxs = torch.arange(0, d_m, 2).double()
        # d_m/2
        denom = torch.pow(1e4, -feature_idxs/d_m)
        # max_len x 1
        pos = torch.arange(0, max_len).unsqueeze(1).double()
        pe = torch.zeros(max_len, d_m)
        # outer product between pos and denom
        # max_len x d_m/2
        division = pos * denom
        pe[:, 0::2] = torch.sin(division)
        pe[:, 1::2] = torch.cos(division)
        # max_len x d_m
        self.pe = nn.Parameter(pe)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # x: B x N x d_m
        act_len = x.size(1)
        # B x N x d_m
        out = x + self.pe[:act_len, :].unsqueeze(0)
        out = self.dropout(out)

        return out

class LayerNorm(nn.Module):
    # layer normalization across features
    # like batch norm, requires scale(gamma) and bias(beta) terms
    # to make sure we can represent identity transform
    def __init__(self, d_m, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gammas = nn.Parameter(torch.ones(d_m))
        self.betas = nn.Parameter(torch.zeros(d_m))
        self.eps = eps
    
    def forward(self, x):
        # x: B x N x d_m
        # B x N x d_m
        means = x.mean(dim=2, keepdim=True)
        # B x N x d_m
        stds = x.std(dim=2, keepdim=True)
        # B x N x d_m
        out = self.gammas * (x - means) / (stds + self.eps) + self.betas

        return out

class Embeddings(nn.Module):
    def __init__(self, vocab_size, d_m):
        super(Embeddings, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_m)
        self.d_m = d_m
        self.vocab_size = vocab_size

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_m)

class EncoderLayer(nn.Module):
    def __init__(self, num_heads, d_k, d_v, d_m, d_hidden, dropout=0.0):
        super(EncoderLayer, self).__init__()
        self.attention = MultiheadAttention(num_heads, d_k, d_v, d_m)
        self.feedforward = FeedforwardNetwork(d_m, d_hidden, dropout=dropout)
        self.norm1 = LayerNorm(d_m)
        self.norm2 = LayerNorm(d_m)

    def forward(self, input_enc):
        # x: B x N x d_m
        # B x N x d_m
        out = self.attention(input_enc, input_enc, input_enc)
        # residual connection
        out = out + input_enc
        # B x N x d_m
        self_att_enc = self.norm1(out)

        # B x N x d_m
        out = self.feedforward(out)
        # residual connection
        out = out + self_att_enc
        # B x N x d_m
        out = self.norm2(out)

        return out
        

class DecoderLayer(nn.Module):
    def __init__(self, num_heads, d_k, d_v, d_m, d_hidden, dropout=0.0):
        super(DecoderLayer, self).__init__()
        # TODO: ADD MASKING
        self.masked_attention = MultiheadAttention(num_heads, d_k, d_v, d_m)
        self.attention = MultiheadAttention(num_heads, d_k, d_v, d_m)
        self.feedforward = FeedforwardNetwork(d_m, d_hidden, dropout=dropout)
        self.norm1 = LayerNorm(d_m)
        self.norm2 = LayerNorm(d_m)
        self.norm3 = LayerNorm(d_m)

    def forward(self, input_enc, output_enc):
        # x: B x N x d_m
        # B x N x d_m
        out = self.masked_attention(Q=output_enc, K=output_enc, V=output_enc)
        # residual connection
        out = out + output_enc
        # B x N x d_m
        self_att_enc = self.norm1(out)

        # B x N x d_m
        out = self.attention(Q=output_enc, K=input_enc, V=input_enc)
        # residual connection
        out = out + self_att_enc
        # B x N x d_m
        cross_att_enc = self.norm2(out)

        # B x N x d_m
        out = self.feedforward(out)
        # residual connection
        out = out + cross_att_enc
        # B x N x d_m
        out = self.norm3(out)

        return out

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, num_layers=1, d_k=64, d_v=64, d_m=512, d_hidden=1024, num_heads=8, dropout=0.0):
        super(Transformer, self).__init__()
        # used for input embedding and output embedding
        self.src_embedding = Embeddings(src_vocab_size, d_m)
        self.tgt_embedding = Embeddings(tgt_vocab_size, d_m)
        self.pos_enc = PositionEncoding(d_m)
        self.encoder_layers = nn.Sequential(*[
            EncoderLayer(num_heads, d_k, d_v, d_m, d_hidden, dropout) for _ in range(num_layers)
        ])
        self.decoder_layers = nn.Sequential(*[
            DecoderLayer(num_heads, d_k, d_v, d_m, d_hidden, dropout) for _ in range(num_layers)
        ])
        self.proj = nn.Linear(d_m, tgt_vocab_size)
    
    def forward(self, inputs, outputs):
        # inputs: B x N_in
        # outputs: B x N_out
        # where input/output < vocab_size
        assert torch.all(inputs < self.src_embedding.vocab_size)
        assert torch.all(outputs < self.tgt_embedding.vocab_size)
        # B x N x d_m
        input_enc = self.src_embedding(inputs)
        # B x N x d_m
        input_enc = self.pos_enc(input_enc)
        # B x N x d_m
        input_enc = self.encoder_layers(input_enc)
        output_enc = self.tgt_embedding(outputs)
        # B x N x d_m
        output_enc = self.pos_enc(output_enc)
        # B x N x d_m
        for decoder_layer in self.decoder_layers:
            # input_enc stays the same while output_enc gets updated
            # what happens if we match sublayers of encoder to sublayers of decoder?
            output_enc = decoder_layer(input_enc, output_enc)
        # B x N x vocab_size
        out = self.proj(output_enc)

        return out





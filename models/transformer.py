# Implementation of Transformer from scratch 
import math
import torch
import torch.nn as nn
import numpy as np

import pdb

def get_non_pad_mask(seq_input, pad_label):
    # seq_input: B x N
    # B x N x 1
    return seq_input.ne(pad_label).float().unsqueeze(-1)


# In decoder self attention, representation of each word need to mask out subsequent words to prevent look-ahead.
# This function gives a square matrix for each sentence to mask out subsequent words. 
def get_subsequent_mask(seq_output):
    # enc: B x N
    max_len = seq_output.size(1)
    # N x N
    subsequent_mask = torch.triu(
        torch.ones((max_len, max_len), 
            device=seq_output.device,
            # turn into ByteTensor
            dtype=torch.uint8),
        # representation of w_i could depend on itself, as we are predicting for next word w_{i+1}
        diagonal=1
    )
    # B x N x N
    subsequent_mask = subsequent_mask.unsqueeze(0).repeat(seq_output.size(0), 1, 1)
    return subsequent_mask

# representation of each word does not need information from padding
def get_att_padding_mask(seq_q, seq_k, pad_label):
    # seq_q: B x N_q
    # seq_k: B x N_k
    max_len_q = seq_q.size(1)
    max_len_k = seq_k.size(1)
    # B x N_k
    padding_mask = seq_k.eq(pad_label)
    # B x N_q x N_k
    padding_mask = padding_mask.unsqueeze(1).repeat(1, max_len_q, 1)
    return padding_mask

class MultiheadAttention(nn.Module):
    # scaled, multi-head attention with scaling on Q, K and V matrices
    def __init__(self, num_heads, d_k, d_v, d_m, dropout=0.1):
        super(MultiheadAttention, self).__init__()
        self.proj_K = nn.ModuleList([nn.Linear(d_m, d_k) for _ in range(num_heads)])
        self.proj_Q = nn.ModuleList([nn.Linear(d_m, d_k) for _ in range(num_heads)])
        self.proj_V = nn.ModuleList([nn.Linear(d_m, d_v) for _ in range(num_heads)])
        self.proj_O = nn.Linear(d_v*num_heads, d_m)
        self.softmax = nn.Softmax(dim=2)
        self.dropout = nn.Dropout(dropout)

    # attention representation of word w_i is always a function of the
    # word w_i itself, which is required by the weight matrix to be
    # the Query vector. 
    def forward(self, Q, K, V, zero_mask=None):
        # Q(query): B x N_q x d_m
        # K(key), V(value): B x N_k x d_m
        # zero_mask: B x N_q x N_k
        # a list to hold weighted_V on the fly. BETTER WAY?
        weighted_Vs = []
        for head_idx in range(len(self.proj_K)):
            # B x N_q x d_k
            Q_proj = self.proj_Q[head_idx](Q)
            # B x N_k x d_k
            K_proj = self.proj_K[head_idx](K)
            # B x N_k x d_v
            V_proj = self.proj_V[head_idx](V)
            # B x N_q x N_k
            scaled_weight = torch.bmm(Q_proj, K_proj.permute(0, 2, 1)) / math.sqrt(Q_proj.size(2))
            # VERY IMPORTANT to first mask then softmax, which ensures two things:
            # 1. representation of w_i does not depends on any word w_{i+k} after w_i,
            # as otherwise softmax representation will include future words' information
            # 2. total weights of all Value vectors sum to 1

            # for a word w_i in a sentence, if the jth weight value is masked to 0, then 
            # the representation of w_i from this MultiheadAttention layer excludes information from word w_j.
            if zero_mask is not None:
                # AVOID in-place operation
                # scaled_weight[zero_mask] = -np.inf
                scaled_weight = scaled_weight.masked_fill(zero_mask, -np.inf)
            scaled_weight = self.softmax(scaled_weight)
            scaled_weight = self.dropout(scaled_weight)
            # B x N_q x d_v
            # each row of weighted_V is a weighted sum of V_proj where weight is determined by K_proj and Q_proj
            weighted_V = torch.bmm(scaled_weight, V_proj)
            weighted_Vs.append(weighted_V)
        # B x N_q x d_v*num_heads
        cat_V = torch.cat(weighted_Vs, dim=2)
        # B x N_q x d_m
        out = self.proj_O(cat_V)

        return out


class FeedforwardNetwork(nn.Module):
    # Positionwise feedforward neural network 
    # performs more projection in addition to the output projection W_O in MultiheadAttention
    def __init__(self, d_m, d_hidden, dropout=0.1):
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
    def __init__(self, d_m, max_len=1000, dropout=0.1):
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
        self.pe = nn.Parameter(pe, requires_grad=False)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # x: B x N x d_m
        act_len = x.size(1)
        # B x N x d_m
        out = x + self.pe[:act_len, :].unsqueeze(0)
        out = self.dropout(out)

        return out

class LayerNorm(nn.Module):
    # layer normalization across features: each word embedding gets normalized
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
    def __init__(self, vocab_size, d_m, padding_idx=None):
        super(Embeddings, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_m, padding_idx=padding_idx)
        self.d_m = d_m
        self.vocab_size = vocab_size

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_m)

class EncoderLayer(nn.Module):
    def __init__(self, num_heads, d_k, d_v, d_m, d_hidden, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attention = MultiheadAttention(num_heads, d_k, d_v, d_m)
        self.feedforward = FeedforwardNetwork(d_m, d_hidden, dropout=dropout)
        self.norm1 = LayerNorm(d_m)
        self.norm2 = LayerNorm(d_m)

    def forward(self, input_enc, self_att_mask=None, non_pad_mask=None):
        # x: B x N x d_m
        # padding_mask: B x N x N, leave padding tokens out when encoding words
        # B x N x d_m
        out = self.self_attention(input_enc, input_enc, input_enc, zero_mask=self_att_mask)
        # residual connection
        out = out + input_enc
        # B x N x d_m
        self_att_enc = self.norm1(out)
        self_att_enc *= non_pad_mask

        # B x N x d_m
        out = self.feedforward(out)
        # residual connection
        out = out + self_att_enc
        # B x N x d_m
        out = self.norm2(out)
        out *= non_pad_mask

        return out
        

class DecoderLayer(nn.Module):
    def __init__(self, num_heads, d_k, d_v, d_m, d_hidden, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiheadAttention(num_heads, d_k, d_v, d_m)
        self.enc_dec_attention = MultiheadAttention(num_heads, d_k, d_v, d_m)
        self.feedforward = FeedforwardNetwork(d_m, d_hidden, dropout=dropout)
        self.norm1 = LayerNorm(d_m)
        self.norm2 = LayerNorm(d_m)
        self.norm3 = LayerNorm(d_m)

    def forward(self, input_enc, output_enc, self_att_mask=None, enc_dec_att_mask=None, non_pad_mask=None):
        # x: B x N x d_m
        # need to mask out following words for each word's representation
        # B x N x d_m
        out = self.self_attention(Q=output_enc, K=output_enc, V=output_enc, zero_mask=self_att_mask)
        # residual connection
        out = out + output_enc
        # B x N x d_m
        self_att_enc = self.norm1(out)
        # mask out padding
        self_att_enc *= non_pad_mask

        # B x N x d_m
        out = self.enc_dec_attention(Q=self_att_enc, K=input_enc, V=input_enc, zero_mask=enc_dec_att_mask)
        # residual connection
        out = out + self_att_enc
        # B x N x d_m
        cross_att_enc = self.norm2(out)
        cross_att_enc *= non_pad_mask

        # B x N x d_m
        out = self.feedforward(cross_att_enc)
        # residual connection
        out = out + cross_att_enc
        # B x N x d_m
        out = self.norm3(out)
        out *= non_pad_mask

        return out

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, src_vocab_vectors=None, tgt_vocab_vectors=None, 
                num_layers=1, d_k=64, d_v=64, d_m=512, d_hidden=1024, num_heads=8, dropout=0.1, pad_label=1):
        super(Transformer, self).__init__()
        # used for input embedding and output embedding
        # self.src_embedding = Embeddings(src_vocab_size, d_m, padding_idx=pad_label)
        self.src_embedding = nn.Linear(src_vocab_size, d_m, bias=False)
        self.tgt_embedding = Embeddings(tgt_vocab_size, d_m, padding_idx=pad_label)
        # if src_vocab_vectors is not None:
        #     assert d_m == src_vocab_vectors.size(1)
        #     assert src_vocab_size == src_vocab_vectors.size(0)
        #     self.src_embedding.embedding.weight.data.copy_(src_vocab_vectors)
        if tgt_vocab_vectors is not None:
            assert d_m == tgt_vocab_vectors.size(1)
            assert tgt_vocab_size == tgt_vocab_vectors.size(0)
            self.tgt_embedding.embedding.weight.data.copy_(tgt_vocab_vectors)
        self.pos_enc = PositionEncoding(d_m)
        self.encoder_layers = nn.Sequential(*[
            EncoderLayer(num_heads, d_k, d_v, d_m, d_hidden, dropout) for _ in range(num_layers)
        ])
        self.decoder_layers = nn.Sequential(*[
            DecoderLayer(num_heads, d_k, d_v, d_m, d_hidden, dropout) for _ in range(num_layers)
        ])
        # reuse the target embedding matrix as a form of regularization
        self.proj = nn.Linear(d_m, tgt_vocab_size)
        self.proj.weight = self.tgt_embedding.embedding.weight
        self.pad_label = pad_label
    
    def forward(self, inputs, outputs):
        # inputs: B x N_in
        # where if inputs[b, i] > 0, the word i is active
        B, N_in = inputs.size()

        # B x N_in x N_in
        diag_inputs = torch.zeros(B, N_in, N_in).to(inputs.device)
        for b in range(B):
            diag_inputs[b] = torch.diag(inputs[b])
        # pad_mask = inputs.eq(0)
        # inputs[pad_mask] = self.pad_label

        # outputs: B x N_out
        # where input/output < vocab_size
        assert torch.all(outputs < self.tgt_embedding.vocab_size)
        # use the first N-1 words(given output) to predict last N-1 words(target) 
        outputs = outputs[:, :-1]
        # B x N_in x d_m
        input_enc = self.src_embedding(diag_inputs)
        # B x N_in x d_m
        input_enc = self.pos_enc(input_enc)
        # encoder padding mask
        # B x N_in x 1
        enc_non_pad_mask = get_non_pad_mask(inputs, self.pad_label)
        # encoder self attention mask, leave out padding token
        # B x N_in x N_in
        enc_self_att_mask = get_att_padding_mask(seq_q=inputs, seq_k=inputs, pad_label=self.pad_label)
        for encoder_layer in self.encoder_layers:
            # B x N_in x d_m
            input_enc = encoder_layer(input_enc, self_att_mask=enc_self_att_mask, non_pad_mask=enc_non_pad_mask)
        # B x N_out x d_m
        output_enc = self.tgt_embedding(outputs)
        # B x N_out x d_m
        output_enc = self.pos_enc(output_enc)
        # decoder padding mask
        # B x N_out x 1
        dec_non_pad_mask = get_non_pad_mask(outputs, self.pad_label)
        # decoder self attention mask, leave out padding token and subsequent words
        # B x N_out x N_out
        dec_self_att_mask = (get_att_padding_mask(seq_q=outputs, seq_k=outputs, pad_label=self.pad_label) + get_subsequent_mask(outputs)).gt(0)
        # encoder-decoder attention mask, leave out padding token
        # B x N_out x N_in
        enc_dec_att_mask = get_att_padding_mask(seq_q=outputs, seq_k=inputs, pad_label=self.pad_label)
        
        for decoder_layer in self.decoder_layers:
            # input_enc stays the same while output_enc gets updated
            # what happens if we match sublayers of encoder to sublayers of decoder?
            # B x N_out x d_m
            output_enc = decoder_layer(input_enc, output_enc, self_att_mask=get_subsequent_mask(outputs), enc_dec_att_mask=enc_dec_att_mask, non_pad_mask=dec_non_pad_mask)
        # B x N_out x vocab_size
        out = self.proj(output_enc)

        return out

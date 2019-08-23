# Implementation of Transformer 
import math
import torch
import torch.nn as nn
import numpy as np

import pdb

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, pad_label, d_model=512, nhead=8, num_encoder_layers=2, num_decoder_layers=2, dim_feedforward=2048, 
        dropout=0.1, custom_encoder=None, custom_decoder=None):
        super(Transformer, self).__init__()
        # used for input embedding and output embedding
        self.src_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_label)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=pad_label)
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers, 
            dim_feedforward=dim_feedforward, dropout=dropout, custom_encoder=custom_encoder, custom_decoder=custom_decoder)
        # reuse the target embedding matrix as a form of regularization
        self.proj = nn.Linear(d_model, tgt_vocab_size, bias=False)
        self.proj.weight = self.tgt_embedding.weight
        self.pad_label = pad_label
    
    def forward(self, inputs, outputs):
        # inputs: B x N_in
        # outputs: B x N_out
        # use the first N-1 words(given output) to predict last N-1 words(target) 
        outputs = outputs[:, :-1]
        # N_in x B x d_m
        input_emb = self.src_embedding(inputs).permute(1, 0, 2)
        # N_out x B x d_m
        output_emb = self.tgt_embedding(outputs).permute(1, 0, 2)
        src_key_padding_mask = inputs.eq(self.pad_label)
        # N_out x N_out
        tgt_mask = self.transformer.generate_square_subsequent_mask(outputs.size(1)).to(inputs.device)
        tgt_key_padding_mask = outputs.eq(self.pad_label)
        # N_out x B x d_m
        out = self.transformer(input_emb, output_emb, tgt_mask=tgt_mask, 
            src_key_padding_mask=src_key_padding_mask, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=src_key_padding_mask)
        # B x N_out x vocab_size
        out = self.proj(out).permute(1, 0, 2)
        return out


        

# implement filter-based image captioning
import torch
import torch.nn as nn
from models import resnet, transformer

import pdb

class Ensemble(nn.Module):
    def __init__(self, filter_vocab_size, tgt_vocab_size, tgt_vocab_vectors=None, pad_label=1):
        super(Ensemble, self).__init__()
        # output pooled feats directly without projection
        self.resnet = resnet.ResNet('resnet18', num_classes=-1)
        self.transformer = transformer.Transformer(
            src_vocab_size=filter_vocab_size, 
            tgt_vocab_size=tgt_vocab_size,
            tgt_vocab_vectors=tgt_vocab_vectors, 
            num_layers=1,
            d_k=25,
            d_v=25, 
            d_m=100,
            d_hidden=512,
            num_heads=4,
            dropout=0.1,
            pad_label=pad_label
        )
        self.pad_label = pad_label
        self.relu = nn.ReLU()
    
    def forward(self, inputs, outputs):
        # inputs(images): B x C_in x W x H
        # outputs(words): B x N_out
        # B x C_act
        activations = self.resnet(inputs)
        assert activations.size(1) == 512
        # check the best performance under self-prediction
        # activations = torch.ones_like(activations)
        # B x N_out x vocab_size
        out = self.transformer(activations, outputs)
        return out

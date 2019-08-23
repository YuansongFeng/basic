# implement filter-based image captioning
import torch
import torch.nn as nn
from models import resnet, transformer
import torchvision.models as models


import pdb

class Ensemble(nn.Module):
    def __init__(self, filter_vocab_size, tgt_vocab_size, tgt_vocab_vectors=None, pad_label=1):
        super(Ensemble, self).__init__()
        # output pooled feats directly without projection
        # self.resnet = resnet.ResNet('resnet18', num_classes=-1)
        # try use pre-trained resnet 
        resnet_model = models.resnet101(pretrained=True)
        # gives spatial features before avg. pooling
        self.resnet = nn.Sequential(*list(resnet_model.children())[:-2])
        # only use resnet for feature extraction
        for param in self.resnet.parameters():
            param.requires_grad = False
        self.transformer = transformer.Transformer(
            src_vocab_size=filter_vocab_size, 
            tgt_vocab_size=tgt_vocab_size,
            tgt_vocab_vectors=tgt_vocab_vectors, 
            num_layers=1,
            d_k=25,
            d_v=25, 
            d_m=100,
            d_hidden=1024,
            num_heads=4,
            dropout=0.1,
            pad_label=pad_label
        )
        self.pad_label = pad_label
        # ISSUE: over time, gradient of the weight becomes zero. 
        self.linear = nn.Linear(2048, 100)
        self.batchnorm = nn.BatchNorm1d(100)
        self.relu = nn.ReLU()

    def forward(self, inputs, outputs):
        # inputs(images): B x C_in x W x H
        # outputs(words): B x N_out
        # B x C_act x W x H(7)
        activations = self.resnet(inputs)
        # pdb.set_trace()
        B, C, W, H = activations.size()
        # B x W*H x C(512)
        activations = activations.view(B, C, -1).permute(0, 2, 1)
        # activations = torch.ones_like(activations).to(inputs.device)
        # B x W*H x d_m(100)
        activations = self.linear(activations)
        activations = self.relu(activations)
        # activations = self.batchnorm(activations.view(B*W*H, -1)).view(B, W*H, -1)
        # pdb.set_trace()
        # activations = torch.ones_like(activations).to(activations.device)
        # check the best performance under self-prediction
        # activations = torch.zeros(inputs.size(0), W*H, 100).to(inputs.device)
        # ISSUE: gradients of encoder and enc_dec attention become really small
        # B x N_out x vocab_size
        out = self.transformer(activations, outputs)
        return out

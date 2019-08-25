# implement filter-based image captioning
import torch
import torch.nn as nn
# from models import resnet, transformer
import torchvision.models as models

import pdb

class Ensemble(nn.Module):
    def __init__(self, tgt_vocab_size, pad_label, d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048, 
            dropout=0.1, custom_encoder=None, custom_decoder=None):
        super(Ensemble, self).__init__()
        resnet_model = models.resnet101(pretrained=True)
        # gives spatial features before avg. pooling
        self.resnet = nn.Sequential(*list(resnet_model.children())[:-2])
        # only use resnet for feature extraction
        for param in self.resnet.parameters():
            param.requires_grad = False
        # TODO change 2048 to be channel number from self.resnet
        self.act_proj = nn.Linear(2048, d_model, bias=False)
        self.relu = nn.ReLU()
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=pad_label)
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers, 
            dim_feedforward=dim_feedforward, dropout=dropout, custom_encoder=custom_encoder, custom_decoder=custom_decoder)
        # reuse the target embedding matrix as a form of regularization
        self.proj = nn.Linear(d_model, tgt_vocab_size, bias=False)
        # Surprisingly, this makes the language model predict exactly itself without training 
        self.proj.weight = self.tgt_embedding.weight
        self.pad_label = pad_label


    def forward(self, inputs, outputs):
        # inputs(images): B x C_in x W x H
        # outputs(words): B x N_out
        # use the first N-1 words(given output) to predict last N-1 words(target) 
        outputs = outputs[:, :-1]
        # B x C_act x W x H(7)
        activations = self.resnet(inputs)
        # pdb.set_trace()
        B, C, W, H = activations.size()
        # B x W*H x C
        activations = activations.view(B, C, -1).permute(0, 2, 1)
        # B x W*H x d_m
        act_emb = self.act_proj(activations)
        # W*H(N_in) x B x d_m
        act_emb = self.relu(act_emb).permute(1, 0, 2)
        # N_out x B x d_m
        output_emb = self.tgt_embedding(outputs).permute(1, 0, 2)
        # N_out x N_out
        tgt_mask = self.transformer.generate_square_subsequent_mask(outputs.size(1)).to(inputs.device)
        tgt_key_padding_mask = outputs.eq(self.pad_label)
        # N_out x B x d_m
        out = self.transformer(act_emb, output_emb, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)
        # B x N_out x vocab_size
        out = self.proj(out).permute(1, 0, 2)
        return out

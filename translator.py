# custom translator to output language based on 
# given image or another language 
import os
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models.transformer import Transformer
from models.ensemble import Ensemble
import dataset
import utils
import pdb

def translate(inputs, model, bos_label, eos_label, pad_label, beam_size=3, max_len=50):
    # inputs: B x C x W x H
    B, C, W, H = inputs.size()

    # beam search variables
    # [B*beam_size x [<s>, <pad>]], notice <pad> is only a placeholder and can be any label
    candidate_seqs = [[bos_label, pad_label]] * (B*beam_size)
     # [B*beam_size x 0.0]
    candidate_log_probs = [0.0] * (B*beam_size)
    # B*beam_size x C x W x H
    inputs = inputs.unsqueeze(1).repeat(1, beam_size, 1, 1, 1).view(B*beam_size, C, W, H)
    # generate more words until max_len is hit
    while len(candidate_seqs[0]) < max_len:
        # candidate_seqs: [B*beam_size x [word_0, word_1, ...]]
        # B*beam_size x K(current outputs length)
        candidate_outputs = torch.tensor(candidate_seqs).to(inputs.device).long()
        # B*beam_size x K x vocab_size
        # for each sentence, first word from candidate_outputs is removed and the next predicted word probabilities are attached to the end
        next_outputs = model(inputs, candidate_outputs)
        # B*beam_size x vocab_size
        next_word_probs = F.softmax(next_outputs[:, -1, :].squeeze(dim=1), dim=1)
        log_next_word_probs = next_word_probs.log()
        
        # perform beam search based on log_next_word_probs
        candidate_seqs, candidate_log_probs = beam_search_step(log_next_word_probs.cpu().detach().numpy(), candidate_seqs, candidate_log_probs, beam_size)

    # remove words after EOS label
    candidate_seqs = [terminate_sent(candidate_seq, eos_label, pad_label) for candidate_seq in candidate_seqs]
    # cast to tensor
    # B*beam_size x N_out
    candidate_seqs = torch.IntTensor(candidate_seqs)
    return candidate_seqs

def beam_search_step(log_next_word_probs, curr_seqs, curr_log_probs, beam_size):
    # batch size
    B = len(curr_seqs) // beam_size
    # current length 
    curr_len = len(curr_seqs[0])
    curr_seqs = np.array(curr_seqs)
    curr_log_probs = np.array(curr_log_probs)
    next_seqs = np.zeros((B*beam_size, curr_len+1), dtype=np.int16)
    next_log_probs = np.zeros(B*beam_size)
    for sent_idx in range(B):
        # indices of all candidates for the current sentence
        candidates_idx = np.arange(sent_idx*beam_size, (sent_idx+1)*beam_size)
        # [beam_size x [word_0, word_1, ..., word_n]]
        curr_seq = curr_seqs[candidates_idx]
        # beam_size x vocab_size
        log_next_word_prob = log_next_word_probs[candidates_idx, :]
        beam_size, vocab_size = log_next_word_prob.shape
        # beam_size 
        curr_log_prob = np.array(curr_log_probs[candidates_idx])
        # beam_size x vocab_size
        curr_log_prob = np.tile(curr_log_prob[:, np.newaxis], (1, vocab_size))
        # beam_size x vocab_size
        # sum_log_prob[i][j] is the log probability of curr_seq[i] followed by j'th word
        sum_log_prob = log_next_word_prob + curr_log_prob
        # top beam_size candidates 
        if len(curr_seq[0]) == 2:
            # initialize the sequence, all current beams are the same ([<s>, <pad>])
            top_b_beam_indices = np.zeros(beam_size, dtype=np.int16)
            top_b_vocab_indices = np.argsort(sum_log_prob[0])[-beam_size:]
        else:
            top_b_indices = np.argsort(sum_log_prob.reshape(-1))[-beam_size:]
            top_b_beam_indices = top_b_indices // vocab_size
            top_b_vocab_indices = top_b_indices % vocab_size
        # [beam_size x [word_0, word_1, ..., word_{n+1}]]
        updated_seq = []
        # [beam_size x (-2.3)]
        updated_log_prob = []
        for idx in range(beam_size):
            beam_idx = top_b_beam_indices[idx]
            vocab_idx = top_b_vocab_indices[idx]
            # insert the target vocab before the last <pad> label
            next_seq = np.insert(curr_seq[beam_idx], -1, vocab_idx)
            updated_seq.append(next_seq)
            updated_log_prob.append(sum_log_prob[beam_idx, vocab_idx])
        next_seqs[candidates_idx] = updated_seq
        next_log_probs[candidates_idx] = updated_log_prob

    return next_seqs, next_log_probs


def terminate_sent(input_seq, eos_label, pad_label):
    # input_seq: [word_0, word_1, ..., word_{n+1}]
    start_pad = False
    for idx, word in enumerate(input_seq):
        if start_pad:
            input_seq[idx] = pad_label
        if word == eos_label:
            start_pad = True
    return input_seq

# transform loaded annotation sentences to torch tensors
def anno_transform(anno_field, annotations):
    # annotations: [K(caps_per_img) x [B x caption]]
    K, B = len(annotations), len(annotations[0])
    # [K*B x caption] captions from same image are separated
    merged = []
    for annotation in annotations:
        merged += annotation
    # tokenization
    # [K*B x [word1, word2, ...]]
    tokenized = [anno_field.preprocess(anno) for anno in merged]
    # apply padding and numericalize
    # K*B x N(max_len)
    annotations = anno_field.process(tokenized).transpose(0, 1)
    # B*K x N captions from same image are grouped
    annotations = annotations.view(K, B, -1).transpose(0, 1).contiguous().view(B*K, -1)
    return annotations

def test_translate():
    pretrained_checkpoint = 'checkpoints/3_layer_best.pth.tar'
    image_val_dir = '/data/feng/coco/images/val2014'
    anno_val_dir = '/data/feng/coco/annotations/captions_val2014.json'
    batch_size = 2
    # TODO add diversity to the beams, otherwise all beams are the same 
    beam_size = 1
    device = torch.device('cuda:0')

    # load vocab to numericalize annotations
    anno_field = dataset.load_annotation_field('anno_field.pl')
    anno_vocab = anno_field.vocab
    assert anno_vocab is not None
    pad_label = anno_vocab.stoi[dataset.PAD_TOKEN]
    bos_label = anno_vocab.stoi[dataset.BOS_TOKEN]
    eos_label = anno_vocab.stoi[dataset.EOS_TOKEN]

    # define transforms
    image_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # expects a tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])

    # define dataloder
    val_loader = DataLoader(
        datasets.CocoCaptions(
            root=image_val_dir, 
            annFile=anno_val_dir, 
            transform=image_transform
        ), 
        batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4
    )

    # load model 
    model = Ensemble(
        tgt_vocab_size=len(anno_vocab),
        pad_label=pad_label,
        d_model=512, 
        nhead=8, 
        num_encoder_layers=3,
        num_decoder_layers=3,
        dim_feedforward=2048,
        dropout=0.1
    )
    model.to(device)
    model.load_state_dict(torch.load(pretrained_checkpoint))

    model.eval()
    for batch_idx, (imgs, annotations) in enumerate(val_loader):
        # imgs: B x C x W x H
        # annotations: [K(caps_per_img) x [B x caption]]
        B, C, W, H = imgs.size()
        K, B = len(annotations), len(annotations[0])
        # B*K x C x W x H
        inputs = imgs.unsqueeze(1).repeat(1, K, 1, 1, 1).view(B*K, C, W, H)
        # B*K x N(max_len)
        targets = anno_transform(anno_field, annotations)
        # move to cuda
        inputs = inputs.to(device)
        targets = targets.to(device)
        # B*K*beam_size x N_out
        predict_seqs = translate(inputs, model, bos_label, eos_label, pad_label, beam_size=beam_size, max_len=30)
        # B*K*beam_size x N_target
        targets = targets.unsqueeze(1).repeat(1, beam_size, 1).view(targets.size(0)*beam_size, -1)
        utils.print_batch_itos(None, anno_vocab, None, targets, predict_seqs, K=1)

test_translate()
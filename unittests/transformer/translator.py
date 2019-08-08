# custom translator to output language based on 
# given image or another language 
import os
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

from models.transformer import Transformer
import dataset
import utils
import pdb

def translate(inputs, model, bos_label, eos_label, pad_label, beam_size=3, max_len=50):
    # inputs: B x N_in
    B, N_in = inputs.size()

    # beam search variables
    # [B*beam_size x [<s>, <pad>]], notice <pad> is only a placeholder and can be any label
    candidate_seqs = [[bos_label, pad_label]] * (B*beam_size)
     # [B*beam_size x 0.0]
    candidate_log_probs = [0.0] * (B*beam_size)
    # B*beam_size x N_in
    inputs = inputs.repeat(1, beam_size).view(B*beam_size, N_in)

    # generate more words until max_len is hit
    while len(candidate_seqs[0]) < max_len:
        # candidate_seqs: [B*beam_size x [word_0, word_1, ...]]
        # B*beam_size x K(current outputs length)
        candidate_outputs = torch.tensor(candidate_seqs).to(inputs.device).long()
        # B*beam_size x K x vocab_size
        # for each sentence, first word from candidate_outputs is removed and the next predicted word probabilities are attached to the end
        next_outputs = model(inputs, candidate_outputs)
        # B*beam_size x vocab_size
        next_word_probs = F.softmax(next_outputs[:, -1, :].squeeze(), dim=1)
        log_next_word_probs = next_word_probs.log()
        
        # perform beam search based on log_next_word_probs
        candidate_seqs, candidate_log_probs = beam_search_step(log_next_word_probs.cpu().detach().numpy(), candidate_seqs, candidate_log_probs, beam_size)

    # remove words after EOS label
    candidate_seqs = [terminate_sent(candidate_seq, eos_label, pad_label) for candidate_seq in candidate_seqs]
    return candidate_seqs

def beam_search_step(log_next_word_probs, curr_seqs, curr_log_probs, beam_size):
    # batch size
    B = len(curr_seqs) // beam_size
    # current length 
    curr_len = len(curr_seqs[0])
    curr_seqs = np.array(curr_seqs)
    curr_log_probs = np.array(curr_log_probs)
    next_seqs = np.zeros((B*beam_size, curr_len+1))
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

    return next_seqs.astype(int), next_log_probs


def terminate_sent(input_seq, eos_label, pad_label):
    # input_seq: [word_0, word_1, ..., word_{n+1}]
    start_pad = False
    for idx, word in enumerate(input_seq):
        if start_pad:
            input_seq[idx] = pad_label
        if word == eos_label:
            start_pad = True
    return input_seq

def test():
    pretrained_checkpoint = 'unittests/transformer/checkpoints/best_acc.pth.tar'
    data_dir = 'unittests/transformer/data'
    batch_size = 20

    dataloaders = dataset.get_dataloader(
        os.path.join(data_dir, 'train.txt'),
        os.path.join(data_dir, 'valid.txt'),
        batch_size=batch_size
    )
    en_vocab = dataloaders['en_vocab']
    ch_vocab = dataloaders['ch_vocab']
    pad_label = en_vocab.stoi[dataset.PAD_TOKEN]
    bos_label = en_vocab.stoi[dataset.BOS_TOKEN]
    eos_label = en_vocab.stoi[dataset.EOS_TOKEN]

    model = Transformer(
        src_vocab=en_vocab,
        tgt_vocab=ch_vocab,
        num_layers=4,
        d_k=25,
        d_v=25,
        d_m=100,
        d_hidden=256,
        num_heads=4,
        dropout=0.1,
        pad_label=pad_label
    )
    model = nn.DataParallel(model).to(torch.device('cuda'))
    model.load_state_dict(torch.load(pretrained_checkpoint))

    model.eval()
    for batch_idx, batch in enumerate(dataloaders['valid']):
        # B x N(max_len)
        inputs, targets = batch.src.transpose(0,1), batch.trg.transpose(0,1)
        predict_seqs = translate(inputs, model, bos_label, eos_label, pad_label, beam_size=3, max_len=30)
        pdb.set_trace()
        utils.print_batch_itos(en_vocab, ch_vocab, inputs, targets, predict_seqs, K=5)

test()
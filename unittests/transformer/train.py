# Test the performance of custom Transformer on the task 
# of Chinese to English translation 

# Tests the implementation of custom transformer by 
# training on the task of English-Chinese translation. 
# Check loss and accuracy curves, as well as final accuracy 
# to get the sense of the performance of the model 
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torchnet.meter as meter

# from models.transformer import Transformer
import dataset
# import Transformer from delldu's repo
from transformer.Models import Transformer
import transformer.Constants as Constants
import pdb

PAD_LABEL = None

def main():
    global PAD_LABEL
    # parameters
    data_dir = 'unittests/transformer/data'
    checkpoint_dir = 'unittests/transformer/checkpoints'
    # it helps to gradually decrease learning rate
    learning_rate = 1e-3
    weight_decay = 1e-4
    batch_size = 196
    num_epochs = 500
    # pretrained_checkpoint = 'unittests/transformer/checkpoints/best_acc.pth.tar'

    # load custom dataloader for ch-en translation dataset
    dataloaders = dataset.get_dataloader(
        os.path.join(data_dir, 'train.txt'),
        os.path.join(data_dir, 'valid.txt'),
        batch_size=batch_size
    )
    en_vocab = dataloaders['en_vocab']
    ch_vocab = dataloaders['ch_vocab']
    PAD_LABEL = en_vocab.stoi[dataset.PAD_TOKEN]
    assert en_vocab.stoi[dataset.PAD_TOKEN] == ch_vocab.stoi[dataset.PAD_TOKEN]

    # load model
    # As we don't have much training data, use small feature dimension(dk, dv, dm)
    # However, we still need some depth(num_layers) greater than 1. 
    # **Depth** is the key to the improved validation accuracy, even though deeper 
    # network introduces several times more parameters: 1 layer -> 4 layers ==> 50% ->  90%
    # model = Transformer(
    #     src_vocab_size=len(en_vocab),
    #     tgt_vocab_size=len(ch_vocab),
    #     num_layers=4,
    #     d_k=32,
    #     d_v=32,
    #     d_m=128,
    #     d_hidden=256,
    #     num_heads=4,
    #     dropout=0.1,
    #     pad_label=PAD_LABEL
    # )
    model = Transformer(
        len(en_vocab),
        len(ch_vocab),
        100,
        tgt_emb_prj_weight_sharing=True,
        emb_src_tgt_weight_sharing=False,
        d_k=32,
        d_v=32,
        d_model=128,
        d_word_vec=128,
        d_inner=256,
        # use shallow net cause we have small training data
        n_layers=4,
        n_head=4,
        # dropout helps
        dropout=0.1)
    # DataParallel helps the most if batch_size is big, in order to justify the communication cost
    model = nn.DataParallel(model).to(torch.device('cuda'))

    if 'pretrained_checkpoint' in locals() and pretrained_checkpoint is not None:
        model.load_state_dict(torch.load(pretrained_checkpoint))

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.98), weight_decay=weight_decay)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, patience=15)

    # training loop 
    train_acc_hist = []
    train_loss_hist = []
    val_acc_hist = []
    val_loss_hist = []
    best_acc = 0.0
    for epoch in range(num_epochs):
        loss, acc = train(model, dataloaders['train'], criterion, optimizer)
        train_acc_hist.append(acc)
        train_loss_hist.append(loss)

        loss, acc = evaluate(model, dataloaders['valid'], criterion, en_vocab, ch_vocab)
        val_acc_hist.append(acc)
        val_loss_hist.append(loss)

        # reduce learning rate on plateau of validation loss
        scheduler.step(loss)

        if acc > best_acc:
            save_path = os.path.join(checkpoint_dir, 'best_acc.pth.tar')
            torch.save(model.state_dict(), save_path)
            best_acc = acc
            print('model with accuracy %f saved to path %s' % (acc, save_path))
        
        print('****** epoch: %i val loss: %f val acc: %f best_acc: %f ******' % (epoch, loss, acc, best_acc))

        # upload loss and acc curves to visdom
        output_history_graph(train_acc_hist, val_acc_hist, train_loss_hist, val_loss_hist)


def train(model, dataloader, criterion, optimizer):
    acc_meter = meter.AverageValueMeter()
    loss_meter = meter.AverageValueMeter()

    model.train()

    for batch_idx, batch in enumerate(dataloader):
        # B x N(max_len)
        inputs, targets = batch.src.transpose(0,1), batch.trg.transpose(0,1)

        # input_pos TODO: test on delldu's transformer
        input_pos = torch.arange(1, inputs.size(1)+1).unsqueeze(0).repeat(inputs.size(0), 1).cuda()
        pad_mask = inputs.eq(PAD_LABEL)
        input_pos = input_pos.masked_fill(pad_mask, 0)

        # input_pos TODO: test on delldu's transformer
        target_pos = torch.arange(1, targets.size(1)+1).unsqueeze(0).repeat(targets.size(0), 1).cuda()
        pad_mask = targets.eq(PAD_LABEL)
        target_pos = target_pos.masked_fill(pad_mask, 0)

        # both translation source and target are inputted to the model
        # B x N-1 x vocab_size
        outputs = model(inputs, input_pos, targets, target_pos)
        # B x N-1
        preds = outputs.argmax(2)
        # use the first N-1 words(targets[:, :-1]) to predict last N-1 words(targets[:, 1:])
        # we use masking inside attention to prevent peak-ahead
        targets = targets[:, 1:]
        # check that the size matches
        assert torch.equal(torch.tensor(preds.size()), torch.tensor(targets.size())), 'prediction and target size mismatch'
        loss = calculate_loss(outputs, targets, criterion, label_smoothing=True)
        # calculate and store partial derivatives
        loss.backward()
        # update all parameters based on partial derivatives
        optimizer.step()
        # make sure to ZERO OUT all parameter gradients to prepare a clean slate for the next batch update
        optimizer.zero_grad()

        acc = calculate_acc(preds, targets)
        acc_meter.add(acc)
        loss_meter.add(loss.item())

        if batch_idx % 100 == 0:
            print('batch: %i loss: %f acc: %f' % (batch_idx, loss_meter.mean, acc_meter.mean))
    return loss_meter.mean, acc_meter.mean

    
def evaluate(model, dataloader, criterion, en_vocab, ch_vocab):
    acc_meter = meter.AverageValueMeter()
    loss_meter = meter.AverageValueMeter()

    model.eval()

    for batch_idx, batch in enumerate(dataloader):
        inputs, targets = batch.src.transpose(0,1), batch.trg.transpose(0,1)

        # input_pos
        input_pos = torch.arange(1, inputs.size(1)+1).unsqueeze(0).repeat(inputs.size(0), 1).cuda()
        pad_mask = inputs.eq(PAD_LABEL)
        input_pos = input_pos.masked_fill(pad_mask, 0)

        # input_pos
        target_pos = torch.arange(1, targets.size(1)+1).unsqueeze(0).repeat(targets.size(0), 1).cuda()
        pad_mask = targets.eq(PAD_LABEL)
        target_pos = target_pos.masked_fill(pad_mask, 0)

        outputs = model(inputs, input_pos, targets, target_pos)
        preds = outputs.argmax(2)
        targets = targets[:, 1:]
        
        # print_batch_itos(en_vocab, ch_vocab, inputs, targets, preds, K=10)

        assert torch.equal(torch.tensor(preds.size()), torch.tensor(targets.size())), 'prediction and target size mismatch'

        loss = calculate_loss(outputs, targets, criterion, label_smoothing=True)
        acc = calculate_acc(preds, targets)
        acc_meter.add(acc)
        loss_meter.add(loss.item())
    return loss_meter.mean, acc_meter.mean

def print_batch_itos(input_vocab, output_vocab, inputs, targets, outputs, K=2):
    words = [inputs, targets, outputs]
    words_label = ['inputs', 'targets', 'outputs']
    if K > inputs.size(0):
        K = inputs.size(0)
    for k in range(K):
        for w in range(len(words)):
            print(words_label[w])    
            vocab = input_vocab if  words_label[w] == 'inputs' else output_vocab
            print(' '.join([vocab.itos[word] for word in words[w][k]]))
        print()

def calculate_loss(outputs, targets, criterion, label_smoothing=False):
    # outputs: B x N x vocab_size
    # targets: B x N
    vocab_size = outputs.size(2)
    eps = 0.1
    # B*N x vocab_size
    outputs = outputs.contiguous().view(-1, vocab_size)
    # B*N
    targets = targets.contiguous().view(-1)
    assert torch.all(targets < vocab_size), "targets contain word index outside of vocabulary"
    # assumes a true probability of (1-eps) for correct class and eps/(K-1) for false class
    if label_smoothing:
        # B*N x vocab_size
        one_hot = torch.zeros_like(outputs).scatter(dim=1, index=targets.unsqueeze(1), value=1)
        true_prob = one_hot * (1-eps) + (1-one_hot) * eps / (vocab_size - 1)
        # B*N x vocab_size
        log_pred_prob = nn.functional.log_softmax(outputs, dim=1)
        # B*N
        loss = (- true_prob * log_pred_prob).sum(dim=1)
        non_pad_mask = targets.ne(PAD_LABEL)
        loss = loss.masked_select(non_pad_mask).sum()
    else:
        loss = criterion(outputs, targets)
    
    return loss

def calculate_acc(predictions, targets):
    # predictions: B x N, tensor
    # targets: B x N, tensor
    assert PAD_LABEL is not None
    # not counting padding 
    mask = targets.ne(PAD_LABEL)
    total_words = mask.sum().item()
    correct_words = predictions.eq(targets).masked_select(mask).sum().item()

    return correct_words / total_words

def output_history_graph(train_acc_history, val_acc_history, train_loss_history, val_loss_history):
    epochs = len(train_acc_history)
    # output training and validation accuracies
    plt.figure(0)
    plt.plot(list(range(epochs)), train_acc_history, label='train')
    plt.plot(list(range(epochs)), val_acc_history, label='val')
    plt.legend(loc='upper left')
    plt.savefig('acc.png')
    plt.clf()

    plt.figure(1)
    plt.plot(list(range(epochs)), train_loss_history, label='train')
    plt.plot(list(range(epochs)), val_loss_history, label='val')
    plt.legend(loc='upper left')
    plt.savefig('loss.png')
    plt.clf()

if __name__ == '__main__':
    main()



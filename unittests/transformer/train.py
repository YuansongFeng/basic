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

from models.transformer import Transformer
import dataset
import utils
import pdb

PAD_LABEL = None

def main():
    global PAD_LABEL
    # parameters
    data_dir = 'unittests/transformer/data'
    checkpoint_dir = 'unittests/transformer/checkpoints'
    # it helps to gradually decrease learning rate
    learning_rate = 1e-3
    weight_decay = 0
    # As a sanity check, try to overfit the model with ONE example and expect a training acc. of 100%
    batch_size = 16
    num_epochs = 50000
    # pretrained_checkpoint = 'unittests/transformer/checkpoints/best_acc.pth.tar'

    # load custom dataloader for ch-en translation dataset
    dataloaders = dataset.get_dataloader(
        os.path.join(data_dir, 'train.txt'),
        os.path.join(data_dir, 'valid.txt'),
        batch_size=batch_size,
        shuffle=False
    )
    en_vocab = dataloaders['en_vocab']
    ch_vocab = dataloaders['ch_vocab']
    PAD_LABEL = en_vocab.stoi[dataset.PAD_TOKEN]
    assert en_vocab.stoi[dataset.PAD_TOKEN] == ch_vocab.stoi[dataset.PAD_TOKEN]

    # model = Transformer(
    #     src_vocab_size=len(en_vocab),
    #     tgt_vocab_size=len(ch_vocab),
    #     # src_vocab_vectors=en_vocab.vectors,
    #     num_layers=2,
    #     d_k=64,
    #     d_v=64,
    #     d_m=512,
    #     d_hidden=2048,
    #     num_heads=8,
    #     dropout=0.1,
    #     pad_label=PAD_LABEL
    # ) 
    # transformer built on top of nn.Transformer
    model = Transformer(
        src_vocab_size=len(en_vocab),
        tgt_vocab_size=len(ch_vocab),
        pad_label=PAD_LABEL
    )
    # DataParallel helps the most if batch_size is big, in order to justify the communication cost
    model.to(torch.device('cuda'))

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
        # TODO: testing if model can overfit 
        if epoch % 20 == 0:
            utils.output_history_graph(train_acc_hist, None, train_loss_hist, None)
        print('epoch %i' % epoch)
        continue
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
        utils.output_history_graph(train_acc_hist, val_acc_hist, train_loss_hist, val_loss_hist)


def train(model, dataloader, criterion, optimizer):
    acc_meter = meter.AverageValueMeter()
    loss_meter = meter.AverageValueMeter()

    model.train()

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx > 0:
            break
        # B x N(max_len)
        inputs, targets = batch.src.transpose(0,1), batch.trg.transpose(0,1)
        # both translation source and target are inputted to the model
        # B x N-1 x vocab_size
        outputs = model(inputs, targets)
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
        # check the gradient is properly flowing 
        # if batch_idx % 100 == 0:
        #     utils.plot_grad_flow(model.named_parameters())
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
        outputs = model(inputs, targets)
        preds = outputs.argmax(2)
        targets = targets[:, 1:]
        
        # utils.print_batch_itos(en_vocab, ch_vocab, inputs, targets, preds, K=10)

        assert torch.equal(torch.tensor(preds.size()), torch.tensor(targets.size())), 'prediction and target size mismatch'

        loss = calculate_loss(outputs, targets, criterion, label_smoothing=True)
        acc = calculate_acc(preds, targets)
        acc_meter.add(acc)
        loss_meter.add(loss.item())
    return loss_meter.mean, acc_meter.mean

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

if __name__ == '__main__':
    main()



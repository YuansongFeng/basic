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
import torchnet.meter as meter

from models.transformer import Transformer
import dataset
# import Transformer from delldu's repo
# from transformer.Models import Transformer
# import transformer.Constants as Constants

PAD_LABEL = None

def main():
    global PAD_LABEL
    # parameters
    data_dir = 'unittests/transformer/data'
    checkpoint_dir = 'unittests/transformer/checkpoints'
    learning_rate = 1e-4
    weight_decay = 0.0
    batch_size = 32
    num_epochs = 80
    # pretrained_checkpoint = 'unittests/transformer/checkpoints/best_acc.pth.tar'
    pretrained_checkpoint = None

    # load custom dataloader for ch-en translation dataset
    dataloaders = dataset.get_dataloader(
        os.path.join(data_dir, 'train.txt'), 
        os.path.join(data_dir, 'valid.txt'),
        batch_size=batch_size
    )
    en_vocab = dataloaders['en_vocab']
    ch_vocab = dataloaders['ch_vocab']
    PAD_LABEL = en_vocab.stoi[dataset.PAD_TOKEN]

    # load model 
    model = Transformer(
        src_vocab_size=len(en_vocab), 
        tgt_vocab_size=len(ch_vocab),
        pad_label=PAD_LABEL
    ).to(torch.device('cuda'))
    # # init model 
    # def weights_init(m):
    #     if isinstance(m, nn.Linear):
    #         nn.init.xavier_uniform(m.weight)
    # model.apply(weights_init)

    if pretrained_checkpoint is not None:
        model.load_state_dict(torch.load(pretrained_checkpoint))

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.98), weight_decay=weight_decay)
    
    # training loop 
    train_acc_hist = []
    train_loss_hist = []
    val_acc_hist = []
    val_loss_hist = []
    best_acc = 0.0
    for epoch in range(num_epochs):
        loss, acc = train(model, dataloaders['train'], criterion, optimizer, en_vocab)
        train_acc_hist.append(acc)
        train_loss_hist.append(loss)

        loss, acc = evaluate(model, dataloaders['valid'], criterion, en_vocab)
        val_acc_hist.append(acc)
        val_loss_hist.append(loss)

        if acc > best_acc:
            save_path = os.path.join(checkpoint_dir, 'best_acc.pth.tar')
            torch.save(model.state_dict(), save_path)
            best_acc = acc
            print('model with accuracy %f saved to path %s' % (acc, save_path))
        
        print('****** epoch: %i val loss: %f val acc: %f best_acc: %f ******' % (epoch, loss, acc, best_acc))

        # upload loss and acc curves to visdom
        output_history_graph(train_acc_hist, val_acc_hist, train_loss_hist, val_loss_hist)


def train(model, dataloader, criterion, optimizer, en_vocab):
    acc_meter = meter.AverageValueMeter()
    loss_meter = meter.AverageValueMeter()

    model.train()

    for batch_idx, batch in enumerate(dataloader):
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

    
def evaluate(model, dataloader, criterion, en_vocab):
    acc_meter = meter.AverageValueMeter()
    loss_meter = meter.AverageValueMeter()

    model.eval()

    for batch_idx, batch in enumerate(dataloader):
        inputs, targets = batch.src.transpose(0,1), batch.trg.transpose(0,1)
        outputs = model(inputs, targets)
        preds = outputs.argmax(2)
        targets = targets[:, 1:]
        assert torch.equal(torch.tensor(preds.size()), torch.tensor(targets.size())), 'prediction and target size mismatch'

        loss = calculate_loss(outputs, targets, criterion)
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



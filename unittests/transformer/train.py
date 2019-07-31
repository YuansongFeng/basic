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

# from models.transformer import Transformer
import dataset
# import Transformer from delldu's repo
from transformer.Models import Transformer
import transformer.Constants as Constants

def main():
    # parameters
    data_dir = 'unittests/transformer/data'
    checkpoint_dir = 'unittests/transformer/checkpoints'
    learning_rate = 1e-3
    weight_decay = 0.0
    batch_size = 32
    num_epochs = 50

    # load custom dataloader for ch-en translation dataset
    dataloaders = dataset.get_dataloader(
        os.path.join(data_dir, 'train.txt'), 
        os.path.join(data_dir, 'valid.txt'),
        batch_size=batch_size
    )
    en_vocab = dataloaders['en_vocab']
    ch_vocab = dataloaders['ch_vocab']
    # load model 
    # model = Transformer(src_vocab_size=len(en_vocab), tgt_vocab_size=len(ch_vocab))
    # transformer as implemented by delldu
    model = Transformer(
        len(en_vocab),
        len(ch_vocab),
        100,
        tgt_emb_prj_weight_sharing=True,
        emb_src_tgt_weight_sharing=False,
        d_k=64,
        d_v=64,
        d_model=512,
        d_word_vec=512,
        d_inner=1024,
        n_layers=4,
        n_head=8,
        dropout=0.0).to(torch.device('cuda'))

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

        # temporarily generate the src_pos and tgt_pos tensors
        input_range = torch.arange(1., inputs.size(1)+1)
        # B x N
        input_pos = input_range.unsqueeze(0).repeat(inputs.size(0), 1).cuda().long()
        # mask out padding values
        input_pad_mask = inputs.eq(1)
        # B x N
        input_pos[input_pad_mask] = torch.tensor(0).cuda()

        # temporarily generate the src_pos and tgt_pos tensors
        target_range = torch.arange(1., targets.size(1)+1)
        # B x N
        target_pos = target_range.unsqueeze(0).repeat(targets.size(0), 1).cuda().long()
        # mask out padding values
        target_pad_mask = targets.eq(1)
        # B x N
        target_pos[target_pad_mask] = torch.tensor(0).cuda()
        
        # both translation source and target are inputted to the model
        # B x N x vocab_size
        outputs = model(src_seq=inputs, src_pos=input_pos, tgt_seq=targets, tgt_pos=target_pos)
        # B x N
        # remove first BOS label from targets
        targets = targets[:, 1:]
        # TODO: revert this back to fit my custom Transformer
        outputs = outputs.view(inputs.size(0), -1, outputs.size(1))
        preds = outputs.argmax(2)
        # check that the size matches
        # assert torch.equal(torch.tensor(preds.size()), torch.tensor(targets.size())), 'prediction and target size mismatch'
        loss = calculate_loss(outputs, targets, criterion)
        # calculate and store partial derivatives
        loss.backward()
        # update all parameters based on partial derivatives
        optimizer.step()
        # make sure to ZERO OUT all parameter gradients to prepare a clean slate for the next batch update
        optimizer.zero_grad()

        acc = calculate_acc(preds, targets, pad_label=en_vocab.stoi['<pad>'], sos_label=en_vocab.stoi['<sos>'])
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

         # temporarily generate the src_pos and tgt_pos tensors
        input_range = torch.arange(1., inputs.size(1)+1)
        # B x N
        input_pos = input_range.unsqueeze(0).repeat(inputs.size(0), 1).cuda().long()
        # mask out padding values
        input_pad_mask = inputs.eq(1)
        # B x N
        input_pos[input_pad_mask] = torch.tensor(0).cuda()

        # temporarily generate the src_pos and tgt_pos tensors
        target_range = torch.arange(1., targets.size(1)+1)
        # B x N
        target_pos = target_range.unsqueeze(0).repeat(targets.size(0), 1).cuda().long()
        # mask out padding values
        target_pad_mask = targets.eq(1)
        # B x N
        target_pos[target_pad_mask] = torch.tensor(0).cuda()
        # both translation source and target are inputted to the model

        # B x N x vocab_size
        outputs = model(inputs, input_pos, targets, target_pos)

        # remove first BOS label from targets
        targets = targets[:, 1:]
        # TODO: revert this back to fit my custom Transformer
        outputs = outputs.view(inputs.size(0), -1, outputs.size(1))

        # B x N
        preds = outputs.argmax(2)
        assert torch.equal(torch.tensor(preds.size()), torch.tensor(targets.size())), 'prediction and target size mismatch'
        loss = calculate_loss(outputs, targets, criterion)
        acc = calculate_acc(preds, targets, pad_label=en_vocab.stoi['<pad>'], sos_label=en_vocab.stoi['<sos>'])
        acc_meter.add(acc)
        loss_meter.add(loss.item())
    return loss_meter.mean, acc_meter.mean

def calculate_loss(outputs, targets, criterion, label_smoothing=False):
    # outputs: B x N x vocab_size
    # targets: B x N
    # B*N x vocab_size
    outputs_expand = outputs.contiguous().view(-1, outputs.size(-1))
    targets_expand = targets.contiguous().view(-1)
    if label_smoothing:
        # TODO: implement label smoothing
        return None
    else:
        loss = criterion(outputs_expand, targets_expand)
    
    return loss

def calculate_acc(predictions, targets, pad_label, sos_label):
    # predictions: B x N, tensor
    # targets: B x N, tensor
    # not counting <sos> and padding
    pad_label = Constants.PAD
    sos_label = Constants.BOS
    mask = targets.ne(pad_label) * targets.ne(sos_label)
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



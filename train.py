# Training script for the resnet + transformer ensemble.
# Ensemble uses pooled features from resnet as sequential 
# inputs to the transformer model

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchnet.meter as meter
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import spacy 
import time

from models.ensemble import Ensemble
import dataset
import pdb

PAD_LABEL = None

def main():
    global PAD_LABEL
    # parameters
    image_train_dir = '/data/feng/coco/images/train2014'
    image_val_dir = '/data/feng/coco/images/val2014'
    anno_train_dir = '/data/feng/coco/annotations/captions_train2014.json'
    anno_val_dir = '/data/feng/coco/annotations/captions_val2014.json'
    checkpoint_dir = 'checkpoints'
    learning_rate = 1e-3
    weight_decay = 0.0
    # effective batch size is batch_size*K(caps_per_img)
    batch_size = 10
    num_epochs = 150
    pretrained_checkpoint = None

    # load vocab to numericalize annotations
    anno_field = dataset.load_annotation_field('anno_field.pl')
    anno_vocab = anno_field.vocab
    assert anno_vocab is not None
    PAD_LABEL = anno_vocab.stoi[dataset.PAD_TOKEN]

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
    train_loader = DataLoader(
        datasets.CocoCaptions(
            root=image_train_dir, 
            annFile=anno_train_dir, 
            transform=image_transform
        ), 
        batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4
    )
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
        filter_vocab_size=512, 
        tgt_vocab_size=len(anno_vocab),
        pad_label=PAD_LABEL
    ).to(torch.device('cuda'))

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
        loss, acc = train(model, train_loader, criterion, optimizer, anno_field)
        train_acc_hist.append(acc)
        train_loss_hist.append(loss)

        loss, acc = evaluate(model, val_loader, criterion, anno_field)
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


def train(model, dataloader, criterion, optimizer, anno_field):
    acc_meter = meter.AverageValueMeter()
    loss_meter = meter.AverageValueMeter()
    time_meter = meter.AverageValueMeter()

    model.train()

    for batch_idx, (imgs, annotations) in enumerate(dataloader):
        start = time.time()
        # imgs: B x C x W x H
        # annotations: [K(caps_per_img) x [B x caption]]
        B, C, W, H = imgs.size()
        K, B = len(annotations), len(annotations[0])
        # B*K x N(max_len)
        targets = anno_transform(anno_field, annotations)
        # B*K x C x W x H
        inputs = imgs.unsqueeze(0).repeat(K, 1, 1, 1, 1).view(B*K, C, W, H)
        # move to cuda
        inputs = inputs.to(torch.device('cuda'))
        targets = targets.to(torch.device('cuda'))
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
        time_meter.add(time.time() - start)

        if batch_idx % 100 == 0:
            epoch_time = len(dataloader) * time_meter.mean
            print('training: batch: %i loss: %f acc: %f epoch time: %fs' % (batch_idx, loss_meter.mean, acc_meter.mean, epoch_time))
        if batch_idx > 2000:
            break
    return loss_meter.mean, acc_meter.mean

    
def evaluate(model, dataloader, criterion, anno_field):
    acc_meter = meter.AverageValueMeter()
    loss_meter = meter.AverageValueMeter()

    model.eval()

    for batch_idx, (imgs, annotations) in enumerate(dataloader):
        # imgs: B x C x W x H
        # annotations: [K(caps_per_img) x [B x caption]]
        B, C, W, H = imgs.size()
        K, B = len(annotations), len(annotations[0])
        targets = anno_transform(anno_field, annotations)
        inputs = imgs.unsqueeze(0).repeat(K, 1, 1, 1, 1).view(B*K, C, W, H)
        # move to cuda
        inputs = inputs.to(torch.device('cuda'))
        targets = targets.to(torch.device('cuda'))
        outputs = model(inputs, targets)
        preds = outputs.argmax(2)
        targets = targets[:, 1:]
        assert torch.equal(torch.tensor(preds.size()), torch.tensor(targets.size())), 'prediction and target size mismatch'

        loss = calculate_loss(outputs, targets, criterion, label_smoothing=True)
        acc = calculate_acc(preds, targets)
        acc_meter.add(acc)
        loss_meter.add(loss.item())

        if batch_idx % 100 == 0:
            print('validation: batch: %i loss: %f acc: %f' % (batch_idx, loss_meter.mean, acc_meter.mean))
        if batch_idx > 200:
            break
    return loss_meter.mean, acc_meter.mean

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


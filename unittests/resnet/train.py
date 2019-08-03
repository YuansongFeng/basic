# Tests the implementation of custom resnet by 
# training on Places365 dataset(mini). 
# Check loss and accuracy curves, as well as final accuracy 
# to get the sense of the performance of the model 
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.optim as optim
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchnet.meter as meter
from torch.utils.tensorboard import SummaryWriter
import cv2

from models.resnet import ResNet
import vis
import pdb

features_blobs = []
proj_weight = None

def main():
    global proj_weight, features_blobs
    # parameters
    data_dir = '/data/feng/places365_mini'
    checkpoint_dir = 'unittests/resnet/checkpoints'
    learning_rate = 1e-4
    weight_decay = 5e-4
    batch_size = 64
    num_epochs = 150
    pretrained_weight = 'unittests/resnet/checkpoints/acc_82.pth.tar'

    # load model 
    model = ResNet('resnet18', num_classes=10)
    if pretrained_weight is not None:
        model.load_state_dict(torch.load(pretrained_weight))
    model.cuda()

    def hook_feature(module, input, output):
        features_blobs.append(input[0])
    model._modules.get('avg_pool').register_forward_hook(hook_feature)
    proj_weight = model._modules.get('projection').weight.transpose(0, 1)

    # define data transform and data loader 
    train_loader = DataLoader(datasets.ImageFolder(
        os.path.join(data_dir, 'train'), 
        transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(224, scale=(0.25, 1.0)), 
            transforms.ToTensor(),
            # expects a tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])), 
        batch_size=batch_size, 
        shuffle=True, 
        # data go from SSD -> CPU-RAM -> GPU
        # speed up I/O between CPU and GPU, but takes page-blocked memory on CPU
        pin_memory=True,
        # more num_workers consume more memory but speed up the I/O 
        num_workers=4 
    )
    val_loader = DataLoader(datasets.ImageFolder(
        os.path.join(data_dir, 'val'), 
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])),batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4
    )

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # tensorboard writer
    # writer = SummaryWriter()

    # training loop 
    train_acc_hist = []
    train_loss_hist = []
    val_acc_hist = []
    val_loss_hist = []
    best_acc = 0.0
    for epoch in range(num_epochs):
        # loss, acc = train(model, train_loader, criterion, optimizer)
        # train_acc_hist.append(acc)
        # train_loss_hist.append(loss)

        loss, acc = evaluate(model, val_loader, criterion)
        val_acc_hist.append(acc)
        val_loss_hist.append(loss)

        return 

        if acc > best_acc:
            save_path = os.path.join(checkpoint_dir, 'best_acc.pth.tar')
            torch.save(model.state_dict(), save_path)
            best_acc = acc
            print('model with accuracy %f saved to path %s' % (acc, save_path))
        
        print('****** epoch: %i val loss: %f val acc: %f best_acc: %f ******' % (epoch, loss, acc, best_acc))

        # upload loss and acc curves to tensorboard
        # writer.add_text('rnn', 'This is an rnn', 10)
        output_history_graph(train_acc_hist, val_acc_hist, train_loss_hist, val_loss_hist)

def train(model, dataloader, criterion, optimizer):
    acc_meter = meter.AverageValueMeter()
    loss_meter = meter.AverageValueMeter()

    model.train()

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs = inputs.cuda()
        targets = targets.cuda()
        # outputs did NOT go thru softmax. per-class values don't sum to one
        outputs = model(inputs)
        preds = outputs.argmax(dim=1)
        # check that the size matches
        assert torch.equal(torch.tensor(preds.size()), torch.tensor(targets.size())), 'prediction and target size mismatch'

        loss = loss = calculate_loss(outputs, targets, criterion, label_smoothing=True)
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

    
def evaluate(model, dataloader, criterion):
    acc_meter = meter.AverageValueMeter()
    loss_meter = meter.AverageValueMeter()

    model.eval()

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs = inputs.cuda()
        targets = targets.cuda()
        outputs = model(inputs)
        preds = outputs.argmax(dim=1)
        # check that the size matches
        assert torch.equal(torch.tensor(preds.size()), torch.tensor(targets.size())), 'prediction and target size mismatch'

        loss = calculate_loss(outputs, targets, criterion)
        acc = calculate_acc(preds, targets)
        acc_meter.add(acc)
        loss_meter.add(loss.item())

        # output cam
        cams = vis.cam(inputs.cpu(), features_blobs[-1].cpu(), proj_weight.cpu(), targets.cpu())
        for idx, cam in enumerate(cams):
            cv2.imwrite('output/cam_%i.jpg' % idx, cam)
            print('cam image output to output/cam_%i.jpg' % idx)
        return None, None
    return loss_meter.mean, acc_meter.mean

def calculate_loss(outputs, targets, criterion, label_smoothing=False):
    # outputs: B x num_class
    # targets: B
    num_class = outputs.size(1)
    eps = 0.1
    assert torch.all(targets < num_class), "targets contain class index outside of allowed classes"
    # assumes a true probability of (1-eps) for correct class and eps/(K-1) for false class
    if label_smoothing:
        # B x num_class
        one_hot = torch.zeros_like(outputs).scatter(dim=1, index=targets.unsqueeze(1), value=1)
        true_prob = one_hot * (1-eps) + (1-one_hot) * eps / (num_class - 1)
        # B x num_class
        log_pred_prob = nn.functional.log_softmax(outputs, dim=1)
        # B
        loss = (- true_prob * log_pred_prob).sum()
    else:
        loss = criterion(outputs, targets)
    return loss

# predictions and targets are both torch tensors 
def calculate_acc(predictions, targets):
    return (predictions == targets).sum().item() / predictions.size(0)
    
def output_history_graph(train_acc_history, val_acc_history, train_loss_history, val_loss_history):
    epochs = len(train_acc_history)
    # output training and validation accuracies
    plt.figure(0)
    plt.plot(list(range(epochs)), train_acc_history, label='train')
    plt.plot(list(range(epochs)), val_acc_history, label='val')
    plt.legend(loc='upper left')
    plt.savefig('unittests/resnet/acc.png')
    plt.clf()

    plt.figure(1)
    plt.plot(list(range(epochs)), train_loss_history, label='train')
    plt.plot(list(range(epochs)), val_loss_history, label='val')
    plt.legend(loc='upper left')
    plt.savefig('unittests/resnet/loss.png')
    plt.clf()

if __name__ == '__main__':
    main()
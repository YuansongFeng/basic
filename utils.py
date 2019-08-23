import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
import random

# output a list of images to the output_dir
def output_list_imgs(list_imgs, output_dir='output'):
    if not os.path.exists(output_dir): 
        os.makedirs(output_dir)
    for idx, img in enumerate(list_imgs):
        cv2.imwrite(os.path.join(output_dir, 'image_%i.jpg' % idx), img)
        print('image output to ' + os.path.join(output_dir, 'image_%i.jpg' % idx))

# output a list of list of images to output_dir
def output_list_list_imgs(list_list_imgs, output_dir='output'):
    for idx, list_imgs in enumerate(list_list_imgs):
        curr_output_dir = os.path.join(output_dir, str(idx))
        output_list_imgs(list_imgs, curr_output_dir)

# Count the number of parameters in a model.
def count_parameters(model):
    return sum([p.numel() for p in model.parameters()])

def print_batch_itos(input_vocab, output_vocab, inputs, targets, outputs, K=2):
    assert targets.size(0) == outputs.size(0)
    if inputs is not None:
        assert inputs.size(0) == outputs.size(0)
    words = [inputs, targets, outputs] if inputs is not None else [targets, outputs]
    words_label = ['inputs', 'targets', 'outputs'] if inputs is not None else ['targets', 'outputs']
    if K > inputs.size(0):
        K = inputs.size(0)
    rand_idxs = random.sample(range(inputs.size(0)), K)
    for k in rand_idxs:
        for w in range(len(words)):
            print(words_label[w])    
            vocab = input_vocab if  words_label[w] == 'inputs' else output_vocab
            print(' '.join([vocab.itos[word] for word in words[w][k]]))
        print()

def output_history_graph(train_acc_history, val_acc_history, train_loss_history, val_loss_history):
    epochs = len(train_acc_history)
    # output training and validation accuracies
    plt.figure(0)
    plt.plot(list(range(epochs)), train_acc_history, label='train')
    if val_acc_history is not None:
        plt.plot(list(range(epochs)), val_acc_history, label='val')
    plt.legend(loc='upper left')
    plt.savefig('acc.png')
    plt.clf()

    plt.figure(1)
    plt.plot(list(range(epochs)), train_loss_history, label='train')
    if val_loss_history is not None:
        plt.plot(list(range(epochs)), val_loss_history, label='val')
    plt.legend(loc='upper left')
    plt.savefig('loss.png')
    plt.clf()

def plot_grad_flow(named_parameters):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n) and ('norm' not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean().cpu())
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    plt.xticks(range(len(ave_grads)), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.savefig('grad.png')
    print('gradient flow graph saved to grad.png')
    plt.clf()
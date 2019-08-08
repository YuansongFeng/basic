import os
import cv2

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
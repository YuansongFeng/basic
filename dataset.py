# Use torchtext to obtain a vocab of Coco annotations 
# Later apply vocab.stoi to numericalize annotations from
# torchvision.datasets.CocoCaptions
import torch
import torchvision.datasets as datasets
from torchtext import data
from torch.utils.data import DataLoader
import spacy
import dill

import pdb
import time

BOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"
PAD_TOKEN = "<pad>"

def load_annotation_field(field_path='anno_field.pl'):
    with open(field_path, 'rb') as f:
        anno_field = dill.load(f)
    return anno_field

def get_annotation_field(annotation_path, min_freq=2, save_to_path='anno_field.pl'):
    # use spacy english tokenizer
    anno_field = data.Field(init_token=BOS_TOKEN, eos_token=EOS_TOKEN, pad_token=PAD_TOKEN, tokenize='spacy', lower=True)
    # define dataset
    annotation_dataset = CocoAnnotationDataset(annotation_path, anno_field)
    # build vocab out of all annotations
    anno_field.build_vocab(annotation_dataset, min_freq=min_freq, vectors='glove.6B.100d', vectors_cache='.glove_cache')
    # save field to save_to_path
    if save_to_path is not None:
        with open(save_to_path, 'wb') as f:
            dill.dump(anno_field, f)
    return anno_field

class CocoAnnotationDataset(data.Dataset):
    def __init__(self, annotation_path, anno_field, **kwargs):
        fields = [("annotation", anno_field)]
        examples = torch.load('examples.pl')
        # use datasets.CocoCaptions to read in annotations
        # TODO: change this to train2014 and rebuild vocab
        # captions = datasets.CocoCaptions(root='/data/feng/coco/images/val2014', annFile=annotation_path)
        # count = 0.0
        # done = 0.0
        # examples = []
        # for (img, target) in captions:
        #     # don't need img
        #     for caption in target:
        #         example = data.Example()
        #         setattr(example, "annotation", anno_field.preprocess(caption))
        #         examples.append(example)
        #     count += 1
        #     if count/float(len(captions)) - done > 0.05:
        #         done = count/float(len(captions))
                # print('current percentage done: %f' % done)
            
        print('loaded annotation dataset from %s of size %i ' % (annotation_path, len(examples)))
        super(CocoAnnotationDataset, self).__init__(examples, fields, **kwargs)

def test():
    field = get_annotation_field('/data/feng/coco/annotations/captions_val2014.json')
    pdb.set_trace()

# test()

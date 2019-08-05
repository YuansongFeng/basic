# Load English-Chinese translation datasets
# Notice how the tokenized dataset requires TWO scans of the corpus:
# first to build the vocab, second to load the tokenized examples

# data source: https://github.com/EstellaCheng/Seq2Seq-model-by-pytorch: train/valid/test.txt
# data loader: torchtext
# tokenizer: spacy & jieba 

from torchtext import data
from torchtext.data import Iterator

import spacy
import jieba

# define tokenizer
spacy_en = spacy.load('en')
def tokenize_ch(text):
    # return [word for word in jieba.cut(text) if word != ' ']
    return text.split(' ')
def tokenize_en(text):
    return [token.text for token in spacy_en.tokenizer(text)]

# define torchtext Fields for both chinese and english sources
# Field characterizes how to turn a specific datatype to Tensor
# Each field contains its own vocab, but can only be built AFTER a data.Dataset is constructed
BOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"
PAD_TOKEN = "<pad>"
en_field = data.Field(init_token=BOS_TOKEN, eos_token=EOS_TOKEN, pad_token=PAD_TOKEN, tokenize=tokenize_en, lower=True)
ch_field = data.Field(init_token=BOS_TOKEN, eos_token=EOS_TOKEN, pad_token=PAD_TOKEN, tokenize=tokenize_ch, lower=True)

# define dataset
class ChineseEnglishDataset(data.Dataset):
    def __init__(self, dataset_path, src_field, trg_field, **kwargs):
        fields = [("src", src_field), ("trg", trg_field)]
        examples = []
        with open(dataset_path, encoding="utf-8") as f:
            for line in f.readlines():
                pair = line.split("\t")
                assert len(pair) == 2
                src = pair[0]
                trg = pair[1].replace("\n", "")
                # example is a single translation pair. words are tokenized
                e = data.Example()
                setattr(e, "src", src_field.preprocess(src))
                setattr(e, "trg", trg_field.preprocess(trg))
                examples.append(e)
                # examples.append(data.Example.fromlist([src, trg], fields=fields))
        print('loaded translation dataset from %s of size %i ' % (dataset_path, len(examples)))
        super(ChineseEnglishDataset, self).__init__(examples, fields, **kwargs)

def get_dataloader(train_path, val_path, batch_size=64, device='cuda', shuffle=True):
    # train and val datasets
    train_dataset = ChineseEnglishDataset(train_path, src_field=en_field, trg_field=ch_field)
    val_dataset = ChineseEnglishDataset(val_path, src_field=en_field, trg_field=ch_field)

    # build vocab for ch/en field based on the training dataset
    en_field.build_vocab(train_dataset, min_freq=2)
    ch_field.build_vocab(train_dataset, min_freq=2)

    # train and val iterators
    # apply padding to sents in the same batch
    train_iterator, valid_iterator = Iterator.splits(
        (train_dataset, val_dataset),
        batch_size=batch_size,
        device=device,
        shuffle=shuffle,
        # sents with similar lengths are batched together to minimize padding
        sort_key=lambda x: (len(x.src), len(x.trg))
    )

    return {'train': train_iterator, 'valid': valid_iterator, 'en_vocab': en_field.vocab, 'ch_vocab': ch_field.vocab}

def test():
    iterators = get_dataloader("data/train.txt", 'data/valid.txt', shuffle=False)
    batch = next(iter(iterators['train']))
    # max_len(N) x batch_size(B)
    src, trg = batch.src, batch.trg
    print("src: ", type(src), src.size(), src)
    print("trg: ", type(trg), trg.size(), trg)



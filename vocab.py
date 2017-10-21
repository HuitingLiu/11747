
# coding: utf-8

# In[1]:


import argparse
import cloudpickle
import json
import re

from collections import Counter, defaultdict


# In[2]:


UNK = ('<UNK>',)
OPTION = ('<OPT>',)
CHECK_A = ('A', )
CHECK_B = ('B', )
CHECK_C = ('C', )
CHECK_D = ('D', )
CHECK_E = ('E', )
START_R = ('<SOR>',)


# In[3]:


def tokenize(sent, pattern=re.compile('(\W)')):
    return (tok.lower() for tok in pattern.split(sent) if tok and tok != ' ')


# In[4]:


def load_tokens(path):
    with open(path, 'rb') as f:
        for line in f:
            item = json.loads(line.decode('utf-8'))
            yield from tokenize(item['question'])
            for option in item['options']:
                yield from tokenize(option)
            yield from tokenize(item['rationale'])


# In[5]:


def build_vocab(tokens, unk_threshold=2):
    tok_count = Counter(tokens)
    tok2id = defaultdict(lambda: 0, {UNK: 0, OPTION: 1, CHECK_A: 2, CHECK_B: 3, CHECK_C: 4, CHECK_D: 5, CHECK_E: 6, START_R: 7})
    for word in tok_count:
        if tok_count[word] > unk_threshold:
            tok2id[word] = len(tok2id)
    id2tok = {tok2id[word]: word for word in tok2id}
    return tok2id, id2tok


# In[6]:


def load_vocabs(path='./vocab.dmp'):
    with open(path, 'rb') as f:
        return cloudpickle.load(f)


# In[10]:


def main():
    parser = argparse.ArgumentParser(description='Build vocabulary')
    parser.add_argument('--dataset', default='./data/train.tok.json', type=str)
    parser.add_argument('--unk_threshold', default=2, type=int)
    parser.add_argument('--save_to', default='./vocab.dmp', type=str)
    args, _ = parser.parse_known_args()
    
    tok2id, id2tok = build_vocab(load_tokens(args.dataset), args.unk_threshold)
    with open(args.save_to, 'wb') as f:
        cloudpickle.dump((tok2id, id2tok), f)


# In[ ]:


if __name__ == '__main__':
    main()



# coding: utf-8

# In[1]:


import json

from itertools import chain
from vocab import tokenize, load_vocabs, UNK, OPTION, CHECK_A, CHECK_B, CHECK_C, CHECK_D, CHECK_E, START_R


# In[2]:


tok2id, id2tok = load_vocabs()
VOCAB_SIZE = len(tok2id)


# In[3]:


def load_questions(path):
    check2tok = {'A': CHECK_A, 'B': CHECK_B, 'C': CHECK_C, 'D': CHECK_D, 'E': CHECK_E}
    id_seq = lambda tokens: [tok2id[tok] for tok in tokens]
    with open(path, 'rb') as f:
        dataset = []
        for line in f:
            item = json.loads(line.decode('utf-8'))
            input_tokens = chain(tokenize(item['question']), *chain(chain([OPTION], tokenize(option)) for option in item['options']))
            output_tokens = chain([START_R], tokenize(item['rationale']), [check2tok[item['correct']]])
            dataset.append((id_seq(input_tokens), id_seq(output_tokens)))
    return dataset


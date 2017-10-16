# -*- encoding: utf-8 -*-
import json
import numpy as np

UNK_token = 0 #
OPT_token = 1 #
EOR_token = 2 #
EOS_token = 3 #

class VocPool:
    def __init__(self):
        self.word2index = {"<UNK>": 0, "<OPT>":1, "<EOR>":2, "<EOS>": 3}
        self.word2count = {"<UNK>": 1, "<OPT>":1, "<EOR>":1, "<EOS>": 1}
        self.index2word = {0: "<UNK>", 1: "<OPT>", 2: "<EOR>", 3: "<EOS>"}
        self.n_words = 4  # Count OPT, EOR and EOS 

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def wordToIdx(self, word):
        if word not in self.word2index:
            return self.word2index["<UNK>"]
        else:
            return self.word2index[word]

    def idxToWord(self, index):
        if index not in self.index2word:
            return "<UNK>"
        else:
            return self.word2index[index]

    def wordToVec(self, word):
        vec = np.zeros((1, self.n_words))
        idx = self.wordToIdx(word)
        vec[0][idx] = 1.0
        return vec

    def widToVec(self, wid):
        vec = np.zeros((1, self.n_words))
        vec[0][wid] = 1.0
        return vec

    def getVsize(self):
        return self.n_words


class DataSet:
    def __init__(self, vocpool, file_name):
        self.vocpool = vocpool
        self.input_data = []
        self.output_data = []
        with open(file_name) as data_file:
            for json_line in data_file:
                json_data = json.loads(json_line)

                x = [] # compressed word id  need to turn into one-hot word vector as network input
                y = [] # compressed word id, need to turn into one-hot word vector as network input 

                self.vocpool.addSentence(json_data['question'])
                for token in json_data['question'].split():
                    x.append(self.vocpool.wordToIdx(token))

                for option in json_data['options']:
                    self.vocpool.addSentence(option)
                    x.append(self.vocpool.wordToIdx("<OPT>"))
                    for token in option.split():
                        x.append(self.vocpool.wordToIdx(token))
                x.append(self.vocpool.wordToIdx("<EOS>"))
                
                self.vocpool.addSentence(json_data['rationale'])
                for token in json_data['rationale'].split():
                    y.append(self.vocpool.wordToIdx(token))

                y.append(self.vocpool.wordToIdx("<EOR>"))
                self.vocpool.addSentence(json_data['correct'])
                for token in json_data['correct']:
                    y.append(self.vocpool.wordToIdx(token))
                y.append(self.vocpool.wordToIdx("<EOS>"))

                self.input_data.append(x)
                self.output_data.append(y)

        self.data_size = len(self.input_data)
        self._index_in_epoch = 0
        self._epochs_finished = 0
        '''
        # Test Code
        for x, y in zip(self.input_data, self.output_data): 
            print(x, y)
            break
        '''

    def vectorize(self, idx):
        vsize = self.vocpool.getVsize()
        vec_x = np.zeros((len(self.input_data[idx]), vsize))
        vec_y = np.zeros((len(self.output_data[idx]), vsize))

        for idx, wid in enumerate(self.input_data[idx]):
            vec_x[idx] = self.vocpool.widToVec(wid)

        for idx, wid in enumerate(self.output_data[idx]):
            vec_y[idx] = self.vocpool.widToVec(wid)

        return vec_x, vec_y



if __name__ == "__main__":
    vp = VocPool()
    ds = DataSet(vp, './AQuA/dev.tok.json')
    vec_x, vec_y = ds.vectorize(1)
    print(vec_x, vec_y)


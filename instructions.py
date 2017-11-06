from functions import *
import itertools
import json

ROUND_DIGIT = 10
MAX_SEQ_LEN = 10
MAX_ARG = 1e20
CONSTANTS = [0.0, 0.01, 1000.0]

class Corpus(object):
    def __init__(self, dataset = 'dev'):
        self.dataset = dataset
        self.data = self.readData()
        self.instructions = []

    def readData(self):
        data = []
        with open('data/%s.json'%(self.dataset)) as f1, open('data/%s.tok.json'%(self.dataset)) as f2:
            for line1, line2 in zip(f1, f2):
                d1 = json.loads(line1)
                d2 = json.loads(line2)
                data.append({'question': d2['question'], 'options': d1['options'],
                             'rationale': d2['rationale'], 'correct': d1['correct']})
        return data

    def findPathByVal(self, question_toks, rationale_toks, ans_text, verbose = False):

        values = set(CONSTANTS)
        rationale_num = set()
        ok, ans = STR2FLOAT(ans_text)
        if ok:
            ans = round(ans, ROUND_DIGIT)

        for tok in question_toks:
            if tok in ['hours', 'hour', 'minute', 'minutes', 'seconds']:
                values.add(round(60, ROUND_DIGIT))
                continue
            if tok in ['day', 'days']:
                values.add(round(24, ROUND_DIGIT))
                continue
            ok, res = STR2FLOAT(tok)
            if ok:
                values.add(round(res, ROUND_DIGIT))

        for tok in rationale_toks:
            ok, res = STR2FLOAT(tok)
            if ok:
                res = round(res, ROUND_DIGIT)
                if res not in values:
                    rationale_num.add(res)

        if ans == None and len(rationale_num) == 0:
            return None

        if verbose:
            print("Initial Args:",values)
            print("Rationale Intermediates:",rationale_num)
            print("Ans_num:", ans)

        optimal = [len(rationale_num), None]

        path = self.helper([], operations_list, values, rationale_num, ans, optimal)
        if path == None:
            return optimal[1]
        else:
            return path

    def helper(self, path, ops, all_args, rationale, ans, optimal):
        if len(rationale) < optimal[0]:
            optimal[1] = path
        if len(path) == MAX_SEQ_LEN:
            return None
        # if ans == None and len(rationale) == 0:
        #    return None
        possible_steps = []
        for op in ops:
            args_num = argsNum[op]
            for args in itertools.permutations(all_args, args_num):
                ok, res = eval(op)(*args)
                if ok and res < MAX_ARG:
                    res = round(res, ROUND_DIGIT)
                    if res == ans:
                        return path + [(op, args)]
                    if res in rationale:
                        new_path = self.helper(path + [(op, args)], ops, all_args | set([res]), rationale - set([res]), ans,
                                          optimal)
                        if new_path != None:
                            return new_path
                    else:
                        possible_steps += [(op, args, res)]

        for op, args, res in possible_steps:
            new_path = self.helper(path + [(op, args)], ops, all_args | set([res]), set(rationale), ans, optimal)
            if new_path != None:
                return new_path

        return None

    def findPathbyIdx(self, idx, verbose = False):
        if idx >= len(self.data):
            print("Idx out of data range!")
        else:
            record = self.data[idx]
            question_toks = record['question'].split()
            rationale_toks = record['rationale'].split()
            correct_idx = ord(record['correct']) - ord('A')
            ans_text = record['options'][correct_idx][2:]
            print("Question %d: %s" % (idx, record['question']))
            print("Rationale: %s" % (record['rationale']))
            print("Answer: %s" % (ans_text))
            path = self.findPathByVal(question_toks, rationale_toks, ans_text, verbose)
            return path


def main():
    corpus = Corpus()
    path = corpus.findPathbyIdx(3, verbose=True)
    print path


if __name__ == '__main__':
    main()
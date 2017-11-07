from functions import *
import itertools
import timeout_decorator
import json
import sys

ROUND_DIGIT = 10
MAX_SEQ_LEN = 10
MAX_ARG = 1e20
PRUNE_UNK_NUMBER_STEP = 3
PRUNE_RATIONALE_COVERAGE = 0
CONSTANTS = [0.0, 0.01, 1000.0, 3.14, 60.0, 24.0]
PARTITION_NUM = 6

@timeout_decorator.timeout(2, use_signals=False)
def timeWrapper(func, args):
    return func(*args)

def tokenize(text):
    tokens = text.split()
    result = []
    for i, token in enumerate(tokens):
        if token == '%' and i > 0:
            result[-1] += '%'
        else:
            result.append(token)
    return result

class Corpus(object):
    def __init__(self, partition_id = None, dataset = 'dev'):
        self.dataset = dataset
        self.data = self.readData()
        self.instructions = []
        self.partitionID = partition_id

    def readData(self):
        with open('data/%s.json'%(self.dataset)) as f1, open('data/%s.tok.json'%(self.dataset)) as f2:
            for line1, line2 in zip(f1, f2):
                d1 = json.loads(line1)
                d2 = json.loads(line2)
                yield {'question': d2['question'], 'options': d1['options'],
                       'rationale': d2['rationale'], 'correct': d1['correct'],
                       'raw': d2}

    def findPathByVal(self, question_toks, rationale_toks, ans_text, verbose = False):

        values = set(CONSTANTS)
        rationale_num = set()
        #print ans_text
        ok, ans = STR2FLOAT(ans_text)
        if ok:
            ans = round(ans, ROUND_DIGIT)

        for tok in rationale_toks:
            ok, res = STR2FLOAT(tok)
            if ok:
                rationale_num.add(round(res, ROUND_DIGIT))

        values &= rationale_num

        for tok in question_toks:
            ok, res = STR2FLOAT(tok)
            if ok:
                values.add(round(res, ROUND_DIGIT))

        rationale_num -= values

        if ans == None and len(rationale_num) == 0:
            return None

        if verbose:
            print("Initial Args:",values)
            print("Rationale Intermediates:",rationale_num)
            print("Ans_num:", ans)

        optimal = [len(rationale_num), None]

        path = self.DFS([], operations_list, values, rationale_num, ans, optimal, 0)
        if path == None:
            return optimal[1]
        else:
            return path

    def DFS(self, path, ops, all_args, rationale, ans, optimal, unk_step):
        if len(rationale) < optimal[0]:
            optimal[1] = path
        if len(path) == MAX_SEQ_LEN:
            return None
        if unk_step > PRUNE_UNK_NUMBER_STEP:
            return None
        #print path
        if ans == None and len(rationale) == PRUNE_RATIONALE_COVERAGE:
            return path
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
                        new_path = self.DFS(path + [(op, args)], ops, all_args | set([res]), rationale - set([res]), ans,
                                          optimal, 0)
                        if new_path != None:
                            return new_path
                    else:
                        possible_steps += [(op, args, res)]

        for op, args, res in possible_steps:
            new_path = self.DFS(path + [(op, args)], ops, all_args | set([res]), set(rationale), ans, optimal, unk_step + 1)
            if new_path != None:
                return new_path

        return None

    def findPathbyIdx(self, idx, verbose = False):

        confident = False
        for i, record in enumerate(self.data):
            if i != idx:
                continue
            question_toks = tokenize(record['question'])
            rationale_toks = tokenize(record['rationale'])
            correct_idx = ord(record['correct']) - ord('A')
            ans_text = record['options'][correct_idx][2:]
            _, ans = STR2FLOAT(ans_text)
            if ans is not None:
                confident = True
            if verbose:
                print("Question %d: %s" % (idx, record['question']))
                print("Rationale: %s" % (record['rationale']))
                print("Answer: %s" % (ans_text))
            path = self.findPathByVal(question_toks, rationale_toks, ans_text, verbose)
            return path, confident

    def findPath(self, timeout=None, verbose = False):
        paths = []
        if timeout is None:
            def find_path(question_toks, rationale_toks, ans_text):
                return self.findPathByVal(question_toks, rationale_toks, ans_text, verbose)
        else:
            @timeout_decorator.timeout(timeout, use_signals=False)
            def find_path(question_toks, rationale_toks, ans_text):
                return self.findPathByVal(question_toks, rationale_toks, ans_text, verbose)

        confident_count = 0
        for (i, record) in enumerate(self.data):
            if i % PARTITION_NUM != self.partitionID:
                continue
            question_toks = tokenize(record['question'])
            rationale_toks = tokenize(record['rationale'])
            correct_idx = ord(record['correct']) - ord('A')
            ans_text = record['options'][correct_idx][2:]
            _, ans = STR2FLOAT(ans_text)
            confident = ans is not None
            if verbose:
                print("Question %d: %s" % (i, record['question']))
                print("Rationale: %s" % (record['rationale']))
                print("Answer: %s" % (ans_text))
            result = dict(record['raw'])
            try:
                result['path'] = find_path(question_toks, rationale_toks, ans_text)
                result['confident'] = confident
                confident_count += 1
            except: pass
            print json.dumps(result)
            #print("%d Done!" % (i))
        #return paths



def main():
    try:
        partition = int(sys.argv[1])
        assert(partition < PARTITION_NUM)
    except:
        print("please provide partition")
        exit(1)
    corpus = Corpus(dataset='dev', partition_id=partition)
    corpus.findPath(timeout=3)
    '''
    fail = []
    count = 0
    confident_count = 0
    path_total_length = 0
    confident_path_total_length = 0
    for i in range(len(corpus.data)):
        try:
            path, confident = timeWrapper(corpus.findPathbyIdx, [i, False])
            if confident:
                confident_count += 1
                confident_path_total_length += len(path)
            #print("%d success" % (i))
            #print path
        except:
            path = None
            fail.append(i)
        if path!= None:
            count += 1
            path_total_length += len(path)
        if (i+1) % 10 == 0:
            #break
            print("%d done! %d success! %d confident!"%(i + 1, count, confident_count))

    print("%d/%d out of %d parsed" % (confident_count, count, len(corpus.data)))
    print path_total_length / count
    print confident_path_total_length / confident_count
    '''
    #print timeWrapper(corpus.findPathbyIdx, [9, True])
    #corpus.findPath()


if __name__ == '__main__':
    main()
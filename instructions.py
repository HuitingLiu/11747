from functions import *
import itertools
import timeout_decorator
import json
import sys
from nltk.tokenize.stanford import StanfordTokenizer
#from preprocessing.num_parse import rewrite_number
import random
from queue import Queue


jar_path = '/Users/chenhu/Downloads/stanford-postagger-2017-06-09/stanford-postagger.jar'
tokenizer = StanfordTokenizer(path_to_jar=jar_path, options={'normalizeParentheses': False})
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


def tokenize(text, stanford):
    if stanford:
        tokens = tokenizer.tokenize(text)
    else:
        tokens = text.split()
    result = []
    for i, token in enumerate(tokens):
        if token == '%' and i > 0:
            result[-1] += '%'
        else:
            result.append(token)
    return result

class Corpus(object):
    def __init__(self, partition_id = None, dataset = 'dev', partition_num = PARTITION_NUM, ans_parse = 'total'):
        self.dataset = dataset
        self.data = self.readData()
        self.instructions = []
        self.partitionID = partition_id
        self.partition_num = partition_num
        self.ans_parse = ans_parse

    def readData(self):
        with open('data/%s.json'%(self.dataset)) as f1, open('data/%s.tok.json'%(self.dataset)) as f2:
            for line1, line2 in zip(f1, f2):
                d_raw = json.loads(line1)
                d_tok = json.loads(line2)
                yield {'question': d_raw['question'], 'options': d_raw['options'],
                       'rationale': d_raw['rationale'], 'correct': d_raw['correct'],
                       'raw': d_raw}

    def parseAns(self, ans_text):
        if self.ans_parse == 'total':
            return STR2FLOAT(ans_text)
        else:
            ans_text = tokenize(ans_text, stanford=True)
            res = None
            for tok in ans_text:
                ok, current_res = STR2FLOAT(tok)
                if ok:
                    if res is None:
                        res = current_res
                    else:
                        res = None
                        break
            return (res is not None, res)


    def findPathByVal(self, question_toks, rationale_toks, ans_text, verbose = False):
        values = set(CONSTANTS)
        rationale_num = set()
        #print ans_text
        ok, ans = self.parseAns(ans_text)
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

    def solveBySearch(self, question_toks, options, search_type = 'BFS'):
        values  = set([0.0])
        #values = set(CONSTANTS)

        for tok in question_toks:
            if tok in ['hours', 'hour', 'minutes', 'minute', 'seconds']:
                values.add(round(60, ROUND_DIGIT))
            elif tok in ['circle', 'radius', 'diameter']:
                values.add(round(3.14, ROUND_DIGIT))
            elif tok.startswith('kilo'):
                values.add(round(1000, ROUND_DIGIT))
            ok, res = STR2FLOAT(tok)
            if ok:
                values.add(round(res, ROUND_DIGIT))

        if search_type == 'BFS':
            path, choice_idx, hit = self.BFSSolver([], operations_list, values, options)
            return path, choice_idx, hit
        else:
            path, choice_idx, hit =  self.DFSSolver([], operations_list, values, options, 0)
            if hit:
                return path, choice_idx, hit
            else:
                idx = random.randint(0, len(options) - 1)
                return ["Random Guess:", options[idx]], idx, False

    def DFSSolver(self, path, ops, all_args, options, depth):
        if depth >= 3:
            return None, None, False
        for op in ops:
            args_num = argsNum[op]
            for args in itertools.permutations(all_args, args_num):
                ok, res = eval(op)(*args)
                if ok and res < MAX_ARG:
                    res = round(res, ROUND_DIGIT)
                    if res in options:
                        return path, options.index(res), True
                    elif res not in all_args:
                        p, choice_idx, hit = self.DFSSolver(path + [(op, args, res)], ops, all_args | set([res]), options, depth + 1)
                        if hit:
                            return p, choice_idx, hit
        return None, None, False




    def BFSSolver(self, path, ops, all_args, options):
        queue = Queue()
        step = 0
        for op in ops:
            args_num = argsNum[op]
            for args in itertools.permutations(all_args, args_num):
                ok, res = eval(op)(*args)
                if ok and res < MAX_ARG:
                    res = round(res, ROUND_DIGIT)
                    if res in options:
                        return path + [(op, args, res)], options.index(res), True

                    queue.put((step, path[:] + [(op, args, res)], all_args | set([res])))

        print("Step:", step, " Qsize:", queue.qsize())

        # Cut complicate branch
        if queue.qsize() > 128:
            # Random Guess
            idx = random.randint(0,len(options)-1)
            return ["Random Guess:", options[idx]], idx

        while queue.empty() == False:
            (step, cur_path, cur_args) = queue.get()

            # Cut complicate branch
            if step > 1 and queue.qsize() > 30000:
                # Random Guess
                idx = random.randint(0,len(options)-1)
                return ["Random Guess:", options[idx]], idx, False

            # Cut complicate branch
            if step > 3:
                idx = random.randint(0,len(options)-1)
                return ["Random Guess:", options[idx]], idx, False

            for op in ops:
                args_num = argsNum[op]
                for args in itertools.permutations(cur_args, args_num):
                    ok, res = eval(op)(*args)
                    if ok and res < MAX_ARG:
                        res = round(res, ROUND_DIGIT)
                        if res in options:
                            return cur_path + [(op, args, res)], options.index(res), True

                        queue.put((step + 1, cur_path[:] + [(op, args, res)], cur_args | set([res])))

        # Random Guess
        idx = random.randint(0,len(options)-1)
        return ["Random Guess:", options[idx]], idx, False


    def DFS(self, path, ops, all_args, rationale, ans, optimal, unk_step):
        if len(rationale) < optimal[0]:
            optimal[1] = path
            optimal[0] = len(rationale)
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
            question_toks = tokenize(record['question'], True)
            rationale_toks = tokenize(record['rationale'], False)
            correct_idx = ord(record['correct']) - ord('A')
            ans_text = record['options'][correct_idx][2:]
            _, ans = self.parseAns(ans_text)
            if ans is not None:
                confident = True
            if verbose:
                print("Question %d: %s" % (idx, record['question']))
                print("Rationale: %s" % (record['rationale']))
                print("Answer: %s" % (ans_text))
            path = self.findPathByVal(question_toks, rationale_toks, ans_text, verbose)
            return path, confident


    def findAns(self, search_type = 'BFS', timeout = None):
        if timeout is None:
            def find_ans(question_toks, options, search_type):
                return self.solveBySearch(question_toks, options, search_type)

        correct_count = 0
        total_count = 0
        hit_count = 0
        hit_correct = 0
        for (i, record) in enumerate(self.data):
            total_count += 1
            question_toks = tokenize(rewrite_number(record['question']), True)
            rationale_toks = tokenize(rewrite_number(record['rationale']), True)
            correct_idx = ord(record['correct']) - ord('A')
            ans_text = rewrite_number(record['options'][correct_idx][2:])
            #flag, ans = STR2FLOAT(ans_text)
            flag, ans = self.parseAns(ans_text)

            result = dict(record['raw'])

            if flag == False:
                choice_idx = random.randint(0, 4)
                result['path'] = ["Random Guess for no-numeric question"]
                result['choice'] = chr(ord('A') + choice_idx)
                result['hit'] = False
                if choice_idx == correct_idx:
                    correct_count += 1
            else:
                options = []
                for option in record['options']:
                    opt_text = option[2:]
                    #flag, opt = STR2FLOAT(opt_text)
                    flag, opt = self.parseAns(opt_text)
                    options.append(opt)

                try:
                    path, choice_idx, hit = find_ans(question_toks, options, search_type)
                    result['path'] = path
                    result['choice'] = chr(ord('A') + choice_idx)
                    result['hit'] = hit
                    if hit:
                        hit_count += 1

                    if choice_idx == correct_idx:
                        correct_count += 1
                        if hit:
                            hit_correct += 1
                except: pass
            print(json.dumps(result))
        print("Correct Count:", correct_count, " Correct Rate:", correct_count * 1.0 / total_count, " Hit Correct:", hit_correct, " Hit Total:", hit_count, " Hit Correct Rate:",hit_correct * 1.0 / hit_count)


    def findPath(self, timeout=None, verbose = False):
        # paths = []
        if timeout is None:
            def find_path(question_toks, rationale_toks, ans_text):
                return self.findPathByVal(question_toks, rationale_toks, ans_text, verbose)
        else:
            @timeout_decorator.timeout(timeout, use_signals=False)
            def find_path(question_toks, rationale_toks, ans_text):
                return self.findPathByVal(question_toks, rationale_toks, ans_text, verbose)

        count = 0
        success_count = 0
        confident_count = 0
        for (i, record) in enumerate(self.data):
            if i % self.partition_num != self.partitionID:
                continue
            question_toks = tokenize(record['question'], True)
            rationale_toks = tokenize(record['rationale'], True)

            correct_idx = ord(record['correct']) - ord('A')
            ans_text = record['options'][correct_idx][2:]
            _, ans = self.parseAns(ans_text)
            confident = ans is not None
            count += 1

            if verbose:
                print("Question %d: %s" % (i, record['question']))
                print("Rationale: %s" % (record['rationale']))
                print("Answer: %s" % (ans_text))
            result = dict(record['raw'])
            try:
                result['path'] = find_path(question_toks, rationale_toks, ans_text)
                result['confident'] = confident
                success_count += 1
                if confident:
                    confident_count += 1
            except: pass
            print(json.dumps(result))
            #print("%d Done!" % (i))
        print("%d done! %d success! %d confident!" % (count, success_count, confident_count))
        #return paths



def main():
    try:
        #partition_num = int(sys.argv[1])
        #partition = int(sys.argv[2])
        #dataset = sys.argv[3]
        partition_num = 1
        partition = 0
        dataset = 'dev'
        assert(dataset in ['dev', 'train', 'test'])
        assert(partition < partition_num)
        assert(partition >=0)
        assert(partition_num > 0)
    except:
        print("please execute like: \n python instructions.py [partition_num] [partition] [dev | train]")
        exit(1)

    corpus = Corpus(dataset=dataset, partition_id=partition, partition_num= partition_num, ans_parse='tok')
    corpus.findPath(timeout=3, verbose=True)
    #corpus.findAns(search_type='DFS')




if __name__ == '__main__':
    main()

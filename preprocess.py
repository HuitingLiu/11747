
import json
import os
import pickle
import timeout_decorator
import argparse
from preprocessing.parser import parse_question, extract_instructions
from collections import Counter
from sympy.parsing.sympy_parser import parse_expr

UNK = ('<UNK>',)

@timeout_decorator.timeout(10, use_signals=False)
def preprocess(i, d, directory):
    question = parse_question(d['question'])
    opts = []
    for option in d['options']:
        opts.append(parse_question(option))
    instructions = extract_instructions(d['rationale'])
    with open(os.path.join(directory ,'%d.dmp' % (i)), 'wb') as writer:
        pickle.dump(tuple([question, opts, instructions]), writer)


def generateSeparateDumps(directory, dataset, begin, end):
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open("data/%s.json" % (dataset), 'r') as f:
        for i, line in enumerate(f):
            if i < begin or i >= end:
                continue
            d = json.loads(line)
            try:
                preprocess(i, d, directory)
            except:
                print("skip %d" % (i))
            if (i + 1) % 10 == 0:
                print("%d Complete!" % (i + 1))

def combineDumps(directory, dataset, begin, end):
    all_data = []
    for i in range(begin, end):
        try:
            with open(os.path.join(directory,'%d.dmp' % (i)), 'rb') as reader:
                all_data.append(pickle.load(reader))
        except: pass
    with open('data/%s.preprocess%d_%d.dmp' % (dataset, begin, end), 'wb') as writer:
        pickle.dump(all_data, writer)


def build_index(question, opts):
    inp_index = []
    for idx, tok in enumerate(question + [tok for opt in opts for tok in opt]):
        try:
            val = parse_expr(tok)
            if val.is_number:
                found = False
                for key, idx_list in inp_index:
                    if val == key:
                        idx_list.append(idx)
                        found = True
                        break
                if not found:
                    inp_index.append((val, [idx]))
        except:pass
    return inp_index

def query_index(val, query_idx):
    for key, idx_list in query_idx:
        if key == val:
            return idx_list
    return []


def symbolicInstructions(instructions, input_num_index, num2id):
    mem = []
    new_instructions = []
    for idx, instr in enumerate(instructions):
        op = instr[0]
        if op == 'end':
            val = instr[1]
            mem.append((val, idx))
            new_instructions.append(('end',))
        elif op == 'load':
            val = instr[1]
            n_instr = ['load']
            num = float(val)
            if num in num2id:
                n_instr.append(num2id[num])
            else:
                n_instr.append(num2id[UNK])
            n_instr.append(query_index(val, input_num_index))
            n_instr.append(query_index(-val, input_num_index))
            mem_symbols_pos = []
            mem_symbols_neg = []
            for k, symb in mem:
                if k == val:
                    mem_symbols_pos.append(symb)
                if k == -val:
                    mem_symbols_neg.append(symb)
            n_instr.append(mem_symbols_pos)
            n_instr.append(mem_symbols_neg)
            new_instructions.append(n_instr)
        else:
            new_instructions.append(instr)
    new_instructions.append(('exit',))
    return new_instructions


def addSymbolicInstructions(args, num2id):
    filename = "data/%s.preprocess%d_%d.dmp" % (args.dataset, args.begin, args.end)
    with open(filename, 'rb') as reader:
        all_data = pickle.load(reader)
    n_all_data = []
    for i, data in enumerate(all_data):
        #if i != 2:
        #    continue
        question = data[0]
        opts = data[1]
        instructions = data[2]

        input_num_index = build_index(question, opts)
        n_instructions = symbolicInstructions(instructions, input_num_index, num2id)
        n_all_data.append((question, opts, n_instructions))

        if (i + 1) % 100 == 0:
            print("%d Complete!" % (i + 1))

        #if i == 2:
        #    print(question)
        #    print(opts)
        #    for (a, b) in zip(instructions, n_instructions):
        #        print(a)
        #        print(b)
            #print(instructions)
            #print(n_instructions)

    with open(filename + '.symb', 'wb') as writer:
        pickle.dump(n_all_data, writer)


def load_words(filename):
    with open(filename, 'rb') as reader:
        all_data = pickle.load(reader)
    for data in all_data:
        for tok in data[0]:
            yield tok
        for opt in data[1]:
            for tok in opt:
                yield tok


def build_word_vocab(args):
    filename = "data/%s.preprocess%d_%d.dmp" % (args.dataset, args.begin, args.end)
    unk_threshold = args.unk_threshold
    word_count = Counter(load_words(filename))

    word2id = {UNK: 0}
    for word in word_count:
        if word_count[word] > unk_threshold:
            word2id[word] = len(word2id)
    id2word = {word2id[word]: word for word in word2id}
    return word2id, id2word

def load_nums(filename):
    with open(filename, 'rb') as reader:
        all_data = pickle.load(reader)
    for data in all_data:
        nums = set()
        for instruction in data[2]:
            op = instruction[0]
            if op == 'load':
                value = instruction[1]
                nums.add(float(value))
        for num in nums:
            yield num


def build_num_vocab(args):
    filename = "data/%s.preprocess%d_%d.dmp" % (args.dataset, args.begin, args.end)
    unk_num_threshold = args.unk_num_threshold
    num_df = Counter(load_nums(filename))

    num2id = {UNK: 0}
    for num in num_df:
        if num_df[num] > unk_num_threshold:
            num2id[num] = len(num2id)
    id2num = {num2id[num]: num for num in num2id}
    return num2id, id2num


def load_ops(filename):
    with open(filename, 'rb') as reader:
        all_data = pickle.load(reader)
    for data in all_data:
        for instruction in data[2]:
            yield instruction[0]

def build_ops_list(args):
    filename = "data/%s.preprocess%d_%d.dmp" % (args.dataset, args.begin, args.end)
    ops_list = set(load_ops(filename))
    ops_list.add('exit')
    return ops_list





def main():
    parser = argparse.ArgumentParser(description='Build vocabulary')
    parser.add_argument('--dump_dir', default='dump', type=str)
    parser.add_argument('--dataset', default='train', type=str)
    parser.add_argument('--begin', default=0, type=int)
    parser.add_argument('--end', default=100, type=int)
    parser.add_argument('--unk_threshold', default=2, type=int)
    parser.add_argument('--unk_num_threshold', default=500, type=int)
    parser.add_argument('--save_to', default='./vocab.dmp', type=str)
    args, _ = parser.parse_known_args()

    print("Generating Separate Tokenized and Instruction Sequences...")
    generateSeparateDumps(args.dump_dir, args.dataset, args.begin, args.end)
    print("Combining files...")
    combineDumps(args.dump_dir, args.dataset, args.begin, args.end)

    print("Building Word Vocabulary...")
    word2id, id2word = build_word_vocab(args)
    print("Word Vocabulary Size: %d" % (len(word2id)))
    print("Building Num Vocabulary...")
    num2id, id2num = build_num_vocab(args)
    print("Num Vocabulary Size: %d" % (len(num2id)))
    print("Building Operation List...")
    ops_list = build_ops_list(args)

    print("Dumping to Disk...")
    with open(args.save_to, 'wb') as writer:
        pickle.dump((ops_list, word2id, id2word, num2id, id2num), writer)

    print("Generating Symbol Instruction Sequences...")
    addSymbolicInstructions(args, num2id)


    #print(ops_list)
    #print(word2id)
    #print(num2id)



if __name__ == '__main__':
    main()












from preprocessing.parser import parse_question, extract_instructions
import json
import pickle
import timeout_decorator
from sympy.parsing.sympy_parser import parse_expr

@timeout_decorator.timeout(10, use_signals=False)
def preprocess(i, d):
    question = parse_question(d['question'])
    opts = []
    for option in d['options']:
        opts.append(parse_question(option))
    instructions = extract_instructions(d['rationale'])
    with open('dump/%d.dmp' % (i), 'wb') as writer:
        pickle.dump(tuple([question, opts, instructions]), writer)
    # with open('%d.dmp' % (i), 'rb') as reader:
    #    data = pickle.load(reader)


def generateSeparateDumps():
    with open("data/train.json", 'r') as f:
        for i, line in enumerate(f):
            if i >= 100:
                break
            d = json.loads(line)
            try:
                preprocess(i, d)
            except:
                print("skip %d" % (i))
            if (i + 1) % 10 == 0:
                print("%d Complete!" % (i + 1))

def combineDumps(begin, end):
    all_data = []
    for i in range(begin, end):
        try:
            with open('dump/%d.dmp' % (i), 'rb') as reader:
                all_data.append(pickle.load(reader))
        except: pass
    with open('data/train.preprocess%d_%d.dmp' % (begin, end), 'wb') as writer:
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


def symbolicInstructions(instructions, input_num_index):
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
            n_instr.append(query_index(val, input_num_index))
            mem_symbols = []
            for k, symb in mem:
                if k == val:
                    mem_symbols.append(symb)
            n_instr.append(mem_symbols)
            new_instructions.append(n_instr)
        else:
            new_instructions.append(instr)
    return new_instructions


def addSymbolicInstructions(filename):
    with open(filename, 'rb') as reader:
        all_data = pickle.load(reader)
    n_all_data = []
    for i, data in enumerate(all_data):
        #if i != 0:
        #    continue
        question = data[0]
        opts = data[1]
        instructions = data[2]

        #print(question)
        #print(opts)
        print(instructions)

        input_num_index = build_index(question, opts)

        n_instructions = symbolicInstructions(instructions, input_num_index)

        n_all_data.append((question, opts, n_instructions))

        print(n_instructions)
        if n_instructions == []:
            print("======", i, instructions)

    with open(filename + '.symb', 'wb') as writer:
        pickle.dump(n_all_data, writer)









#generateSeparateDumps()
#combineDumps(0, 100)
addSymbolicInstructions("data/train.preprocess0_100.dmp")












from preprocessing.parser import parse_question, extract_instructions
import json
import pickle
import timeout_decorator

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


with open("data/train.json", 'r') as f:
    for i, line in enumerate(f):
        if i != 0:
            continue
        d = json.loads(line)
        try:
            preprocess(i, d)
        except: pass








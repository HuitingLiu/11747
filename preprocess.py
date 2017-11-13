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


def generateSeparateDumps():
    with open("data/train.json", 'r') as f:
        for i, line in enumerate(f):
            #if i == 100:
            #    break
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


#generateSeparateDumps()
combineDumps(0, 100)












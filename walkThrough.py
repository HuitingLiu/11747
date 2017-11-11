import json
import sys
from preprocessing.parser import extract_nums
partition_num = int(sys.argv[1])
partition = int(sys.argv[2])



with open('data/train.json') as f:
    for (i, line) in enumerate(f):
        if i % partition_num != partition:
            continue
        d = json.loads(line)
        print(i)
        print(d['question'])
        print(json.dumps(extract_nums(d['question'])))
        print(d['rationale'])
        print(json.dumps(extract_nums(d['rationale'])))
        for option in d['options']:
            print(option)
            print(json.dumps(extract_nums(option)))
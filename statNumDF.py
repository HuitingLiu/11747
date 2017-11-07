# -*- encoding: utf-8 -*-
import json
import numpy as np
import pickle
import math
from functions import *

'''
Count the document frequency of numbers in a specified field of the data file.
@parameter:
fileName: String
pklName: String
field: String
@ouput:
Storge the counting in a pickle file.
'''
def numberDF(fileName, pklName, field):
    totalTF = {}
    totalDF = {}
    totalDoc = 0
    with open(fileName) as dataFile:
        for jsonLine in dataFile:
            docStat = set([])
            totalDoc += 1
            json_data = json.loads(jsonLine)
            fieldObj = json_data[field]
            if isinstance(fieldObj, list):
                text = ' '.join(fieldObj)
            else:
                text = fieldObj

            words = text.split()
            for word in words:
                (mark, number) = STR2FLOAT(word)
                if mark:
                    docStat.add(number)
                    if number not in totalTF:
                        totalTF[number] = 0
                    totalTF[number] += 1

            for number in docStat:
                if number not in totalDF:
                    totalDF[number] = 0
                else:
                    totalDF[number] += 1

    print("totalDoc:", totalDoc)

    dumpDict = {'totalDF':totalDF, 'totalTF':totalTF, 'totalDoc':totalDoc}
    output = open(pklName, "wb")
    pickle.dump(dumpDict, output)
    output.close()

'''
Select constant number by the distance between two groups of fields.
@parameter
pklTemplate: String
group1: list[String]
group2: list[String]
'''
def selectByDist(pklTemplate, group1, group2):
    group1DF = {}
    for field in group1:
        pklName = pklTemplate % field
        dumpDict = pickle.load(open(pklName, "rb"))
        totalDF = dumpDict['totalDF']
        for number, df in totalDF.items():
            if number not in group1DF:
                group1DF[number] = totalDF[number]
            else:
                group1DF[number] += totalDF[number]


    average = 0
    total = 0
    for number, df in group1DF.items():
        total += group1DF[number]

    average = total * 1.0 / len(group1DF.keys())

    L2 = 0
    N = len(group1DF.keys())
    for number, df in group1DF.items():
        L2 = L2 + (1.0/N) * (df - average) * (df - average)

    SD = math.sqrt(L2)

    print("mean:", average, "sd:", SD, 'N:', N, 'total:', total)

    group2DF = {}
    for field in group2:
        pklName = pklTemplate % field
        dumpDict = pickle.load(open(pklName, "rb"))
        totalDF = dumpDict['totalDF']
        for number, df in totalDF.items():
            if number not in group2DF:
                group2DF[number] = totalDF[number]
            else:
                group2DF[number] += totalDF[number]


    for number, df in group1DF.items():
        #if df > average * 2 and number in group2DF and df > group2DF[number] * 1.5:
        if number in group2DF and df > group2DF[number] + 1000:
            print(number, df, group2DF[number])

'''
Select constant number by the difference between rationale and the combination of question and options.
@parameter:
fileName: String
'''
def selectByDiff(originName, tokName):
    totalDF = {}
    with open(originName) as originFile, open(tokName) as tokFile:
        for originLine, tokLine  in zip(originFile, tokFile):
            originJson = json.loads(originLine)
            tokJson = json.loads(tokLine)

            rationale = tokJson['rationale']
            questAndOpts = tokJson['question'] + ' '.join(originName['options'])

            rationaleWords = rationale.split()
            questOptsWords = questAndOpts.split()

            rationaleSet = set([])
            for word in rationaleWords:
                (mark, number) = STR2FLOAT(word)
                if mark:
                    rationaleSet.add(number)
                if number == 5.0:
                    print(number, word)

            questOptSet = set([])
            for word in questOptsWords:
                (mark, number) = STR2FLOAT(word)
                if mark:
                    questOptSet.add(number)
                if number == 5.0:
                    print(number, word)

            diff = rationaleSet.difference(questOptSet)
            for number in diff:
                if number not in totalDF:
                    totalDF[number] = 0
                totalDF[number] += 1

    pklName = "diff-%s.pkl" % (originName)
    output = open(pklName, "wb")
    pickle.dump(totalDF, output)

    for (key, value) in sorted(totalDF.items(), key=lambda x: x[1], reverse=True)[:100]:
        print(key, value)


if __name__ == "__main__":
    # fields = ['rationale', 'question', 'options']
    # pklTemplate = "numDF-%s.pkl"
    #
    # # count
    # for field in fields:
    #     if field == 'options':
    #         fileName = "./AQuA/train.json"
    #     else:
    #         fileName = "./AQuA/train.tok.json"
    #
    #     pklName =  pklTemplate % field
    #     numberDF(fileName, pklName, field)
    #
    # # stat
    # selectByDist(pklTemplate, ['rationale'], ['question', 'options'])


    selectByDiff("./AQuA/train.json", "./AQuA/train.tok.json")

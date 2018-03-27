from collections import defaultdict
import dynet as dy
import numpy as np
import json

with open("../model/w2iDict.json") as jsondata:
    w2i = json.load(jsondata)
    jsondata.close()
w2i = defaultdict(lambda: 0, w2i)
nwords = len(w2i)

model = dy.Model()
[mlp] = dy.load("../model/mesclass.model", model)

def read_dataset(filename):
    with open(filename, "r") as f:
        for line in f:
            tag, words = line.lower().strip().split(" ||| ")
            yield ([w2i[x] for x in words.split(" ")], tag)
dev = list(read_dataset("../data/test-caller.txt"))

for words, tag in dev:
    scores = mlp(words)
    predict = np.argmax(scores.npvalue())
    print "%d\t%s"%(predict, tag)


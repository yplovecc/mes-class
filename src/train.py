from collections import defaultdict
import time
import json
import random
import dynet as dy
import numpy as np
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

import cnn4nlp

# Functions to read in the corpus
w2i = defaultdict(lambda: len(w2i)) # the default value of newinsert is len(w2i) 
t2i = defaultdict(lambda: len(t2i))
UNK = w2i["<unk>"] # set unknown-word to zero

def read_dataset(filename):
    with open(filename, "r") as f:
        for line in f:
            tag, words = line.lower().strip().split(" ||| ")
            yield ([w2i[x] for x in words.split(" ")], t2i[tag])

# Read in the data
train = list(read_dataset("../data/train.txt"))
# Because we only know the words from train.txt, before we are going to read text.txt, we need to stop w2i
w2i = defaultdict(lambda: UNK, w2i) #set the default value of newinsert to zero
dev = list(read_dataset("../data/test.txt"))
nwords = len(w2i)
ntags = len(t2i)

# Start DyNet and define trainer
model = dy.Model()
trainer = dy.AdamTrainer(model)
mlp = cnn4nlp.OneLayerCNN(model, nwords, ntags, 64, 3, 48, dy.rectify)

test_1_count = 0 
test_0_count = 0 
for _, tag in dev: 
    if tag == 1: 
        test_1_count += 1
test_0_count = len(dev) - test_1_count

for ITER in range(100):
    # Perform training
    random.shuffle(train)
    train_loss = 0.0
    train_correct = 0.0
    start = time.time()
    for words, tag in train:
        scores = mlp(words)
        predict = np.argmax(scores.npvalue())
        if predict == tag:
            train_correct += 1

        my_loss = dy.pickneglogsoftmax(scores, tag)
        train_loss += my_loss.value()
        my_loss.backward()
        trainer.update()
    print("iter %r: train loss/sent=%.4f, acc=%.4f, time=%.2fs" % (ITER, train_loss/len(train), train_correct/len(train), time.time()-start))

    # Perform testing
    targets = []
    predicts = []
    for words, tag in dev:
        scores = mlp(words).npvalue()
        predict = np.argmax(scores)
        predicts.append(predict)
        targets.append(tag)

    # Cal precision & recall
    precision1 = precision_score(targets, predicts, pos_label=1)
    recall1 = recall_score(targets, predicts, pos_label=1)
    f2_score1 = (1 + 2 * 2) * precision1 * recall1 * 1.0 / (2 * 2 * precision1 + recall1)
    print "***type=1 : precision = %.4f, recall = %.4f, f2_score = %.4f" % (precision1, recall1, f2_score1)
    
    precision0 = precision_score(targets, predicts, pos_label=0)
    recall0 = recall_score(targets, predicts, pos_label=0)
    f2_score0 = (1 + 2 * 2) * precision0 * recall0 * 1.0 / (2 * 2 * precision0 + recall0)
    print "***type=0 : precision = %.4f, recall = %.4f, f2_score = %.4f" % (precision0, recall0, f2_score0)

# Save model
json = json.dumps(w2i)
f = open("../model/w2iDict.json", "w")
f.write(json)
f.close()
dy.save("../model/mesclass.model", [mlp])


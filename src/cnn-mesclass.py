from collections import defaultdict
import time
import json
import random
import dynet as dy
import numpy as np
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

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

# Define the model
EMB_SIZE = 64 # Word embedding size; Kernel_height
W_emb = model.add_lookup_parameters((nwords, 1, 1, EMB_SIZE)) # Word embeddings
WIN_SIZE = 3 # Window approach size; Kernel_width
FILTER_SIZE = 48 # CNN kernel_size
W_cnn = model.add_parameters((1, WIN_SIZE, EMB_SIZE, FILTER_SIZE)) # cnn weights 
b_cnn = model.add_parameters((FILTER_SIZE)) # cnn bias

W_sm = model.add_parameters((ntags, FILTER_SIZE))          # Softmax weights
b_sm = model.add_parameters((ntags))                      # Softmax bias

def calc_scores(words):
    dy.renew_cg()
    W_cnn_express = dy.parameter(W_cnn)
    b_cnn_express = dy.parameter(b_cnn)
    W_sm_express = dy.parameter(W_sm)
    b_sm_express = dy.parameter(b_sm)
    if len(words) < WIN_SIZE:
      words += [0] * (WIN_SIZE-len(words))

    cnn_in = dy.concatenate([dy.lookup(W_emb, x) for x in words], d=1)
    cnn_out = dy.conv2d_bias(cnn_in, W_cnn_express, b_cnn_express, stride=(1, 1), is_valid=False)
    pool_out = dy.max_dim(cnn_out, d=1)
    pool_out = dy.reshape(pool_out, (FILTER_SIZE,))
    pool_out = dy.rectify(pool_out)
    """ 
    # Debug
    print "### %s"%words 
    print cnn_in.value() 
    print cnn_in.dim() 
    print W_cnn_express.dim() 
    print cnn_out.dim() 
    print pool_out.dim() 
    """
    return W_sm_express * pool_out + b_sm_express 

test_1_count = 0 
test_0_count = 0 
for _, tag in dev: 
    if tag == 1: 
        test_1_count += 1
test_0_count = len(dev) - test_1_count

for ITER in range(10):
    # Perform training
    random.shuffle(train)
    train_loss = 0.0
    train_correct = 0.0
    start = time.time()
    for words, tag in train:
        scores = calc_scores(words)
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
        scores = calc_scores(words).npvalue()
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


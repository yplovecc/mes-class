# mes-class : 骚扰短信识别
## Introduction 介紹
CNN for NLP. 将卷积神经网络运用到文本分类中。识别短信为骚扰短信，或普通短信。
## Steps 步骤
1. 自定义模型
```
class OneLayerCNN(object):
    def __call__(self, words):
        dy.renew_cg()
        W_cnn_express = dy.parameter(self.W_cnn)
        b_cnn_express = dy.parameter(self.b_cnn)
        W_sm_express = dy.parameter(self.W_sm)
        b_sm_express = dy.parameter(self.b_sm)
        if len(words) < self.win_size:
          words += [0] * (self.win_size-len(words))  
        cnn_in = dy.concatenate([dy.lookup(self.W_emb, x) for x in words], d=1)
        cnn_out = dy.conv2d_bias(cnn_in, W_cnn_express, b_cnn_express, stride=(1, 1), is_valid=False)
        pool_out = dy.max_dim(cnn_out, d=1)
        pool_out = dy.reshape(pool_out, (self.filter_size,))
        pool_out = dy.rectify(pool_out)
        return W_sm_express * pool_out + b_sm_express 
```
 包括词向量化和一层卷积神经网络：多卷积核、max pooling和softmax激励

2. 训练
* 数据

  [SMS Spam Collection](http://www.dt.fee.unicamp.br/~tiago/smsspamcollection/)
* 过程
```
# Start DyNet and define trainer
model = dy.Model()
trainer = dy.AdamTrainer(model)
mlp = cnn4nlp.OneLayerCNN(model, nwords, ntags, 64, 3, 48, dy.rectify)

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
```
3. 预测
```
model = dy.Model()
[mlp] = dy.load("../model/mesclass.model", model)
dev = list(read_dataset("../data/test-caller.txt"))

for words, tag in dev:
    scores = mlp(words)
    predict = np.argmax(scores.npvalue())
    print "%d\t%s"%(predict, tag)
```

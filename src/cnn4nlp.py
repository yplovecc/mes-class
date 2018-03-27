import dynet as dy

class OneLayerCNN(object):
    def __init__(self, model, num_words, num_tags, emb_size, win_size, filter_size, act = dy.rectify):
        self.W_emb = model.add_lookup_parameters((num_words, 1, 1, emb_size)) # Word embeddings
        self.W_cnn = model.add_parameters((1, win_size, emb_size, filter_size)) # cnn weights 
        self.b_cnn = model.add_parameters((filter_size)) # cnn bias
        self.W_sm = model.add_parameters((num_tags, filter_size)) # Softmax weights
        self.b_sm = model.add_parameters((num_tags)) # Softmax bias
        self.model = model
        self.act = act
        self.win_size = win_size
        self.filter_size = filter_size
        self.spec = (num_words, num_tags, emb_size, win_size, filter_size, act)

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

    def param_collection(self):
        return self.model

    @staticmethod
    def from_spec(spec, model):
        num_words, num_tags, emb_size, win_size, filter_size, act = spec
        return OneLayerCNN(model, num_words, num_tags, emb_size, win_size, filter_size, act)


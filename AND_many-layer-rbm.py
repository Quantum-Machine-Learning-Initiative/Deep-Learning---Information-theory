from __future__ import print_function, division
try:
    from builtins import range
except ImportError:
    from __builtin__ import range
import sys
import numpy as np
import tensorflow as tfsys
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from preprocessing_MNIST import PreMnist

class RBM(object):
    def __init__(self, D, M, an_id):
        self.D = D
        self.M = M
        self.id = an_id
        self.build(D, M)

    def set_session(self, session):
        self.session = session

    def build(self, D, M):
        # params
        self.W = tf.Variable(tf.random_normal(shape=(D, M)) * np.sqrt(2.0 / M))
        # note: without limiting variance, you get numerical stability issues
        #self.c = tf.Variable(np.zeros(M).astype(np.float32))
        #self.b = tf.Variable(np.zeros(D).astype(np.float32))

        # data
        self.X_in = tf.placeholder(tf.float32, shape=(None, D))

        # conditional probabilities
        # NOTE: tf.contrib.distributions.Bernoulli API has changed in Tensorflow v1.2
        V = self.X_in
        p_h_given_v = tf.nn.sigmoid(tf.matmul(V, self.W)) #+ self.c)
        self.p_h_given_v = p_h_given_v # save for later
        # self.rng_h_given_v = tf.contrib.distributions.Bernoulli(
        #     probs=p_h_given_v,
        #     dtype=tf.float32
        # )
        r = tf.random_uniform(shape=tf.shape(p_h_given_v))
        H = tf.to_float(r < p_h_given_v)

        p_v_given_h = tf.nn.sigmoid(tf.matmul(H, tf.transpose(self.W))) #+ self.b)
        # self.rng_v_given_h = tf.contrib.distributions.Bsysernoulli(
        #     probs=p_v_given_h,
        #     dtype=tf.float32
        # )
        r = tf.random_uniform(shape=tf.shape(p_v_given_h))
        X_sample = tf.to_float(r < p_v_given_h)


        # build the objective
        objective = tf.reduce_mean(self.free_energy(self.X_in)) - tf.reduce_mean(self.free_energy(X_sample))
        self.train_op = tf.train.AdamOptimizer(1e-2).minimize(objective)
        # self.train_op = tf.train.GradientDescentOptimizer(1e-3).minimize(objective)

        # build the cost
        # we won't use this to optimize the model parameters
        # just to observe what happens during training
        logits = self.forward_logits(self.X_in)
        self.cost = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=self.X_in,
                logits=logits,
            )
        )

    def fit(self, X, epochs, batch_sz, show_fig, show_print):
        N, D = X.shape
        n_batches = N // batch_sz

        costs = []
        if show_print:
          print("training rbm: %s" % self.id)
        for i in range(epochs):
            if show_print:
              print("epoch:", i)
              sys.stdout.flush()sys
            X = shuffle(X)
            for j in range(n_batches):
                batch = X[j*batch_sz:(j*batch_sz + batch_sz)]
                _, c = self.session.run((self.train_op, self.cost), feed_dict={self.X_in: batch})
                #if j % 10 == 0 and show_print:
                #    print("j / n_batches:", j, "/", n_batches, "cost:", c)
                costs.append(c)
        if show_fig:
            plt.plot(costs)
            plt.show()

    def free_energy(self, V):
        #b = tf.reshape(self.b, (self.D, 1))
        #first_term = -tf.matmul(V, b)
        #first_term = tf.reshape(first_term, (-1,))

        second_term = -tf.reduce_sum(
            # tf.log(1 + tf.exp(tf.matmul(V, self.W) + self.c)),
            tf.nn.softplus(tf.matmul(V, self.W)), #+ self.c),
            axis=1
        )

        #return first_term + second_term
        return second_term

    def forward_hidden(self, X):
        return tf.nn.sigmoid(tf.matmul(X, self.W)) #+ self.c)

    def forward_logits(self, X):
        Z = self.forward_hidden(X)
        return tf.matmul(Z, tf.transpose(self.W)) #+ self.b

    def forward_output(self, X):
        return tf.nn.sigmoid(self.forward_logits(X))

    def transform(self, X):
        # accepts and returns a real numpy array
        # unlike forward_hidden and forward_output
        # which deal with tensorflow variablessys
        return self.session.run(self.p_h_given_v, feed_dict={self.X_in: X})
    '''
    def predict(self, X):
        Z = self.forward_hidden(X)
        output = self.forward_output(Z)
        return self.session()
    '''
class DNN(object):
    def __init__(self, D, hidden_layer_sizes, K=0, UnsupervisedModel=RBM):
        self.hidden_layers = []
        count = 0
        input_size = D
        for output_size in hidden_layer_sizes:
            ae = UnsupervisedModel(input_size, output_size, count)
            self.hidden_layers.append(ae)
            count += 1
            input_size = output_size
        self.build_final_layer(D, hidden_layer_sizes[-1], K)

    def set_session(self, session):
        self.session = session
        for layer in self.hidden_layers:
            layer.set_session(session)

    def build_final_layer(self, D, M, K):
        # initialize logistic regression layer
        self.W = tf.Variable(tf.random_normal(shape=(M, K)))
        #self.b = tf.Variable(np.zeros(K).astype(np.float32))

        self.X = tf.placeholder(tf.float32, shape=(None, D))
        labels = tf.placeholder(tf.int32, shape=(None,))
        self.Y = labels
        logits = self.forward(self.X)

        # accepts and returns a real 
        self.cost = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits,
                labels=labels
            )
        )
        self.train_op = tf.train.AdamOptimizer(1e-2).minimize(self.cost)
        self.prediction = tf.argmax(logits, 1)

    #def fit(self, X, '''Y=0, Xtest=0, Ytest=0, pretrain=True''', epochs'''=1''', batch_sz'''=10''', show_print, showfig=false):
    def fit(self, X, epochs, batch_sz, show_print, show_fig):
        N = len(X)

        
        #pretrain_epochs = 10
        #if not pretrain:
        #    pretrain_epochs = 0

        current_input = X
        for ae in self.hidden_layers:
            ae.fit(current_input, epochs, 1,  show_fig, show_print)

            # create current_input for the next layer
            current_input = ae.transform(current_input)

        '''
        n_batches = N // batch_sz        
        costs = []
        
        print("supervised training...")
        for i in range(epochs):
            print("epoch:", i)
            X, Y = shuffle(X, Y)
            for j in range(n_batches):
                Xbatch = X[j*batch_sz:(j*batch_sz + batch_sz)]
                Ybatch = Y[j*batch_sz:(j*batch_sz + batch_sz)]
                self.session.run(sys
                    self.train_op,
                    feed_dict={self.X: Xbatch, self.Y: Ybatch}
                )
                c, p = self.session.run(
                    (self.cost, self.prediction),
                    feed_dict={self.X: Xtest, self.Y: Ytest
                })
                error = error_rate(p, Ytest)
                if j % 10 == 0:
                    print("j / n_batches:", j, "/", n_batches, "cost:", c, "error:", error)
                costs.append(c)   def forward_hidden(self, X):
        return tf.nn.sigmoid(tf.matmul(X, self.W)) #+ self.c)

    def forward_logits(self, X):
        Z = self.forward_hidden(X)
        return tf.matmul(Z, tf.transpose(self.W)) #+ self.b

    def forward_output(self, X):
        return tf.nn.sigmoid(self.forward_logits(X))

        plt.plot(costs)
        plt.show()
        '''
    def forward(self, X):
        current_input = X
        for ae in self.hidden_layers:
            Z = ae.forward_hidden(current_input)
            current_input = Z

        # logistic layer
        logits = tf.matmul(current_input, self.W) #+ self.b
        return logits



def all_parity_pairs(nbit):
    # total number of samples (Ntotal) will be a multiple of 100
    # why did I make it this way? I don't remember.
    N = 2**nbit
    remainder = 100 - (N % 100)
    Ntotal = N + remainder
    X = np.zeros((Ntotal, nbit))
    Y = np.zeros(Ntotal)
    for ii in range(Ntotal):
        i = ii % N
        # now generate the ith sample
        for j in range(nbit):
            if i % (2**(j+1)) != 0:
                i -= 2**j
                X[ii,j] = 1
        Y[ii] = X[ii].sum() % 2
    return X, Y


'''
def test_single_RBM(Xtrain):
    
    Xtrain = Xtrain.astype(np.float32)fit
    _, D = Xtrain.shape
    rbm = RBM(D, 1, 0)
    init_op = tf.global_variables_initializer()
    with tf.Session() as session:
        session.run(init_op)
        rbm.set_session(session)
        rbm.fit(Xtrain, show_fig=True)
        print(session.run(rbm.W))
'''

def main():

    batchSize = -1
    epochs = -1

    args = []

    for arg in sys.argv[1:]:
        args.append(arg)
    
    print("-------------------------------")
    print("len: " + str(len(args)))    
    for string in args:
      print(string)
    print("-------------------------------")
        
    for i, arg in enumerate(args, start=1):
      print(arg)
      if arg == "-size":
        try:    
          batchSize = int(args[i]) if i<len(args) and int(args[i])>0  else -1
        except ValueError:
            pass
          
      if arg == "-epochs":
        try:
          epochs = int(args[i]) if i<len(args) and int(args[i])>0  else -1
        except ValueError:
          pass
          
    if len(sys.argv)>1:
        print("size: " + str(batchSize))
        print("epochs: " + str(epochs))   
        sys.stdout.flush()
        if batchSize<0 or epochs<0:
          quit()            
       


    batch, labels = PreMnist().ET(1, batchSize)

    #PreMnist().printSet(batch[1], batch[1], 0)

    size = batch[1].size

    dnn = DNN(size, [size, size, size, size, size], UnsupervisedModel=RBM)
    init_op = tf.global_variables_initializer()
    with tf.Session() as session:
        session.run(init_op)
        dnn.set_session(session)
        #dnn.fit(Xtrain, pretrain=True, epochs=10)
        dnn.fit(batch, epochs, size, True, False)
        W = []
        for e in dnn.hidden_layers:
            W.append(session.run(e.W))
        #w1 = session.run(dnn.hidden_layers[0].W)
        #w2 = session.run(dnn.hidden_layers[1].W)
        #w3 = session.run(dnn.hidden_layers[2].W)
        #w4 = session.run(dnn.hidden_layers[3].W)
        #print(w1)
        #print(w2)
        #print(w3)
    #W = [w1, w2, w3, w4]
    print('--------------------')
    ##print(W)
    print('Complete learning \n Saving weights to file')
    np.save('rbm_weights_size' + str(batchSize) + '_epochs' + str(epochs), W)
    


    '''
    b = batch[1]
    print(type(b))
    print(b.shape)
    pr=PreMnist()
    PreMnist.printSet(pr, b, b, 0)
    
   
    
    #a = np.array(b, dtype=np.float32 )   
    
    #print(a.shape)
    #print(type(a[0]))      
      
      
    Z = dnn.hidden_layers[0].forward_hidden(batch)
    output = dnn.hidden_layers[0].forward_output(Z)
    print(output)
    print(output.shape)
    print(output[0].shape)
    
    print("\n Array: \n")
    sess = tf.InteractiveSession()
    array = output[0].eval()    
    
    
    PreMnist.printSet(pr, array, array, 0)   
    ''' 
  
    

if __name__ == '__main__':
    main()



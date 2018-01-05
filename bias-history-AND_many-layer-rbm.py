from __future__ import print_function, division
from builtins import range
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.utils import shuffle



        


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
        self.c = tf.Variable(np.random.rand(M).astype(np.float32))
        self.b = tf.Variable(np.random.rand(D).astype(np.float32))


        # data
        self.X_in = tf.placeholder(tf.float32, shape=(None, D))

        # conditional probabilities
        # NOTE: tf.contrib.distributions.Bernoulli API has changed in Tensorflow v1.2
        V = self.X_in
        p_h_given_v = tf.nn.sigmoid(tf.matmul(V, self.W) + self.c)
        self.p_h_given_v = p_h_given_v # save for later
        # self.rng_h_given_v = tf.contrib.distributions.Bernoulli(
        #     probs=p_h_given_v,
        #     dtype=tf.float32
        # )
        r = tf.random_uniform(shape=tf.shape(p_h_given_v))
        H = tf.to_float(r < p_h_given_v)

        p_v_given_h = tf.nn.sigmoid(tf.matmul(H, tf.transpose(self.W)) + self.b)
        # self.rng_v_given_h = tf.contrib.distributions.Bernoulli(
        #     probs=p_v_given_h,
        #     dtype=tf.float32
        # )
        r = tf.random_uniform(shape=tf.shape(p_v_given_h))
        X_sample = tf.to_float(r < p_v_given_h)

        regularizer_W = tf.nn.l2_loss(self.W)
        regularizer_b = tf.nn.l2_loss(self.b)
        regularizer_c = tf.nn.l2_loss(self.c)
        beta = 0.1
        # build the objective
        objective = tf.reduce_mean(self.free_energy(self.X_in)) - tf.reduce_mean(self.free_energy(X_sample)) + beta * regularizer_W + beta * regularizer_b + beta * regularizer_c
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


    def fit(self, X, hidden_layers, epochs=10, batch_sz=4, show_fig=False):
        N, D = X.shape
        n_batches = N // batch_sz

        costs = []
        print("training rbm: %s" % self.id)
        for i in range(epochs+1):
            if i %100  == 0:
                self.save_history_w2(i, hidden_layers)
                #print('a')    
            print("epoch:", i)
            X = shuffle(X)
            for j in range(n_batches):
                batch = X[j*batch_sz:(j*batch_sz + batch_sz)]
                _, c = self.session.run((self.train_op, self.cost), feed_dict={self.X_in: batch})
                #print(self.session.run(self.W))
                if j % 10 == 0:
                    print("j / n_batches:", j, "/", n_batches, "cost:", c)
                costs.append(c) 
        if show_fig:
            plt.plot(costs)
            plt.show()
        

    def free_energy(self, V):
        b = tf.reshape(self.b, (self.D, 1))
        first_term = -tf.matmul(V, b)
        first_term = tf.reshape(first_term, (-1,))

        second_term = -tf.reduce_sum(
            # tf.log(1 + tf.exp(tf.matmul(V, self.W) + self.c)),
            tf.nn.softplus(tf.matmul(V, self.W) + self.c),
            axis=1
        )

        return first_term + second_term

    def forward_hidden(self, X):
        return tf.nn.sigmoid(tf.matmul(X, self.W) + self.c)

    def forward_logits(self, X):
        Z = self.forward_hidden(X)
        return tf.matmul(Z, tf.transpose(self.W)) + self.b

    def forward_output(self, X):
        return tf.nn.sigmoid(self.forward_logits(X))

    def transform(self, X):
        # accepts and returns a real numpy array
        # unlike forward_hidden and forward_output
        # which deal with tensorflow variables
        return self.session.run(self.p_h_given_v, feed_dict={self.X_in: X})

    def save_history_w(self, epoch):
        W_out = []
        b_out = []
        c_out = []
        number_out = self.id
        W_out.append(self.session.run(self.W))
        b_out.append(self.session.run(self.b))
        c_out.append(self.session.run(self.c))
        string1 = 'rbm_weights_' + str(number_out) + '-' + str(epoch)
        string2 = 'rbm_bias_b_' + str(number_out) + '-' + str(epoch)
        string3 = 'rbm_bias_c_' + str(number_out) + '-' + str(epoch)     
        np.save(string1, W_out)
        np.save(string2, b_out)
        np.save(string3, c_out)

    def save_history_w2(self, epoch, hidden_layers):
        W = []
        b = []
        c = []
        number_out = self.id
        for ae in hidden_layers:
            W.append(ae.session.run(ae.W))
            b.append(ae.session.run(ae.b))
            c.append(ae.session.run(ae.c))
        string1 = 'rbm_weights_' + str(number_out) + '-' + str(epoch)
        string2 = 'rbm_bias_b_' + str(number_out) + '-' + str(epoch)
        string3 = 'rbm_bias_c_' + str(number_out) + '-' + str(epoch)     
        np.save(string1, W)
        np.save(string2, b)
        np.save(string3, c)         


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
        self.b = tf.Variable(np.zeros(K).astype(np.float32))

        self.X = tf.placeholder(tf.float32, shape=(None, D))
        labels = tf.placeholder(tf.int32, shape=(None,))
        self.Y = labels
        logits = self.forward(self.X)

        self.cost = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits,
                labels=labels
            )
        )
        self.train_op = tf.train.AdamOptimizer(1e-2).minimize(self.cost)
        self.prediction = tf.argmax(logits, 1)

    def fit(self, X, Y=0, Xtest=0, Ytest=0, pretrain=True, epochs=1, batch_sz=4):
        N = len(X)

        
        pretrain_epochs = 32000
        if not pretrain:
            pretrain_epochs = 0

        epoch = 0
        #self.save_history_all_w(epoch)
        current_input = X
        for ae in self.hidden_layers:
            ae.fit(current_input, self.hidden_layers, epochs=pretrain_epochs, show_fig=True)
            epoch += 32000
            #self.save_history_all_w(epoch) 
            # create current_input for the next layer
            current_input = ae.transform(current_input)

        n_batches = N // batch_sz
        costs = []
        '''
        print("supervised training...")
        for i in range(epochs):
            print("epoch:", i)
            X, Y = shuffle(X, Y)
            for j in range(n_batches):
                Xbatch = X[j*batch_sz:(j*batch_sz + batch_sz)]
                Ybatch = Y[j*batch_sz:(j*batch_sz + batch_sz)]
                self.session.run(
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
                costs.append(c)
        plt.plot(costs)
        plt.show()
        '''
    def forward(self, X):
        current_input = X
        for ae in self.hidden_layers:
            Z = ae.forward_hidden(current_input)
            current_input = Z

        # logistic layer
        logits = tf.matmul(current_input, self.W) + self.b
        return logits

    def save_history_all_w(self, epoch):
        W = []
        b = []
        c = []
        for ae in self.hidden_layers:
            W.append(ae.session.run(ae.W))
            b.append(ae.session.run(ae.b))
            c.append(ae.session.run(ae.c))
        string1 = 'rbm_weights_all' + '-' + str(epoch)
        string2 = 'rbm_bias_b_all' + '-' + str(epoch)
        string3 = 'rbm_bias_c_all' + '-' + str(epoch)     
        np.save(string1, W)
        np.save(string2, b)
        np.save(string3, c)        


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



def test_single_RBM(Xtrain):
    
    Xtrain = Xtrain.astype(np.float32)
    _, D = Xtrain.shape
    rbm = RBM(D, 1, 0)
    init_op = tf.global_variables_initializer()
    with tf.Session() as session:
        session.run(init_op)
        rbm.set_session(session)
        rbm.fit(Xtrain, show_fig=True)
        print(session.run(rbm.W))


def main():
    T, F = 1.0, 0.0 
    training_input = [
        [T, T, T],
        [T, F, F],
        [F, T, F],
        [F, F, F],
    ]

#    training_input = [
#        [T, T, F],
#        [T, F, T],
#        [F, T, T],
#        [F, F, F],
#    ]
    training_input = np.array(training_input)
    #test_single_RBM(training_input)

    Xtrain = training_input.astype(np.float32)
    #Xtest = Xtest.astype(np.float32)
    _, D = Xtrain.shape
    dnn = DNN(D, [3,3,3], UnsupervisedModel=RBM)
    init_op = tf.global_variables_initializer()
    

    with tf.Session() as session:
        #W = []
        #b = []
        #c = []
        session.run(init_op)
        dnn.set_session(session)
        #for e in dnn.hidden_layers:
        #    W.append(session.run(e.W))
        #    b.append(session.run(e.b))
        #    c.append(session.run(e.c))
        #np.save('rbm_weights_0-0', W)
        #np.save('rbm_bias_b_0-0', b)
        #np.save('rbm_bias_c_0-0', c)

        dnn.fit(Xtrain, pretrain=True, epochs=10)
        #W = []
        #b = []
        #c = []
        #for e in dnn.hidden_layers:
        #    W.append(session.run(e.W))
        #    b.append(session.run(e.b))
        #    c.append(session.run(e.c))
        #print(b)
        #print(c)
        #w1 = session.run(dnn.hidden_layers[0].W)
        #w2 = session.run(dnn.hidden_layers[1].W)
        #w3 = session.run(dnn.hidden_layers[2].W)
        #w4 = session.run(dnn.hidden_layers[3].W)
        #print(w1)
        #print(w2)
        #print(w3)
    #W = [w1, w2, w3, w4]
    print('--------------------')
    #print(W)
    #np.save('rbm_weights', W)
    #np.save('rbm_bias_b', b)
    #np.save('rbm_bias_c', c)
if __name__ == '__main__':
    main()



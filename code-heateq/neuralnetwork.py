import time

import numpy as np
import tensorflow as tf


class PhysicsInformedNN:
    def __init__(self, X, u, layers, lb, ub, gamma):
        self.lb = lb
        self.ub = ub

        self.gamma = gamma

        # Here and below the following subtle broadcasting rule is used.
        # Slicing with `0:1` preserves 2D arrays, while slicing with just `0`
        # would return 1D array.
        self.x = X[:,0:1]
        self.t = X[:,1:2]
        self.u = u

        self.layers = layers

        # Initialize NNs
        self.weights, self.biases = self.initialize_NN(layers)

        # tf placeholders and graph
        self.sess = tf.Session(
            config=tf.ConfigProto(allow_soft_placement=True,
                                  log_device_placement=False))

        # Initialize parameters
        self.lambda_ = tf.Variable([-2.718], dtype=tf.float32)

        self.x_tf = tf.placeholder(tf.float32, shape=[None, self.x.shape[1]])
        self.t_tf = tf.placeholder(tf.float32, shape=[None, self.t.shape[1]])
        self.u_tf = tf.placeholder(tf.float32, shape=[None, self.u.shape[1]])

        self.u_pred = self.net_u(self.x_tf, self.t_tf)
        self.f_pred = self.net_f(self.x_tf, self.t_tf)

        self.loss = \
            tf.reduce_mean(tf.square(self.u_tf - self.u_pred)) + \
            self.gamma * tf.reduce_mean(tf.square(self.f_pred))

        ftol = 1.0 * np.finfo(float).eps
        ftol = np.finfo(np.float32).eps
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(
            self.loss,
            method = 'L-BFGS-B',
            options = {'maxiter': 50000,
                       'maxfun': 50000,
                       'maxcor': 50,
                       'maxls': 50,
                       'ftol' : ftol})

        self.optimizer_Adam = tf.train.AdamOptimizer()
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)

        init = tf.global_variables_initializer()
        self.sess.run(init)

        # Counter to control the frequency of output in self.callback.
        self._counter = 0

    def initialize_NN(self, layers):
        """Initialize feedforward neural network.

        Parameters
        ----------
        layers: list
            The length of the list defines the depth of the network, while
            each value in the list defines the width of the corresponding
            layer, with the first and the last values defining the dimension
            of the input and output layers, respectively.

        """
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0, num_layers-1):
            W = self.xavier_init(size=(layers[l], layers[l+1]))
            b = tf.Variable(tf.zeros((1, layers[l+1]), dtype=tf.float32),
                            dtype=tf.float32)
            weights.append(W)
            biases.append(b)
        return weights, biases

    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        return tf.Variable(
            tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev),
            dtype=tf.float32)

    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1

        H = 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0
        for l in range(0, num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    def net_u(self, x, t):
        u = self.neural_net(tf.concat([x, t],1), self.weights, self.biases)
        return u

    def net_f(self, x, t):
        D = self.lambda_
        D = tf.exp(D)
        u = self.net_u(x,t)
        u_t = tf.gradients(u, t)[0]
        u_x = tf.gradients(u, x)[0]
        u_xx = tf.gradients(u_x, x)[0]
        s = tf.exp(-t) * (4*np.pi**2 - 1) * tf.sin(2*np.pi*x)
        #s = tf.exp(-t) * tf.exp(-x**2) - 0.35*2*(2*x**2-1) * tf.exp(-t) * tf.exp(-x**2)
        f = u_t - D*u_xx - s

        return f

    def callback(self, loss, lambda_):
        if self._counter % 100 == 0:
            values = (loss, np.exp(lambda_))
            print('Loss: %e, lambda: %.5f' % values)

        self._counter += 1


    def train(self, nIter):
        tf_dict = {self.x_tf: self.x, self.t_tf: self.t, self.u_tf: self.u}

        start_time = time.time()
        for it in range(nIter):
            self.sess.run(self.train_op_Adam, tf_dict)

            # Print
            if it % 100 == 0:
                elapsed = time.time() - start_time
                loss_value = self.sess.run(self.loss, tf_dict)
                lambda_value = self.sess.run(self.lambda_)
                print('It: %d, Loss: %.3e, Lambda: %.3f, Time: %.2f' %
                      (it, loss_value, lambda_value, elapsed))
                start_time = time.time()

        fetches = [self.loss, self.lambda_]
        self.optimizer.minimize(self.sess,
                                feed_dict=tf_dict,
                                fetches=fetches,
                                loss_callback=self.callback)
        loss_final = self.sess.run(self.loss, feed_dict=tf_dict)
        lambda_final = self.get_pde_params()[0]
        print('=== Loss: %e, lambda: %.5f' % (loss_final, lambda_final))
        print('=== # of iterations ', self._counter)

    def predict(self, X_star):
        tf_dict = {self.x_tf: X_star[:, 0:1], self.t_tf: X_star[:, 1:2]}

        u_star = self.sess.run(self.u_pred, tf_dict)
        f_star = self.sess.run(self.f_pred, tf_dict)

        return u_star, f_star

    def get_pde_params(self):
        """Return PDE parameters as tuple."""
        D = self.sess.run(self.lambda_)
        D = np.exp(D)

        return (D,)

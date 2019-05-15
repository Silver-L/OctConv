'''
# cifar100 classification
# Author: Zhihui Lu
# Date: 2019/05/13
'''

import os
import tensorflow as tf

class resnet_model(object):

    def __init__(self, sess, outdir, input_size, alpha, network, is_training=True, learning_rate=1e-4):

        self._sess = sess
        self._outdir = outdir
        self._input_size = input_size
        self._alpha = alpha
        self._network = network
        self._is_training = is_training
        self._lr = learning_rate

        self.build_graph()

    def build_graph(self):
        with tf.variable_scope('input'):
            self._input = tf.placeholder(tf.float32, shape=[None, self._input_size[0],
                                                            self._input_size[1], self._input_size[2]])
            self._label = tf.placeholder(tf.float32, shape=[None, 100])

            with tf.variable_scope('classify'):
                self._result = self._network(self._input, self._alpha, self._is_training)

            with tf.variable_scope('loss'):
                self._loss = self.train_loss(self._label, self._result)

            with tf.variable_scope('optimizer'):
                optimizer = tf.train.AdamOptimizer(learning_rate=self._lr, beta1=0.5, beta2=0.9)

            with tf.variable_scope('training-step'):
                self._train = optimizer.minimize(self._loss)

            self.saver = tf.train.Saver(max_to_keep=None)
            init = tf.initializers.global_variables()
            self._sess.run(init)

    @staticmethod
    def train_loss(ground_truth, pred):
        loss = tf.keras.backend.categorical_crossentropy(ground_truth, pred)
        return loss

    def update(self, X, Y):
        _, loss = self._sess.run([self._train, self._loss], feed_dict= {self._input: X, self._label: Y})
        return loss

    def save_model(self, index):
        save = self.saver.save(self._sess, os.path.join(self._outdir, 'model', 'model_{}'.format(index)))
        return save

    def restore_model(self, path):
        self.saver.restore(self._sess, path)


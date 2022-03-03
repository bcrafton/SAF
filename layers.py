
import numpy as np
np.set_printoptions(threshold=1000)

import tensorflow as tf

#############

class model:
    def __init__(self, layers):
        self.layers = layers
        
    def forward(self, x):
        y = x
        for layer in self.layers:
            y = layer.forward(y)
        return y

    def get_params(self):
        params = []
        for layer in self.layers:
            params.extend(layer.get_params())
        return params
        
    def save(self, name):
        weights = {}
        for layer in self.layers:
            weights.update(layer.get_weights())
        np.save(name, weights)

#############

class layer:
    layer_id = 0
    weight_id = 0

    def __init__(self):
        assert(False)
        
    def forward(self, x):        
        assert(False)

    def get_params(self):
        assert(False)

    def get_weights(self):
        assert (False)

#############

class dense_block(layer):
    def __init__(self, shape, act=True, weights=None):
        self.layer_id = layer.layer_id
        layer.layer_id += 1

        self.shape = shape
        self.act = act
        self.weights = weights

        if weights:
            w = weights[self.layer_id]['w']
            self.w = tf.Variable(w, dtype=tf.float32)
        else:
            self.w = tf.Variable(np.random.uniform(low=-0.01, high=0.01, size=self.shape), dtype=tf.float32)

    def forward(self, x):
        x = tf.reshape(x, (-1, self.shape[0]))
        y = tf.matmul(x, self.w)
        z = tf.nn.relu(y) if self.act else y
        return z

    def get_params(self):
        return [self.w]

    def get_weights(self):
        return {self.layer_id: {'w': self.w.numpy()}}

#############




        
        
        

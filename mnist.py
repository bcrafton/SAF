
import numpy as np
import tensorflow as tf
from layers import *

####################################

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

####################################

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

assert(np.shape(x_train) == (60000, 28, 28))
x_train = x_train.reshape(60000, 28*28)
x_train = x_train - np.mean(x_train)
x_train = x_train / np.std(x_train)

assert(np.shape(x_test) == (10000, 28, 28))
x_test = x_test.reshape(10000, 28*28)
x_test = x_test - np.mean(x_test)
x_test = x_test / np.std(x_test)

####################################

model = model(layers=[
dense_block(shape=(784, 128), act=True),
dense_block(shape=(128, 10), act=False)
])

params = model.get_params()

####################################

optimizer = tf.keras.optimizers.Adam()

@tf.function(experimental_relax_shapes=False)
def gradients(model, x, y):
    with tf.GradientTape() as tape:
        pred_logits = model.train(x)
        pred_label = tf.argmax(pred_logits, axis=1)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=pred_logits)
        correct = tf.reduce_sum(tf.cast(tf.equal(pred_label, y), tf.float32))
    
    grad = tape.gradient(loss, params)
    return loss, correct, grad

####################################

def predict(model, x, y):
    pred_logits = model.predict(x)
    pred_label = tf.argmax(pred_logits, axis=1)
    correct = tf.reduce_sum(tf.cast(tf.equal(pred_label, y), tf.float32))
    return correct

####################################

batch_size = 50
epochs = 2

####################################

for _ in range(epochs):
    total_correct = 0
    for batch in range(0, len(x_train), batch_size):
        xs = x_train[batch:batch+batch_size].astype(np.float32)
        ys = y_train[batch:batch+batch_size].reshape(-1).astype(np.int64)
        loss, correct, grad = gradients(model, xs, ys)
        optimizer.apply_gradients(zip(grad, params))
        total_correct += correct

    print (total_correct / len(x_train) * 100)

####################################

batch_size = 50
total_correct = 0
for batch in range(0, len(x_test), batch_size):
    xs = x_test[batch:batch+batch_size].astype(np.float32)
    ys = y_test[batch:batch+batch_size].reshape(-1).astype(np.int64)
    correct = predict(model, xs, ys)
    total_correct += correct

print (total_correct / len(x_test) * 100)

####################################

model.save('cifar10_weights')
weights = np.load('cifar10_weights.npy', allow_pickle=True).item()
print (weights.keys())

####################################











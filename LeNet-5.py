import tensorflow as tf
import pickle
import numpy as np
import pandas as pd
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
STEPS = 20000
lamda = 0.004
BATCH_SIZE = 100
sess = tf.InteractiveSession()
train_data = {b'data': [], b'labels': []}
for i in range(5):
    with open('./cifar-10-batches-py/data_batch_' + str(i + 1), mode='rb') as file:
        data = pickle.load(file, encoding='bytes')
        train_data[b'data'] += list(data[b'data'])
        train_data[b'labels'] += data[b'labels']
with open("cifar-10-batches-py/test_batch", mode='rb') as file:
    test_data = pickle.load(file, encoding='bytes')
X_train = np.transpose(np.array(train_data[b'data']).reshape([-1, 3, 32, 32]), [0, 2, 3, 1])/255
y_train = np.array(pd.get_dummies(train_data[b'labels']))
X_test = np.transpose(np.array(test_data[b'data']).reshape([-1, 3, 32, 32]), [0, 2, 3, 1])/255
y_test = np.array(pd.get_dummies(test_data[b'labels']))


def get_weight(shape):
    w = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(w)


def get_bias(shape):
    b = tf.constant(0.1, shape=shape)
    return tf.Variable(b)


def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


def max_pooling_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')


x = tf.placeholder(tf.float32, [None, 32, 32, 3])
y_ = tf.placeholder(tf.float32, [None, 10])

conv1_w = get_weight([5, 5, 3, 6])
conv1_b = get_bias([6])
conv1 = conv2d(x, conv1_w)
relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_b))
pool1 = max_pooling_2x2(relu1)

conv2_w = get_weight([14, 14, 6, 16])
conv2_b = get_bias([16])
conv2 = conv2d(pool1, conv2_w)
relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_b))
pool2 = max_pooling_2x2(relu2)
reshaped = tf.reshape(pool2, [-1, 1024])
fc1_w = get_weight([1024, 120])
fc1_b = get_bias([120])
fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_w) + fc1_b)

fc2_w = get_weight([120, 10])
fc2_b = get_bias([10])
y = tf.nn.softmax(tf.matmul(fc1, fc2_w) + fc2_b)

cross_entropy = tf.reduce_mean(
    -tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y, 1e-8, tf.reduce_max(y))), reduction_indices=[1]))
w1_loss = lamda * tf.nn.l2_loss(fc1_w)
w2_loss = lamda * tf.nn.l2_loss(fc2_w)
loss = w1_loss + w2_loss + cross_entropy
train_step = tf.train.GradientDescentOptimizer(0.05).minimize(cross_entropy)

correct_predict = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_predict, tf.float32))

sess.run(tf.global_variables_initializer())

for i in range(STEPS):
    start = i * BATCH_SIZE % 50000
    train_step.run(feed_dict={x: X_train[start:start + BATCH_SIZE], y_: y_train[start:start + BATCH_SIZE]})
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: X_test[0:200], y_: y_test[0:200]})
        loss_value = cross_entropy.eval(
            feed_dict={x: X_train[start:start + BATCH_SIZE], y_: y_train[start:start + BATCH_SIZE]})
        print('After %d steps, training accuracy is %g loss %g' % (i, train_accuracy, loss_value))

test_accuracy = accuracy.eval(feed_dict={x: X_test, y_: y_test})
print("test accuracy is %g" % test_accuracy)

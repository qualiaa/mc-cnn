import tensorflow as tf

sess = tf.Session()

import tf.contrib.keras as keras
from tf.keras import backend as K

K.set_session(sess)

img = tf.placeholder(tf.float32, shape=(None, 784))

from keras.layers import Dense

x = Dense(128, activation="relu")(img)
x = Dense(128, activation="relu")(x)
preds = Dense(10, activation="softmax")(x)

labels = tf.placeholder(tf.float32, shape=(None,10))

from keras.objectives import categorical_crossentropy as xentropy
loss = tf.reduce_mean(xentropy(labels,preds))

from tensorflow.examples.tutorials.mnist import input_data

mnist_data = input_data.read_data_sets("MNIST_data", one_hot=True)

train_step = tf.train.GradientDescentOptimizer(0.5).minimise(loss)

init_op = tf.global_variables_initializer()
sess.run(init_op)

with sess.as_default():
    for i in range(100):
        batch = mnist_data.train.next_batch(50)
        train_step.run(feed_dict={image:batch[0],
                                  labels:batch[1]})

from keras.metrics import categorical_accuracy as accuracy

acc_value = accuracy(labels, preds)
with sess.as_default():
    print acc_value.eval(feed_dict={img: mnist_data.test.images,
                                    labels: mnist_data.test.labels})
#sess.run(loss,feed_dict=

#!/usr/bin/env python3

import tensorflow as tf

from keras.layers import Input, Conv2D, concatenate, Reshape
from keras.models import Model
import keras.utils.vis_utils as vis

from functools import reduce

""" generic network application helpers """
apply = lambda f, x: f(x)
flip = lambda f: lambda a,b: f(b,a)
apply_sequence = lambda l, x: reduce(flip(apply), l, x)

def _loss_summaries(loss):
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [loss])

    for l in losses + [loss]:
        tf.summary.scalar(l.op.name + " (raw)", l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))

    return loss_averages_op

def inference(left_input,right_input,flatten=False):
    """ network definition """
    C2 = lambda f: Conv2D(f,(3,3),activation="relu")
    FC = lambda f: Conv2D(f,(1,1),activation="relu")
    siamese = [
        C2(64), C2(64), C2(64), C2(64),FC(200)
    ]
    classifier = [
        FC(300),FC(300),FC(300),FC(300),Conv2D(1,(1,1),activation="sigmoid")
    ]

    """ network application """
    left_patch = Input(tensor=left_input,name="left_input", dtype=tf.float32)
    right_patch = Input(tensor=right_input,name="right_input", dtype=tf.float32)

    left_output = apply_sequence(siamese,left_patch)
    right_output = apply_sequence(siamese,right_patch)
    classifier_input = concatenate([left_output,right_output])

    classifier_output = apply_sequence(classifier,classifier_input)
    if flatten:
        classifier_output = Reshape(tuple())(classifier_output)

    """ model creation """
    model = Model(inputs=[left_input,right_input],#inputs=[left,right],
                  outputs=[classifier_output])

    vis.plot_model(model,to_file="model.png")

    model.summary()

    return classifier_output

def train(loss, global_step):
    """
    num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

    lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                    global_step,
                                    decay_steps,
                                    LEARNING_RATE_DECAY_FACTOR,
                                    staircase=True)
    tf.summary.scalar("learning_rate", lr)

    """

    lr = 0.1

    loss_averages_op = _loss_summaries(loss)
    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.GradientDescentOptimizer(lr)
        grads = opt.compute_gradients(loss,aggregation_method=1)

    apply_gradients_op = opt.apply_gradients(grads,global_step=global_step)

    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name,var)

    for grad, var in grads:
        if grad is not None:
            tf.summary.histogram(var.op.name+"/gradients", grad)

    with tf.control_dependencies([apply_gradients_op]):
        train_op = tf.no_op(name="train")

    return train_op

def loss(ypred,ytrue):
    from keras.objectives import binary_crossentropy as xentropy
    return tf.reduce_mean(xentropy(ytrue, ypred))

def accuracy(logits,labels):
    ypred = tf.cast(tf.round(logits),tf.int32)
    ytrue = tf.cast(labels,tf.int32)
    from keras.metrics import binary_accuracy as accuracy
    return tf.reduce_mean(accuracy(ytrue,ypred))

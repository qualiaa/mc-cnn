#!/usr/bin/env python3

import tensorflow as tf
from flags import *

import mc_cnn_input


# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1       # Initial learning rate.

def _layer_summary(h,w,b):
    #tf.summary.histogram('biases', b)
    #tf.summary.histogram('weights', w)
    tf.summary.histogram('activations/hist', h)
    tf.summary.scalar('activations/sparsity',
                      tf.nn.zero_fraction(h))

def _loss_summaries(loss):
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [loss])

    for l in losses + [loss]:
        tf.summary.scalar(l.op.name + " (raw)", l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))

    return loss_averages_op

def _bias_variable(size,name="bias"):
    init = tf.zeros([size])
    return tf.Variable(init,name=name);

def _fan_in(shape):
    return tf.to_float(tf.reduce_prod(shape[:-1]))

def _leaky_relu(x, name=None):
    with tf.name_scope(name or "leaky_relu"):
        h = tf.maximum(0.0,x) + tf.minimum(0.0, x/10)
    return h

def _weight_variable(shape,name="weights"):
    with tf.name_scope(name):
        fan_in = _fan_in(shape)
        stddev=tf.sqrt(1.0/fan_in)
        init = tf.random_normal(shape,stddev=stddev)
        weights = tf.Variable(init,name=name)
    return weights

def _relu_weight_variable(shape,name="weights"):
    # use initialization suggested by He & al.
    # https://arxiv.org/pdf/1502.01852.pdf
    with tf.name_scope(name):
        fan_in = _fan_in(shape)
        stddev=tf.sqrt(2.0/fan_in)
        init = tf.random_normal(shape,stddev=stddev)
        weights = tf.Variable(init,name=name)
    return weights

def _appropriate_weight_initializer(act_fn):
    return (_relu_weight_variable
            if act_fn==tf.nn.relu or act_fn==_leaky_relu
            else _weight_variable)

def _linear(input_,w,b,act_fn=tf.nn.relu):
    wx = tf.matmul(input_, w)
    activations = act_fn(wx + b,name="activations")
    _layer_summary(activations,w,b)
    return activations

def _convolve(input_,w,b,act_fn=tf.nn.relu,padding="SAME"):
    wx = tf.nn.conv2d(input_,w,strides=[1,1,1,1],padding=padding)
    activations = act_fn(wx + b,name="activations")
    _layer_summary(activations,w,b)
    return activations

def _conv_layer(input_,shape,name="conv",act_fn=tf.nn.relu,padding="SAME"):
    with tf.name_scope(name) as scope:
        # initialise weights and biases
        b = _bias_variable(shape[-1])

        weight_fn = _appropriate_weight_initializer(act_fn)
        w = weight_fn(shape)

        # calculate activations
        activations = _convolve(input_,w,b,act_fn,padding)
    return activations

def _fc_layer(input_,input_size,output_size,name="fc",act_fn=tf.nn.relu):
    with tf.name_scope(name) as scope:
        # initialise weights and biases
        b = _bias_variable(output_size)

        weight_shape = tf.stack([input_size,output_size])
        weight_fn = _appropriate_weight_initializer(act_fn)
        w = weight_fn(weight_shape)

        # calculate activations
        activations = _linear(input_,w,b)
    return activations

def _flatten(input_):
    return tf.reshape(input_,[tf.shape(input_)[0],-1], name="flatten")

def conv_inference(left, right, channels=1):

    with tf.name_scope("tied_layers"):
        with tf.name_scope("tied_weights"):
            tied_weights1 = _relu_weight_variable([5,5,channels,32])
            tied_weights2 = _relu_weight_variable([5,5,32,32])
            tied_weights3 = _relu_weight_variable([1,1,32,200])

        def tied_layers(input_):
            with tf.name_scope("conv1"):
                bias1 = _bias_variable(32)
                conv1 = _convolve(input_,tied_weights1,bias1,padding="VALID")
            with tf.name_scope("conv2"):
                bias2 = _bias_variable(32)
                conv2 = _convolve(conv1, tied_weights2,bias2,padding="VALID")
            with tf.name_scope("fc1"):
                bias3 = _bias_variable(200)
                fc1   = _convolve(conv2, tied_weights3,bias3)
            return fc1

        with tf.name_scope("left"):
            left = tied_layers(left)
        with tf.name_scope("right"):
            right = tied_layers(right)
        concat = tf.concat([left, right],axis=3)

    act_fn = tf.nn.relu

    fc2 = _conv_layer(concat,[1,1,400,300],name="fc2", act_fn=act_fn)
    fc3 = _conv_layer(fc2,[1,1,300,300],name="fc3", act_fn=act_fn)
    fc4 = _conv_layer(fc3,[1,1,300,300],name="fc4", act_fn=act_fn)
    fc5 = _conv_layer(fc4,[1,1,300,300],name="fc5", act_fn=act_fn)

    # defer softmax activation to loss calculation
    id_ = lambda x, name: x
    logits = _conv_layer(fc5,[1,1,300,2],name="softmax",act_fn=id_)

    return logits

def inference(left, right, channels=1):
    with tf.name_scope("tied_layers"):
        with tf.name_scope("tied_weights"):
            tied_weights1 = _relu_weight_variable([5,5,channels,32])
            tied_weights2 = _relu_weight_variable([32*9*9,200])
            tied_weights3 = _relu_weight_variable([200,200])

        def tied_layers(input_):
            with tf.name_scope("conv1"):
                b = _bias_variable(32)
                conv1 = _convolve(input_,tied_weights1,b,padding="VALID")
                conv1 = _flatten(conv1)
            with tf.name_scope("fc1"):
                b = _bias_variable(200)
                fc1 = _linear(conv1,tied_weights2,b)
            with tf.name_scope("fc2"):
                b = _bias_variable(200)
                fc2 = _linear(fc1,tied_weights3,b)
            return fc2

        with tf.name_scope("left"):
            left = tied_layers(left)
        with tf.name_scope("right"):
            right = tied_layers(right)
        concat = tf.concat([left, right],axis=1)

    act_fn = tf.nn.relu
    """
    def soft_relu(x, name=None):
        with tf.name_scope(name or "soft_relu"):
            h = tf.maximum(0.0,x) + tf.minimum(0.0, x/10)
        return h
    """

    fc3 = _fc_layer(concat,400,300,name="fc3", act_fn=act_fn)
    fc4 = _fc_layer(fc3,300,300,name="fc4", act_fn=act_fn)
    fc5 = _fc_layer(fc4,300,300,name="fc5", act_fn=act_fn)
    fc6 = _fc_layer(fc5,300,300,name="fc6", act_fn=act_fn)

    # defer softmax activation to loss calculation
    id_ = lambda x, name: x
    logits = _fc_layer(fc6,300,2,name="softmax",act_fn=id_)

    return logits

def loss(logits, labels):
    cross_entropy = (#tf.Print(
            tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                    labels=labels,
                                                    name="example_xentropy"))#,
            #[logits,labels], summarize=10)
    cross_entropy = tf.reduce_mean(cross_entropy)
    tf.add_to_collection("losses", cross_entropy)

    return tf.add_n(tf.get_collection('losses'), name="total_loss")

def accuracy(logits, labels):
    results = tf.nn.softmax(logits)
    correct_prediction = tf.equal(tf.argmax(results,1),tf.argmax(labels,1))
    accuracy = tf.reduce_mean(tf.to_float(correct_prediction))
    return accuracy

def current_epoch():
    return tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES,
                             "filename_queue/limit_epochs/epochs")[0]

def _learning_rate(epoch_op):
    with tf.name_scope("learning_rate"):
        """
        lr = tf.Variable(0.01)
        """
        lr1 = lambda:tf.constant(0.01)
        lr2 = lambda:tf.constant(0.001)
        lr3 = lambda:tf.constant(0.0001)
        lr = tf.case([(tf.greater_equal(epoch_op, 15),lr3),
                      (tf.greater_equal(epoch_op, 13),lr2)],
                     default=lr1)
    return lr


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

    lr = _learning_rate(current_epoch())

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


if __name__ == "__main__":
    example_batch,label_batch = mc_cnn_input.example_queue(data_dir="training", num_epochs=2)

    op = inference(example_batch[:,0,:,:,:],example_batch[:,1,:,:,:])

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        writer = tf.summary.FileWriter("mc-cnn_logs", graph=sess.graph)
        summarize = lambda s: None if s is None else writer.add_summary(sess.run(s))
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())
        threads = tf.train.start_queue_runners(coord=coord)

        sess.run(op)
        summarize(tf.summary.merge_all())

        coord.request_stop()
        coord.join(threads)
        writer.close()

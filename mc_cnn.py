#!/usr/bin/env python3

import tensorflow as tf
from flags import *

import mc_cnn_input


# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1       # Initial learning rate.

def _layer_summary(name,h,w,b):
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
    #init = tf.zeros([size])
    init = tf.random_normal([size],stddev=5e-1)
    return tf.Variable(init,name=name);

def _weight_variable(shape,name="weights"):
    init = tf.random_normal(shape,stddev=5e-1)
    return tf.Variable(init,name=name)

def _relu_weight_variable(shape,name="weights"):
    # use initialization suggested by He & al.
    # https://arxiv.org/pdf/1502.01852.pdf
    return _weight_variable(shape,name)
    init = tf.random_normal(shape)*tf.sqrt(2.0/tf.to_float(tf.reduce_prod(shape)))
    return tf.Variable(init,name=name)


def _conv_layer(input_,shape,name="conv",activation_fn=tf.nn.relu):
    with tf.name_scope(name) as scope:
        b = _bias_variable(shape[-1])
        w = _weight_variable(shape)

        wx = tf.nn.conv2d(input_,w,strides=[1,1,1,1],padding="VALID")
        z = tf.add(wx,b)
        activations = activation_fn(z, name=scope)
        _layer_summary(scope,activations,w,b)
    return activations

def _fc_layer(input_,input_size,output_size,name="fc",activation_fn=tf.nn.relu):
    with tf.name_scope(name) as scope:
        weight_shape = tf.stack([input_size,output_size])
        b = _bias_variable(output_size)
        weight_initializer = (_relu_weight_variable
                if activation_fn == tf.nn.relu
                else _weight_variable)

        w = weight_initializer(weight_shape)

        wx = tf.matmul(input_, w)
        z = tf.add(wx,b)
        activations = activation_fn(z, name=scope)
        _layer_summary(scope,activations,w,b)
    return activations

def _flatten(input_):
    return tf.reshape(input_,[tf.shape(input_)[0],-1], name="flatten")

def conv_inference(left, right, channels=1):
    
    def split_layers(input_):
        with tf.name_scope("conv_layers"):
            conv1 = _conv_layer(input_,[5,5,channels,32])
            conv2 = _conv_layer(conv1,[5,5,32,32])
            fc1 = _conv_layer(conv2,[1,1,32,200])

        return fc1
            
    """
    left = tf.Print(
            conv_layers(left),
            [left,right],summarize=10)
    """
    left = split_layers(left)
    right = split_layers(right)

    concat = tf.concat([left, right],axis=2)

    def soft_relu(x, name=None):
        with tf.name_scope(name or "soft_relu"):
            h = tf.maximum(0.0,x) + tf.minimum(0.0, x/10)
        return h

    fc3 = _conv_layer(concat,[1,1,400,300],name="fc3", activation_fn=soft_relu)
    fc4 = _conv_layer(fc3,[1,1,300,300],name="fc4", activation_fn=soft_relu)
    fc5 = _conv_layer(fc4,[1,1,300,300],name="fc5", activation_fn=soft_relu)
    fc6 = _conv_layer(fc5,[1,1,300,300],name="fc6", activation_fn=soft_relu)
    
    # defer softmax activation to loss calculation
    id_ = lambda x, name: x
    logits = _conv_layer(fc6,[1,1,300,2],name="softmax",activation_fn=id_)

    return logits

def inference(left, right, channels=1):
    
    def split_layers(input_):
        with tf.name_scope("conv_layers"):
            conv1 = _conv_layer(input_,[5,5,channels,32])
            fc1 = _fc_layer(_flatten(conv1),32*5*5,200)
            fc2 = _fc_layer(fc1,200,200)

        return fc1
            
    """
    left = tf.Print(
            conv_layers(left),
            [left,right],summarize=10)
    """
    left = split_layers(left)
    right = split_layers(right)

    concat = tf.concat([left, right],axis=1)

    def soft_relu(x, name=None):
        with tf.name_scope(name or "soft_relu"):
            h = tf.maximum(0.0,x) + tf.minimum(0.0, x/10)
        return h

    fc3 = _fc_layer(concat,400,300,name="fc3", activation_fn=soft_relu)
    fc4 = _fc_layer(fc3,300,300,name="fc4", activation_fn=soft_relu)
    fc5 = _fc_layer(fc4,300,300,name="fc5", activation_fn=soft_relu)
    fc6 = _fc_layer(fc5,300,300,name="fc6", activation_fn=soft_relu)
    
    # defer softmax activation to loss calculation
    id_ = lambda x, name: x
    logits = _fc_layer(fc6,300,2,name="softmax",activation_fn=id_)

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

    lr = tf.constant(0.00001)
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

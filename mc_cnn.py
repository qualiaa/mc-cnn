#!/usr/bin/env python3

import tensorflow as tf

import mc_cnn_input as examples

def _layer_summary(name,h,w,b):
    tf.summary.histogram('biases', b)
    tf.summary.histogram('weights', b)
    tf.summary.histogram('activations/hist', h)
    tf.summary.scalar('activations/sparsity',
                      tf.nn.zero_fraction(h))

def _bias_variable(size,name="bias"):
    init = tf.zeros([size])
    return tf.Variable(init,name=name);

def _weight_variable(shape,name="weights"):
    init = tf.random_normal(shape,stddev=5e-2)
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
        w = _weight_variable(weight_shape)

        wx = tf.matmul(input_, w)
        z = tf.add(wx,b)
        activations = activation_fn(z, name=scope)
        _layer_summary(scope,activations,w,b)
    return activations

def _flatten(input_):
    return tf.reshape(input_,[tf.shape(input_)[0],-1], name="flatten")


def inference(left, right, channels=1):
    
    def conv_layers(input_):
        with tf.name_scope("conv_layers"):
            conv1 = _conv_layer(input_,[5,5,channels,32])
            fc1 = _fc_layer(_flatten(conv1),32*5*5,200)
            fc2 = _fc_layer(fc1,200,200)

        return fc1
            
    left = conv_layers(left)
    right = conv_layers(right)

    concat = tf.concat([left, right],axis=1)

    fc3 = _fc_layer(concat,400,300)
    fc4 = _fc_layer(fc3,300,300)
    fc5 = _fc_layer(fc4,300,300)
    fc6 = _fc_layer(fc5,300,300)
    
    softmax = _fc_layer(fc6,300,2,name="softmax",activation_fn=tf.nn.softmax)

    return softmax

if __name__ == "__main__":
    example_batch,label_batch = examples.example_queue(data_dir="training", num_epochs=2)

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

#!/usr/bin/env python3

import tensorflow as tf

import mc_cnn_input as examples

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
        z = tf.add(wx,b,axis=2)
        activations = activation_fn(z, name=scope.name)
    return activations

def _fc_layer(input_,size,name="fc",activation_fn=tf.nn.relu):
    with tf.name_scope(name) as scope:
        weight_shape = tf.stack(tf.size(input_),size)
        b = _bias_variable(size)
        w = _weight_variable(weight_shape)

        wx = tf.matmul(w,input_)
        z = tf.add(wx,b)
        activations = activation_fn(z, name=scope.name)
    return activations


def inference(left, right, channels=1):
    
    def conv_layers(input_):
        with tf.name_scope("conv_layers"):
            conv1 = _conv_layer(input_,[1,5,5,channels])
            fc1 = _fc_layer(conv1,200)
            fc2 = _fc_layer(fc1,200)


        return fc2
            
    left = conv_layers(left)
    right = conv_layers(right)

    concat = tf.concat([left, right],axis=0)
    
    softmax = _fc_layer(concat,2,name="softmax",activation_fn=tf.nn.softmax)

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
        #summarize(tf.summary.merge_all())

        coord.request_stop()
        coord.join(threads)
        writer.close()

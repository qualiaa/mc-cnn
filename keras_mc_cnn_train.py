#!/usr/bin/env python3

import tensorflow as tf

from keras import backend as K
from flags import *
from LoggerHook import *
import keras_mc_cnn
import mc_cnn_input

string_flag("output_dir", "/tmp/mc_cnn_train",
            """Directory to save logs and learned parameters.""")
int_flag("max_steps", None,
         """Number of training batches.""")
int_flag('num_epochs', 16,
         """Number of repeated exposures to input data.""")
bool_flag('log_device_placement', False,
          """Whether to log device placement.""")
bool_flag("conv", False,
         """Use fully convolutional architecture""")

def validation(inference_fn=keras_mc_cnn.inference):
    # XXX: Very hacky to use large batch rather than full validation set.
    # Need to look into tf.contrib.data.Dataset for iterable queue
    batch_size = 100
    with tf.name_scope("validation"):
        with tf.device("/cpu:0"):
            validation_examples, validation_labels = mc_cnn_input.example_queue(
                    "validation",batch_size,shuffle=False)

        left_examples = validation_examples[:,0,...]
        right_examples = validation_examples[:,1,...]

        validation_logits = inference_fn(left_examples,right_examples)
        accuracy = keras_mc_cnn.accuracy(validation_logits,validation_labels)
    return accuracy

def train():
    """ set session """

    with tf.Graph().as_default():

        global_step = tf.train.get_or_create_global_step()
        examples, labels = mc_cnn_input.example_queue(
                "training",FLAGS.batch_size,shuffle=FLAGS.shuffle,
                                            num_epochs=FLAGS.num_epochs)

        with tf.name_scope("left_examples"):
            left_examples = examples[:,0,...]
        with tf.name_scope("right_examples"):
            right_examples = examples[:,1,...]
        
        inference = keras_mc_cnn.conv_inference if FLAGS.conv else keras_mc_cnn.inference

        logits = inference(left_examples,right_examples,flatten=True)

        loss = keras_mc_cnn.loss(logits,labels)
        train_op = keras_mc_cnn.train(loss,global_step)

        validation_accuracy = validation(inference)

        hooks = [tf.train.NanTensorHook(loss),
                 LoggerHook(loss,validation_accuracy)]
        if FLAGS.max_steps:
            hooks.append(tf.train.StopAtStepHook(last_step=FLAGS.max_steps))

        with tf.train.MonitoredTrainingSession(
                checkpoint_dir=FLAGS.output_dir,
                hooks=hooks,
                config=tf.ConfigProto(
                    log_device_placement=FLAGS.log_device_placement)
                ) as mon_sess:
            K.set_session(mon_sess)
            #mon_sess.run(tf.global_variables_initializer())
            while not mon_sess.should_stop():
                mon_sess.run(train_op)

def main(argv=None):
    if tf.gfile.Exists(FLAGS.output_dir):
        tf.gfile.DeleteRecursively(FLAGS.output_dir)
    tf.gfile.MakeDirs(FLAGS.output_dir)
    train()
                   
if __name__ == "__main__":
    tf.app.run()

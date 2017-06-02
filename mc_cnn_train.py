#!/usr/bin/env python3

import tensorflow as tf
from flags import *

import time
from datetime import datetime

import mc_cnn
import mc_cnn_input

string_flag("output_dir", "/tmp/mc_cnn_train",
            """Directory to save logs and learned parameters.""")
int_flag("max_steps", 1000000,
         """Number of training batches.""")
int_flag("log_frequency", 10,
         """How often to print logs in terms of batches seen.""")
int_flag('num_epochs', None,
         """Number of repeated exposures to input data.""")
bool_flag('log_device_placement', False,
          """Whether to log device placement.""")

def train():
    with tf.Graph().as_default():
        global_step = tf.contrib.framework.get_or_create_global_step()
        examples, labels = mc_cnn_input.example_queue(
                "training",shuffle=FLAGS.shuffle)

        with tf.name_scope("left_examples"):
            left_examples = examples[:,0,...]
        with tf.name_scope("right_examples"):
            right_examples = examples[:,1,...]
        logits = mc_cnn.inference(left_examples,right_examples)
        loss = mc_cnn.loss(logits,labels)
        train_op = mc_cnn.train(loss,global_step)

        class LoggerHook(tf.train.SessionRunHook):
            def begin(self):
                self._step = -1
                self._start_time = time.time()

            def before_run(self, run_context):
                self._step += 1
                return tf.train.SessionRunArgs(loss)

            def after_run(self, run_context, run_values):
                if self._step % FLAGS.log_frequency == 0:
                    current_time = time.time()
                    duration = current_time - self._start_time
                    self._start_time = current_time

                    loss_value = run_values.results
                    examples_per_sec = (FLAGS.log_frequency * 
                            FLAGS.batch_size/duration)
                    sec_per_batch = duration / float(FLAGS.log_frequency)
                    loss_value = run_values.results

                    format_str = ("{}: step {:d}, loss = {:.2f} "
                            "({:.1f} examples/sec; {:.3f} sec/batch)")
                    print (format_str.format(datetime.now(), self._step, loss_value,
                                   examples_per_sec, sec_per_batch))

        with tf.train.MonitoredTrainingSession(
                checkpoint_dir=FLAGS.output_dir,
                hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
                       tf.train.NanTensorHook(loss),
                       LoggerHook()],
                config=tf.ConfigProto(
                    log_device_placement=FLAGS.log_device_placement)
                ) as mon_sess:
            while not mon_sess.should_stop():
                mon_sess.run(train_op)

def main(argv=None):
    if tf.gfile.Exists(FLAGS.output_dir):
        tf.gfile.DeleteRecursively(FLAGS.output_dir)
    tf.gfile.MakeDirs(FLAGS.output_dir)
    train()
                   
if __name__ == "__main__":
    tf.app.run()

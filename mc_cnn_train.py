#!/usr/bin/env python3

import tensorflow as tf
from flags import *

import time
from datetime import datetime

import mc_cnn
import mc_cnn_input

string_flag("output_dir", "/tmp/mc_cnn_train",
            """Directory to save logs and learned parameters.""")
int_flag("max_steps", None,
         """Number of training batches.""")
int_flag("log_frequency", 10,
         """How often to print logs in terms of batches seen.""")
int_flag('num_epochs', 16,
         """Number of repeated exposures to input data.""")
bool_flag('log_device_placement', False,
          """Whether to log device placement.""")
int_flag("validation_steps", 10000,
         """Number of batches between evaluations""")
bool_flag("conv", False,
         """Use fully convolutional architecture""")

def validation(inference_fn=mc_cnn.inference):
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
        accuracy = mc_cnn.accuracy(validation_logits,validation_labels)
    return accuracy

def train():
    with tf.Graph().as_default():
        global_step = tf.contrib.framework.get_or_create_global_step()
        examples, labels = mc_cnn_input.example_queue(
                "training",FLAGS.batch_size,shuffle=FLAGS.shuffle,
                                            num_epochs=FLAGS.num_epochs)

        with tf.name_scope("left_examples"):
            left_examples = examples[:,0,...]
        with tf.name_scope("right_examples"):
            right_examples = examples[:,1,...]
        
        inference = mc_cnn.conv_inference if FLAGS.conv else mc_cnn.inference

        logits = inference(left_examples,right_examples)

        loss = mc_cnn.loss(logits,labels)
        train_op = mc_cnn.train(loss,global_step)

        validation_accuracy = validation(inference)

        epoch_op = mc_cnn.current_epoch()

        class LoggerHook(tf.train.SessionRunHook):
            def begin(self):
                self._step = -1
                self._start_time = time.time()
                with open("accuracy.csv","w") as f:
                    f.write("Step,Accuracy\n")

            def before_run(self, run_context):
                self._step += 1
                return tf.train.SessionRunArgs(loss)

            def after_run(self, run_context, run_values):
                sess = run_context.session
                if self._step % FLAGS.log_frequency == 0:
                    current_time = time.time()
                    duration = current_time - self._start_time
                    self._start_time = current_time

                    loss_value = run_values.results
                    examples_per_sec = (FLAGS.log_frequency * 
                            FLAGS.batch_size/duration)
                    sec_per_batch = duration / float(FLAGS.log_frequency)
                    loss_value = run_values.results

                    epoch = sess.run(epoch_op)

                    format_str = ("{}: epoch {:d}, step {:d}, loss = {:.2f} "
                            "({:.1f} examples/sec; {:.3f} sec/batch)")
                    print (format_str.format(datetime.now(), epoch,
                                   self._step, loss_value,
                                   examples_per_sec, sec_per_batch))

                if (self._step+1) % FLAGS.validation_steps == 0:
                    print("Running validation...")
                    accuracy = 0
                    n_batches = 1000
                    for _ in range(n_batches):
                        accuracy += sess.run(validation_accuracy)
                    accuracy /= n_batches
                    print("Accuracy: {} %".format(100*accuracy))
                    with open("accuracy.csv","a") as f:
                        f.write("{:d},{:f}\n".format(self._step,accuracy))

        hooks = [tf.train.NanTensorHook(loss),
                 LoggerHook()]
        if FLAGS.max_steps:
            hooks.append(tf.train.StopAtStepHook(last_step=FLAGS.max_steps))

        with tf.train.MonitoredTrainingSession(
                checkpoint_dir=FLAGS.output_dir,
                hooks=hooks,
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

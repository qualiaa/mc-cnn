from flags import *

import time
from datetime import datetime

int_flag("log_frequency", 10,
         """How often to print logs in terms of batches seen.""")
int_flag("validation_steps", 10000,
         """Number of batches between evaluations""")

class LoggerHook(tf.train.SessionRunHook):
    def __init__(self,loss):
        self._loss = loss

    def begin(self):
        self._step = -1
        self._start_time = time.time()
        with open("accuracy.csv","w") as f:
            f.write("Step,Accuracy\n")

    def before_run(self, run_context):
        self._step += 1
        return tf.train.SessionRunArgs(self._loss)

    def print_log(self,run_context,run_values):
        current_time = time.time()
        duration = current_time - self._start_time
        self._start_time = current_time

        loss_value = run_values.results
        examples_per_sec = (FLAGS.log_frequency * 
                FLAGS.batch_size/duration)
        sec_per_batch = duration / float(FLAGS.log_frequency)
        loss_value = run_values.results

        sess = run_context.session
        epoch = 0#sess.run(epoch_op)

        format_str = ("{}: epoch {:d}, step {:d}, loss = {:.2f} "
                "({:.1f} examples/sec; {:.3f} sec/batch)")
        print (format_str.format(datetime.now(), epoch,
                       self._step, loss_value,
                       examples_per_sec, sec_per_batch))
    
    def run_validation(self, run_context, run_values):
        sess = run_context.session
        print("Running validation...")
        accuracy = 0
        n_batches = 1000
        for _ in range(n_batches):
            accuracy += sess.run(validation_accuracy)
        accuracy /= n_batches
        print("Accuracy: {} %".format(100*accuracy))
        with open("accuracy.csv","a") as f:
            f.write("{:d},{:f}\n".format(self._step,accuracy))

    def after_run(self, run_context, run_values):
        if self._step % FLAGS.log_frequency == 0:
            self.print_log(run_context,run_values)

        if (self._step+1) % FLAGS.validation_steps == 0:
            self.run_validation(run_context,run_values)


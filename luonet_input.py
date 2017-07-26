#!/usr/bin/env python3

import tensorflow as tf
from flags import *

import os
import sys
import re

from operator import add, truth
from functools import reduce
from itertools import repeat

string_flag("data_root", "luonet_data",
            """Relative path from execution dir to KITTI dataset.""")
string_flag("infile_regex", "(\d{6})_10.tfrecord",
            """Regular expression describing input image filenames.""")
int_flag('queue_threads', 1,
         """Number of CPU threads for queuing examples.""")

int_flag("l0", 0.5,
         """Correct pixel probability. Probabilities must sum to 1.""")
int_flag("l1", 0.2,
         """1px error probability. Probabilities must sum to 1.""")
int_flag("l2", 0.05,
         """2px error probability. Probabilities must sum to 1.""")
int_flag('batch_size', 100,
         """Number of examples per batch.""")
bool_flag("low_gpu_mem", False,
          """Reduce the amount of data stored on the GPU, reducing """
          """performance""")

bool_flag('shuffle', True,
          """Whether to shuffle input files.""")

min_after_dequeue = 1000


def _sorted_file_lists(data_dir, sub_dirs):
    """ convert a list of dataset folders into a list of lists of files from
    those folders matching a regex """
    # file_lists is a list of possibly many directory listings
    file_lists = [os.listdir(os.path.join(FLAGS.data_root, data_dir, sub_dir))
            for sub_dir in sub_dirs]

    # replace filename strings with regex Match instances
    match_fn = lambda f: re.fullmatch(FLAGS.infile_regex, f)
    file_lists = [[*map(match_fn, l)] for l in file_lists]

    # remove non-matches
    file_lists = [[*filter(truth,l)] for l in file_lists]

    def sort_key(m):
        """ sort the matches by the concatenated integer groups """
        return int(reduce(add,m.groups()))

    file_lists = [sorted(l, key=sort_key) for l in file_lists]

    # TODO: check sort_key for files in same row is the same
    # (can transpose here using sort_key as pivot)

    # convert back to filenames with full paths
    match_to_path = lambda m,d: os.path.join(FLAGS.data_root, data_dir, d, m.string)
    file_lists = [[*map(match_to_path, l, repeat(sub_dir))]
            for l, sub_dir
            in zip(file_lists, sub_dirs)]

    # transpose list structure
    file_lists = [list(x) for x in zip(*file_lists)]

    file_lists = tf.convert_to_tensor(file_lists, dtype=tf.string,
                                                  name="input_paths")

    return file_lists

def _sorted_file_list(data_dir):
    # file_lists is a list of possibly many directory listings
    file_list = os.listdir(os.path.join(FLAGS.data_root,data_dir))

    # replace filename strings with regex Match instances
    file_list = [re.fullmatch(FLAGS.infile_regex, f) for f in file_list]

    # remove non-matches
    file_list = filter(truth,file_list)

    # sort 
    def sort_key(m):
        """ sort the matches by the concatenated integer groups """
        return int(reduce(add,m.groups()))
    file_list = sorted(file_list, key=sort_key)

    # convert back to filenames with full paths
    file_list = [os.path.join(FLAGS.data_root, data_dir,m.string) for m in file_list]
    print(file_list)


    return file_list

def read_record_file(filename_queue,patch_size=9,max_disparity=128,channels=1,
        normalize=True):
    """ Reads filenames from a queue three at a time to acquire pair + ground
        truth. Assumes order is [ground_truth, left image, right image]
    """
    with tf.name_scope("load_files"):
        compression = tf.python_io.TFRecordCompressionType.ZLIB
        options = tf.python_io.TFRecordOptions(compression)

        reader = tf.TFRecordReader(options=options)
        key, record = reader.read(filename_queue)
        example = tf.parse_single_example(record, features={
                'left': tf.FixedLenFeature((),tf.string),
                'right': tf.FixedLenFeature((),tf.string),
                'label': tf.FixedLenFeature((max_disparity),tf.float32)
            })

        right_patch_width = (patch_size - 1) + max_disparity
        left = tf.decode_raw(example['left'],tf.uint8)
        left = tf.to_float(left)
        left = tf.reshape(left,(patch_size,patch_size,channels))
        right = tf.decode_raw(example['right'],tf.uint8)
        right = tf.to_float(right)
        right = tf.reshape(right,(patch_size,right_patch_width,channels))
        label = example['label']
        if normalize:
            left = left/255.0
            right = right/255.0

        ed = tf.expand_dims
        tf.summary.image("label",ed(ed(ed(label,1),0),0))
        tf.summary.image("left_image",tf.to_float(ed(left,0)))
        tf.summary.image("right_image",tf.to_float(ed(right,0)))

        #label = tf.Print(label,[tf.shape(label)],"")
        print(label.shape)

        return left,right,label

def batch_examples(left,right,labels,
        batch_size,
        patch_size=9,
        max_disparity=128,
        shuffle=False,
        channels=1):

    input_tensor=[left,right,labels]
    kwargs={"shapes":[(patch_size,patch_size,channels),
                      (patch_size,(patch_size-1)+max_disparity,channels),
                      (max_disparity)],
            "capacity":(min_after_dequeue + (FLAGS.batch_size * 1.1) *
                FLAGS.queue_threads),
            "enqueue_many":False,
            "batch_size":batch_size,
            "num_threads":FLAGS.queue_threads}

    with tf.name_scope("example_batches"):
        if shuffle:
            batch = tf.train.shuffle_batch(
                    input_tensor,
                    min_after_dequeue=min_after_dequeue,
                    **kwargs)
            return batch
        else:
            batch = tf.train.batch(
                    input_tensor,
                    **kwargs)
            return batch

def example_queue(data_dir,
        batch_size,
        shuffle=True,
        num_epochs=None,
        patch_size=9,
        max_disparity=128,
        color_channels=1):
    filenames = _sorted_file_list(data_dir)
    filename_queue = tf.train.string_input_producer(filenames,
                                                    shuffle=shuffle,
                                                    num_epochs=num_epochs)
    input_data = read_record_file(filename_queue,
                                  patch_size=patch_size,
                                  max_disparity=max_disparity)
    return batch_examples(*input_data,batch_size,
                          patch_size=patch_size,
                          max_disparity=max_disparity,
                          shuffle=shuffle)

if __name__ == "__main__":
    left,right,label = example_queue("training",100)
    op = tf.reduce_sum(left)
    #op = example_queue("training",100)
    with tf.Session() as sess:
        coord = tf.train.Coordinator()

        writer = tf.summary.FileWriter("luoin_logs", graph=sess.graph)
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())
        threads = tf.train.start_queue_runners(coord=coord)

        summary_op = tf.summary.merge_all()
        writer.add_summary(sess.run(summary_op))
        
        print(sess.run(op))

        coord.request_stop()
        coord.join(threads)
        writer.close()


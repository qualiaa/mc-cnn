#!/usr/bin/env python3

import tensorflow as tf
from flags import *

import os
import sys
import re

from operator import add, truth
from functools import reduce
from itertools import repeat

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
int_flag('batch_size', 128,
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
    file_lists = [os.listdir(os.path.join(data_dir, sub_dir))
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
    match_to_path = lambda m,d: os.path.join(data_dir, d, m.string)
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
    file_list = os.listdir(data_dir)

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
    file_list = [os.path.join(data_dir,m.string) for m in file_list]

    return file_list

def read_record_file(filename_queue,patch_size=9,max_disparity=128,channels=1):
    """ Reads filenames from a queue three at a time to acquire pair + ground
        truth. Assumes order is [ground_truth, left image, right image]
    """
    with tf.name_scope("load_files"):
        reader = tf.TFRecordReader()
        key, record = reader.read(filename_queue)
        example = tf.parse_single_example(record, features={
                'left': tf.FixedLenFeature((),tf.string),
                'right': tf.FixedLenFeature((),tf.string),
                'label': tf.FixedLenFeature((128),tf.float32)
            })

        left = tf.decode_raw(example['left'],tf.uint8)
        left = tf.reshape(left,(patch_size,patch_size,channels))
        right = tf.decode_raw(example['right'],tf.uint8)
        right = tf.reshape(right,(patch_size,max_disparity+patch_size,channels))
        label = example['label']

        tf.summary.image("label",tf.expand_dims(tf.expand_dims(label,0),0))
        tf.summary.image("left_image",tf.to_float(tf.expand_dims(left,0)))
        tf.summary.image("right_image",tf.to_float(tf.expand_dims(right,0)))

    return (left,right), label

def examples_from_stereo_pair(stereo_pair, gt, max_disparity=128, patch_size=9, channels=1):
    pass
"""
    l, r = stereo_pair
    with tf.name_scope("example_generation"):
        gt = tf.identity(gt, name="ground_truth_image")
        zero = tf.constant(0, dtype=gt.dtype)
        image_shape = tf.to_int64(tf.shape(gt))

        def normalize(i):
            with tf.name_scope("normalize"):
                i = i - tf.reduce_mean(i)
                i = i / tf.sqrt(tf.reduce_mean(i**2))
            return i

        l,r = [normalize(i) for i in [l, r]]

        def extract_patches(t, height=patch_size, width=patch_size, name=None):
"""
"""         take [x,y,channel] image
            return [x,y,patch_size*patch_size*channels] patches """
"""
            with tf.name_scope(name or "extract_patches"):
                patch_dims = [1, height, width, channels]
                strides = [1, 1, 1, channels]
                rates = [1, 1, 1, 1]
                padding = "SAME"
                patches = tf.extract_image_patches(
                        tf.expand_dims(t,0), patch_dims, strides, rates, padding)
                patches = tf.squeeze(patches)
            return patches

        def reshape_patches(p, height=patch_size, width=patch_size):
"""
"""         take [x,y,patch_size*patch_size*channels] patches and
            return [x*y,patch_width,patch_width,channels] """
"""
            return tf.reshape(p, [-1, height, width, channels])

        right_patch_width = max_disparity + patch_size
        left_patches = extract_patches(l,name="left_patches")
        right_patches = extract_patches(r,width=right_patch_width,name="right_patches")

        with tf.name_scope("ground_truth_coordinates"):
            # ground truth is sparse, so only keep patches whose class is known
            gt_non_zero = tf.greater(gt, zero)
            known_gt_coords = tf.where(gt_non_zero,name="find_gt_coords")
            gt_values = tf.gather_nd(gt, known_gt_coords, name="index_gt")

            # restrict matches to those for which we have a correct match patch
            # (ie, x > 0)
            valid_indices = tf.where(
                    tf.greater(known_gt_coords[:,1] - gt_values, 0),
                    name="find_valid_indices")
            known_gt_coords = tf.gather_nd(known_gt_coords, valid_indices,
                    name="find_valid_coords")
            gt_values = tf.gather_nd(gt, known_gt_coords, name="valid_gt")

        left_patches = tf.gather_nd(left_patches, known_gt_coords,
                                    name="index_left_patches")

        # calculate a good example and bad example from right image
        num_gt = tf.shape(known_gt_coords,name="num_examples")[0]
        right_patch_offset = right_patch_width // 2 - patch_size // 2 
        print("Right offset is {}".format(right_patch_offset))
        right_patch_offset = tf.constant([0,right_patch_offset],dtype=tf.int64)
        right_patch_coords = known_gt_coords + right_patch_offset
        right_patches_are_valid = tf.where(tf.less_equal(right_patch_coords[:,1],image_shape[1]))
        right_patch_coords = tf.gather_nd(right_patch_coords,right_patches_are_valid)
        right_patches = tf.gather_nd(right_patches,right_patch_coords,name="right_patches")

        # remove left patches and gt without a valid right patch
        left_patches = tf.gather_nd(left_patches,right_patches_are_valid)
        gt_values = tf.gather(gt_values,right_patches_are_valid)


        left_patches = reshape_patches(left_patches)
        right_patches = reshape_patches(right_patches,width=right_patch_width)

        tf.summary.image("left_patches",left_patches)
        tf.summary.image("right_patches",right_patches)

        with tf.name_scope("label_generation"):
            l0=FLAGS.l0;l1=FLAGS.l1;l2=FLAGS.l2
            fill_val = tf.reshape(tf.tile([l2,l1,l0,l1,l2], [max_disparity]),
                                  [max_disparity,5])
            indices = tf.stack([gt_values-2,
                                gt_values-1,
                                gt_values,
                                gt_values+1,
                                gt_values+2],
                               axis=1)
            shape = [tf.shape(gt_values)[0],tf.constant(max_disparity)]
            labels = tf.scatter_nd(indices=indices,updates=fill_val,shape=shape)
            labels = tf.Print(labels,[labels])



        # use CPU to avoid OOM on GPU
        #dev = "/cpu:0" if FLAGS.low_gpu_mem else None
        #with tf.device(dev):
        examples = tf.stack([left_patches,right_patches], axis=1)

        one_v = tf.ones_like(gt_values)
        zero_v = tf.zeros_like(gt_values)
        labels = tf.concat([tf.stack([one_v,zero_v],axis=1),
                            tf.stack([zero_v,one_v],axis=1)], axis=0)

        # examples and labels must be the same length
        tf.assert_equal(tf.shape(examples)[0],tf.shape(labels)[0])
        shuffle_indices = tf.random_shuffle(tf.range(tf.shape(examples)[0]))
        examples = tf.gather(examples,shuffle_indices)
        labels = tf.gather(labels,shuffle_indices)

        #tf.summary.image("left_patches",left_patches)
        #tf.summary.image("pos_patches",pos_patches)
        #tf.summary.image("neg_patches",neg_patches)

    return examples,labels
        """

def batch_examples(left,right,labels,
        batch_size,
        patch_size=9,
        max_disparity=128,
        shuffle=False,
        channels=1):

    input_tensor=[left,right,labels]
    kwargs={"shapes":[(patch_size,patch_size,channels),
                      (patch_size,patch_size+max_disparity,channels),
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
    examples, labels = read_record_file(filename_queue,
                                        patch_size=patch_size,
                                        max_disparity=max_disparity)
    return batch_examples(*examples,labels,batch_size,
                          patch_size=patch_size,
                          max_disparity=max_disparity,
                          shuffle=shuffle)

if __name__ == "__main__":
    op = example_queue("luonet_data/training",100)
    with tf.Session() as sess:
        sess.run(op)
        writer = tf.summary.FileWriter("luoin_logs", graph=sess.graph)
        summarize = lambda s: None if s is None else writer.add_summary(sess.run(s))
        summarize(tf.summary.merge_all())
        writer.close()


#!/usr/bin/env python3

import tensorflow as tf
from flags import *

import os
import sys
import re

from operator import add, truth
from functools import reduce
from itertools import repeat

string_flag("kitti_root", "kitti_stereo_2012",
            """Relative path from execution dir to KITTI dataset.""")
string_flag("infile_regex", "(\d{6})_10.png",
            """Regular expression describing input image filenames.""")
int_flag('queue_threads', 1,
         """Number of CPU threads for queuing examples.""")

int_flag("n_low", 4,
         """Lower bound for negative example offset from ground truth.""")
int_flag("n_high", 8,
         """Upper bound for negative example offset from ground truth.""")
int_flag("p_high", 1,
         """Upper bound for positive example offset from ground truth.""")
int_flag('batch_size', 128,
         """Number of examples per batch.""")
bool_flag("low_gpu_mem", False,
          """Reduce the amount of data stored on the GPU, reducing """
          """performance""")

bool_flag('shuffle', True,
          """Whether to shuffle input files.""")

min_after_dequeue = 1000

def _random_sign(shape=[], name = None):
    with tf.name_scope(name or "random_sign"):
        val =  tf.random_uniform(shape,
                minval=0,
                maxval=2,
                dtype=tf.int64, name="random_sign")*2-1
    return val


def _sorted_file_lists(data_dir, sub_dirs):
    """ convert a list of dataset folders into a list of lists of files from
    those folders matching a regex """
    # file_lists is a list of possibly many directory listings
    file_lists = [os.listdir(os.path.join(FLAGS.kitti_root, data_dir, sub_dir))
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
    match_to_path = lambda m,d: os.path.join(FLAGS.kitti_root, data_dir, d, m.string)
    file_lists = [[*map(match_to_path, l, repeat(sub_dir))]
            for l, sub_dir
            in zip(file_lists, sub_dirs)]

    # transpose list structure
    file_lists = [list(x) for x in zip(*file_lists)]

    file_lists = tf.convert_to_tensor(file_lists, dtype=tf.string,
                                                  name="input_paths")

    return file_lists

def kitti_filename_queue(data_dir, capacity=30, shuffle=True, num_epochs=None):
    """ produce FIFOQueue which produces ground truth, left and right
    images paths in order
    
    similar to string_input_producer, but for a 2D string tensor
    https://www.tensorflow.org/api_docs/python/tf/train/string_input_producer
    https://github.com/tensorflow/tensorflow/blob/r1.1/tensorflow/python/training/input.py#L174
    """

    with tf.name_scope("filename_queue"):

        sub_dirs = ["disp_occ","image_0","image_1"]
        files = _sorted_file_lists(data_dir,sub_dirs)

        if shuffle:
            files = tf.random_shuffle(files)

        files = tf.train.limit_epochs(files, num_epochs)


        files = tf.reshape(files, [-1], name="flatten")

        q = tf.FIFOQueue(capacity, [tf.string])
        enq = q.enqueue_many(files)
        tf.train.add_queue_runner(
                tf.train.QueueRunner(q, [enq]))

    return q

def read_stereo_pair(filename_queue, channels=1):
    """ Reads filenames from a queue three at a time to acquire pair + ground
        truth. Assumes order is [ground_truth, left image, right image]
    """
    with tf.name_scope("load_files"):
        reader = tf.WholeFileReader()
        def read_image(channels, dtype=tf.uint8,name=None):
            with tf.name_scope(name or "read_png"):
                key, png = reader.read(filename_queue)
                uint_image = tf.image.decode_png(png,channels=channels,
                                                     dtype=dtype)
                float_image = tf.to_float(uint_image)
            return float_image

        gt = read_image(channels=1,dtype=tf.uint16,name="ground_truth")/256
        gt = tf.squeeze(gt)
        gt = tf.to_int64(tf.round(gt))
        with tf.control_dependencies([gt]):
            l = read_image(channels,name="left_image")
            with tf.control_dependencies([l]):
                r = tf.Print(read_image(channels,name="right_image"),
                        [[]],"Loaded next stereo pair")

        """
        tf.summary.image("ground_truth",tf.to_float(tf.expand_dims(tf.expand_dims(gt,0),3)))
        tf.summary.image("left_image",tf.expand_dims(l,0))
        tf.summary.image("right_image",tf.expand_dims(r,0))
        """

    stereo_pair = (l,r)

    return stereo_pair, gt

def examples_from_stereo_pair(stereo_pair, gt, patch_size=9, channels=1):
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

        def extract_patches(t, name=None):
            """ take [x,y,channel] image
            return [x,y,patch_size*patch_size*channels] patches """
            with tf.name_scope(name or "extract_patches"):
                patch_dims = [1, patch_size, patch_size, channels]
                strides = [1, 1, 1, channels]
                rates = [1, 1, 1, 1]
                padding = "SAME"
                patches = tf.extract_image_patches(
                        tf.expand_dims(t,0), patch_dims, strides, rates, padding)
                patches = tf.squeeze(patches)
            return patches

        def reshape_patches(p):
            """ take[x,y,patch_size*patch_size*channels] patches and
            return [x*y,patch_width,patch_width,channels] """
            return tf.reshape(p, [-1, patch_size, patch_size, channels])

        left_patches = extract_patches(l,name="left_patches")
        right_patches = extract_patches(r,name="right_patches")

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
        def choose_example(offset_min, offset_max, name=None):
            with tf.name_scope(name or "random_offset"):
                num_gt = tf.shape(known_gt_coords,name="num_examples")[0]
                offsets = _random_sign([num_gt])* tf.random_uniform([num_gt],
                                                          minval=offset_min,
                                                          maxval=offset_max,
                                                          dtype=gt.dtype,
                                                          name="x_offsets")
                offsets = tf.add(gt_values, offsets,name="offsets_plus_gt")
                zeros = tf.zeros([num_gt],dtype=gt.dtype,name="zeros")
                offsets = tf.stack([zeros,offsets], 1,name="xy_offsets")
                coords = tf.subtract(known_gt_coords, offsets,name="coords")
                coords = tf.maximum(coords, 0)
                coords = tf.minimum(coords, image_shape-1)
                patches = tf.gather_nd(right_patches,coords,name="patches")
            return patches

        pos_patches = choose_example(0,FLAGS.p_high,name="positive_examples")
        neg_patches = choose_example(
                FLAGS.n_low, FLAGS.n_high, name="negative_examples")

        left_patches = reshape_patches(left_patches)
        pos_patches = reshape_patches(pos_patches)
        neg_patches = reshape_patches(neg_patches)

        # use CPU to avoid OOM on GPU
        dev = "/cpu:0" if FLAGS.low_gpu_mem else None
        with tf.device(dev):
            # join left patches to both positive and negative matches
            positive_examples = tf.stack([left_patches,pos_patches], axis=1)
            negative_examples = tf.stack([left_patches,neg_patches], axis=1)
            examples = tf.concat([positive_examples,negative_examples],axis=0)

            one_v = tf.ones_like(gt_values)
            zero_v = tf.zeros_like(gt_values)
            labels = tf.concat([tf.stack([one_v,zero_v],axis=1),
                                tf.stack([zero_v,one_v],axis=1)], axis=0)

            # examples and labels must be the same length
            tf.assert_equal(tf.shape(examples)[0],tf.shape(labels)[0])
            shuffle_indices = tf.random_shuffle(tf.range(tf.shape(examples)[0]))
            examples = tf.gather(examples,shuffle_indices)
            labels = tf.gather(labels,shuffle_indices)

        """
        tf.summary.image("left_patches",left_patches)
        tf.summary.image("pos_patches",pos_patches)
        tf.summary.image("neg_patches",neg_patches)
        """

    return examples,labels

def batch_examples(examples,labels,
        batch_size,
        shuffle=False,
        channels=1,
        window_size=9,
        enqueue_many=True):
    input_tensor=[examples,labels]
    kwargs={"shapes":[[2,window_size,window_size,channels],[2]],
            "capacity":(min_after_dequeue + (FLAGS.batch_size *1.1) *
                FLAGS.queue_threads),
            "enqueue_many":enqueue_many,
            "batch_size":batch_size,
            "num_threads":FLAGS.queue_threads}

    with tf.name_scope("example_batches"):
        if shuffle:
            example_batch, label_batch = tf.train.shuffle_batch(
                    input_tensor,
                    min_after_dequeue=min_after_dequeue,
                    **kwargs)
            return example_batch, label_batch
        else:
            example_batch, label_batch = tf.train.batch(
                    input_tensor,
                    **kwargs)
            return example_batch, label_batch
        """
        q = tf.RandomShuffleQueue(
                capacity=min_after_dequeue +
                (FLAGS.batch_size*1.1)*FLAGS.queue_threads,
                min_after_dequeue=min_after_dequeue,
                dtypes=[examples.dtype,labels.dtype],
                names=["examples","labels"],
                shapes=[[2,window_size,window_size,channels],[]])
        enq = q.enqueue_many({"examples":examples,
                              "labels":labels})
        tf.train.add_queue_runner(
                tf.train.QueueRunner(q,[enq]))

        deque = q.dequeue_many(batch_size)
    return deque
    """

def example_queue(data_dir,
        batch_size,
        shuffle=True,
        num_epochs=None,
        patch_size=9,
        color_channels=1):
    filename_queue = kitti_filename_queue(data_dir,
                                          shuffle=shuffle,
                                          num_epochs=num_epochs)
    stereo_pair, gt = read_stereo_pair(filename_queue,channels=color_channels)
    examples, labels = examples_from_stereo_pair(stereo_pair,gt,
                                                 channels=color_channels)
    return batch_examples(examples,labels,batch_size,shuffle=shuffle)

#!/usr/bin/env python3

import tensorflow as tf

import os
import sys
import re

from operator import add, truth
from functools import reduce
from itertools import repeat


N_low = 4
N_high = 8
P_high = 1

batch_size = 1000
min_after_dequeue = 1000
num_threads = 1

root_dir = "../../stereo/kitti_stereo_2012/"
infile_regex = "(\d{6})_10.png"

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
    file_lists = [os.listdir(os.path.join(root_dir, data_dir, sub_dir))
            for sub_dir in sub_dirs]

    # replace filename strings with regex Match instances
    match_fn = lambda f: re.fullmatch(infile_regex, f)
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
    match_to_path = lambda m,d: os.path.join(root_dir, data_dir, d, m.string)
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


        #files = tf.Print(tf.reshape(files, [-1]), [files])
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
                uint_image = tf.image.decode_png(png,channels=channels,dtype=dtype)
                float_image = tf.Print(tf.to_float(uint_image),[key])
            return float_image

        gt = read_image(channels=1,dtype=tf.uint16,name="ground_truth")/256
        gt = tf.squeeze(gt)
        gt = tf.to_int64(tf.round(gt))
        with tf.control_dependencies([gt]):
            l = read_image(channels,name="left_image")
            with tf.control_dependencies([l]):
                r = read_image(channels,name="right_image")

        tf.summary.image("ground_truth",tf.to_float(tf.expand_dims(tf.expand_dims(gt,0),3)))
        tf.summary.image("left_image",tf.expand_dims(l,0))
        tf.summary.image("right_image",tf.expand_dims(r,0))

    stereo_pair = (l,r)

    return stereo_pair, gt

def examples_from_stereo_pair(stereo_pair, gt, patch_size=9, channels=1):
    l, r = stereo_pair
    with tf.name_scope("example_generation"):
        gt = tf.identity(gt, name="ground_truth_image")
        zero = tf.constant(0, dtype=gt.dtype)
        image_shape = tf.to_int64(tf.shape(gt))

        num_gt_outputs = 3
        #summaries = []

        def extract_patches(t, name=None):
            """ take [x,y,channel] image
            return [x,y,patch_size*patch_size*channels] patches """
            with tf.name_scope(name or "extract_patches"):
                patch_dims = [1, patch_size, patch_size, channels]
                strides = [1, 1, 1, channels]
                rates = [1, 1, 1, 1]
                padding = "SAME"
                patches = tf.extract_image_patches(tf.expand_dims(t,0), patch_dims, strides, rates, padding)
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
            known_gt_coords = tf.where(tf.greater(gt, zero),name="find_gt_coords")
            gt_values = tf.gather_nd(gt, known_gt_coords, name="index_gt")

            # restrict matches to those for which we have a correct match patch
            # (ie, x > 0)
            valid_indices = tf.where(
                    tf.greater(known_gt_coords[:,1] - gt_values, 0),
                    name="find_valid_indices")
            known_gt_coords = tf.gather_nd(known_gt_coords, valid_indices,
                    name="find_valid_coords")
            gt_values = tf.Print(
                    tf.gather_nd(gt, known_gt_coords, name="valid_gt"),
                    [
                        tf.reduce_max(known_gt_coords,axis=0),
                        tf.reduce_min(known_gt_coords,axis=0),
                        image_shape,
                        tf.shape(left_patches)
                    ],
                    "Max index vs ")

        left_patches = tf.gather_nd(left_patches, known_gt_coords, name="index_left_patches")

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

        pos_patches = choose_example(0,P_high,name="positive_examples")
        neg_patches = choose_example(N_low,N_high,name="negative_examples")

        left_patches = reshape_patches(left_patches)
        pos_patches = reshape_patches(pos_patches)
        neg_patches = reshape_patches(neg_patches)

        # join left patches to both positive and negative matches
        positive_examples = tf.stack([left_patches,pos_patches], axis=1)
        negative_examples = tf.stack([left_patches,neg_patches], axis=1)
        examples = tf.concat([positive_examples,negative_examples],axis=0)
        labels = tf.concat([gt_values,gt_values],axis=0)
        labels = tf.Print(
                tf.identity(labels),
                [tf.shape(examples),tf.shape(labels)],
                summarize=100)
        """
        summaries.append(tf.summary.image("left_patches",left_patches))
        summaries.append(tf.summary.image("pos_patches",pos_patches))
        summaries.append(tf.summary.image("neg_patches",neg_patches))
        """
    return examples,labels

def batch_examples(examples,labels,
        shuffle=False,
        channels=1,
        window_size=9,
        enqueue_many=True):
    with tf.name_scope("example_batches"):
        if shuffle:
            example_batch, label_batch = tf.train.shuffle_batch(
                    [examples,labels],
                    batch_size=batch_size,
                    capacity=(min_after_dequeue + (batch_size *1.1) * num_threads),
                    min_after_dequeue=min_after_dequeue,
                    enqueue_many=enqueue_many,
                    shapes=[[2,window_size,window_size,channels],[]],
                    num_threads=num_threads)
            return example_batch, label_batch
        else:
            example_batch, label_batch = tf.train.batch(
                    [examples,labels],
                    batch_size=batch_size,
                    capacity=(min_after_dequeue + (batch_size *1.1) * num_threads),
                    enqueue_many=enqueue_many,
                    shapes=[[2,window_size,window_size,channels],[]],
                    num_threads=num_threads)
            return example_batch, label_batch
        """
        q = tf.RandomShuffleQueue(
                capacity=min_after_dequeue + (batch_size*1.1)*num_threads,
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
    dequeue = batch_examples(examples,labels,shuffle=shuffle)
    return dequeue

def _main(argv):
    op = example_queue(data_dir="training", num_epochs=2)

    if type(op) != "list":
        op = [op]

    """
    with tf.control_dependencies(op):
        op = tf.no_op()
        """

    from tensorflow.python import debug as tfdbg

    with tfdbg.LocalCLIDebugWrapperSession(tf.Session()) as sess:

        coord = tf.train.Coordinator()
        writer = tf.summary.FileWriter("mc-cnn_logs", graph=sess.graph)
        summarize = lambda s: None if s is None else writer.add_summary(sess.run(s))
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())
        threads = tf.train.start_queue_runners(coord=coord)

        sess.run(op)
        #summarize(tf.summary.merge_all())
        """
        try:
            while not coord.should_stop():
                sess.run(op))
                summarize(tf.summary.merge_all())
        except tf.errors.OutOfRangeError as e:
            coord.request_stop(e)
        """
        coord.request_stop()
        coord.join(threads)
        writer.close()

if __name__ == "__main__":
    _main(sys.argv)

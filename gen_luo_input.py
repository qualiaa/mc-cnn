#!/usr/bin/env python3

from operator import truth, add
from functools import reduce
from itertools import repeat
import numpy as np
from scipy.misc import imread
import os, os.path
import re
import sys
import math
import matplotlib.cm
import matplotlib.pyplot as plt
import random as rand
import time

import tensorflow as tf

patch_size = 9
max_disparity = 128
input_dir = "kitti_stereo_2012"
output_root = "luonet_data3"
infile_regex = "(\d{6})_10.png"

output_compression = tf.python_io.TFRecordCompressionType.ZLIB
output_options = tf.python_io.TFRecordOptions(output_compression)

dataset_names = ["testing","training","validation"]
subdirs = ["disp_occ","image_0","image_1"]

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
def _floatlist_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _sorted_file_lists(path, sub_dirs):
    """ convert a list of dataset folders into a list of lists of files from
    those folders matching a regex """
    # file_lists is a list of possibly many directory listings
    file_lists = [os.listdir(os.path.join(path, sub_dir))
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
    match_to_path = lambda m,d: os.path.join(path, d, m.string)
    file_lists = [[*map(match_to_path, l, repeat(sub_dir))]
            for l, sub_dir
            in zip(file_lists, sub_dirs)]

    # transpose list structure
    file_lists = [list(x) for x in zip(*file_lists)]

    return file_lists

def extract_patches(image,patch_shape,patch_offset=None,border_val=0,dtype=None,dbg=False):
    if dtype is None: dtype = image.dtype
    if patch_offset is None: patch_offset = np.zeros_like(patch_shape)

    po = patch_offset
    hs = tuple(a//2 for a in patch_shape)

    output_shape = (*image.shape, *patch_shape)

    output = np.full(output_shape,border_val,dtype)

    for y in np.arange(output_shape[0]):
        for x in np.arange(output_shape[1]):
            y0 = y - hs[0] + po[0]; y1 = y + hs[0] + po[0]
            x0 = x - hs[1] + po[1]; x1 = x + hs[1] + po[1]
            if patch_shape[0] % 2 == 0: y0 += 1
            if patch_shape[1] % 2 == 0: x0 += 1

            y0_edge = y0 < 0; x0_edge = x0 < 0
            y1_edge = y1 >= image.shape[0]; x1_edge = x1 >= image.shape[1]

            if not (x0_edge or x1_edge or y0_edge or y1_edge):
                output[y,x,:,:] = image[y0:y1+1, x0:x1+1]
            else:
                # input patch coords
                iy0 = max(y0,0); ix0 = max(x0,0)
                iy1 = min(y1+1,image.shape[0])
                ix1 = min(x1+1,image.shape[1])
                dy = iy1-iy0
                dx = ix1-ix0

                # output patch coords
                oy0 = ox0 = 0
                oy1, ox1 = patch_shape
                if y0_edge:
                    dbg and print("y0 edge")
                    oy0 = patch_shape[0] - dy
                    oy1 = patch_shape[0]
                elif y1_edge:
                    dbg and print("y1 edge")
                    oy1 = dy
                if x0_edge:
                    dbg and print("x0 edge")
                    ox0 = patch_shape[1] - dx
                    ox1 = patch_shape[1]
                elif x1_edge:
                    dbg and print("x1 edge")
                    ox1 = dx

                if dbg:
                    print(patch_shape)
                    print(image.shape)
                    print ("r: {}, {}".format(x,y))
                    print ("x-range: {}, {}".format(x0,x1))
                    print ("y-range: {}, {}".format(y0,y1))
                    print ("ri: {}, {} to {}, {}".format(ix0,iy0,ix1,iy1))
                    print ("ro: {}, {} to {}, {}".format(ox0,oy0,ox1,oy1))
                    print ("dr: {}, {}".format(dx,dy))
                    print("="*30)

                output[y,x,oy0:oy1,ox0:ox1] = image[iy0:iy1,ix0:ix1]

    return output

def display(left_patches,right_patches,labels):
    print("Displaying image")

    def one_example(left,right,label):
        gt = right.shape[1]-label.argmax()
        fig = plt.figure()
        ax=plt.subplot(311)
        ax.imshow(left,vmin=0,vmax=255,cmap=matplotlib.cm.gray)
        ax=plt.subplot(312)
        ax.imshow(right,vmin=0,vmax=255,cmap=matplotlib.cm.gray)
        ax.axvline(x=gt,c="red")

        ax=plt.subplot(313)
        ax.invert_xaxis()
        ax.plot(label)

    for _ in range(10):
        i = rand.randrange(0,labels.shape[0])
        one_example(left_patches[i,:,:],right_patches[i,:,:],labels[i,:])
    plt.show()



def generate_examples_from_inputs(gt,left_image,right_image, patch_size):
    right_patch_width = (patch_size - 1) + max_disparity
    right_patch_offset = -(right_patch_width+1)//2 + patch_size//2 + 1

    x = np.meshgrid(range(gt.shape[1]),range(gt.shape[0]))[0]
    np_and = np.logical_and
    valid_gt_mask = np_and(
            #gt != 0,
            gt > 2,
            np_and(gt <= x,gt < max_disparity+1-2))
    del x

    gt_vals = gt[valid_gt_mask] - 1

    invalid_vals = gt_vals[gt_vals>=max_disparity]
    if invalid_vals:
        print("Invalid disp. vals: ")
        print(invalid_vals)
    del gt
    left_patches=extract_patches(left_image,
                                 (patch_size,patch_size))
    right_patches=extract_patches(right_image,
                                  (patch_size,right_patch_width),
                                  (0, right_patch_offset))
    del left_image,right_image
    left_patches_new = left_patches[valid_gt_mask]
    del left_patches; left_patches = left_patches_new
    right_patches_new = right_patches[valid_gt_mask]
    del right_patches; right_patches = right_patches_new
    del valid_gt_mask

    labels = np.zeros([gt_vals.size,max_disparity])

    l0 = 0.5; l1 = 0.2; l2 = 0.05

    for i,val in enumerate(gt_vals):
        if val-2 >= 0 and val-2<max_disparity: labels[i,val-2] = l2
        if val-1 >= 0 and val-1<max_disparity: labels[i,val-1] = l1
        if val   < max_disparity:              labels[i,val]   = l0
        if val+1 < max_disparity:              labels[i,val+1] = l1
        if val+2 < max_disparity:              labels[i,val+2] = l2

    invalid_label_mask = abs(np.sum(labels,axis=1) - 1) > sys.float_info.epsilon
    num_invalid_labels = np.sum(invalid_label_mask)
    if num_invalid_labels != 0:
        print("{} invalid labels found, e.g.".format(num_invalid_labels))
        print(labels[invalid_label_mask][:3,:])
        print("which sum to")
        print(np.sum(labels[invalid_label_mask][:3,:],axis=1))

    #display(left_patches,right_patches,labels)

    return left_patches,right_patches,labels

def load_inputs(instance_files):
    gt = (imread(instance_files[0])/255).astype(np.int16,copy=False)
    left = imread(instance_files[1])
    right = imread(instance_files[2])
    if left.ndim == 3: left = left[:,:,0]
    if right.ndim == 3: left = left[:,:,0]
    return gt, left, right

def save_outputs(examples,output_path):
    with tf.python_io.TFRecordWriter(path=output_path,
                                     options=output_options) as writer:
        for left_patch,right_patch,label in examples:
            features = tf.train.Features(feature={
                'left': _bytes_feature(left_patch.tobytes()),
                'right': _bytes_feature(right_patch.tobytes()),
                'label': _floatlist_feature(label)})
            output = tf.train.Example(features=features)
            writer.write(output.SerializeToString())

def process_instance(instance_files,output_path):
    start_time = time.time()

    inputs = load_inputs(instance_files)
    examples = generate_examples_from_inputs(
            *inputs,patch_size)
    print("\tGenerated {} examples".format(len(examples[0])))
    print("\tSaving to {}".format(output_path))
    examples = zip(*examples)
    del inputs
    save_outputs(examples,output_path)
    del examples

    duration = time.time()-start_time

    print("\tFinished in {} s".format(duration))


def process_dataset(dataset_name):
    output_dir = os.path.join(output_root,dataset_name)
    os.makedirs(output_dir, mode=0o755, exist_ok=True)
    if dataset_name != "testing":
        file_lists = _sorted_file_lists(os.path.join(input_dir,dataset_name),subdirs)
        current_file_num = 0

        for instance_files in file_lists:
            current_file_num += 1
            output_filename=os.path.basename(instance_files[1]).replace(".png",".tfrecord")
            output_path = os.path.join(output_dir,output_filename)

            if os.access(output_path,os.F_OK):
                print("Skipping file {}: already exists.".format(current_file_num))
                continue

            print("Processing file {} out of {}".format(
                current_file_num, len(file_lists)))

            process_instance(instance_files,output_path)
    else:
        pass

if __name__ == "__main__":
    for dataset_name in dataset_names:
        process_dataset(dataset_name)


#!/usr/bin/env python3

from operator import truth, add
from functools import reduce
from itertools import repeat
import numpy as np
from scipy.misc import imread,imsave
from sklearn.feature_extraction.image import extract_patches_2d
import os, os.path
import re
import math
import matplotlib.cm
import matplotlib.pyplot as plt
import random as rand
import png

patch_size = 9
max_disparity = 128
input_dir = "kitti_stereo_2012"
output_dir = "luo_data"
infile_regex = "(\d{6})_10.png"

datasets = ["testing","training","validation"]
subdirs = ["disp_occ","image_0","image_1"]

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

def extract_patches(image,patch_shape,patch_offset=None,sentinel=0,dtype=None):
    if dtype is None: dtype = image.dtype
    if patch_offset is None: patch_offset = np.zeros_like(patch_shape)

    po = patch_offset
    hs = tuple(a//2 for a in patch_shape)

    output_shape = (*image.shape, *patch_shape)

    output = np.full(output_shape,sentinel,dtype)

    for y in np.arange(output_shape[0]):
        for x in np.arange(output_shape[1]):
            y0 = y - hs[0] + po[0]; y1 = y + hs[0] + po[0] 
            x0 = x - hs[1] + po[1]; x1 = x + hs[1] + po[1]
            y0_edge = y0 < 0; x0_edge = x0 < 0
            y1_edge = y1 >= image.shape[0]; x1_edge = x1 >= image.shape[1]
            if not (x0_edge or x1_edge or y0_edge or y1_edge):
                output[y,x,:,:] = image[y-hs[0]+po[0]:y+hs[0]+po[0]+1,
                                        x-hs[1]+po[1]:x+hs[1]+po[1]+1]
            else:
                # input patch coords
                """
                print("r0: {}, {}".format(x0,y0))
                print("r1: {}, {}".format(x1,y1))
                """
                iy0 = max(y0,0); ix0 = max(x0,0)
                iy1 = min(y1+1,image.shape[0])
                ix1 = min(x1+1,image.shape[1])
                dy = iy1-iy0
                dx = ix1-ix0


                # output patch coords
                oy0 = ox0 = 0
                oy1, ox1 = patch_shape
                if y0_edge:
                    oy0 = patch_shape[0] - dy
                    oy1 = patch_shape[0]
                elif y1_edge:
                    oy1 = dy
                if x0_edge:
                    ox0 = patch_shape[1] - dx
                    ox1 = patch_shape[1]
                elif x1_edge:
                    ox1 = dx

                """
                print("dr: {}, {}".format(dx,dy))
                print("In coords: {}, {} to {}, {}".format(ix0,iy0,ix1,iy1))
                print("Out coords: {}, {} to {}, {}".format(ox0,oy0,ox1,oy1))
                """

                output[y,x,oy0:oy1,ox0:ox1] = image[iy0:iy1,ix0:ix1]
                

                """
                for py in np.arange(patch_shape[0]):
                    for px in np.arange(patch_shape[1]):
                        iy,ix = (py-hs[0]-1,px-hs[1]-1)
                        try:
                            output[y,x,py,px] = image[y+iy,x+ix]
                        except IndexError:
                            pass
                """
    return output

def display(left_patches,right_patches,labels):
    print("Displaying image")
    def one_example(left,right,label):
        gt = right.shape[1]-label.argmax()
        fig = plt.figure()
        ax=plt.subplot(311)
        ax.imshow(left,vmin=0,vmax=1,cmap=matplotlib.cm.gray)
        ax=plt.subplot(312)
        ax.imshow(right,vmin=0,vmax=1,cmap=matplotlib.cm.gray)
        ax.axvline(x=gt,c="red")

        ax=plt.subplot(313)
        ax.invert_xaxis()
        ax.plot(label)

    for _ in range(10):
        i = rand.randrange(0,labels.shape[0])
        one_example(left_patches[i,:,:],right_patches[i,:,:],labels[i,:])
    plt.show()

    

def generate_examples_and_labels(gt,left,right, patch_size):
    gt = gt.astype(np.int32,copy=False)

    right_patch_width = patch_size + max_disparity 
    right_patch_offset = -right_patch_width//2 + patch_size//2

    x = np.meshgrid(range(gt.shape[1]),range(gt.shape[0]))[0]
    valid_gt_mask = np.logical_and(gt != 0,gt <= x,gt < max_disparity)
    del x

    gt_vals = gt[valid_gt_mask]
    del gt
    left_patches=extract_patches(left,
                                 (patch_size,patch_size))
    right_patches=extract_patches(right,
                                  (patch_size,right_patch_width),
                                  (0, right_patch_offset))
    del left,right
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
        if val < max_disparity: labels[i,val] = l0
        if val+1 < max_disparity: labels[i,val+1] = l1
        if val+2 < max_disparity: labels[i,val+2] = l2

    #display(left_patches,right_patches,labels)
    #input()

    return left_patches,right_patches,labels
                
# write_png = imsave
def write_png(path,arr):
    with open(path, "wb") as f:
        w = png.Writer(*arr.shape[::-1],greyscale=True)
        w.write(f,arr)

if __name__ == "__main__":
    instance_count = 0
    for dataset in datasets:
        folder = os.path.join(output_dir,dataset)
        os.makedirs(folder, mode=0o755, exist_ok=True)
        if dataset != "testing":
            file_lists = _sorted_file_lists(os.path.join(input_dir,dataset),subdirs)
            file_count = 0
            for single_instance in file_lists:
                print("Processing file {} out of {}".format(file_count,
                                                            len(file_lists)))
                examples_and_labels = generate_examples_and_labels(
                        *[imread(x)/255 for x in single_instance], patch_size)

                z = zip(*examples_and_labels)
                for left,right,label in z:
                    path = os.path.join(folder,"{}_{}.png")
                    write_png(path.format(str(instance_count),"left"),left)
                    write_png(path.format(str(instance_count),"right"),right)
                    write_png(path.format(str(instance_count),"label"),
                           np.expand_dims(label,axis=0))
                    instance_count += 1
                del examples_and_labels, z 
                file_count += 1

        else:
            pass


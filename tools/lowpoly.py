#!/usr/bin/python

# setup paths
import sys
sys.path.append('/home/jsupanci/workspace/StructuredForests/')

# import dependencies
from scipy import misc
import matplotlib.pyplot as plt
import os
from contextlib import contextmanager
import numpy as np
import skimage

@contextmanager
def pushd(newDir):
    previousDir = os.getcwd()
    os.chdir(newDir)
    yield
    os.chdir(previousDir)
    
def get_se():
    rand = np.random.RandomState(1)

    options = {
        "rgbd": 0,
        "shrink": 2,
        "n_orient": 4,
        "grd_smooth_rad": 0,
        "grd_norm_rad": 4,
        "reg_smooth_rad": 2,
        "ss_smooth_rad": 8,
        "p_size": 32,
        "g_size": 16,
        "n_cell": 5,

        "n_pos": 10000,
        "n_neg": 10000,
        "fraction": 0.25,
        "n_tree": 8,
        "n_class": 2,
        "min_count": 1,
        "min_child": 8,
        "max_depth": 64,
        "split": "gini",
        "discretize": lambda lbls, n_class:
            discretize(lbls, n_class, n_sample=256, rand=rand),

        "stride": 2,
        "sharpen": 2,
        "n_tree_eval": 4,
        "nms": True,
    }

    with pushd('/home/jsupanci/workspace/StructuredForests/'):
        import StructuredForests as SE     
        model = SE.StructuredForests(options, rand=rand)
    return model

def main():
    # load the original image
    imgpath = '/mnt/vlad/jsupanci/data/celebA/072348.jpg'
    img = misc.imread(imgpath)
    plt.imshow(img)
    #plt.show()
    
    # get the edge image
    se = get_se()
    im_edges = se.predict(skimage.img_as_float(img))
    plt.imshow(im_edges)
    plt.show()
    
    # 

if __name__ == '__main__':
    main()
    code.interact(local=locals())
    

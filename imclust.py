#!/usr/bin/env -S python3 -u # -*- python -*-
# imclust.py (c) R.Jaksa 2021

import sys,os

# -------------------------- parse command-line arguments: dirname and no. of clusters

HELP = f"""
NAME
    imclust.py - image clustering demo

USAGE
    imclust.py [OPTIONS] DIRECTORY...

DESCRIPTION
    Image clusteuring demo imclust.py will cluster images in
    the DIRECTORY, and produce a html visualization of results.

OPTIONS
    -h  This help.
    -c  Requested number of clusters.

VERSION
    imclust.py 0.1 (c) R.Jaksa 2021
"""

import argparse
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("-h","--help",action="store_true")
parser.add_argument("-c","--clusters",type=int,default=16)
parser.add_argument("path",type=str,nargs='*')
args = parser.parse_args()

if args.help or len(args.path)==0:
    print(HELP)
    exit(0)

# ---------------------------------------------------------- get image names from dirs
from glob import glob
import random

path = []
for dir in args.path:
  path += glob(dir+"/**/*.png",recursive=True)
  path += glob(dir+"/**/*.jpg",recursive=True)
random.shuffle(path)
# for p in path: print(p)

# ------------------------------------------------------------------------ load images
from imageio import imread
from skimage.transform import resize
import numpy as np
np.warnings.filterwarnings("ignore",category=np.VisibleDeprecationWarning)

SIZE = (224,224,3)
images = np.array([imread(str(p)).astype(np.float32) for p in path])
images = np.asarray([resize(image,SIZE,0) for image in images])
print(f"images: {len(images)}")
print(f"single image shape: {images[0].shape}")

# ------------------------------------------------------------------------- load model
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

model = tf.keras.applications.resnet50.ResNet50(include_top=False,weights="imagenet",input_shape=(224,224,3))

# ------------------------------------------------------------- get embeddings vectors

vector = model.predict(images)
print(f"model output shape: {vector[0].shape}")
vector = vector.reshape(vector.shape[0],-1)
print(f"reshaped to 1D: {vector[0].shape}")

# ----------------------------------------------------------------------- cluster them
from sklearn.cluster import KMeans

CLUSTERS = args.clusters
clustering = KMeans(n_clusters=CLUSTERS)
clustering.fit(vector)
cluster = clustering.predict(vector)
print(f"clusters: {cluster}")

# ------------------------------------------------ copy images according their cluster

# import shutil
# for i in range(len(images)):
#   if not os.path.exists(f"output/cluster{cluster[i]}"): os.makedirs(f"output/cluster{cluster[i]}")
#   print(f"cp {path[i]} output/cluster{cluster[i]}")
#   shutil.copy2(f"{path[i]}",f"output/cluster{cluster[i]}")
  
# -------------------------------------------------------------------------- make html
from web import *

# make html section for every cluster
section = [""]*CLUSTERS
for i in range(len(images)):
  section[cluster[i]] += addimg(f"{path[i]}",f"cluster {cluster[i]}",f"{path[i]}")

# build the page
BODY = ""
for i in range(len(section)):
  BODY += f"<h2>cluster {i}<h2>\n"
  BODY += section[i]
  BODY += "\n\n"
html = HTML.format(BODY=BODY,CSS=CSS)

# save html
print("write: index.html")
with open("index.html","w") as fd:
  fd.write(html)

# ------------------------------------------------------------------------------------

import nnvm
import nnvm.frontend.darknet
import nnvm.testing.darknet
import matplotlib.pyplot as plt
import numpy as np
import tvm
import sys
import cv2

from ctypes import *
#from tvm.contrib.download import download
from nnvm.testing.darknet import __darknetffi__
from tvm.contrib import graph_runtime

DARKNET_LIB = 'libdarknet.so'
CFG_NAME = 'yolov3.cfg'
WEIGHTS_NAME = 'yolov3.weights'


# load from current directory
darknet_lib = __darknetffi__.dlopen('./' + DARKNET_LIB)
cfg = "./" + str(CFG_NAME)
weights = "./" + str(WEIGHTS_NAME)

net = darknet_lib.load_network(cfg.encode('utf-8'), weights.encode('utf-8'), 0)

dtype = 'float32'
batch_size = 1
print("Converting darknet to nnvm symbols...")
sym, params = nnvm.frontend.darknet.from_darknet(net, dtype)

target = 'llvm'
ctx = tvm.cpu(0)
data = np.empty([batch_size, net.c, net.h, net.w], dtype)
shape = {'data': data.shape}
print("Compiling the model...")
with nnvm.compiler.build_config(opt_level=2):
    graph, lib, params = nnvm.compiler.build(sym, target, shape, dtype, params)

test_image = 'dog.jpg'
print("Loading the test image...")
data = nnvm.testing.darknet.load_image(test_image, net.w, net.h)

m = graph_runtime.create(graph, lib, ctx)

# set inputs
m.set_input('data', tvm.nd.array(data.astype(dtype)))
m.set_input(**params)
# execute
print("Running the test image...")

m.run()
# get outputs
out_shape = (net.outputs,)
tvm_out = m.get_output(0).asnumpy().flatten()

# do the detection and bring up the bounding boxes
thresh = 0.24
hier_thresh = 0.5
img = nnvm.testing.darknet.load_image_color(test_image)

#print("dtype:{}".format(img.dtype))
#cv2.imwrite('detection-2.png', np.flip(img, 0).transpose(1, 2, 0)*255)
#sys.exit(0)

_, im_h, im_w = img.shape
probs = []
boxes = []
region_layer = net.layers[net.n - 1]
boxes, probs = nnvm.testing.yolo2_detection.get_region_boxes(
    region_layer, im_w, im_h, net.w, net.h,
    thresh, probs, boxes, 1, tvm_out)

boxes, probs = nnvm.testing.yolo2_detection.do_nms_sort(
    boxes, probs,
    region_layer.w*region_layer.h*region_layer.n, region_layer.classes, 0.3)

coco_name = 'coco.names'
coco_url = 'https://github.com/siju-samuel/darknet/blob/master/data/' + coco_name + '?raw=true'
font_name = 'arial.ttf'
font_url = 'https://github.com/siju-samuel/darknet/blob/master/data/' + font_name + '?raw=true'

from tvm.contrib.download import download

download(coco_url, coco_name)
download(font_url, font_name)


with open(coco_name) as f:
    content = f.readlines()

names = [x.strip() for x in content]

nnvm.testing.yolo2_detection.draw_detections(
    img, region_layer.w*region_layer.h*region_layer.n,
    thresh, boxes, probs, names, region_layer.classes)

#cv2.imshow('detection',img.transpose(1, 2, 0))
cv2.imwrite('detection.png',np.flip(img, 0).transpose(1, 2, 0)*255)
#plt.imshow(img.transpose(1, 2, 0))
#plt.show()
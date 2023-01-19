import json
import sys
import os
import time
import numpy as np
import cv2
import onnx
import onnxruntime
from onnx import numpy_helper
 
model_dir ="./output"
model=model_dir+"/model.onnx"
path="../datasets/test_images/Bathroom1.jpg" # sys.argv[1]

ade20k_info_path = "../datasets/ade20k_label_colors.txt"

def read_ade20k_info(info_path=ade20k_info_path):
    with open(info_path) as fp:
        lines = fp.readlines()

        labels = [line[:-1].replace(';', ',').split(',')[0] for line in lines]
        colors = np.array([line[:-1].replace(';', ',').split(',')[-3:] for line in lines]).astype(np.int)

    return colors, labels

colors, labels = read_ade20k_info()
    
def util_draw_seg(seg_map, image, alpha = 0.5):

    # Convert segmentation prediction to colors
    color_segmap = cv2.resize(image, (seg_map.shape[1], seg_map.shape[0]))
    color_segmap[seg_map>0] = colors[seg_map[seg_map>0]]

    # Resize to match the image shape
    color_segmap = cv2.resize(color_segmap, (image.shape[1],image.shape[0]))

    # Fuse both images
    combined_img = None
    if alpha == 0:
        combined_img = np.hstack((image, color_segmap))
    else:
        combined_img = cv2.addWeighted(image, alpha, color_segmap, (1-alpha),0)

    cv2.imwrite("./output/predictions.png", combined_img)
    return 
 
#Preprocess the image
input_img = cv2.imread(path)
#img = np.dot(img[...,:3], [0.299, 0.587, 0.114])
img = cv2.resize(input_img, dsize=(512, 512), interpolation=cv2.INTER_AREA)
#img = img.transpose(2, 0, 1)
img = img.astype("float32").transpose(2, 0, 1)
#img.resize((1, 1, 28, 28))
 
data = np.array(img).astype('float32')
session = onnxruntime.InferenceSession(model, None)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
print(input_name)
print(output_name)
 
result = session.run([output_name], {input_name: data})
prediction = np.argmax(np.array(result[0]).squeeze(), axis=0)
#prediction=int(np.argmax(np.array(result).squeeze(), axis=0))
print(prediction.shape)
print(prediction)

util_draw_seg(prediction, input_img, 0)
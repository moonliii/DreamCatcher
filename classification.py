from keras.models import *
from keras import *
import cv2
import numpy as np
import base64

# global params
img_size = 32

# transparent to white
def to_white(img):
    width = img.shape[0]
    height = img.shape[1]
    for w in range(width):
        for h in range(height):
            if (img[w,h,3] == 0):
                img[w,h] = [255,255,255,255]
    return img[:,:,:-1]

def load_preprocess(image_base64):
    # base64转opencv图片
    image_buffer = base64.b64decode(image_base64)
    img_np_arr = np.frombuffer(image_buffer, np.uint8)
    img = cv2.imdecode(img_np_arr, cv2.IMREAD_UNCHANGED)
    if(img.shape[2] == 4):  # has alpha 
        img = to_white(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size))
    img = img.astype('float32')/255.
    img = np.reshape(img,(-1,img.shape[0],img.shape[1],img.shape[2]))
    return img

def get_result(image_base64):
    # load the pre-trained model
    model = load_model('vgg_model.h5')
    # load data and preprocess
    img = load_preprocess(image_base64)
    # predict
    predict =  model.predict(img)
    return predict

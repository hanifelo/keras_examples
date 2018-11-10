
#U-Net for segmenting seismic images with keras
# link: https://www.depends-on-the-definition.com/unet-keras-segmenting-images/
import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("ggplot")
%matplotlib inline

from tqdm import tqdm_notebook, tnrange
from itertools import chain
from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from sklearn.model_selection import train_test_split

import tensorflow as tf

from keras.models import Model, load_model
from keras.layers import Input, BatchNormalization, Activation,  Dropout,UpSampling2D
from keras.layers.core import Lambda, RepeatVector, Reshape
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.layers.pooling import MaxPooling2D, GlobalMaxPool2D
from keras.layers.merge import concatenate, add
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

#---------------------------------------
# Set some parameters
im_width = 128
im_height = 128
border = 5
path_train = '/media/zahra/66048aa8-5d56-45a6-9c13-d1145235d158/data/download/bmpBRATS/train/'
path_test = '/media/zahra/66048aa8-5d56-45a6-9c13-d1145235d158/data/download/bmpBRATS/test/'

#Load the images

# Get and resize train images and masks
def get_data(path, train=True):
    ids = next(os.walk(path + "images"))[2]
    X = np.zeros((len(ids), im_height, im_width, 1), dtype=np.float32)
    if train:
        y = np.zeros((len(ids), im_height, im_width, 1), dtype=np.float32)
    print('Getting and resizing images ... ')
    for n, id_ in tqdm_notebook(enumerate(ids), total=len(ids)):
        # Load images
        img = load_img(path + '/images/' + id_, grayscale=True)
        x_img = img_to_array(img)
        x_img = resize(x_img, (128, 128, 1), mode='constant', preserve_range=True)

        # Load masks
        if train:
            mask = img_to_array(load_img(path + '/masks/' + id_, grayscale=True))
            mask = resize(mask, (128, 128, 1), mode='constant', preserve_range=True)

        # Save images
        X[n, ..., 0] = x_img.squeeze() / 255
        if train:
            y[n] = mask / 255
    print('Done!')
    if train:
        return X, y
    else:
        return X
    
X, y = get_data(path_train, train=True)

# Split train and valid
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.15, random_state=2018)
# Check if training data looks all right
ix = random.randint(0, len(X_train))
has_mask = y_train[ix].max() > 0

fig, ax = plt.subplots(1, 2, figsize=(20, 10))

ax[0].imshow(X_train[ix, ..., 0], cmap='seismic', interpolation='bilinear')
if has_mask:
    ax[0].contour(y_train[ix].squeeze(), colors='k', levels=[0.5])
ax[0].set_title('Image')

ax[1].imshow(y_train[ix].squeeze(), interpolation='bilinear', cmap='gray')
ax[1].set_title('Mask');



#-----------------------------------------------

#---------------------------------------------------------------

def conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):
    # first layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x


def get_segnet(input_img,kernel_size, dropout, n_labels,batchnorm):
    c1 = conv2d_block(input_img, n_filters=64, kernel_size=3, batchnorm=batchnorm)
    c1 = Dropout(dropout*0.5)(c1)
    c2 = conv2d_block(c1, n_filters=64, kernel_size=3, batchnorm=batchnorm)
    c2= Dropout(dropout*0.5)(c2)
    c2= MaxPooling2D((2, 2))(c2)
# 
    c3 = conv2d_block(c2, n_filters=128, kernel_size=3, batchnorm=batchnorm)
    c3 = Dropout(dropout*0.5)(c3)
    c4 = conv2d_block(c3, n_filters=128, kernel_size=3, batchnorm=batchnorm)
    c4= Dropout(dropout*0.5)(c4)
    c4= MaxPooling2D((2, 2))(c4)
# 
    c5 = conv2d_block(c4, n_filters=256, kernel_size=3, batchnorm=batchnorm)
    c5 = Dropout(dropout*0.5)(c5)
    c6 = conv2d_block(c5, n_filters=256, kernel_size=3, batchnorm=batchnorm)
    c6= Dropout(dropout*0.5)(c6)
    c7 = conv2d_block(c6, n_filters=256, kernel_size=3, batchnorm=batchnorm)
    c7= Dropout(dropout*0.5)(c7)
    c7= MaxPooling2D((2, 2))(c7)
# 
    c8 = conv2d_block(c7, n_filters=512, kernel_size=3, batchnorm=batchnorm)
    c8 = Dropout(dropout*0.5)(c8)
    c9 = conv2d_block(c8, n_filters=512, kernel_size=3, batchnorm=batchnorm)
    c9= Dropout(dropout*0.5)(c9)
    c10 = conv2d_block(c9, n_filters=512, kernel_size=3, batchnorm=batchnorm)
    c10= Dropout(dropout*0.5)(c10)
    c10= MaxPooling2D((2, 2))(c10)
# 
    c11 = conv2d_block(c10, n_filters=512, kernel_size=3, batchnorm=batchnorm)
    c11 = Dropout(dropout*0.5)(c11)
    c12 = conv2d_block(c11, n_filters=512, kernel_size=3, batchnorm=batchnorm)
    c12= Dropout(dropout*0.5)(c12)
    c13 = conv2d_block(c12, n_filters=512, kernel_size=3, batchnorm=batchnorm)
    c13= Dropout(dropout*0.5)(c13)
    c13= MaxPooling2D((2, 2))(c13)
# 
   #decoding_layers  
    p1= UpSampling2D(size=(2,2))(c13)
    # 
    c14 = conv2d_block(p1, n_filters=512, kernel_size=3, batchnorm=batchnorm)
    c14 = Dropout(dropout*0.5)(c14)
    c15 = conv2d_block(c14, n_filters=512, kernel_size=3, batchnorm=batchnorm)
    c15= Dropout(dropout*0.5)(c15)
    c16 = conv2d_block(c15, n_filters=512, kernel_size=3, batchnorm=batchnorm)
    c16= Dropout(dropout*0.5)(c16)
    
    
    p2= UpSampling2D(size=(2,2))(c16)
    # 
    c17 = conv2d_block(p2, n_filters=512, kernel_size=3, batchnorm=batchnorm)
    c17 = Dropout(dropout*0.5)(c17)
    c18 = conv2d_block(c17, n_filters=512, kernel_size=3, batchnorm=batchnorm)
    c18= Dropout(dropout*0.5)(c18)
    c19 = conv2d_block(c18, n_filters=256, kernel_size=3, batchnorm=batchnorm)
    c19= Dropout(dropout*0.5)(c19)
    
    
    p3= UpSampling2D(size=(2,2))(c19)
    # 
    c20 = conv2d_block(p3, n_filters=256, kernel_size=3, batchnorm=batchnorm)
    c20 = Dropout(dropout*0.5)(c20)
    c21 = conv2d_block(c20, n_filters=256, kernel_size=3, batchnorm=batchnorm)
    c21= Dropout(dropout*0.5)(c21)
    c22 = conv2d_block(c21, n_filters=128, kernel_size=3, batchnorm=batchnorm)
    c22= Dropout(dropout*0.5)(c22)
   
    
    p4= UpSampling2D(size=(2,2))(c22)
    # 
    c23 = conv2d_block(p4, n_filters=128, kernel_size=3, batchnorm=batchnorm)
    c23 = Dropout(dropout*0.5)(c23)
    c24 = conv2d_block(c23, n_filters=64, kernel_size=3, batchnorm=batchnorm)
    c24= Dropout(dropout*0.5)(c24)
    
    p5= UpSampling2D(size=(2,2))(c24)
    # 
    c25 = conv2d_block(p5, n_filters=64, kernel_size=3, batchnorm=batchnorm)
    c25 = Dropout(dropout*0.5)(c25)

    #end layer number of labels
    c26 = Conv2D(filters=n_labels, kernel_size=(1, 1), kernel_initializer="he_normal",
               padding="same")(c25)
    outputs=BatchNormalization()(c26)
    model = Model(inputs=[input_img], outputs=[outputs])
    return model

input_img = Input((im_height, im_width, 1), name='img')

model = get_segnet(input_img,kernel_size=3,  dropout=0.05, n_labels=1,batchnorm=True)
#---------------------------------------------------------------------------------------------

model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy"])
model.summary()

callbacks = [
    EarlyStopping(patience=10, verbose=1),
    ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose=1),
    ModelCheckpoint('model-tgs-salt.h5', verbose=1, save_best_only=True, save_weights_only=True)
]

results = model.fit(X_train, y_train, batch_size=4, epochs=1, callbacks=callbacks,
                    validation_data=(X_valid, y_valid))
#--------------------------------------------------------------------








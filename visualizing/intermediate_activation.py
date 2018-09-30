#-------------------------
#there are 3 ways to visulizing what convnet learn
#1. visulaizing intermediate activation:useful to undrastanding how layers transform the nput
#2. visulaizing convnet filters
#3. visulaizing heatmaps of class
#--------------------------------

#******************************************************************************
#1. visulaizing intermediate activation
#******************************************************************************
#pre processing a single image
img_path='/media/zahra/66048aa8-5d56-45a6-9c13-d1145235d158/pythonCode/catDog/baseDir/test/cat/1524.jpg'
from keras.preprocessing import image
import numpy as np
img=image.load_img(img_path,target_size=(150,150))
img_tensor=image.img_to_array(img)
img_tensor=np.expand_dims(img_tensor,axis=0)
img_tensor/=255
print(img_tensor.shape)

#display the picture
import matplotlib.pyplot as plt
plt.imshow(img_tensor[0])
plt.show()

#load h5 model
from keras import models
from keras.models import load_model
model=load_model('cat_and-dig.h5')
model.summary()

#instanting a model from an input and a list of output tensord
layer_outputs=[layer.output for layer in model.layers[:8]]
activation_model=models.Model(input=model.input,output=layer_outputs)
#running model on predict mode
activations=activation_model.predict(img_tensor)
first_layer_activation=activations[1]
print('fisrt kayer activation:',first_layer_activation)
import matplotlib.pyplot as plt
plt.matshow(first_layer_activation[0,:,:,4],cmap='viridis')
#-------------------------------------------------
#show all channel in all layers
#------------------------------------------------
activations=activation_model.predict(img_tensor)
layer_names=[]
for layer in model.layers[:8]:
    layer_names.append(layer.name)
image_per_row=16
for layer_name,layer_activation in zip(layer_names,activations):
    n_feauters=layer_activation.shape[-1]
    size=layer_activation.shape[1]
    n_cols=n_feauters//image_per_row
    display_grid=np.zeros((size*n_cols,image_per_row*size))
    for col in range(n_cols):
        for row in range(image_per_row):
            channel_image=layer_activation[0,:,:,col*image_per_row+row]
            channel_image-=channel_image.mean()
            channel_image/=channel_image.std()
            channel_image*=64
            channel_image+=128
            channel_image=np.clip(channel_image,0,255).astype('uint8')
            display_grid[col*size:(col+1)*size,row*size:(row+1)*size]=channel_image
            scale=1./size
            plt.figure(figsize=(scale*display_grid.shape[1],scale*display_grid.shape[0]))
            plt.title(layer_name)
            plt.imshow(display_grid,aspect='auto',cmap='viridis')

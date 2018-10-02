
'''
there are 3 ways to visulizing what convnet learn
1. visulaizing intermediate activation:useful to undrastanding how layers transform the nput
2. visulaizing convnet filters
3. visulaizing heatmaps of class
'''

#******************************************************************************
#2. visulaizing convnet filters
#******************************************************************************

from keras.applications import VGG16
from keras import backend as K
import matplotlib.pyplot as plt
model=VGG16(weights='imagenet',include_top=False)
layer_name='block3_conv1'
filter_index=0

def deprocess_image(x):
    x-=x.mean()
    x/=(x.std()+0.000001)
    x*=0.1
    x+=0.5
    x=np.clip(x,0,1)
    x*=255
    x=np.clip(x,0,255).astype('uint8')
    return x
 
def generator_pattern(layer_name,filer_index,size=150):
    layer_output=model.get_layer(layer_name).output
    loss=K.mean(layer_output[:,:,:,filter_index])
    grads=K.gradients(loss,model.input)[0]
    grads/=K.sqrt(K.mean(K.square(grads)))+0.00001
    itrator=K.function([model.input],[loss,grads])
    input_image_data=np.random.random((1,size,size,3))*20+128
    step=1
    for i in range(40):
        loss_value,grads_value=itrator([input_image_data])
        input_image_data+=grads_value*step
    img=input_image_data[0]
    return deprocess_image(img)

plt.imshow(generator_pattern(layer_name,filter_index))

#-----------------------
'''
grid all filters
'''
layer_name='block5_conv2'
size=64
margin=5
results=np.zeros((8*size+7*margin,8*size+7*margin,3))
for i in range(8):
    for j in range(8):
        filter_img=generator_pattern(layer_name,i+(j*8),size=size)
        horizonta_start=i*size+i*margin
        horizontal_end=horizonta_start+size
        vertical_start=j*size+j*margin
        vertical_end=vertical_start+size
        results[horizonta_start:horizontal_end,vertical_start:vertical_end,:]=filter_img
plt.figure(figsize=(20,20))
plt.imshow(results)

#---------------------------------
#use fine_tunning to lerarn cat-dog model
#steps to fine_tunning:
# 1: add my custom model on top of VGG16 
# 2: freez VGG16 layers
# 3: train new part tht i added
# 4: unfreez VGG16 (or some layers)
# 5: train both VGG16 and new layers
#------------------------------------

#steps 1 to 3
#read data by data generator and train a frozen nework

from keras.preprocessing.image import ImageDataGenerator

base_dir='/media/zahra/66048aa8-5d56-45a6-9c13-d1145235d158/pythonCode/catDog/baseDir'
train_dir='/media/zahra/66048aa8-5d56-45a6-9c13-d1145235d158/pythonCode/catDog/baseDir/train'
valid_dir='/media/zahra/66048aa8-5d56-45a6-9c13-d1145235d158/pythonCode/catDog/baseDir/valid'

train_datagen=ImageDataGenerator(
 rescale=1./255,
 rotation_range=40,
 width_shift_range=0.2,
 height_shift_range=0.2,
 shear_range=0.2,
 zoom_range=0.2,
horizontal_flip=True,)
valid_datagen=ImageDataGenerator( rescale=1./255)
test_datagen=ImageDataGenerator( rescale=1./255)

train_datagenerator=train_datagen.flow_from_directory(
    train_dir,
    batch_size=20,
    target_size=(150,150),
    class_mode='binary')
valid_generator=valid_datagen.flow_from_directory(
    valid_dir,
    batch_size=20,
    target_size=(150,150),
    class_mode='binary')
test_generator=test_datagen.flow_from_directory(
    test_dir,
    batch_size=20,
    target_size=(150,150),
    class_mode='binary')


#freez VGG6 and train top layers
from keras.applications import VGG16
from keras import layers
from keras import models
from keras import optimizers
base_conv=VGG16(weights='imagenet',include_top=False,input_shape=(150,150,3))
model=models.Sequential()
model.add(base_conv)
model.add(layers.Flatten())
model.add(layers.Dense(256,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))
base_conv.trainable=False
#compile and fit model
model.compile(loss='binary_crossentropy',optimizer=optimizers.RMSprop(lr=0.0001),metrics=['acc'])
history=model.fit_generator(
 train_datagenerator,
 steps_per_epoch=50,
 epochs=20,
 validation_data=valid_generator,
 validation_steps=50)
#save results
model.save('pretaed_1_cat_and-dig.h5')
#-----------------------------------------------------
#step 4: unfreez some layers onVGG16
# i freezing all layers up to special one
#------------------------------------
base_conv.trainable=True
set_trainable=False
for layer in base_conv.layers:
    if layer.name=='block5_conv1':
        set_trainable=True
    if set_trainable:
        layer.trainable=True
    else:
        layer.trainable=False
#now train network again
#compile and fit model
model.compile(loss='binary_crossentropy',optimizer=optimizers.RMSprop(lr=0.0001),metrics=['acc'])
history=model.fit_generator(
 train_datagenerator,
 steps_per_epoch=50,
 epochs=20,
 validation_data=valid_generator,
 validation_steps=50)
model.save('fine_tunning_cat_and-dig.h5')
test_loss,test_acc=model.evaluate_generator(test_generator,steps=50)
print('final test los is:    ',test_acc)
#------------------------------------------------------
#show result by smooth points
#-------------------------------------------------------
def smooth_curve(points,factor=0.8):
    smoothed_points=[]
    for point in points:
        if smoothed_points:
            previous=smoothed_points[-1]
            smoothed_points.append(previous*factor+point*(1-factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

import matplotlib.pyplot as plt
acc=history.history['acc']
loss=history.history['loss']
val_acc=history.history['val_acc']
val_loss=history.history['val_loss']
epochs=range(1,len(acc)+1)
plt.plot(epochs,smooth_curve(acc),'bo',label='tarining acc')
plt.plot(epochs,smooth_curve(val_acc),label='validation acc')
plt.show()

plt.plot(epochs,smooth_curve(loss),'bo',label='tarining loss')
plt.plot(epochs,smooth_curve(val_loss),label='validation loss')
plt.show()

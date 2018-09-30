#---------------------
#train cat dog dataset with VGG16  and data agumation
#--------------

#train, test and validation address
base_dir='/media/zahra/66048aa8-5d56-45a6-9c13-d1145235d158/pythonCode/catDog/baseDir'
train_dir='/media/zahra/66048aa8-5d56-45a6-9c13-d1145235d158/pythonCode/catDog/baseDir/train'
valid_dir='/media/zahra/66048aa8-5d56-45a6-9c13-d1145235d158/pythonCode/catDog/baseDir/valid'
test_dir='/media/zahra/66048aa8-5d56-45a6-9c13-d1145235d158/pythonCode/catDog/baseDir/test'
#-----------------------------------------
#create model , first layers from VGG16 and frees VGG16 layers
#------------------------------------------
from keras.applications import VGG16
from keras import models
from keras import layers
base_conv=VGG16(weights='imagenet',include_top=False,input_shape=(150,150,3))
model=models.Sequential()
model.add(base_conv)
model.add(layers.Flatten())
model.add(layers.Dense(156,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))
model.summary()
print('number of parametrs before freezing:',len(model.trainable_weights))
base_conv.trainable=False
print('number of parametrs after freezing:',len(model.trainable_weights))

#------------------------------------------------------
#now fit data on model
#--------------------------------------------------------

from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
#data generators
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
 target_size=(150,150),
 batch_size=20,
 class_mode='binary')
valid_generator=test_datagen.flow_from_directory(
 valid_dir,
 target_size=(150,150),
 batch_size=20,
 class_mode='binary')
test_generator=test_datagen.flow_from_directory(
 test_dir,
 target_size=(150,150),
 batch_size=20,
 class_mode='binary')
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
#------------------------------------------------------
#show result
#-------------------------------------------------------
import matplotlib.pyplot as plt
acc=history.history['acc']
loss=history.history['loss']
val_acc=history.history['val_acc']
val_loss=history.history['val_loss']
epochs=range(1,len(acc)+1)
plt.plot(epochs,acc,'bo',label='tarining acc')
plt.plot(epochs,val_acc,label='validation acc')
plt.show()

plt.plot(epochs,loss,'bo',label='tarining loss')
plt.plot(epochs,val_loss,label='validation loss')
plt.show()




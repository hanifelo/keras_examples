import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
base_dir='/media/zahra/66048aa8-5d56-45a6-9c13-d1145235d158/pythonCode/catDog/baseDir'
train_dir='/media/zahra/66048aa8-5d56-45a6-9c13-d1145235d158/pythonCode/catDog/baseDir/train'
valid_dir='/media/zahra/66048aa8-5d56-45a6-9c13-d1145235d158/pythonCode/catDog/baseDir/valid'
test_dir='/media/zahra/66048aa8-5d56-45a6-9c13-d1145235d158/pythonCode/catDog/baseDir/test'
#-------------------------------------------------------
#a fuction to read batch data and extract thers featurs
#-----------------------------------------------------
datagen=ImageDataGenerator(rescale=1./255)
batch_size=20
def extrcat_features(directory,sample_count):
    features=np.zeros(shape=(sample_count,4,4,512))
    labels=np.zeros(shape=(sample_count))
    generator=datagen.flow_from_directory(
        directory,
        target_size=(150,150),
        batch_size=batch_size,
        class_mode='binary')
    i=0
    for inputs_batch,labels_batch in generator:
        features_batch=conv_base.predict(inputs_batch)
        features[i*batch_size:(i+1)*batch_size]=features_batch
        labels[i*batch_size:(i+1)*batch_size]=labels_batch
        i+=1
        if i*batch_size>=sample_count:
            break
    return features,labels
#-------------------------------------------------------
#use VGG16 and extract features from cat dog data set
#-----------------------------------------------------
from keras.applications import VGG16
conv_base=VGG16(weights='imagenet', include_top=False,input_shape=(150,150,3))
train_features,train_labels=extrcat_features(train_dir,2000)
valid_features,valid_labels=extrcat_features(valid_dir,500)
test_features,test_labels=extrcat_features(test_dir,500)

train_features=np.reshape(train_features,(2000,4*4*512))
valid_features=np.reshape(valid_features,(500,4*4*512))
test_features=np.reshape(test_features,(500,4*4*512))
#-------------------------------------------------------
#run model
#-----------------------------------------------------

from keras import models
from keras import layers
from keras import optimizers
model=models.Sequential()
model.add(layers.Dense(256,activation='relu',input_dim=4*4*512))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1,activation='sigmoid'))
model.compile(optimizer=optimizers.RMSprop(lr=0.00001),
              loss='binary_crossentropy',
              metrics=['acc'])

history=model.fit(train_features,train_labels,epochs=10,batch_size=20,validation_data=(valid_features,valid_labels))
model.save('pre_trained_cat_and-dig.h5')
model.summary()
#-------------------------------------------------------
#show results
#-----------------------------------------------------
import matplotlib.pyplot as plt
acc=history.history['acc']
loss=history.history['loss']
val_acc=history.history['val_acc']
val_loss=history.history['val_loss']
epochs=range(1,len(acc)+1)
plt.plot(epochs,acc,'bo',label='tarining_acc')
plt.plot(epochs,val_acc,'b',label='validation_acc')
plt.title('tarinig and validation acc')
plt.legend()
plt.show()

plt.figure()
plt.plot(epochs,loss,'bo',label='tarining_loss')
plt.plot(epochs,val_loss,'b',label='validation_loss')
plt.title('tarinig and validation loss')
plt.legend()
plt.show()





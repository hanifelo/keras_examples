#----------------------------
#crete train test validation set
#---------------------------------
import os, shutil
orginal_dir='/media/zahra/66048aa8-5d56-45a6-9c13-d1145235d158/pythonCode/catDog/resizeData'
baseDir='/media/zahra/66048aa8-5d56-45a6-9c13-d1145235d158/pythonCode/catDog/baseDir'
catDir=os.path.join(orginal_dir,'cat')
dogDir=os.path.join(orginal_dir,'dog')
os.mkdir(baseDir)

trainDir=os.path.join(baseDir,'train')
os.mkdir(trainDir)
validDir=os.path.join(baseDir,'valid')
os.mkdir(validDir)
testDir=os.path.join(baseDir,'test')
os.mkdir(testDir)

trainCatDir=os.path.join(trainDir,'cat')
os.mkdir(trainCatDir)
trainDogDir=os.path.join(trainDir,'dog')
os.mkdir(trainDogDir)

validCatDir=os.path.join(validDir,'cat')
os.mkdir(validCatDir)
validDogDir=os.path.join(validDir,'dog')
os.mkdir(validDogDir)

testCatDir=os.path.join(testDir,'cat')
os.mkdir(testCatDir)
testDogDir=os.path.join(testDir,'dog')
os.mkdir(testDogDir)

fnames=['{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
 src=os.path.join(catDir,fname)
 dst=os.path.join(trainCatDir,fname)
 shutil.copyfile(src,dst)

fnames=['{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
 src=os.path.join(dogDir,fname)
 dst=os.path.join(trainDogDir,fname)
 shutil.copyfile(src,dst)

fnames=['{}.jpg'.format(i) for i in range(1000,1500)]
for fname in fnames:
 src=os.path.join(catDir,fname)
 dst=os.path.join(validCatDir,fname)
 shutil.copyfile(src,dst)

fnames=['{}.jpg'.format(i) for i in range(1000,1500)]
for fname in fnames:
 src=os.path.join(dogDir,fname)
 dst=os.path.join(validDogDir,fname)
 shutil.copyfile(src,dst)

fnames=['{}.jpg'.format(i) for i in range(1500,2000)]
for fname in fnames:
 src=os.path.join(catDir,fname)
 dst=os.path.join(testCatDir,fname)
 shutil.copyfile(src,dst)

fnames=['{}.jpg'.format(i) for i in range(1500,2000)]
for fname in fnames:
 src=os.path.join(dogDir,fname)
 dst=os.path.join(testDogDir,fname)
 shutil.copyfile(src,dst)


print('total cat  test data:')
print(len(os.listdir(testCatDir)))

#---------------------------------------
#preprocessing and data augumation
#--------------------------------------
trainDir='/media/zahra/66048aa8-5d56-45a6-9c13-d1145235d158/pythonCode/catDog/baseDir/train'
validDir='/media/zahra/66048aa8-5d56-45a6-9c13-d1145235d158/pythonCode/catDog/baseDir/valid'
testDir='/media/zahra/66048aa8-5d56-45a6-9c13-d1145235d158/pythonCode/catDog/baseDir/test'

from keras.preprocessing.image import ImageDataGenerator
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
 trainDir,
 target_size=(150,150),
 batch_size=20,
 class_mode='binary')
valid_generator=test_datagen.flow_from_directory(
 validDir,
 target_size=(150,150),
 batch_size=20,
 class_mode='binary')
test_generator=test_datagen.flow_from_directory(
 testDir,
 target_size=(150,150),
 batch_size=20,
 class_mode='binary')
#-----------------------------------
#show some images
#---------------------------------
import os
from keras.preprocessing import image
import matplotlib.pyplot as plt
trainCatDir='/media/zahra/66048aa8-5d56-45a6-9c13-d1145235d158/pythonCode/catDog/baseDir/train/cat'
fnames=[os.path.join(trainCatDir,fname) for
 fname in os.listdir(trainCatDir)]
img_path=fnames[30]
img=image.load_img(img_path,target_size=(150,150))
x=image.img_to_array(img)
x=x.reshape((1,)+x.shape)
i=0
for batch in train_datagen.flow(x,batch_size=1):
 plt.figure(i)
 implot=plt.imshow(image.array_to_img(batch[0]))
 i+=1
 if i%4 ==0:
  break
plt.show()
#-------------------------------------------------------------
#create model
#---------------------------------------------------------------
from keras import models
from keras import layers
from keras import optimizers
from keras import regularizers
model=models.Sequential()
model.add(layers.Conv2D(32,(3,3),activation='relu',kernel_regularizer=regularizers.l2(0.005),input_shape=(150,150,3)))
model.add(layers.Dropout(0.05))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Conv2D(64,(3,3),kernel_regularizer=regularizers.l2(0.005),activation='relu'))
model.add(layers.Dropout(0.05))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Conv2D(128,(3,3),kernel_regularizer=regularizers.l2(0.005),activation='relu'))
model.add(layers.Dropout(0.1))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Conv2D(128,(3,3),kernel_regularizer=regularizers.l2(0.005),activation='relu'))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer=optimizers.RMSprop(lr=0.0001),metrics=['acc'])
history=model.fit_generator(
 train_datagenerator,
 steps_per_epoch=20,
 epochs=5,
 validation_data=valid_generator,
 validation_steps=50)

model.save('cat_and-dig.h5')
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




#---------------------------------------
#preprocessing and data augumation
#--------------------------------------
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
trainCatDir='/media/zahra/66048aa8-5d56-45a6-9c13-d1145235d158/pythonCode/catDog/baseDir/train/cat'
fnames=[os.path.join(trainCatDir,fname) for
 fname in os.listdir(trainCatDir)]
img_path=fnames[3]
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
plt.show()

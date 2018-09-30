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
 steps_per_epoch=100,
 epochs=100,
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


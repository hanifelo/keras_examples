from keras.datasets import mnist
(trainData,trainTarget),(testData,testTarget)=mnist.load_data()
from keras import models
from keras import layers
from keras import regularizers
from keras.utils import to_categorical
trainData=trainData.reshape(60000,28,28,1)
trainData=trainData.astype('float32')/255
testData=testData.reshape(10000,28,28,1)
testData=testData.astype('float32')/255
trainTarget=to_categorical(trainTarget)
testTarget=to_categorical(testTarget)

myNet=models.Sequential()
myNet.add(layers.Conv2D(32,(3,3),kernel_regularizer=regularizers.l2(0.005),activation='relu',input_shape=(28,28,1)))
myNet.add(layers.Dropout(0.05))
myNet.add(layers.MaxPooling2D(2,2))
myNet.add(layers.Conv2D(64,(3,3),kernel_regularizer=regularizers.l2(0.005),activation='relu'))
myNet.add(layers.Dropout(0.05))
myNet.add(layers.MaxPooling2D(2,2))
myNet.add(layers.Conv2D(64,(3,3),kernel_regularizer=regularizers.l2(0.005),activation='relu'))
myNet.add(layers.Dropout(0.05))
myNet.add(layers.Flatten())
myNet.add(layers.Dense(64,activation='relu'))
myNet.add(layers.Dense(10,activation='softmax'))
myNet.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
myNet.fit(trainData,trainTarget,epochs=5,batch_size=8)

testLoss,testAcc=myNet.evaluate(testData,testTarget)
print(testLoss)
print(testAcc)


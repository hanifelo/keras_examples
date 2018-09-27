
from keras.datasets import mnist
from keras.utils import to_categorical
from keras import models
from keras import layers


(trainData,trainTarget),(testData,testTarget)=mnist.load_data()

trainData=trainData.reshape(60000,28*28)
trainData=trainData.astype('float32')/255
testData=testData.reshape(10000,28*28)
testData=testData.astype('float32')/255
trainTarget=to_categorical(trainTarget)
testTarget=to_categorical(testTarget)

myNet=models.Sequential()
myNet.add(layers.Dense(512,activation='relu',input_shape=(28*28,)))
myNet.add(layers.Dense(10,activation='softmax'))
myNet.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
myNet.fit(trainData,trainTarget,epochs=5,batch_size=128)
testLoss,tesAcc=myNet.evaluate(testData,testTarget)
print(testLoss)
print(tesAcc)

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


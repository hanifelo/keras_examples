import os, shutil
import  sys
import Image
orginal_dir='/media/zahra/66048aa8-5d56-45a6-9c13-d1145235d158/pythonCode/catDog/PetImages'
baseDir='/media/zahra/66048aa8-5d56-45a6-9c13-d1145235d158/pythonCode/catDog/resizeData'
catDir=os.path.join(orginal_dir,'Cat')
dogDir=os.path.join(orginal_dir,'Dog')

os.mkdir(baseDir)
resizeCatDir=os.path.join(baseDir,'cat')
os.mkdir(resizeCatDir)
resizeDogDir=os.path.join(baseDir,'dog')
os.mkdir(resizeDogDir)

size = 150, 150

fnames=['{}.jpg'.format(i) for i in range(4000)]
for fname in fnames:
 src=os.path.join(catDir,fname)
 dst=os.path.join(resizeCatDir,fname)
 im=Image.open(src).convert('RGB')
 im.thumbnail(size, Image.ANTIALIAS)
 im.save(dst, "JPEG")

fnames=['{}.jpg'.format(i) for i in range(4000)]
for fname in fnames:
 src=os.path.join(dogDir,fname)
 dst=os.path.join(resizeDogDir,fname)
 im=Image.open(src).convert('RGB')
 im.thumbnail(size, Image.ANTIALIAS)
 im.save(dst, "JPEG")

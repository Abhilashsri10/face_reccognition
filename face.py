# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 14:20:12 2018

@author: Abhilash Srivastava
"""

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import math
import copy
from keras.models import Sequential,Model
from keras.layers import Input,Dense,Flatten,Dropout,Activation,Lambda,Permute,Reshape
from keras.layers import Convolution2D,MaxPooling2D
import cv2
from keras import backend as K
K.set_image_data_format('channels_last')
from scipy.io import loadmat
from keras.preprocessing.image import load_img,img_to_array
#model architecture
def convblock(cdim,nb,bits=3):
    L=[]
    for k in range(1,bits+1):
        convname='conv'+str(nb)+'_'+str(k)
        L.append( Convolution2D(cdim, kernel_size=(3,3), padding='same', activation='relu',name=convname))
    L.append(MaxPooling2D((2,2),strides=(2,2)))
    return L

def vgg_face_blank():
    withDo=True
    if True:
        mdl=Sequential()
        mdl.add(Permute((1,2,3),input_shape=(224,224,3)))
        for l in convblock(64,1,bits=2):
            mdl.add(l)
        for l in convblock(128,2,bits=2):
            mdl.add(l)
        for l in convblock(256,3,bits=3):
            mdl.add(l)
        for l in convblock(512,4,bits=3):
            mdl.add(l)
        for l in convblock(512,5,bits=3):
            mdl.add(l)
        mdl.add(Convolution2D(4096,kernel_size=(7,7),activation='relu',name='fc6'))
        if withDo:
            mdl.add(Dropout(0.5))
        mdl.add(Convolution2D(4096,kernel_size=(1,1),activation='relu',name='fc7'))
        if withDo:
            mdl.add(Dropout(0.5))
        
        mdl.add(Convolution2D(2622,kernel_size=(1,1),activation='relu',name='fc8'))
        mdl.add( Flatten() )
        mdl.add( Activation('softmax') )
        return mdl
    else:
        raise ValueError('not implemented')
facemodel=vgg_face_blank()
#facemodel.summary()

data=loadmat('vgg-face.mat',matlab_compatible=False,struct_as_record=False)
l=data['layers']
description=data['meta'][0,0].classes[0,0].description

def weight_compare(kmodel):
    kerasname=[lr.name for lr in kmodel.layers]
    prmt=(3,2,0,1)
    for i in range(l.shape[1]):
        matname=l[0,i][0,0].name[0]
        mattype=l[0,i][0,0].type[0]
        if matname in kerasname:
            kindex=kerasname.index(matname)
            print(matname,mattype)
            print(l[0,i][0,0].weights[0,0].transpose(prmt).shape,l[0,i][0,0].weights[0,1].shape)
            print(kmodel.layers[kindex].get_weights()[0].shape,kmodel.layer[kindex].get_weights()[1].shape)
            print('-------------------------')
        else:
            print('MISSING:',matname,mattype)
            print('-------------------------')
    
def copy_mat_to_keras(kmodel):
    kerasname=[lr.name for lr in kmodel.layers]
    prmt=(0,1,2,3)
    for i in range(l.shape[1]):
        matname = l[0,i][0,0].name[0]
        if matname in kerasname:
            kindex = kerasname.index(matname)
            #print matname
            l_weights = l[0,i][0,0].weights[0,0]
            l_bias = l[0,i][0,0].weights[0,1]
            f_l_weights = l_weights.transpose(prmt)

            assert (f_l_weights.shape == kmodel.layers[kindex].get_weights()[0].shape)
            assert (l_bias.shape[1] == 1)
            assert (l_bias[:,0].shape == kmodel.layers[kindex].get_weights()[1].shape)
            assert (len(kmodel.layers[kindex].get_weights()) == 2)
            kmodel.layers[kindex].set_weights([f_l_weights, l_bias[:,0]])
copy_mat_to_keras(facemodel)
def pred(kmodel,crpimg,transform=False):
    imarr=np.array(crpimg).astype(np.float32)
    if transform:
        imarr[:,:,0]-=129.1863
        imarr[:,:,1]-=104.7624
        imarr[:,:,2]-=93.5940
        aux=copy.copy(imarr)
    imarr=np.expand_dims(imarr,axis=0)
    out=kmodel.predict(imarr)
    
    best_index=np.argmax(out,axis=1)[0]
    best_name=description[best_index,0]
    print(best_index,best_name[0],out[0,best_index],[np.min(out),np.max(out)])
'''
im=Image.open('aamir.png')
im=im.resize((224,224))
plt.imshow(np.asarray(im))
crim=im
pred(facemodel,crim,transform=False)
pred(facemodel,crim,transform=True)
'''
#features
featuremodel = Model(inputs=facemodel.layers[0].input, outputs=facemodel.layers[-2].output)
def features(featmodel, crpimg, transform=False):
    imarr = np.array(crpimg).astype(np.float32)

    if transform:
        imarr[:,:,0] -= 129.1863
        imarr[:,:,1] -= 104.7624
        imarr[:,:,2] -= 93.5940
        aux = copy.copy(imarr)

    imarr = np.expand_dims(imarr, axis=0)

    fvec = featmodel.predict(imarr)[0,:]
    # normalize
    normfvec = math.sqrt(fvec.dot(fvec))
    return fvec/normfvec
'''
#WEBCAM
# Camera 0 is the integrated web cam on my netbook
camera_port = 0
 
#Number of frames to throw away while the camera adjusts to light levels
ramp_frames = 30
 
# Now we can initialize the camera capture object with the cv2.VideoCapture class.
# All it needs is the index to a camera port.
camera = cv2.VideoCapture(camera_port)
 
# Captures a single image from the camera and returns it in PIL format
#def get_image():
 # read is the easiest way to get a full image out of a VideoCapture object.
retval, im = camera.read()
 #return im
 
# Ramp the camera - these frames will be discarded and are only used to allow v4l2
# to adjust light levels, if necessary
for i in range(ramp_frames):
 temp = im
print("Taking image...")
# Take the actual image we want to keep
camera_capture = im
file = "C://Users//Abhilash Srivastava//Documents//python//deep_learning//face_classification//id_image.png"
# A nice feature of the imwrite method is that it will automatically choose the
# correct format based on the file extension you provide. Convenient!
cv2.imwrite(file, camera_capture)
 
# You'll want to release the camera, otherwise you won't be able to create a new
# capture object until your script exits
del(camera)
'''
#detection using opencv
path='abhi3.jpg'
cascade_file='haarcascade_frontalface_default.xml'
facecascade=cv2.CascadeClassifier(cascade_file)
image=cv2.imread(path)
gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
faces=facecascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30,30))
faces=facecascade.detectMultiScale(gray,1.2,5)
print("found {0} faces".format(len(faces)))
for(x,y,w,h)in faces:
    cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
plt.imshow(image)
im = Image.open(path)
(x, y, w, h) = faces[0]
center_x = x+w/2
center_y = y+h/2
b_dim = min(max(w,h)*1.2,im.width, im.height) # WARNING : this formula in incorrect
#box = (x, y, x+w, y+h)
box = (center_x-b_dim/2, center_y-b_dim/2, center_x+b_dim/2, center_y+b_dim/2)
# Crop Image
crpim = im.crop(box).resize((224,224))

#plt.imshow(np.asarray(crpim))
'''
pred(facemodel, crpim, transform=False)
pred(facemodel, crpim, transform=True)
'''
str1='ABHILASH'
str2='MANMEET'
str3='MOHINI'
str4='ABHINAV'
face_name={str1:'abhi.jpg',str2:'manmeet1.jpg',str3:'moh1.jpg',str4:'ABHINAV1.JPG'}
def get_im(strx):
# load an image from file
    imarr_i = load_img(face_name[strx], target_size=(224, 224))
# convert the image pixels to a numpy array
    imarr_i = img_to_array(imarr_i)
# reshape data for the model
    imarr_i = imarr_i.reshape((1, imarr_i.shape[0], imarr_i.shape[1], imarr_i.shape[2]))
    return imarr_i

imarr1=get_im(str1)
imarr2=get_im(str2)
imarr3=get_im(str3)
#imarr3=get_im(str3)
#imarr4=get_im(str4)

imarr_t=crpim
imarr_t = img_to_array(imarr_t)
# reshape data for the model
imarr_t = imarr_t.reshape((1, imarr_t.shape[0], imarr_t.shape[1], imarr_t.shape[2]))
#imarr2=Image.open(crpim)
face_id={str1:imarr1,str2:imarr2,str3:imarr3}
featuremodel = Model( input = facemodel.layers[0].input,output = facemodel.layers[-2].output )
from scipy.spatial.distance import cosine as dcos
for j in face_id:
    fvec1 = featuremodel.predict(face_id[j])[0,:]
    fvec2 = featuremodel.predict(imarr_t)[0,:]
    dcos_1_2 = dcos(fvec1, fvec2)
    if(dcos_1_2<0.2):
        print('\n \n \n RECOGNISED {0}'.format(j))


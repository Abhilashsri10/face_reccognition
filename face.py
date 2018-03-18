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

#detection using opencv
path='abhi2.jpg'
cascade_file='haarcascade_frontalface_default.xml'
facecascade=cv2.CascadeClassifier(cascade_file)
image=cv2.imread(path)
gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
faces=facecascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30,30))
faces=facecascade.detectMultiScale(gray,1.2,5)
print("found{0} faces".format(len(faces)))
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

# load an image from file
imarr1 = load_img('abhi.jpg', target_size=(224, 224))
# convert the image pixels to a numpy array
imarr1 = img_to_array(imarr1)
# reshape data for the model
imarr1 = imarr1.reshape((1, imarr1.shape[0], imarr1.shape[1], imarr1.shape[2]))
imarr2=crpim
imarr2 = img_to_array(imarr2)
# reshape data for the model
imarr2 = imarr2.reshape((1, imarr2.shape[0], imarr2.shape[1], imarr2.shape[2]))
#imarr2=Image.open(crpim)

featuremodel = Model( input = facemodel.layers[0].input,output = facemodel.layers[-2].output )
from scipy.spatial.distance import cosine as dcos
fvec1 = featuremodel.predict(imarr1)[0,:]
fvec2 = featuremodel.predict(imarr2)[0,:]
dcos_1_2 = dcos(fvec1, fvec2)
if(dcos_1_2<0.2):
    print('RECOGNISED:Abhilash')
else:
    print('NOT RECOGNISED')

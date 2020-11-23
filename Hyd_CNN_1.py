

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 04:31:50 2020

@author: krajna4ever
"""




# Import all the libraries required for the training
from os import environ, chdir
from keras.preprocessing.image import ImageDataGenerator  #preprocessing method for the image data generator 
from keras import models, layers, optimizers, callbacks 



# Specify our working directory that consists of eval, train, valid set 
environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
chdir(r'/home/krajna4ever/Desktop/')

#Augmentation of data, where to find the pictures and which transformations to carry out on them  
images_tr_dg = ImageDataGenerator(rescale=1./255, zoom_range=[1.0,1.25],width_shift_range=0.1, height_shift_range=0.1, fill_mode='reflect')
images_train = images_tr_dg.flow_from_directory(directory=r'Dataset/train', target_size=(100,100), class_mode='binary', batch_size=125, shuffle= True)
tovalid_idg = ImageDataGenerator(rescale=1./255)
valid_g = tovalid_idg.flow_from_directory(directory=r'Dataset/valid', target_size=(100,100), class_mode='binary', batch_size=125, shuffle= True)

######################################################################################################################################################

#CNN ARCHITECTURE 

hydrangea_model = models.Sequential()
hydrangea_model.add(layers.Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), input_shape=(100, 100, 3)))
hydrangea_model.add(layers.Activation('relu'))
hydrangea_model.add(layers.Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1)))
hydrangea_model.add(layers.Activation('softmax'))
hydrangea_model.add(layers.MaxPooling2D(pool_size=(3, 3)))
hydrangea_model.add(layers.Dropout(rate=0.3))
hydrangea_model.add(layers.Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1)))
hydrangea_model.add(layers.Activation('relu'))
hydrangea_model.add(layers.MaxPooling2D(pool_size=(3, 3))) 
hydrangea_model.add(layers.Flatten())
hydrangea_model.add(layers.Dense(units=10))
hydrangea_model.add(layers.Activation('relu'))
hydrangea_model.add(layers.Dense(units=1))
hydrangea_model.add(layers.Activation('sigmoid'))


print(hydrangea_model.summary())


#Loss and Optimization functions

compile_hmodel = hydrangea_model.compile(optimizer = optimizers.RMSprop(learning_rate=0.17, rho=0.9, momentum=0.1), loss='binary_crossentropy', metrics=['accuracy'])

#setting callbacks- tools for allowing us flexible options during training , checkpoint and reduced learning rate

chkpt=callbacks.ModelCheckpoint(filepath='hydrangea_model_{val_accuracy:.2f}.h5', monitor='val_accuracy',verbose=1,save_best_only=True, save_weights_only=False)
reduce_lr=callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.95,patience=5,verbose=1,cooldown=2)
callb_l=[chkpt, reduce_lr]


#Training options setting them 
fitara = hydrangea_model.fit_generator(generator=images_train, steps_per_epoch=22, epochs=80, verbose=1, callbacks=callb_l, validation_data= valid_g, validation_steps=4)


#Saving the model
hydrangea_model.save(filepath=r'hydrangea_cnn.h5', overwrite=True)




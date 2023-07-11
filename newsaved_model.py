import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Dropout,Flatten,Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np
import os

main_dir="New Masks Dataset"
train_dir=os.path.join(main_dir,"Train")
test_dir=os.path.join(main_dir,"Test")
validation_dir=os.path.join(main_dir,"Validation")

train_mask_dir=os.path.join(train_dir,"Mask")
train_nomask_dir=os.path.join(train_dir,"Non Mask")

train_masks_names=os.listdir(train_mask_dir)

train_nomask_names=os.listdir(train_nomask_dir)

train_datagen=ImageDataGenerator(rescale=1./255,
                                zoom_range=0.2,
                                rotation_range=40,
                                horizontal_flip=True
                                )
test_datagen=ImageDataGenerator(rescale=1./25)
validation_datagen=ImageDataGenerator(rescale=1./25)
train_generator=train_datagen.flow_from_directory(train_dir,
                                                 target_size=(150,150),
                                                 batch_size=32,
                                                 class_mode='binary'
                                                 )
test_generator=test_datagen.flow_from_directory(test_dir,
                                                 target_size=(150,150),
                                                 batch_size=32,
                                                 class_mode='binary'
                                                 )
validation_generator=validation_datagen.flow_from_directory(validation_dir,
                                                 target_size=(150,150),
                                                 batch_size=32,
                                                 class_mode='binary'
                                                 )

train_generator.class_indices

model=keras.Sequential()
model.add(Conv2D(32,(3,3),padding='SAME',activation='relu',input_shape=(150,150,3)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))
#
model.add(Conv2D(32,(3,3),padding='SAME',activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))
#
model.add(Flatten())
#
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1,activation='sigmoid'))
model.summary()

model.compile(Adam(lr=0.001),loss='binary_crossentropy',metrics=['accuracy'])

history=model.fit(train_generator,
                 epochs=30,
                 validation_data=validation_generator)

model.evaluate(test_generator)

model.save('newsaved_model.h5')

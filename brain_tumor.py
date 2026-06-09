import numpy as np
import pandas as pd
import cv2
import os
from PIL import Image
from sklearn.model_selection import train_test_split
from keras.utils import normalize
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *

image_directory = r'C:\Users\saini\OneDrive\ONLY FOR ME\ai\deeplearning\braintumor\brain_tumor_dataset\\'


no_tumor_images= os.listdir(image_directory + 'no/')
yes_tumor_images= os.listdir(image_directory + 'yes/')
#print(no_tumor_images)

dataset=[]
label=[]
input_size=128

for i, image_name in enumerate(no_tumor_images):
    if(image_name.split('.')[1]=='jpg'):
        image=cv2.imread(image_directory+'no/'+image_name,cv2.IMREAD_GRAYSCALE)
        #image=Image.fromarray(image,'RGB')

        image = cv2.resize(image, (128,128))
        dataset.append(np.array(image))

        #image=image.resize((input_size,input_size))
        #dataset.append(np.array(image))
        label.append(0)


for i, image_name in enumerate(yes_tumor_images):
    if(image_name.split('.')[1]=='jpg'):
        image=cv2.imread(image_directory+'yes/'+image_name,cv2.IMREAD_GRAYSCALE)

        image = cv2.resize(image, (128,128))
        dataset.append(np.array(image))


        #image=Image.fromarray(image,'RGB')
        #image=image.resize((input_size,input_size))
        #dataset.append(np.array(image))
        label.append(1)


dataset=np.array(dataset)

dataset = np.expand_dims(dataset, axis=-1)
label=np.array(label)

x_train,x_test,y_train,y_test=train_test_split(dataset,label,test_size=0.2,random_state=0)
#print(x_train.shape)

x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    horizontal_flip=True
)
datagen.fit(x_train)
model=Sequential()

model.add(Conv2D(32,(3,3),input_shape=(input_size,input_size,1)))
#model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32,(3,3),kernel_initializer='he_uniform'))
#model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32,(3,3),kernel_initializer='he_uniform'))
#model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(128,Activation('relu')))
model.add(Dropout(0.5))
model.add(Dense(1,Activation('sigmoid')))


#binary cross entropy loss= 1 dense layer and sigmoid
#cross entropy loss = 2 dense layer(no. of classes) and softmax

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=6,
    restore_best_weights=True
)
model.fit(
          datagen.flow(x_train,y_train,batch_size=16),
          verbose=1,
          epochs=50,
          validation_data=(x_test,y_test),
          shuffle=False,
          callbacks=[early_stop])

from sklearn.metrics import classification_report

y_pred = (model.predict(x_test) > 0.5).astype(int)
print(classification_report(y_test, y_pred))

from sklearn.metrics import roc_auc_score
y_prob = model.predict(x_test)
print("AUC:", roc_auc_score(y_test, y_prob))

model.save("brain_tumor_cnn.keras")
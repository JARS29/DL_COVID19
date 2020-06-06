# -*- coding: utf-8 -*-
"""
Created on Thu May 28 00:39:10 2020

@author: JARS
"""

import tensorflow as tf  


def train_validation_data(data_directory, model, height, width, batch_size, class_type, val_split):
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    if model=='in_resV2':
        from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
        pre_function=preprocess_input
    elif model=='mobv2':
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
        pre_function=preprocess_input
    elif model=='Xception':
        from tensorflow.keras.applications.xception import preprocess_input
    else:
        pre_function=None

    train_datagen = ImageDataGenerator(  
    rescale=1. / 255,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    horizontal_flip = True,
    rotation_range = 40, 
    zoom_range = 0.2,
    shear_range = 0.2,
    fill_mode = "nearest",
    validation_split=val_split,
    preprocessing_function=pre_function)
    
    val_datagen=ImageDataGenerator(rescale=1. / 255,     
                               validation_split=val_split,
                               preprocessing_function=pre_function)
    
    train_generator = train_datagen.flow_from_directory(  
    data_directory,
    target_size=(height, width),
    batch_size=batch_size,
    class_mode=class_type,
    subset='training',
    shuffle=True)
     
    validation_generator = val_datagen.flow_from_directory(  
    data_directory,
    target_size=(height, width),
    batch_size=batch_size,
    class_mode=class_type,
    subset='validation',
    shuffle=False)
    
    return train_generator, validation_generator

def unfreeze(model, a,b):
    i=0
    for layer in model.layers:
        if i<a*len(model.layers)//b:
            layer.trainable=False
        else:
            layer.trainable=True
        i=i+1
    
def model_TL(model,  height, width, channels): #models= VGG, InceptionResNETV2, MobileNet, Xception, Dense
    if model=='in_resV2':
        from tensorflow.keras.applications import inception_resnet_v2
        pre_trained_model = inception_resnet_v2(
            weights = "imagenet",#NULL,#imagenet", Random weight
            include_top = False,
            input_shape = (height, width, channels)
          )
    elif model=='mobv2':
        from tensorflow.keras.applications import MobileNetV2
        pre_trained_model = MobileNetV2(
            weights = "imagenet",#NULL,#imagenet", Random weight
            include_top = False,
            input_shape = (height, width, channels)
          )
    elif model=='Xception':
        from tensorflow.keras.applications import xception    
        pre_trained_model = xception(
            weights = "imagenet",#NULL,#imagenet", Random weight
            include_top = False,
            input_shape = (height, width, channels)
          )
    elif model=='VGG':
        from tensorflow.keras.applications import VGG16
        pre_trained_model = VGG16(
            weights = "imagenet",#NULL,#imagenet", Random weight
            include_top = False,
            input_shape = (height, width, channels)
          )
    elif model=='dense':
        from tensorflow.keras.applications import DenseNet201
        pre_trained_model = DenseNet201(
            weights = "imagenet",#NULL,#imagenet", Random weight
            include_top = False,
            input_shape = (height, width, channels)
          )
    #pre_trained_model.summary()
    unfreeze(pre_trained_model, 3,4)
    # i=0
    # for layer in pre_trained_model.layers:
    #     if i<3*len(pre_trained_model.layers)//4:
    #         layer.trainable=False
    #     else:
    #         layer.trainable=True
    #     i=i+1
    

    final_model=tf.keras.models.Sequential([
        pre_trained_model,
        tf.keras.layers.Flatten(),
        # tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(256,activation='relu'),
        tf.keras.layers.Dropout(0.5),
        #tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128,activation='relu'),#kernel_regularizer=regularizers.l2(0.001)
        tf.keras.layers.Dropout(0.2),
        #tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1,activation='sigmoid'),
    ])
    
    final_model.summary()
    return final_model

def training_model(model, epochs, batch_size, learning_rate, training_data, validating_data, patience, stage, model_name):
    from keras.callbacks import EarlyStopping, ModelCheckpoint  

    if stage=='val':
        unfreeze(model, 4,5)
        
    model.compile(
    #optimizer=tf.keras.optimizers.Adam(lr = 0.0001)
    optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.99),
    loss='binary_crossentropy',
    metrics= ["accuracy"])
    
    es = EarlyStopping(monitor='val_acc', mode='max', verbose=1, patience=patience)
    mc = ModelCheckpoint('models\\best_model_'+model_name+'.h5', 
                         monitor='val_acc', mode='max', verbose=1, save_best_only=True)
    
    history=model.fit_generator(
    training_data,
    steps_per_epoch = training_data.n//batch_size+1,  
    epochs = epochs,
    validation_data = validating_data,
    validation_steps = validating_data.n//batch_size+1,
    callbacks=[es, mc])
    
    return history


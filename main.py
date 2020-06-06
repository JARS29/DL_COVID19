# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 23:35:18 2020

@author: JARS
"""

from models_full import *
from utils import *
import tensorflow as tf  
from keras import backend as K  
import time
import os  
path_base = os.getcwd()
data_base= path_base+'\\SARS-Cov-2'
tf.debugging.set_log_device_placement(True)


K.set_learning_phase(1)
width=224
height = 224
channels=3
epochs = 50 
batch_size = 16  
n_classes = 2  
class_type="binary"
model='VGG'

train_generator, validation_generator= train_validation_data(data_base, 
                                                             model,
                                                             height, 
                                                             width, 
                                                             batch_size, 
                                                             class_type, 
                                                             val_split=0.2)

pre_trained_model=model_TL(model,  
                           height,
                           width, 
                           channels) 

#Training
t0=time.clock()

history=training_model(pre_trained_model,
               epochs=epochs,
               batch_size=batch_size, 
               learning_rate=0.0001, 
               training_data=train_generator, 
               validating_data=validation_generator, 
               patience=15, 
               stage='train',
               model_name=model)

training_time=time.time()-t0


test_lost, test_acc= pre_trained_model.evaluate_generator(validation_generator)
print ("Test Accuracy:", test_acc)

true_label=validation_generator.classes


from numpy import round
predict_label=round(pre_trained_model.predict_generator(validation_generator, 
                                                        validation_generator.n//batch_size))
plotting_save(history, 
         true_label=true_label, 
         predict_label=predict_label, 
         model_name=model, 
         stage='train',
         times=None)


#Validating
from tensorflow.keras.models import load_model
K.set_learning_phase(0)

loaded_model=load_model('models\\best_model_'+model+'.h5')

t0=time.clock()
history_lm=training_model(loaded_model,
               epochs=25,
               batch_size=batch_size//2, 
               learning_rate=0.0001, 
               training_data=train_generator, 
               validating_data=validation_generator, 
               patience=10, 
               stage='val',
               model_name=model)

training_time_LM=time.time()-t0

test_lost_lm, test_acc_lm= loaded_model.evaluate_generator(validation_generator)
print ("Test Accuracy:", test_acc)
print ("Test Accuracy - LM:", test_acc_lm)

from numpy import round
t0=time.clock()
predict_label=round(loaded_model.predict_generator(validation_generator, 
                                                        validation_generator.n//batch_size))
prediction_time=time.clock-t0
plotting_save(history, 
              true_label=true_label, 
              predict_label=predict_label, 
              model_name=model,
              stage='val',
              times=[training_time, training_time_LM, prediction_time])


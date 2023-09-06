##### MODEL ####
import tensorflow as tf
import numpy as np
import pickle
import os
import csv
import tensorflow.keras.backend as K
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from sklearn.metrics import mean_squared_error
from dataTestProcess import dataTestProcess
from tensorflow.keras import activations

#####CHANGE THE MODEL NAME######
model_directory='./forecast_model' # directory to save model history after every epoch 

class StoreModelHistory(keras.callbacks.Callback):

  def on_epoch_end(self,batch,logs=None):
    if ('lr' not in logs.keys()):
      logs.setdefault('lr',0)
      logs['lr'] = K.get_value(self.model.optimizer.lr)

    if not ('model_history.csv' in os.listdir(model_directory)):
      with open(model_directory+'model_history.csv','a') as f:
        y=csv.DictWriter(f,logs.keys())
        y.writeheader()

    with open(model_directory+'/model_history.csv','a') as f:
      y=csv.DictWriter(f,logs.keys())
      y.writerow(logs)

class forecastModel():

    ######  DATA SHOULD COME HERE AS A PANDAS DATAFRAME SO IT'S EASY TO USE ADAPT IT WHILE
    ###### WHILE WORKING
    def __init__(self, model_name = ""):
        self.model = None
        self.dTP = dataTestProcess()

    def constructModel(self, x_training_data):
        rnn = Sequential()
        rnn.add(LSTM(units = 64, return_sequences = True, input_shape = (x_training_data.shape[1], x_training_data.shape[2])))

        #Perform some dropout regularization
        rnn.add(Dropout(0.2))
        rnn.add(LSTM(units = 64))
        rnn.add(Dropout(0.2))

        #Adding our output layer
        rnn.add(Dense(units = 1))
        #Compiling the recurrent neural network

        rnn.compile(optimizer =keras.optimizers.Adam(), loss = 'mae')

        self.model = rnn

    def fitModel(self, Xtrain, ytrain, Xval, yval, save= False, test= False):
        early = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=2)
        if(test == False):
          self.model.fit(Xtrain, ytrain, batch_size= 72, epochs= 20, 
                         validation_data=(Xval, yval), callbacks=[early, StoreModelHistory()])
        else:
          self.model.fit(Xtrain, ytrain, batch_size= 96, epochs= 5, 
                         validation_data=(Xval, yval), callbacks=[early])

    def predictModel(self, Xtest, ytest):
        predict_values = self.model.predict(Xtest)
        return predict_values, ytest

    def saveModel(self, filename):
        if not os.path.exists(filename):
          os.mkdir(filename)
        filename = os.path.join(filename, filename)
        pickle.dump(self.model, open(filename, 'wb'))

    def loadModel(self, filename):
        self.model = pickle.load(open(filename, 'rb'))

    def multiStepForecast(self, data, timestep = 5, init_time = 0):
        window_length = 24  if timestep == 5 else 8
        seq_len = 144
        seq_init = init_time
        dataCopy = data
        forecasts= []
        seq_final = seq_init+seq_len
        inputModel = dataCopy[seq_init:seq_final].to_numpy()
        forecast = self.model.predict(np.array([inputModel]))
        forecasts.append(forecast)
        for i in range (0, window_length-1):
            seq_init +=1
            seq_final +=1
            dataCopy.at[seq_final,'previous_y'] = forecast
            inputModel = dataCopy[seq_init:seq_final].to_numpy()
            forecast = self.model.predict(np.array([inputModel]))
            forecasts.append(forecast)
        
        return forecasts 
            
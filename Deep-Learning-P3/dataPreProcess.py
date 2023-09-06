##### DATA MANAGER #####
import pandas as pd
import csv
from numpy import percentile
import os 
import matplotlib.pyplot as plt
from IPython.display import display
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from numpy import array
from visualization import * 


class dataManager: 

    def __init__(self, path:str = "", name_model=""):
        self.pth = path
        self.vis = visualizator(name_model)
        
    #######################################    
    ######### PUBLIC FUNCTIONS ############
    #######################################

    def getData (self, normalization= "minmax", altered_forecast = False):
        pathAct= os.getcwd()
        pathDataTrain = os.path.join(pathAct, "data", "no1_train.csv")
        pathDataVal = os.path.join(pathAct, "data", "no1_validation.csv")

        #######CHANGE FOR GET THE FILE IN DEMO##############
        
        #pdSetTrain= pd.read_csv("/content/no1_train.csv")
        #pdSetVal = pd.read_csv("/content/no1_validation.csv")

        pdSetTrain= pd.read_csv(pathDataTrain)
        pdSetVal = pd.read_csv(pathDataVal)
        
        pdSetTrain["flow"] = -pdSetTrain["flow"]
        pdSetVal["flow"] = -pdSetVal["flow"]

        ####################################################
        
        ########DATA PREPROCESSING#########

        if(altered_forecast == True):
          print("entro altered forecast")
          pdSetTrain = self.altered_forecast_pre(pdSetTrain)
          print("validation set")
          pdSetVal = self.altered_forecast_pre(pdSetVal)

        plt = self.vis.correlationMap(pdSetTrain)
        self.vis.save_figure("correlation_Map", plt)

        pdSetTrain= self.__nonUsefulVars(pdSetTrain)
        pdSetVal= self.__nonUsefulVars(pdSetVal)
        
        pdSetTrain = self.__addLaggedVar(pdSetTrain)
        pdSetVal = self.__addLaggedVar(pdSetVal)

        plt = self.vis.plotData(pdSetTrain['y'])
        self.vis.save_figure("y_values_before_preprocessing(outliers)", plt)

        pdSetTrain = self.__OutliersProcess(pdSetTrain)
        pdSetVal = self.__OutliersProcess(pdSetVal)
        plt = self.vis.plotData(pdSetTrain['y'])
        self.vis.save_figure("y_values_after_preprocessing(outliers)", plt)

        self.__addCategoricalVars(pdSetTrain)
        self.__addCategoricalVars(pdSetVal)

        pdSetTrain.drop('start_time', inplace=True, axis=1)
        pdSetVal.drop('start_time', inplace=True, axis=1)
        
        pdSetTrain = self.normalizeData(pdSetTrain, normalization)
        pdSetVal = self.normalizeData(pdSetVal, normalization)

        return pdSetTrain, pdSetVal

    def altered_forecast_pre(self, data):
      dataCopy = data
      dataCopy['suma_cols'] = data['flow'].add(data['total'])
      print("suma de columnes")
      print(dataCopy['suma_cols'])

      dataCopy['suma_inter'] = data['flow'].add(data['total']).interpolate(method = "spline", order= 4)
      print("interpol de columnes")
      print(dataCopy['suma_inter'])

      dataCopy['resta_cols'] = dataCopy['suma_inter'].sub(dataCopy['suma_cols'])
      print("resta de columnes")
      print(dataCopy['resta_cols'])

      dataCopy['y'] = dataCopy['y'].sub(dataCopy['resta_cols'])
      dataCopy.drop('suma_cols', axis =1)
      dataCopy.drop('suma_inter', axis =1)
      dataCopy.drop('resta_cols', axis =1)
      print("inside funcio")
      print(dataCopy['y'])
      return dataCopy
    
    def getTarget(self, dataframe, timeStep = 144, test = False):
        if (test == True):
            dataframe.reset_index(drop= True, inplace= True)
        dataY = dataframe["y"].copy().sort_index(axis=0)
        dataX = dataframe.drop("y", axis = 1)
        last_dim = len(dataX.columns)
        ###DADES DE TEST
        if (test == True):
            return dataX, dataY
        X, y = self.split_sequence(dataX, dataY, timeStep,test)
        X = X.reshape((X.shape[0], X.shape[1], last_dim))
        y = y.reshape((y.shape[0], 1))
        return X, y

    ### DATA SHAPING ### 
    def split_sequence(self, dataX, dataY, n_steps_in, test):
        X, y = list(), list()
        for i in range(len(dataX)):
            # find the end of this pattern
            end_ix = i + n_steps_in
            # check if we are beyond the sequence
            if end_ix >= len(dataX):
                break
            # gather input and output parts of the pattern
            seq_x, seq_y = dataX[i:end_ix], dataY[end_ix]
            X.append(seq_x)
            y.append(seq_y)
        return array(X), array(y)

    #######################################
    ## NORMALIZATION/STANDARIZATION CODE ##
    #######################################
    

    def __normalization(self, data):
        max = data.max()
        norm_data = data/max
        return norm_data

    def normalizeData(self, data, type:str):
      dataCopy = data
      if (type == "norm"):
          for column in data:
              dataCopy[column]= self.__normalization(dataCopy[column]).values
        
      elif(type== 'minmax'):
          scaler = MinMaxScaler(feature_range=(-1,1))
          dataCopy = pd.DataFrame(scaler.fit_transform(dataCopy.values), columns=dataCopy.columns, index=dataCopy.index)
      
      elif(type == "std"):
          ss = StandardScaler()
          dataCopy = pd.DataFrame(ss.fit_transform(dataCopy),columns = dataCopy.columns)
      
      return dataCopy


   
    #######################################
    ######## ADDITION OF VARIABLES ########
    #######################################

    '''
    Add a column of variables that defines the time of the day:
    'N' : Night time --> 1
    'M' : Morning --> 2
    'A' : Afternoon --> 3
    '''
    def __addHourVars(self, timeData):
        hours = []
        for _, hour in timeData.items():
            if(hour > 21 and hour < 7): hours.append(1)
            elif(hour > 7 and hour < 15 ): hours.append(2)
            else: hours.append(3)
        return hours
                
    '''
    S'HA D'ACABAR

    Add a column of variables that defines the time of the day:
        Week days --> 1
        Weekend --> 2
    '''
    def __addWeekVars(self, timeData):
        weekMom = []
        for _, value in timeData.items():
            if (value < 4):
                weekMom.append(0)
            
            else: weekMom.append(1)
        
        return weekMom

    '''
    Add a column of variables that defines the time of the day:
        Winter --> 1
        Spring --> 2
        Summer --> 3
        Autumn --> 4
    '''
    def __addYearVars(self,timeData):
        seasons=[]
        for _, season in timeData.items():
            if(season == 12 or season < 4): seasons.append(1)
            elif(season >= 4 and season < 6): seasons.append(2)
            elif(season >= 6 and season <= 8): seasons.append(3)
            else: seasons.append(4)
        
        return seasons
 
    ######CHANGE THIS SO IT ADDS VARIABLES 

    '''
    Returns a pandas dataset with time variables
    '''    
    def __addCategoricalVars(self, data):
        data['start_time'] = pd.to_datetime(data['start_time'],format='%Y-%m-%d %H:%M:%S', errors='coerce')
        data['season'] = self.__addYearVars(data['start_time'].dt.month)
        data['moment_week']=self.__addWeekVars(data['start_time'].dt.dayofweek) 
        data['time_of_the_day'] = self.__addHourVars(data['start_time'].dt.hour) 
        
    '''
    Returns a pandas dataset with lagged variable
    '''
    def __addLaggedVar(self, data, previous_y = True, prev_24h_y = True):
      data_new = data
      y_column = data.loc[:,'y']
      if(previous_y):
        y_column1 = y_column.iloc[:-1].values
        data_new = data_new.iloc[1:,:]
        data_new['previous_y']= y_column1.tolist()
      
      if(prev_24h_y):
        y_column2 = y_column.iloc[:-289].values
        data_new = data_new.iloc[288:, :]
        data_new['prev_24h_y']= y_column2.tolist()
      
      data_new.reset_index(drop= True, inplace= True)
      data_new.info
      return data_new

    #######################################
    ##### PRE-PROCESSING OF VARIABLES #####
    #######################################

    '''
    Returns a pandas dataset without outliers
    '''
    def __OutliersProcess(self, pandDataSet):
        pdDataSet= pandDataSet.loc[:,'y'].to_list()
        qLow, qUp = percentile(pdDataSet, 0.5), percentile(pdDataSet, 99.5)
        for i in range(1,len(pdDataSet)):
            if(pdDataSet[i] < qLow or pdDataSet[i] > qUp):
                pandDataSet.at[i,'y'] = (pdDataSet[i-1]+pdDataSet[i+1])/2
        return pandDataSet

    def __nonUsefulVars(self, pandDataSet):
        newpd= pandDataSet.drop('river', axis = 1)
        return newpd
      

###################################################################
###################################################################
###################################################################
    
#### DATA TEST PROCESS ####

import pandas as pd
from numpy import percentile
import os 
import matplotlib.pyplot as plt 
from IPython.display import display
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import SplineTransformer
from numpy import array
from visualization import visualizator
import numpy as np

class dataTestProcess:

    def __init__(self, path:str = "" ):
        self.pth = path
        self.vis = visualizator()
    

    def __dateTimeToIndex(self, datetime, data):
        dataCopy = data
        #dataCopy['start_time'] = pd.to_datetime(dataCopy['start_time'],format='%Y-%m-%d %H:%M:%S', errors='coerce')
        return dataCopy.index(dataCopy['start_time'] == datetime)

    #######################################    
    ######### PUBLIC FUNCTIONS ############
    #######################################

    def getData (self, datetime, normalization = "minmax", altered_forecast = False):
        pathAct= os.getcwd()
        pathDataTest = os.path.join(pathAct, "data", self.pth)

        #######CHANGE FOR GET THE FILE IN DEMO##############
        
        pdSetTrain= pd.read_csv(pathDataTest)

        ####################################################
        
        ########DATA PREPROCESSING#########
        self.vis.correlationMap(pdSetTrain)
        pdSetTrain= self.__nonUsefulVars(pdSetTrain)

        print("-------------------")
        print("NONE USEFUL VARIABLES PROCESSED")
        print("-------------------")

        if(altered_forecast == True):
          pdSetTrain = self.altered_forecast_pre(pdSetTrain)

        pdSetTrain = self.__OutliersProcess(pdSetTrain)
        print("-------------------")
        print("OUTLIERS PROCESSED")
        print("-------------------")

        print("add previous y variable? (yes/no)")
        inpt_previous_y = input()
        previous_y = True
        if(inpt_previous_y == "no"):
            previous_y = False
        
        print("add previous 24h y variable? (yes/no)")
        inpt_previous_y = input()
        previous_24h_y = True
        if(inpt_previous_y == "no"):
            previous_24h_y = False

        pdSetTrain = self.__addLaggedVar(pdSetTrain, previous_y, previous_24h_y)

        print("-------------------")
        print("LAGGED VARIABLES ADDED")
        print("-------------------")

        self.__addCategoricalVars(pdSetTrain)
        print("-------------------")
        print("CATEGORICAL VARIABLES ADDED")
        print("-------------------")
        
        if (not datetime == -1):
            index = self.__dateTimeToIndex(datetime, pdSetTrain)

        else:
            index = -1

        pdSetTrain.drop('start_time', inplace=True, axis=1)
        pdSetTrain = self.normalizeData(pdSetTrain, normalization)
        print("-------------------")
        print("DATA NORMALIZED")
        print("-------------------")

        return pdSetTrain, index

    def altered_forecast_pre(self, data):
        spline = SplineTransformer(degree=2, n_knots=3)
        flow = data.loc[:, 'flow'].to_numpy()
        total = data.loc[:,'total'].to_numpy()
        y = data.loc[:,'y'].to_numpy()
        suma_cols = np.sum([total, flow], axis=0)
        suma_cols = suma_cols.reshape(-1,1)
        interp = spline.fit_transform(suma_cols)[:][0]

        resta = np.subtract(interp, suma_cols)
        y = np.subtract(y, resta)
        data.drop("y", axis = 1, inplace= True)
        data["y"] = pd.Series(y)

    #######################################
    ## NORMALIZATION/STANDARIZATION CODE ##
    #######################################
    

    def __normalization(self, data):
        max = data.max()
        norm_data = data/max
        return norm_data

    def normalizeData(self, data, type:str):
      dataCopy = data
      if (type == "normalize"):
          for column in data:
              dataCopy[column]= self.__normalization(dataCopy[column]).values
        
      elif(type== 'minmax'):
          print("entro a minmax")
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
                if(i < 2):
                    pandDataSet.at[i,'y'] = (pdDataSet[i+1]+pdDataSet[i+2]+
                    pdDataSet[i+3] + pdDataSet[i+4])/4
                
                elif(i > len(pdDataSet)-2):
                    pandDataSet.at[i,'y'] = (pdDataSet[i-1]+pdDataSet[i-2]+
                    pdDataSet[i-3] + pdDataSet[i-4])/4
                else:
                    pandDataSet.at[i,'y'] = (pdDataSet[i-2]+pdDataSet[i-1]+
                    pdDataSet[i+1] + pdDataSet[i+2])/4
        
        return pandDataSet

    def __nonUsefulVars(self, pandDataSet):
        newpd= pandDataSet.drop('river', axis = 1)
        return newpd
        
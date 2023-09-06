######  VISUALIZATOR #####

from unicodedata import name
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
from statsmodels.graphics.tsaplots import plot_acf
import os
import pandas as pd


class visualizator:

    def __init__(self, name_model = ""):
        self.name_model = name_model

    def __validColumn(self, nameColumn):
        valid = ["hydro", "flow", "sys_reg", "total", "wind", "thermal", "micro" ]
        if nameColumn in valid: return True
        else: return False
    
    def correlationMap(self, data):
        corrMat = data.corr()
        sn.heatmap(corrMat, annot=True)
        plt.title('Correlation Map')
        plt.show()
        return plt
        

    def plotDataY(self, data, pre_outliers):
        plt.scatter(x = [range(0,len(data))], y = data)
        if(pre_outliers == True):
          plt.title('Y values pre outliers process')
        else:
          plt.title('Y values post outliers process')
        plt.show()
        return plt


    def plotValues(self, data):
        figNum = 0
        for columnName, columnData in data.iteritems():
            if(self.__validColumn(columnName)):
                y = columnData
                plt.figure(figNum)
                fig, ax = plt.subplots(figsize=(20, 6))
                ax.plot(y,marker='.', linestyle='-', linewidth=0.5, label= columnName)
                plt.title('Values of a column name' + columnName)
                ax.set_ylabel('Values')
                ax.legend()
                figNum = figNum +1 
        
        plt.show()
        return plt
    
    def __plotRange(self, series, true_val, pred):
      if (not len(series)== 1):
        if (true_val.min() < pred.min() and true_val.min() < series.min()):
          min = true_val.min()
        elif(pred.min() < true_val.min() and pred.min() < series.min()):
          min = pred.min()        
        else:
          min = series.min()
          
        if (true_val.max() > pred.max() and true_val.max() > series.max()):
          max = true_val.max()
        elif(pred.max() > true_val.max() and pred.max() > series.max()):
          max = pred.max()
        else:
          max = series.max()
      
      else:
        if (true_val.min() < pred.min()):
          min = true_val.min()
        else:
          min = pred.min()
            
        if (true_val.max() > pred.max()):
          max = true_val.max()
        else:
          max = pred.max()
        
      return min-0.15, max+0.15


    def plotForecasts2(self, y_true, y_pred, init_time):
        plt.plot(y_true, label='Actual')
        plt.plot(y_pred, label='Predicted')
        min, max = self.__plotRange(np.array([0]), np.array(y_true), np.array(y_pred))
        plt.ylim([min, max])
        plt.title('Target and predicted values' + str(init_time))
        plt.legend()
        plt.show()
        return plt

    def plotForecasts(self, series,  n_test, forecasts, init_time):
        # plot the entire dataset in blue
        min, max = self.__plotRange(np.array(series), np.array(n_test), np.array(forecasts))
        plt.ylim([min, max])
        plt.plot(series, label='Historical')
        off_s = len(series) 
        off_e = len(series) + len(forecasts) 
        xaxis = [x for x in range(off_s, off_e)]
        plt.plot(xaxis, forecasts, color = "red", label='Predicted')
        plt.plot(xaxis, n_test, color = "green", label= "Real Values")
        plt.title('Historic, target and predicted values' + str(init_time))
        plt.legend()
        plt.show()
        return plt

    def lossPlotEpoch(self,model_directory, epoch):
        EPOCH = epoch # number of epochs the model has trained for

        history_dataframe = pd.read_csv(model_directory+'/model_history.csv',sep=',')


        # Plot training & validation loss values
        plt.style.use("ggplot")
        plt.plot(range(1,EPOCH+1),
                history_dataframe['loss'])
        plt.plot(range(1,EPOCH+1),
                history_dataframe['val_loss'],
                linestyle='--')
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()
        return plt

    
    def save_figure(self, name_plot, plot):
      path = os.path.join(self.name_model, name_plot)
      plot.savefig(path)


###################################################################
###################################################################
###################################################################
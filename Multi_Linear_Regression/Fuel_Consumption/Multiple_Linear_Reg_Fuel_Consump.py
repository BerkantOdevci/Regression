#######################################################################
#   Lineer Regression For Fuel Consumption in 2014
#######################################################################
#
#   @Class Name(s): LinearRegressionProcess
#
#   @Description:   Analysis of fuel consumption of vehicles according to data obtained in 2014
#
#
#   @Note:  It is used to analyze the dataset and predict the output of the next model.
#
#   Version 0.0.1:  LinearRegressionProcessclass
#                   dataframe_read(self)
#                   visualize_features(self)
#                   train_test_split(self)
#                   linear_regression(self)
#                   evaluation(self)
#                   28 KASIM 2022 Pztesi, 16:30 - Hasan Berkant Ödevci
#
#
#
#   @Author(s): Hasan Berkant Ödevci
#
#   @Mail(s):   berkanttodevci@gmail.com
#
#   28 KASIM 2022 Pztesi, 16:30 tarihinde oluşturuldu.
#
#
########################################################################

# Libraries
try:
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import pylab as pl
    from sklearn import linear_model
    from sklearn.metrics import r2_score #Evaluation such as MSE
except ImportError:
    print("Please Check Library")

class MultipleLinearRegression():
    def __init__(self):
        self.path = "D:/Python_Proje/Machine_Learning/Datasets/FuelComsumption/FuelConsumption.csv"

    def dataframe_read(self):
        self.dataframe = pd.read_csv(self.path, decimal=',' , delimiter=';')
        self.cdf = self.dataframe[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY','FUELCONSUMPTION_COMB','CO2EMISSIONS']]

    def multiple_linear_regression(self):
        self.msk = np.random.rand(len(self.dataframe)) < 0.8
        self.train = self.cdf[self.msk]
        self.test = self.cdf[~self.msk]

        self.regression = linear_model.LinearRegression()
        x = np.asanyarray(self.train[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','FUELCONSUMPTION_CITY']])
        y = np.asanyarray(self.train[['CO2EMISSIONS']])
        self.regression.fit (x, y)

        # The coefficients
        print ('Coefficients: ', self.regression.coef_)

    def prediction(self):
        y_hat= self.regression.predict(self.test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','FUELCONSUMPTION_CITY']])
        x = np.asanyarray(self.test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','FUELCONSUMPTION_CITY']])
        y = np.asanyarray(self.test[['CO2EMISSIONS']])
        print("Residual sum of squares: %.2f"
         % np.mean((y_hat - y) ** 2))

        # Explained variance score: 1 is perfect prediction
        print('Variance score: %.2f' % self.regression.score(x, y))


# Assign Class to variable
multi_linear_reg = MultipleLinearRegression()

# Access Function

multi_linear_reg.dataframe_read()

multi_linear_reg.multiple_linear_regression()

multi_linear_reg.prediction()
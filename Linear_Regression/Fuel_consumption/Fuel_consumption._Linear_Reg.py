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
#   Version 0.0.1:  ExtractFrameProcessclass
#                   VideoLoad(self)
#                   MakeFolder(self)
#                   InitialiseParameter(self)
#                   VideoClose(self)
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
    from sklearn import linear_model
    from sklearn.metrics import r2_score #Evaluation such as MSE
except ImportError:
    print("Please Check Library")

class LinearRegression():
    def __init__(self):
        self.path = "D:/Python_Proje/Machine_Learning/Datasets/FuelComsumption/FuelConsumption.csv"

    def dataframe_read(self):
        self.dataframe = pd.read_csv(self.path, decimal=",", delimiter=";")
        # Not: Decimal işlevi bizim ondalık sayıları virgülle ayıran yerleri "." şeklinde ifade etmemizi sağlıyor. 
        # Delimiter ise veri tablosundaki birleşik veri dizilerini ; ayrılmış olanları düzeltmemize yarıyor.
        describe_df = self.dataframe.describe()
        #self.dataframe = self.dataframe.head(9)
        self.cdf = self.dataframe[["ENGINESIZE","CYLINDERS","FUELCONSUMPTION_COMB","CO2EMISSIONS"]]
    

    def visualize_features(self):
        # Visualize and Plot FuelConsumption_Comb vs Co2Emissions
        plt.scatter(self.cdf.FUELCONSUMPTION_COMB,self.cdf.CO2EMISSIONS,color='blue')
        plt.xlabel("Fuel_Consumption")
        plt.ylabel("Co2Emission")
        plt.show()

        # Visualize and Plot EngineSize vs Co2Emissions
        plt.scatter(self.cdf.ENGINESIZE, self.cdf.CO2EMISSIONS, color= 'red')
        plt.xlabel("Engine_Size")
        plt.ylabel("Co2Emission")
        #plt.show()

        plt.scatter(self.cdf.CYLINDERS,self.cdf.CO2EMISSIONS,color='green')
        plt.xlabel("Cylinders")
        plt.ylabel("Co2Emission")
        #plt.show()

    def train_test_split(self):
        # Create patch that the dimension of train set is lower than %80 percent. Other part is test set which is %20 percent.
        self.msk = np.random.rand(len(self.dataframe)) < 0.8
        self.train = self.cdf[self.msk]
        self.test = self.cdf[~self.msk]

        plt.scatter(self.train.FUELCONSUMPTION_COMB, self.train.CO2EMISSIONS, color = 'blue')
        plt.xlabel("FUELCONSUMPTION_COMB")
        plt.ylabel('Co2Emissions')
        plt.show()

    def linear_regression(self):
        # Linear Regression modelini çağırıyorum
        self.regression = linear_model.LinearRegression()
        
        # Train edilecek verileri array haline getiriyorum
        train_x = np.asanyarray(self.train[['FUELCONSUMPTION_COMB']], dtype=float)   # asanyarray() işlevi, girişi bir diziye dönüştürmek istediğimizde kullanılır. 
        train_y = np.asanyarray(self.train[['CO2EMISSIONS']], dtype=float)     # Ancak ndarray alt sınıflarını . Girdiler skalerler, listeler, demet listeleri, demetler, demetler demetleri, listeler demetleri ve ndarray'ler olabilir

        # Train edilecek verileri linear regression ile fit hale getiriyorum.
        self.regression.fit(train_x,train_y)

        print("The coefficiet: ", self.regression.coef_)
        print("The intercept: ", self.regression.intercept_)
        # Lineer regresyon fonksiyonumuzun denklemi y = ax+b şeklindedir. Tek bir bağımsız değişkene bağlıdır. Biz train_y ve train_x arraylerimizi
        # regression.fit fonksiyonuna sokarak a ve b değerlerini buluyoruz.

        plt.scatter(self.train.FUELCONSUMPTION_COMB, self.train.CO2EMISSIONS, color = 'blue') 
        plt.plot(train_x,self.regression.coef_[0][0]*train_x + self.regression.intercept_[0], '-r')
        plt.xlabel("FUELCONSUMPTION_COMB")
        plt.ylabel("Co2Emission")
        plt.show()

    def evaluation(self):
        # We compare the actual values and predicted values to calculate the accuracy of a regression model. 
        # Evaluation metrics provide a key role in the development of a model, as it provides insight to areas that require improvement.
        test_x = np.asanyarray(self.train[['FUELCONSUMPTION_COMB']])
        test_y = np.asanyarray(self.train[['CO2EMISSIONS']])
        test_y_ = self.regression.predict(test_x)

        print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
        print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))
        print("R2-score: %.2f" % r2_score(test_y , test_y_) )


# Assign Class Variable
linear_regression = LinearRegression()

# Attribute Definitions
linear_regression.dataframe_read()

linear_regression.visualize_features()

linear_regression.train_test_split()

linear_regression.linear_regression()

linear_regression.evaluation()

        




